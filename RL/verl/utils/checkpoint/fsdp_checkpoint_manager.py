# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from typing import Optional, Union

import psutil
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager
import time

class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def _log_memory(self, tag: str):
        """Log current CPU and GPU memory usage."""
        proc = psutil.Process()
        cpu_gb = proc.memory_info().rss / (1024 ** 3)
        msg = f"[rank-{self.rank}][{tag}] CPU RSS: {cpu_gb:.2f} GB"
        if torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            msg += f" | GPU allocated: {gpu_alloc_gb:.2f} GB, reserved: {gpu_reserved_gb:.2f} GB"
        print(msg)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # Auto-detect checkpoint format: DCP sharded vs legacy per-rank .pt
        model_dcp_dir = os.path.join(path, "model")
        if os.path.isdir(model_dcp_dir):
            self._load_dcp_checkpoint(path)
        else:
            self._load_legacy_checkpoint(path)

    def _load_dcp_checkpoint(self, path: str):
        """Load from DCP sharded checkpoint (memory-efficient, supports resharding)."""
        model_ckpt_dir = os.path.join(path, "model")
        optim_ckpt_dir = os.path.join(path, "optim")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        sharded_options = StateDictOptions(full_state_dict=False, cpu_offload=True)

        # Load model: get sharded template, fill via DCP, set into model, then free
        self._log_memory("before dcp load model")
        model_state_dict = get_model_state_dict(self.model, options=sharded_options)
        print(f"[rank-{self.rank}]: Loading sharded model from {os.path.abspath(model_ckpt_dir)}.")
        dcp.load(model_state_dict, checkpoint_id=model_ckpt_dir)
        set_model_state_dict(self.model, model_state_dict=model_state_dict, options=sharded_options)
        del model_state_dict
        gc.collect()
        self._log_memory("after dcp load model")

        # Load optimizer: same pattern
        self._log_memory("before dcp load optim")
        optim_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=sharded_options)
        print(f"[rank-{self.rank}]: Loading sharded optimizer from {os.path.abspath(optim_ckpt_dir)}.")
        dcp.load(optim_state_dict, checkpoint_id=optim_ckpt_dir)
        set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=optim_state_dict, options=sharded_options)
        del optim_state_dict
        gc.collect()
        self._log_memory("after dcp load optim")

        # Load extra state (small, per-rank)
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        extra_state_dict = torch.load(extra_path, weights_only=False)
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def _load_legacy_checkpoint(self, path: str):
        """Load from legacy per-rank .pt checkpoint (backward compatibility)."""
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str):
        """Save checkpoint using DCP sharded format.

        Each rank saves only its own FSDP shard directly from GPU to disk,
        avoiding the CPU OOM caused by gathering the full state dict.

        Directory layout::

            path/
              model/          # DCP sharded model checkpoint
              optim/          # DCP sharded optimizer checkpoint
              extra_state_world_size_*_rank_*.pt
              huggingface/    # config + tokenizer (rank 0 only)

        To gather into a full HuggingFace model later, see
        ``gather_sharded_to_hf()`` or use ``dcp.load`` with
        ``full_state_dict=True`` on a single machine.
        """
        path = self.local_mkdir(path)
        dist.barrier()
        print(f"[CheckpointManager/rank-{self.rank}]")

        model_ckpt_dir = os.path.join(path, "model")
        optim_ckpt_dir = os.path.join(path, "optim")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        # Keep sharded DTensors on GPU — no full gather, no CPU copy
        sharded_options = StateDictOptions(full_state_dict=False, cpu_offload=False)

        # ---- model ----
        self._log_memory("before get_model_state_dict (sharded)")
        model_state_dict = get_model_state_dict(self.model, options=sharded_options)
        self._log_memory("after get_model_state_dict (sharded)")
        print(f"[rank-{self.rank}]: Saving sharded model to {os.path.abspath(model_ckpt_dir)}.")
        dcp.save(model_state_dict, checkpoint_id=model_ckpt_dir)
        self._log_memory("after dcp.save model")
        del model_state_dict
        gc.collect()
        self._log_memory("after del model_state_dict + gc")

        # ---- optimizer ----
        self._log_memory("before get_optimizer_state_dict (sharded)")
        optim_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=sharded_options)
        self._log_memory("after get_optimizer_state_dict (sharded)")
        print(f"[rank-{self.rank}]: Saving sharded optimizer to {os.path.abspath(optim_ckpt_dir)}.")
        dcp.save(optim_state_dict, checkpoint_id=optim_ckpt_dir)
        self._log_memory("after dcp.save optim")
        del optim_state_dict
        gc.collect()
        self._log_memory("after del optim_state_dict + gc")

        # ---- extra state (small, per-rank) ----
        print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
        extra_state_dict = {
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "rng": self.get_rng_state(),
        }
        torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            assert isinstance(self.model._fsdp_wrapped_module, PreTrainedModel)
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_path)
            self.model._fsdp_wrapped_module.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        dist.barrier()

    # ------------------------------------------------------------------
    # Offline gather: convert sharded DCP checkpoint → HuggingFace model
    # ------------------------------------------------------------------
    @staticmethod
    def gather_sharded_to_hf(ckpt_path: str, output_path: str):
        """Gather a DCP sharded checkpoint into a full HuggingFace model.

        Run **offline** on a single machine with enough CPU RAM (no GPU needed)::

            python -c "
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            FSDPCheckpointManager.gather_sharded_to_hf('path/to/ckpt', 'path/to/hf_output')
            "

        This only needs enough CPU RAM to hold one full copy of the model.
        """
        import torch.distributed.checkpoint as _dcp
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

        hf_cfg_path = os.path.join(ckpt_path, "huggingface")
        model_ckpt_dir = os.path.join(ckpt_path, "model")

        print(f"Loading config from {hf_cfg_path}")
        config = AutoConfig.from_pretrained(hf_cfg_path)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

        print(f"Loading sharded model weights from {model_ckpt_dir}")
        _dcp.load(model.state_dict(), checkpoint_id=model_ckpt_dir)
        # The loaded state dict is modified in-place; set it back
        model.load_state_dict(model.state_dict())

        os.makedirs(output_path, exist_ok=True)
        print(f"Saving full HuggingFace model to {output_path}")
        model.save_pretrained(output_path)

        # Copy tokenizer / processor
        try:
            tok = AutoTokenizer.from_pretrained(hf_cfg_path)
            tok.save_pretrained(output_path)
        except Exception:
            pass  # tokenizer may not exist

        print("Done.")

