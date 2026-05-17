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

import os
from contextlib import contextmanager
import pdb
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.tokenizer import get_processor
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
from tools.actors import StepHashServer, SampleHashServer
from verl.workers.reward.function import FunctionRuleBasedJudgeManager
import ray
import re


# monet
from monet_models.vllm.latent_recorder import LatentRecorder
import os, json, shutil, tempfile, pathlib

# Key in vLLM ``SamplingParams.extra_args`` for per-request latent block length; see
# https://docs.vllm.ai/en/latest/api/vllm/sampling_params/ — GPUModelRunner reads this
# (with legacy ``sampling_params.monet_latent_size`` fallback) for new requests.
_MONET_LATENT_KEY = "monet_latent_size"
# Sentinel: key was not present in ``extra_args`` before this generate.
_MONET_LATENT_SIZE_MISSING = object()


def _set_monet_latent_size_for_generate(
    sampling_params: Any, ovr: Optional[int]
) -> tuple[Any, bool]:
    """
    Merge :data:`_MONET_LATENT_KEY` into vLLM ``SamplingParams.extra_args`` for this
    ``generate`` call. ``extra_args`` is the supported extension point; it is carried
    with each scheduled request to ``GPUModelRunner`` without using ``LATENT_SIZE`` env.
    """
    if ovr is None:
        return _MONET_LATENT_SIZE_MISSING, False
    val = int(ovr)
    ea = getattr(sampling_params, "extra_args", None)
    if ea is not None and not isinstance(ea, dict):
        ea = dict(ea)
    prev = (ea or {}).get(_MONET_LATENT_KEY, _MONET_LATENT_SIZE_MISSING)
    new_ea = {**(ea or {}), _MONET_LATENT_KEY: val}
    try:
        sampling_params.extra_args = new_ea
    except Exception:
        try:
            object.__setattr__(sampling_params, "extra_args", new_ea)
        except Exception as e2:
            raise type(e2)(
                f"Could not set sampling_params.extra_args[{_MONET_LATENT_KEY!r}]={val}. "
                f"Check vLLM SamplingParams on {type(sampling_params)}."
            ) from e2
    return prev, True


def _restore_monet_latent_size_after_generate(
    sampling_params: Any, old: Any, was_set: bool
) -> None:
    if not was_set:
        return
    ea = getattr(sampling_params, "extra_args", None)
    if ea is not None and not isinstance(ea, dict):
        ea = dict(ea)
    new_ea = {**(ea or {})}
    if old is _MONET_LATENT_SIZE_MISSING:
        new_ea.pop(_MONET_LATENT_KEY, None)
    else:
        new_ea[_MONET_LATENT_KEY] = old
    out: Optional[Dict[str, Any]] = new_ea if new_ea else None
    try:
        sampling_params.extra_args = out
    except Exception:
        object.__setattr__(sampling_params, "extra_args", out)


def _per_row_token_lists_to_1d_object_array(raw) -> np.ndarray:
    """
    One list (or 1D int segment) of token ids per batch row, shape (B,) dtype=object.

    Dataloader/collate often already provides ``(B,)`` ``dtype=object`` with
    element type ``list``; keep that layout with a view-safe copy.

    Do not use ``np.array([...], dtype=object)`` on *plain lists* of equal-length
    inner lists: NumPy can still form a 2D (B, L) array, which then disagrees
    with other ranks' (B,) object layout in ``DataProto.concat``.
    """
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1 and raw.dtype == object:
            return raw.copy()
        if raw.ndim == 2 and np.issubdtype(raw.dtype, np.integer):
            out = np.empty(raw.shape[0], dtype=object)
            for i in range(int(raw.shape[0])):
                out[i] = raw[i].tolist()
            return out
    rows = [list(x) for x in raw]
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        out[i] = r
    return out


def _is_qwen2vl_processor(processor: Any) -> bool:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return False
    return ip.__class__.__name__ in ("Qwen2VLImageProcessor", "Qwen2_5_VLImageProcessor")


def _prompt_positions_via_rope_index(
    processor: Any,
    raw_list: List[int],
    mm_inputs_row: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    mrope position_ids for the same token sequence vLLM uses (``raw_prompt_ids``), using
    the same ``get_rope_index`` path as the dataloader. Required when ``input_ids`` are
    processor-expanded and cannot be matched token-for-token to ``raw_prompt_ids``.
    """
    if processor is None or not _is_qwen2vl_processor(processor):
        return None
    try:
        from verl.models.transformers.qwen2_vl import get_rope_index
    except ImportError:
        return None

    raw_tensor = torch.tensor(raw_list, dtype=torch.long, device=device)
    attn = torch.ones_like(raw_tensor)
    image_grid_thw = None
    video_grid_thw = None
    second_per_grid_ts = None
    if mm_inputs_row:
        ig = mm_inputs_row.get("image_grid_thw")
        if isinstance(ig, torch.Tensor):
            image_grid_thw = ig.to(device=device)
        vg = mm_inputs_row.get("video_grid_thw")
        if isinstance(vg, torch.Tensor):
            video_grid_thw = vg.to(device=device)
        sp = mm_inputs_row.get("second_per_grid_ts")
        if isinstance(sp, torch.Tensor):
            second_per_grid_ts = sp.to(device=device)
    try:
        pos = get_rope_index(
            processor,
            raw_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attn,
        )
    except Exception:
        return None
    if pos.dim() != 2 or pos.size(0) != 3 or pos.size(-1) != len(raw_list):
        return None
    return pos.detach().cpu().clone()


def _fallback_prompt_positions_non_mrope(L: int) -> torch.Tensor:
    """1D positions ``0..L-1`` (CPU), same style as non-mrope dataloader path; ``extend_position_ids`` accepts this."""
    return torch.arange(L, dtype=torch.long)


def _snapshot_row_for_latent_necessity(
    device: torch.device,
    raw_prompt_ids: Any,
    multi_modal_data: Any,
    processor: Optional[Any] = None,
    mm_inputs_row: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Store vLLM rollout inputs for latent necessity: ``raw_prompt_ids`` (exact vLLM prompt),
    ``multi_modal_data``, and unpadded position_ids for that **raw** sequence only.

    Positions come from ``get_rope_index`` for Qwen2-VL / Qwen2.5-VL (plus ``multi_modal_inputs``
    grids when present). No padded ``input_ids`` / alignment is used.
    """
    raw_list = list(raw_prompt_ids)
    L = len(raw_list)
    if L <= 0:
        return None

    if processor is not None and _is_qwen2vl_processor(processor):
        pos_u = _prompt_positions_via_rope_index(processor, raw_list, mm_inputs_row, device)
        if pos_u is None:
            return None
    else:
        pos_u = _fallback_prompt_positions_non_mrope(L)

    return {
        "prompt_token_ids": raw_list,
        "multi_modal_data": multi_modal_data,
        "prompt_position_ids_unpadded": pos_u.numpy(),
    }


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None



def remove_text_config_inplace(path: str) -> bool:
    """Remove the entire 'text_config' field from config.json on disk.
    - `path` can be a directory containing config.json or a direct path to config.json.
    - Only the 'text_config' key is removed; nothing else is changed.
    - Returns True if a write happened, False if 'text_config' did not exist.
    """
    # Resolve config.json path
    cfg_path = path
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found at: {cfg_path}")

    # Load current JSON
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # If no text_config, nothing to do
    if "text_config" not in cfg:
        return False

    # Remove the entire text_config section
    del cfg["text_config"]

    # Atomic write-back to avoid partial writes
    dir_name = os.path.dirname(cfg_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=dir_name) as tmp:
        json.dump(cfg, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name

    os.replace(tmp_name, cfg_path)
    return True


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer, processor:  Optional[ProcessorMixin], 
                 hash_server: Optional[Union[StepHashServer, SampleHashServer]] = None,
                 rule_based_judge_server: Optional[FunctionRuleBasedJudgeManager] = None,
                 embed_model = None,
                 embed_tokenizer = None):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.processor = processor
        self.tokenizer = tokenizer
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        #print("config.prompt_length: ", config.prompt_length)
        #print("config.response_length: ", config.response_length)
        #print("config.gpu_memory_utilization", config.gpu_memory_utilization)
        #model_for_vllm = _make_vllm_shadow_model_dir(model_path)

        remove_text_config_inplace(model_path)

        self.inference_engine = LLM(
            model=model_path,
            #tokenizer=model_for_vllm,
            #tokenizer_mode="mmap",
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="auto",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={"image": config.limit_images},
            #disable_mm_preprocessor_cache=True,
            #mm_processor_cache_gb=config.mm_processor_cache_gb,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        self.hash_server = hash_server
        self.rule_based_judge_server = rule_based_judge_server

        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer

        self.latent_size = int(os.getenv("ABS_VIS_LATENT_SIZE", '0'))

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        plain = bool(prompts.meta_info.get("vllm_plain_generate", False))
        batch_sample_idx = None
        if self.config.sampling_strategy in ["monet"] and not plain:
            batch_sample_idx = list(non_tensor_batch.pop("global_index"))

        precomputed_arr = non_tensor_batch.pop("precomputed_vllm_inputs", None)
        if precomputed_arr is not None:
            if len(precomputed_arr) != batch_size:
                raise RuntimeError(
                    f"vllm precomputed_vllm_inputs length {len(precomputed_arr)} != batch_size {batch_size}"
                )
            vllm_inputs = []
            for i in range(batch_size):
                d = precomputed_arr[i]
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(d["prompt_token_ids"]),
                        "multi_modal_data": d.get("multi_modal_data"),
                    }
                )
        else:
            if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                raise RuntimeError("vllm sharding manager is not work properly.")

            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
                ):
                    vllm_inputs.append(
                        {"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data}
                    )
            else:
                vllm_inputs = [
                    {"prompt_token_ids": list(raw_prompt_ids)}
                    for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                ]

        # users can customize different sampling_params at different run
        batch_min_mean_correct_resp_lens = []

        with self.update_sampling_params(**prompts.meta_info):
            ovr = prompts.meta_info.get("override_latent_size")
            old_m, did_m = _set_monet_latent_size_for_generate(self.sampling_params, ovr)
            try:
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(False and self.rank == 0)
                )
            finally:
                _restore_monet_latent_size_after_generate(self.sampling_params, old_m, did_m)
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            #breakpoint()

            
            if self.config.sampling_strategy in ["monet"] and not plain and batch_sample_idx is not None:
                response_ids = []
                for completion, global_id in zip(completions, batch_sample_idx):
                    for output in completion.outputs:
                        response_ids.append(output.token_ids)
                    min_len, mean_len = ray.get(self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id))
                    batch_min_mean_correct_resp_lens.extend([min_len] * self.sampling_params.n if min_len<float("inf") else [mean_len] * self.sampling_params.n)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                # value = torch.tensor([1, 2, 3])
                # value = value.repeat_interleave(2, dim=0)
                # output: tensor([1, 1, 2, 2, 3, 3])
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        if self.config.sampling_strategy in ["monet"] and not plain:
            non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    @torch.no_grad()
    def generate_sequences_monet(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        gts = list(non_tensor_batch["ground_truth"])
        questions = list(non_tensor_batch["problem"])
        orig_raw_prompt_ids = None
        orig_multi_modal_data = None
        if "raw_prompt_ids" in non_tensor_batch:
            orig_raw_prompt_ids = _per_row_token_lists_to_1d_object_array(
                non_tensor_batch["raw_prompt_ids"]
            )
        if "multi_modal_data" in non_tensor_batch:
            orig_multi_modal_data = np.array(non_tensor_batch["multi_modal_data"], dtype=object)
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # Always capture vLLM prompt snapshots for latent-necessity metrics / dumps; training
        # only adds Rn to rewards when ``latent_necessity_reward`` is enabled (driver).
        store_ln_snapshots = True
        ln_snapshots: Optional[List[Optional[Dict[str, Any]]]] = None
        if store_ln_snapshots:
            raws = non_tensor_batch["raw_prompt_ids"]
            if "multi_modal_data" in non_tensor_batch:
                mm_iter = non_tensor_batch["multi_modal_data"]
            else:
                mm_iter = [None] * batch_size
            mm_inputs_arr = non_tensor_batch.get("multi_modal_inputs")
            ln_snapshots = []
            for i in range(batch_size):
                mm_inputs_row = None
                if mm_inputs_arr is not None and i < len(mm_inputs_arr):
                    mm_inputs_row = mm_inputs_arr[i]
                    if mm_inputs_row is not None and not isinstance(mm_inputs_row, dict):
                        mm_inputs_row = None
                ln_snapshots.append(
                    _snapshot_row_for_latent_necessity(
                        input_ids[i].device,
                        raws[i],
                        mm_iter[i],
                        self.processor,
                        mm_inputs_row,
                    )
                )

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        batch_min_mean_correct_resp_lens = []

        with self.update_sampling_params(**prompts.meta_info):
            ovr = prompts.meta_info.get("override_latent_size")
            old_m, did_m = _set_monet_latent_size_for_generate(self.sampling_params, ovr)
            with LatentRecorder(set_env=True, prefer_tcp=True, filter_rank=self.rank) as rec:
                try:
                    completions: List[RequestOutput] = self.inference_engine.generate(
                        prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(False and self.rank == 0)
                    )
                finally:
                    _restore_monet_latent_size_after_generate(self.sampling_params, old_m, did_m)

            min_req_id = 99999
            for completion in completions:
                min_req_id = min(min_req_id, int(completion.request_id))
            
            non_tensor_batch['latents'] = rec.to_object_array_auto(bsz=batch_size, rollout_n=self.sampling_params.n, min_req_id=min_req_id)

            #if self.rank == 0:
            #    pdb.set_trace()
            #breakpoint()
            response_ids = []
            for completion, global_id in zip(completions, batch_sample_idx):
                for output in completion.outputs:
                    response_ids.append(output.token_ids)
                min_len, mean_len = ray.get(self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id))
                batch_min_mean_correct_resp_lens.extend([min_len] * self.sampling_params.n if min_len<float("inf") else [mean_len] * self.sampling_params.n)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)


            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                # value = torch.tensor([1, 2, 3])
                # value = value.repeat_interleave(2, dim=0)
                # output: tensor([1, 1, 2, 2, 3, 3])
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            if orig_raw_prompt_ids is not None:
                o = orig_raw_prompt_ids
                if self.sampling_params.n > 1:
                    o = np.repeat(o, self.sampling_params.n, axis=0)
                non_tensor_batch["orig_raw_prompt_ids"] = o
            if orig_multi_modal_data is not None:
                omm = orig_multi_modal_data
                if self.sampling_params.n > 1:
                    omm = np.repeat(omm, self.sampling_params.n, axis=0)
                non_tensor_batch["orig_multi_modal_data"] = omm

        if store_ln_snapshots and ln_snapshots is not None:
            orig_bsz_ln = len(ln_snapshots)
            sn_arr = np.empty(orig_bsz_ln, dtype=object)
            for _i, _s in enumerate(ln_snapshots):
                sn_arr[_i] = _s
            if self.sampling_params.n > 1:
                sn_arr = np.repeat(sn_arr, self.sampling_params.n, axis=0)
            non_tensor_batch["monet_rollout_vllm_input"] = sn_arr

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens) 
        non_tensor_batch["ground_truth"] = _repeat_interleave(np.array(gts), self.sampling_params.n)
        non_tensor_batch["problem"] = _repeat_interleave(np.array(questions), self.sampling_params.n)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    

