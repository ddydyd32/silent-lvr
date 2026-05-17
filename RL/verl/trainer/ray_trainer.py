import os
import pdb
import uuid
import json
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, Type, Set

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
import torch.distributed as dist
import sys
import glob
import shutil
from transformers import AutoProcessor
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager, FunctionRuleBasedJudgeManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from tools.api_judge import api_batch_judge
from tools.custom_api import build_deepseek_client, build_gemini_client, build_vllm_client

import random
from torch.utils.data import Dataset
from ..utils.dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
#from tools.compute_embeds import compute_embeds_fn
from tools.actors import StepHashServer, SampleHashServer
from tools.actors import EmbedServer
import matplotlib.pyplot as plt
import re

def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'<abs_vis_token>.*?</abs_vis_token>', flags=re.DOTALL)
    s = pattern.sub('', s).strip()
    pattern = re.compile(r'<\|lvr_start\|>.*?<\|lvr_end\|>', flags=re.DOTALL)
    s = pattern.sub('', s).strip()
    return s

def remove_latent_from_text(s: str) -> str:
    pattern = re.compile(r'<abs_vis_token>.*?</abs_vis_token>', flags=re.DOTALL)
    s = pattern.sub('', s).strip()
    pattern = re.compile(r'<\|lvr_start\|>.*?<\|lvr_end\|>', flags=re.DOTALL)
    s = pattern.sub('', s).strip()
    return s

def strip_latent_tokens_from_dataproto(dp, start_id=None, end_id=None):
    """
    Create a new DataProto (single sample, batch_size=1) with latent tokens
    (from start_id to end_id, inclusive) removed from the response portion.
    Non-latent tokens are compacted left and the tail is zero-padded.
    """
    from tensordict import TensorDict

    if start_id is None:
        start_id = int(os.environ.get('ABS_VIS_START_ID', '151666'))
    if end_id is None:
        end_id = int(os.environ.get('ABS_VIS_END_ID', '151667'))

    responses = dp.batch['responses'][0].clone()      # (resp_len,)
    response_mask = dp.batch['response_mask'][0].clone()  # (resp_len,)
    input_ids = dp.batch['input_ids'][0].clone()       # (total_len,)
    attention_mask = dp.batch['attention_mask'][0].clone()  # (total_len,)

    resp_len = responses.shape[0]
    total_len = input_ids.shape[0]
    prompt_len = total_len - resp_len

    # Build boolean mask: True for positions inside <vis_start> ... <vis_end>
    in_latent = torch.zeros(resp_len, dtype=torch.bool, device=responses.device)
    inside = False
    for j in range(resp_len):
        tok = responses[j].item()
        if tok == start_id:
            inside = True
        if inside:
            in_latent[j] = True
        if tok == end_id and inside:
            inside = False

    if not in_latent.any():
        return dp  # nothing to strip

    keep_mask = ~in_latent
    n_keep = keep_mask.sum().item()

    # --- responses ---
    new_responses = torch.zeros_like(responses)
    new_responses[:n_keep] = responses[keep_mask]

    # --- response_mask ---
    new_response_mask = torch.zeros_like(response_mask)
    new_response_mask[:n_keep] = response_mask[keep_mask]

    # --- input_ids (only the response tail changes) ---
    new_input_ids = input_ids.clone()
    new_input_ids[prompt_len:] = new_responses

    # --- attention_mask ---
    new_attention_mask = attention_mask.clone()
    resp_attn = attention_mask[prompt_len:]
    new_resp_attn = torch.zeros_like(resp_attn)
    new_resp_attn[:n_keep] = resp_attn[keep_mask]
    new_attention_mask[prompt_len:] = new_resp_attn

    # Build new tensor dict from all keys, replacing the four we modified
    new_tensors = {}
    for key in dp.batch.keys():
        if key == 'responses':
            new_tensors[key] = new_responses.unsqueeze(0)
        elif key == 'response_mask':
            new_tensors[key] = new_response_mask.unsqueeze(0)
        elif key == 'input_ids':
            new_tensors[key] = new_input_ids.unsqueeze(0)
        elif key == 'attention_mask':
            new_tensors[key] = new_attention_mask.unsqueeze(0)
        elif key == 'position_ids':
            pos = dp.batch['position_ids'][0].clone()
            if pos.dim() == 1:  # (total_len,)
                # recompute sequential positions for the response portion
                last_prompt_pos = pos[prompt_len - 1].item() if prompt_len > 0 else -1
                pos[prompt_len:prompt_len + n_keep] = torch.arange(
                    n_keep, dtype=pos.dtype, device=pos.device) + last_prompt_pos + 1
                pos[prompt_len + n_keep:] = 0
            new_tensors[key] = pos.unsqueeze(0)
        else:
            new_tensors[key] = dp.batch[key][0:1].clone()

    # Copy non_tensor_batch
    new_non_tensor = {}
    for key, val in dp.non_tensor_batch.items():
        if isinstance(val, np.ndarray):
            new_non_tensor[key] = val.copy()
        else:
            new_non_tensor[key] = val

    new_td = TensorDict(source=new_tensors, batch_size=[1])
    return DataProto(batch=new_td, non_tensor_batch=new_non_tensor,
                     meta_info=dp.meta_info.copy() if dp.meta_info else {})


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)
    data.batch["token_level_kl"] = kld

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(config: PPOConfig, data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, sampling_strategy: str = "greedy") -> DataProto:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if sampling_strategy in ["greedy"]:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
        elif sampling_strategy in ["monet"]:
            advantages, returns = core_algos.compute_grpo_latent_advantage(token_level_rewards, response_mask, index)

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
        rule_based_judge: Optional[FunctionRuleBasedJudgeManager] = None,
        #embed_model: Optional[torch.nn.Module] = None,
        #embed_tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.trank = dist.get_rank() if dist.is_initialized() else 0
        self.has_latent_pre_generate_monet = {'true': 0, 'false': 0}
        self.latent_rate_in_drop_due_to_highacc = []
        self.latent_necessity_reward_n = float(os.environ.get("LATENT_NECESSITY_REWARD_N", "1.0"))
        print(f"[rank {self.trank}] [ray_trainer.py] __init__ latent_necessity_reward_n: {self.latent_necessity_reward_n}")
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.rule_based_judge = rule_based_judge
        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")


        if self.config.data.pr_batch_size != -1:
            if config.data.pr_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.pr_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )

        else:
            if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")
        print('reward_fn:', type(self.reward_fn), self.reward_fn)

        self.base_dataset = self.train_dataloader.dataset
        self.correct_pool = defaultdict(list)
        self.correct_gen_out_pool = defaultdict(list)
        self.selected_sample_statistics = defaultdict(int)

        if self.config.worker.rollout.sampling_strategy in ["monet"]:
            self.sample_hash_server_main = SampleHashServer.options(
                name="sample_hash_server_main"
            ).remote()
            #print("type of self.sample_hash_server_main:", type(self.sample_hash_server_main))
            self.config.worker.rollout.monet.hash_server_name = "sample_hash_server_main"
            ray.get(self.sample_hash_server_main.ping.remote())

        if "api" in self.config.worker.rule_based_judge.judge_function_name:
            if os.environ.get("LLM_JUDGE", "") == "vllm":
                self.client = build_vllm_client()
            elif self.config.worker.rule_based_judge.api_name in ['deepseek-chat', 'deepseek']:
                self.client = build_deepseek_client()
            elif self.config.worker.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.client = build_gemini_client()
            else:
                self.client = None
                raise ValueError(f"API {self.config.worker.rule_based_judge.api_name} not supported.")
        print(f"[rank {self.trank}] [ray_trainer.py] self.client: {self.client}")

        self.post_generate_dump_file = None
        if self.trank == 0:
            exp_root = self.config.trainer.save_checkpoint_path
            exp_name = str(self.config.trainer.experiment_name)
            if os.path.basename(os.path.normpath(exp_root)) != exp_name:
                exp_root = os.path.join(exp_root, exp_name)
            root = os.path.join(exp_root, "post_generate_outputs")
            os.makedirs(root, exist_ok=True)
            self.post_generate_dump_file = os.path.join(root, "attn_reward_stats.jsonl")

    def _maybe_dump_rollout_outputs(self, step: int, gen_batch_output: DataProto) -> None:
        if self.rollout_dump_file is None:
            return

        every_n = max(1, int(self.config.worker.rollout.rollout_output_every_n_steps))
        if step % every_n != 0:
            return

        max_n = max(1, int(self.config.worker.rollout.rollout_output_max_samples_per_step))
        n = min(len(gen_batch_output), max_n)

        responses = gen_batch_output.batch["responses"]
        response_mask = gen_batch_output.batch["response_mask"]
        ntb = gen_batch_output.non_tensor_batch

        def _maybe_get(ntb_dict, key, i, default=None):
            if key not in ntb_dict:
                return default
            try:
                return ntb_dict[key][i]
            except Exception:
                return default

        rows = []
        for i in range(n):
            resp_len = int(response_mask[i].sum().item())
            valid_response_ids = responses[i][:resp_len]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            rows.append(
                {
                    "global_step": int(step),
                    "index": int(i),
                    "problem": _maybe_get(ntb, "problem", i, None),
                    "ground_truth": _maybe_get(ntb, "ground_truth", i, None),
                    "response": response_text,
                    "response_len": resp_len,
                    "global_index": _maybe_get(ntb, "global_index", i, None),
                }
            )

        with open(self.rollout_dump_file, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_correctness_flags(non_tensor_batch: Dict[str, Any], expected_n: int) -> Optional[List[Optional[bool]]]:
        raw = non_tensor_batch.get("correctness", None)
        if raw is None:
            return None

        if isinstance(raw, torch.Tensor):
            vals = raw.detach().cpu().tolist()
        elif isinstance(raw, np.ndarray):
            vals = raw.tolist()
        elif isinstance(raw, (list, tuple)):
            vals = list(raw)
        else:
            return None

        flags: List[Optional[bool]] = []
        for v in vals[:expected_n]:
            if isinstance(v, (bool, np.bool_)):
                flags.append(bool(v))
            elif isinstance(v, (int, float, np.number)):
                flags.append(bool(int(v)))
            elif isinstance(v, str):
                vv = v.strip().lower()
                if vv in {"1", "true", "t", "yes", "y"}:
                    flags.append(True)
                elif vv in {"0", "false", "f", "no", "n"}:
                    flags.append(False)
                else:
                    flags.append(None)
            else:
                flags.append(None)

        if len(flags) < expected_n:
            flags.extend([None] * (expected_n - len(flags)))
        return flags

    @staticmethod
    def _to_object_list(raw: Any, expected_n: int) -> Optional[List[Any]]:
        if raw is None:
            return None

        if isinstance(raw, torch.Tensor):
            vals = raw.detach().cpu().tolist()
        elif isinstance(raw, np.ndarray):
            vals = raw.tolist()
        elif isinstance(raw, (list, tuple)):
            vals = list(raw)
        else:
            vals = [raw]

        vals = vals[:expected_n]
        if len(vals) < expected_n:
            vals.extend([None] * (expected_n - len(vals)))
        return vals

    def _maybe_dump_post_generate_outputs(
        self,
        batch: DataProto,
        attn: Optional[torch.Tensor],
        attn_image: Optional[torch.Tensor],
        timing_raw: Dict[str, Any],
        kl: Optional[torch.Tensor] = None,
        actor_per_sample_pg_loss: Optional[List[Optional[float]]] = None,
        actor_per_sample_total_loss: Optional[List[Optional[float]]] = None,
        ld: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.post_generate_dump_file is None:
            return
        if "token_level_rewards" not in batch.batch:
            return

        tlr = batch.batch["token_level_rewards"].detach().float().cpu()
        tls = batch.batch.get("token_level_scores", None)
        tls_cpu = tls.detach().float().cpu() if tls is not None else None
        seq_rewards = tlr.sum(dim=-1).tolist()
        token_level_rewards = tlr.tolist()
        token_level_scores = tls_cpu.tolist() if tls_cpu is not None else None

        attn_list = attn.detach().float().cpu().tolist() if attn is not None else None
        attn_image_list = attn_image.detach().float().cpu().tolist() if attn_image is not None else None
        kl_cpu = kl.detach().float().cpu() if kl is not None else None
        token_level_kl = kl_cpu.tolist() if kl_cpu is not None else None
        sequence_kl = kl_cpu.sum(dim=-1).tolist() if kl_cpu is not None else None
        bsz = int(tlr.shape[0])

        latent_start_positions = [-1] * bsz
        latent_end_positions = [-1] * bsz
        start_id = int(os.environ.get("ABS_VIS_START_ID", "151666"))
        end_id = int(os.environ.get("ABS_VIS_END_ID", "151667"))
        responses = batch.batch.get("responses", None)
        if responses is not None:
            responses_cpu = responses.detach().cpu()
            for i in range(min(bsz, int(responses_cpu.shape[0]))):
                seq = responses_cpu[i]

                start_hits = (seq == start_id).nonzero(as_tuple=False)
                if start_hits.numel() > 0:
                    start_pos = int(start_hits[0].item())
                    latent_start_positions[i] = start_pos

                    end_after_start = (seq[start_pos:] == end_id).nonzero(as_tuple=False)
                    if end_after_start.numel() > 0:
                        latent_end_positions[i] = start_pos + int(end_after_start[0].item())
                else:
                    end_hits = (seq == end_id).nonzero(as_tuple=False)
                    if end_hits.numel() > 0:
                        latent_end_positions[i] = int(end_hits[0].item())

        correctness_flags = self._extract_correctness_flags(batch.non_tensor_batch, bsz)
        accuracy = None
        attn_correct: List[float] = []
        attn_incorrect: List[float] = []
        attn_image_correct: List[float] = []
        attn_image_incorrect: List[float] = []
        reward_correct: List[float] = []
        reward_incorrect: List[float] = []
        sequence_kl_correct: List[float] = []
        sequence_kl_incorrect: List[float] = []
        if correctness_flags is not None:
            valid = [int(x) for x in correctness_flags if x is not None]
            if valid:
                accuracy = float(sum(valid) / len(valid))

            for i, flag in enumerate(correctness_flags):
                if flag is None:
                    continue
                if flag:
                    reward_correct.append(float(seq_rewards[i]))
                    if attn_list is not None:
                        attn_correct.append(float(attn_list[i]))
                    if attn_image_list is not None:
                        attn_image_correct.append(float(attn_image_list[i]))
                    if sequence_kl is not None:
                        sequence_kl_correct.append(float(sequence_kl[i]))
                else:
                    reward_incorrect.append(float(seq_rewards[i]))
                    if attn_list is not None:
                        attn_incorrect.append(float(attn_list[i]))
                    if attn_image_list is not None:
                        attn_image_incorrect.append(float(attn_image_list[i]))
                    if sequence_kl is not None:
                        sequence_kl_incorrect.append(float(sequence_kl[i]))

        row = {
            "timestamp": int(time.time()),
            "time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "global_step": int(self.global_step),
            "experiment_name": str(self.config.trainer.experiment_name),
            "batch_size": bsz,
            "timing_raw": {k: float(v) for k, v in timing_raw.items() if isinstance(v, (int, float, np.number))},
            "accuracy": accuracy,
            "correctness": correctness_flags,
            "attn": attn_list,
            "attn_image": attn_image_list,
            "attn_correct": attn_correct,
            "attn_incorrect": attn_incorrect,
            "attn_image_correct": attn_image_correct,
            "attn_image_incorrect": attn_image_incorrect,
            "token_level_kl": token_level_kl,
            "sequence_kl": sequence_kl,
            "sequence_kl_correct": sequence_kl_correct,
            "sequence_kl_incorrect": sequence_kl_incorrect,
            "token_level_rewards": token_level_rewards,
            "token_level_scores": token_level_scores,
            "sequence_rewards": seq_rewards,
            "sequence_rewards_correct": reward_correct,
            "sequence_rewards_incorrect": reward_incorrect,
            "actor_per_sample_pg_loss": actor_per_sample_pg_loss,
            "actor_per_sample_total_loss": actor_per_sample_total_loss,
            "latent_start_positions": latent_start_positions,
            "latent_end_positions": latent_end_positions,
        }
        if ld is not None:
            row["latent_necessity_judge_mode"] = ld.get("latent_necessity_judge_mode")
            row["latent_necessity_rn"] = ld.get("latent_necessity_rn", [None] * bsz)
            row["latent_necessity_a0"] = ld.get("latent_necessity_a0", [None] * bsz)
            row["latent_necessity_a1"] = ld.get("latent_necessity_a1", [None] * bsz)
            row["latent_necessity_applied_to_rewards"] = ld.get("latent_necessity_applied_to_rewards")
        else:
            row["latent_necessity_judge_mode"] = None
            row["latent_necessity_rn"] = [None] * bsz
            row["latent_necessity_a0"] = [None] * bsz
            row["latent_necessity_a1"] = [None] * bsz
            row["latent_necessity_applied_to_rewards"] = None

        with open(self.post_generate_dump_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:

            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index"],
                    # non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index", "ground_truth", 'problem'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                    # non_tensor_batch_keys=["raw_prompt_ids", "ground_truth", 'problem'],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch.meta_info["mode"] = "test"
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]

            output_texts = [replace_abs_vis_token_content(self.tokenizer.decode(ids, skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "") for ids in output_ids]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=False).replace("<|endoftext|>", "").replace("<|im_end|>", "") for ids in output_ids]

            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            if 'api' in self.config.worker.rule_based_judge.judge_function_name:
                #breakpoint()
                correctness_list = api_batch_judge(
                    questions=test_batch.non_tensor_batch["problem"].tolist(),
                    preds=output_texts,
                    gts=test_batch.non_tensor_batch["ground_truth"].tolist(),
                    api_name=self.config.worker.rule_based_judge.api_name,
                    api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                    client=self.client,
                    repetition_penalty=self.config.worker.reward.repetition_penalty,
                )
                #correctness_list = ray.get(self.rule_based_judge.judge.remote(output_texts, test_batch.non_tensor_batch["ground_truth"].tolist()))
                test_batch.non_tensor_batch["correctness"] = correctness_list
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        # val_reward_metrics.update({"val/attn": attn.cpu().numpy()} if attn is not None else {})

        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        print('start building worker group')
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
        print('done building worker group')
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        try:
            print("[_save_checkpoint] free -h:")
            os.system("free -h")
            self.actor_rollout_wg.save_checkpoint(actor_path)
        except Exception as e:
            print('[_save_checkpoint] failed, exit')
            sys.exit(0)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

        if self.config.worker.rollout.sampling_strategy in ["monet"]:
            self.sample_hash_server_main.save_info.remote(filepath=folder_path, overwrite=True)


    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        print("Loaded global step:", self.global_step)
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        self.sample_hash_server_main.load_info.remote(self.config.trainer.load_checkpoint_path)

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def clean_tmp_folder(self):
        try:
            log_dir = f"{os.environ['RAY_TMPDIR']}/session_latest/logs/"
            if os.path.exists(log_dir):
                files = glob.glob(os.path.join(log_dir, '*'))
                for f in files:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
        except Exception as e:
            print(f"Error while cleaning temporary folder: {e}")
            pass
 
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.add_latent_to_reward = os.environ.get("add_latent_to_reward", "0").lower() == "1"
        self.add_attn_to_reward = os.environ.get("add_attn_to_reward", "0").lower() == "1"
        self.add_token_attn_to_reward = os.environ.get("add_token_attn_to_reward", "0").lower() == "1"
        self.add_relative_attn_to_reward = os.environ.get("add_relative_attn_to_reward", "0").lower() == "1"
        self.latent_necessity_reward = os.environ.get("latent_necessity_reward", "0").lower() == "1"
        print(
            "[Monet/RL/verl/trainer/ray_trainer.py] add_latent_to_reward:",
            self.add_latent_to_reward,
            "add_attn_to_reward:",
            self.add_attn_to_reward,
            "self.add_relative_attn_to_reward:",
            self.add_relative_attn_to_reward,
            "latent_necessity_reward:",
            self.latent_necessity_reward,
        )

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            print("[Monet/RL/verl/trainer/ray_trainer.py] _validate started before training...")
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return
        print("[Monet/RL/verl/trainer/ray_trainer.py] _validate finished. Starting training loop...")

        for epoch in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
            self.clean_tmp_folder()
            if self.config.worker.rollout.offline_difficulty_sampling:
                if self.config.worker.rollout.sampling_strategy in ["monet"]:
                    self.pre_generate_monet_offline(epoch=epoch)

                epoch_new_selected_statistics = {}
                for global_id in self.correct_pool.keys():
                    if self.selected_sample_statistics[global_id] == 0:
                        epoch_new_selected_statistics[global_id] = 1
                    self.selected_sample_statistics[global_id] += 1

            if self.config.worker.rollout.online_difficulty_sampling:
                if self.config.worker.rollout.sampling_strategy in ["monet"]:
                    self.correct_pool.clear() # epoch start
                    if self.config.data.shuffle:
                        train_dataloader_generator = torch.Generator()
                        train_dataloader_generator.manual_seed(self.config.data.seed)
                        sampler = RandomSampler(data_source=self.base_dataset, generator=train_dataloader_generator)
                    else:
                        sampler = SequentialSampler(data_source=self.base_dataset)

                    out_data_loader = StatefulDataLoader(
                        dataset=self.base_dataset,
                        batch_size=self.config.data.online_accum_size,
                        sampler=sampler,
                        num_workers=self.config.data.dataloader_num_workers,
                        collate_fn=collate_fn,
                        pin_memory=False,
                        drop_last=False,
                    )

                    for large_batch_dict in tqdm(out_data_loader, desc=f"Traversing all training data, accum batch={self.config.data.online_accum_size}", position=1):
                        self.pre_generate_monet_online(DataProto.from_single_dict(large_batch_dict))
                        if len(self.correct_pool) < self.config.data.rollout_batch_size:
                            print(f"Not enough samples ({len(self.correct_pool)}) to form a batch ({self.config.data.rollout_batch_size}) from {self.config.data.online_accum_size} samples. Continue to the next large batch.")
                            continue
                        self.build_train_dataloader_with_correct_gen_out_pool(self.base_dataset)
                        for batch in tqdm(self.train_dataloader, desc=f"Running train step from gs {self.global_step} to {self.global_step + len(self.train_dataloader)}, max {self.training_steps}:", position=5 if self.config.worker.rollout.sampling_strategy in ["monet"] else 1):
                            self.global_step += 1
                            if self.global_step > self.training_steps:
                                break

                            metrics, timing_raw = {}, {}
                            #pdb.set_trace()
                            with timer("step", timing_raw):
                                with batch.batch.unlock_():
                                    self._balance_batch(batch, metrics=metrics)
                                    self.post_generate_update(metrics, timing_raw, batch)

                            self.logger.log(data=metrics, step=self.global_step)
                        if self.global_step > self.training_steps:
                            break

            else:  # default / offline
                for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=4 if self.config.worker.rollout.sampling_strategy in ["monet"] else 1):
                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    #reakpoint()
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    # pop those keys for generation
                    #breakpoint()
                    batch.meta_info["mode"] = "train_rl_gen"
                    gen_batch = self.build_gen_batch(batch)
                    sample_idx = gen_batch.non_tensor_batch["global_index"]
                    sample_idx = [item for item in sample_idx for _ in range(self.config.worker.rollout.n)]
                    with timer("step", timing_raw):
                        # generate a batch
                        #breakpoint()
                        with timer("gen", timing_raw):  # wg: worker group
                            gen_batch.meta_info["mode"] = "train_rl_gen"
                            raise ValueError("Don't use offline")
                            # gen_batch_output = self.actor_rollout_wg.generate_s equences(gen_batch)

                        self._maybe_dump_rollout_outputs(self.global_step, gen_batch_output)

                        batch = self.post_generate_process(metrics, timing_raw, gen_batch, gen_batch_output)

                        self.post_generate_update(metrics, timing_raw, batch)

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                self.logger.log(data=metrics, step=self.global_step)


        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
        sys.exit(0)

    def pre_generate_monet_online(self, large_batch):
        """
        Traverse the large batch, generate rollouts for each small batch, and collect the samples with correct answers and avg accuracy below threshold
        """
        self.correct_pool.clear()
        ori_bsz = self.config.data.rollout_batch_size
        for b in tqdm(range(0, len(large_batch), ori_bsz), desc=f"[rank {self.trank}] [pre_generate_monet_online] [large_batch={len(large_batch)}] [ori_bsz={ori_bsz}]", position=2):
            metrics, timing_raw = {}, {}
            batch = large_batch[b:b+ori_bsz]
            batch.meta_info["mode"] = "train_pre_gen"
            gen_batch = self.build_gen_batch(batch)
            sample_idx = gen_batch.non_tensor_batch["global_index"]
            sample_idx = [item for item in sample_idx for _ in range(self.config.worker.rollout.n)]
            monet_rl_verl_trainer = os.path.dirname(os.path.abspath(__file__))
            monet_rl = os.path.dirname(os.path.dirname(monet_rl_verl_trainer))
            if os.path.exists(f"{monet_rl}/quit"):
                print("################### quit signal detected. Exiting... #################")
                exit(0)
            with timer("pre_step", timing_raw):
                with timer("pre_gen", timing_raw):
                    gen_batch.meta_info["mode"] = "train_pre_gen_online"
                    gen_out = self.actor_rollout_wg.generate_sequences(gen_batch)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                batch = batch.union(gen_out)
                batch.non_tensor_batch.pop("multi_modal_data", None)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # after all samples in the batch are judged, use ray.get to get the results
                judge_results_and_answer_texts_proto = self.actor_rollout_wg.compute_rule_based_judge(data=batch)
                judge_results, answer_texts = judge_results_and_answer_texts_proto.non_tensor_batch["correctness"], judge_results_and_answer_texts_proto.non_tensor_batch["response_strs"]
                batch.non_tensor_batch["correctness"] = judge_results

                if b == 0:
                    print(f"[rank {self.trank}] Pre-rollout acc of batch {b}:", f"acc={judge_results.sum().item()} / {len(judge_results)}={judge_results.sum().item() / len(judge_results)} from {judge_results.tolist()}") # ori_bsz * rollout.n

                upd_min_correct_len_refs = []
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):
                    resp_tks_len = batch.batch['responses'][i][batch.batch['response_mask'][i].bool()].shape[0]
                    if correct:
                        upd_min_correct_len_ref = self.sample_hash_server_main.update_min_mean_correct_resp_len.remote(sample_id, resp_tks_len)
                        upd_min_correct_len_refs.append(upd_min_correct_len_ref)
                ray.get(upd_min_correct_len_refs) # update the min correct response length for each question

                batch_rollout_record = defaultdict(list)
                batch_update_info = defaultdict(list)
                latent_rate = {}
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):

                    batch_update_info[sample_id].append((int(sample_id), correct))
                    has_latent = (batch.batch['responses'][i] == int(os.environ.get('ABS_VIS_START_ID', 151666))).sum(dim=-1)
                    has_latent = has_latent.item() > 0
                    if sample_id not in latent_rate:
                        latent_rate[sample_id] = []
                    latent_rate[sample_id] += [has_latent]
                    self.has_latent_pre_generate_monet['true' if has_latent else 'false'] += 1
                    if sum(self.has_latent_pre_generate_monet.values()) % 100 == 0:
                        print(f"[rank {self.trank}] [pre_generate_monet_online] has_latent_pre_generate_monet:", self.has_latent_pre_generate_monet, 'has_latent:', has_latent)
                    if b == 0 and i < 3:
                    #     input_text = self.tokenizer.decode(batch.batch['input_ids'][i], skip_special_tokens=False).replace('<|endoftext|>', '').replace("<|image_pad|>", '')
                    #     print(f"[rank {self.trank}] [pre_generate_monet_online] sample_id={sample_id} sample input={input_text}")
                        response_text = self.tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=False).replace('<|endoftext|>', '')
                        gt = batch.non_tensor_batch["ground_truth"].tolist()[i]
                        print(f"[rank {self.trank}] [pre_generate_monet_online] sample_id={sample_id} sample response={response_text} gt={gt} correct={correct}")

                    # Strip latent embeds/ids before saving for later use
                    strip_latent = os.environ.get('STRIP_LATENT_IN_ROLLOUT', '0') == '1'
                    sample_dp = batch[i:i+1]
                    clean_answer_text = answer_text
                    if has_latent and strip_latent:
                        old_response_len = sample_dp.batch['responses'].shape[1]
                        sample_dp = strip_latent_tokens_from_dataproto(sample_dp)
                        new_response_len = sample_dp.batch['responses'].shape[1]
                        if sum(self.has_latent_pre_generate_monet.values()) % 100 == 0:
                            print(f"[rank {self.trank}] [pre_generate_monet_online] [strip] sample_id {sample_id} from {old_response_len} to {new_response_len}.")
                        clean_answer_text = remove_latent_from_text(answer_text)
                    self.correct_gen_out_pool[sample_id].append(sample_dp)

                    if correct:
                        if int(os.environ.get('NO_LATENT_IN_ROLLOUT', '0')) == 1:
                            if not has_latent:
                                batch_rollout_record[sample_id].append(1)
                                self.correct_pool[sample_id].append(clean_answer_text)
                        elif int(os.environ.get('ALL_LATENT_IN_ROLLOUT', '0')) == 1:
                            if has_latent:
                                batch_rollout_record[sample_id].append(1)
                                self.correct_pool[sample_id].append(clean_answer_text)
                        else:
                            batch_rollout_record[sample_id].append(1)
                            self.correct_pool[sample_id].append(clean_answer_text)
                    else:
                        batch_rollout_record[sample_id].append(0)

                # discard samples that are too easy
                static_correct_pool_keys = list(self.correct_pool.keys())
                for sample_id in static_correct_pool_keys:
                    if len(batch_rollout_record[sample_id]) == 0: # this sample_id is from previous batches and not recorded in the batch_rollout_record of the current batch
                        continue
                    sample_pre_rollout_acc = batch_rollout_record[sample_id].count(1)/len(batch_rollout_record[sample_id])
                    if sample_pre_rollout_acc > self.config.worker.rollout.monet.select_acc_threshold:
                        self.correct_pool.pop(sample_id)
                        self.latent_rate_in_drop_due_to_highacc.append(latent_rate[sample_id])
                        mean_rates = [sum(rates) / len(rates) for rates in self.latent_rate_in_drop_due_to_highacc if len(rates) > 0]
                        mean_rate = sum(mean_rates) / len(mean_rates)
                        if len(self.latent_rate_in_drop_due_to_highacc) > 0 and len(self.latent_rate_in_drop_due_to_highacc) % 100 == 0:
                            print(f"[rank {self.trank}] [pre_generate_monet_online] latent_rate_in_drop_due_to_highacc: {mean_rate}")

    def post_generate_process(self, metrics, timing_raw, gen_batch, gen_batch_output):
        batch = gen_batch

        if self.config.algorithm.adv_estimator == "remax":
            with timer("gen_max", timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                raise ValueError("Don't use remax")
                # gen_baseline_output = self.actor_rollout_wg.gener ate_sequences(gen_baseline_batch)

                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
        batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)

        batch = batch.union(gen_batch_output)
        batch.non_tensor_batch.pop("multi_modal_data", None)

        # balance the number of valid tokens on each dp rank.
        # Note that this breaks the order of data inside the batch.
        # Please take care when you implement group based adv computation such as GRPO and rloo
        self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _run_latent_necessity_r1(
        self, batch: DataProto
    ) -> Tuple[Optional[DataProto], Optional[List[int]]]:
        if os.environ.get("LATENT_SIZE").lower() in ("0", "none", "null"):
            return None, None
        if self.config.worker.rollout.sampling_strategy not in ("monet",):
            return None, None
        snaps = batch.non_tensor_batch.get("monet_rollout_vllm_input")
        if snaps is None:
            return None, None
        from .latent_necessity import build_latent_necessity_r1_subbatch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        response_text = self.processor.batch_decode(batch.batch["responses"], skip_special_tokens=False)
        response_text = [t.replace("<|endoftext|>", "").replace("<|image_pad|>", "I") for t in response_text]
        # for i, rt in enumerate(response_text):
        #     print(f"[rank {self.trank}] [_run_latent_necessity_r1] original response[{i}]: {rt}")
        built = build_latent_necessity_r1_subbatch(
            responses=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
            rollout_snapshots=snaps,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            pad_token_id=pad_id,
        )
        if built is None:
            # print(f"[rank {self.trank}] snap:", len(snaps))
            # for snap in snaps:
            #     print(f"[rank {self.trank}] {snap}")
            return None, None
        r1_dp, rows = built
        r1_input_text = self.processor.batch_decode(r1_dp.batch["input_ids"], skip_special_tokens=False)
        r1_input_text = [t.replace("<|endoftext|>", "").replace("<|image_pad|>", "I") for t in r1_input_text]
        # for i, rt in enumerate(r1_input_text):
        #     print(f"[rank {self.trank}] [_run_latent_necessity_r1] re-roll inputs[{i}]: {rt}")
        r1_dp.meta_info = {}
        for k, v in batch.meta_info.items():
            if k in ("mode", "n", "vllm_plain_generate", "override_latent_size"):
                continue
            r1_dp.meta_info[k] = v
        r1_dp.meta_info["mode"] = "train_rl_gen_latent_necessity"
        r1_dp.meta_info["vllm_plain_generate"] = True
        r1_dp.meta_info["n"] = 1
        r1_dp.meta_info["override_latent_size"] = int(
            os.environ.get("LATENT_NECESSITY_R1_LATENT_SIZE", "0")
        )
        r1_padded, pad = pad_dataproto_to_divisor(r1_dp, self.actor_rollout_wg.world_size)
        r1_out = self.actor_rollout_wg.generate_sequences(r1_padded)
        # for i, r1o in enumerate(r1_out.batch["responses"]):
        #     print(f"[rank {self.trank}] [_run_latent_necessity_r1] r1 output[{i}]: {processor.decode(r1o, skip_special_tokens=False).replace('<|endoftext|>', '').replace('<|image_pad|>', 'I')}")
        r1_out = unpad_dataproto(r1_out, pad_size=pad)
        return r1_out, rows

    def post_generate_update(self, metrics, timing_raw, batch):
        self._post_generate_latent_necessity = None
        with timer("latent_necessity_r1", timing_raw):
            r1_out, r1_rows = self._run_latent_necessity_r1(batch)
        with timer("reward", timing_raw):
            # batch.non_tensor_batch should have "correctness" here
            reward_ref = self.reward_fn.compute_reward.remote(batch)

        # recompute old_log_probs
        #breakpoint()
        with timer("old", timing_raw):
            batch.meta_info["output_attentions"] = True
            old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
            attn = old_log_probs.batch.pop('attn_weights_lst', None)
            attn_image = old_log_probs.batch.pop('attn_image_weights_lst', None)
            attn_token_weights_lst = old_log_probs.batch.pop('attn_token_weights_lst', None)
            if attn is not None:
                attn = attn.detach().clone() # (batch_size, )
            if attn_image is not None:
                attn_image = attn_image.detach().clone() # (batch_size, )
            if attn_token_weights_lst is not None:
                attn_token_weights_lst = attn_token_weights_lst.detach().clone() # (batch_size, seqlen, )
            batch = batch.union(old_log_probs)

        # compute ref_log_probs
        kl_for_dump = None
        if self.use_reference_policy:
            with timer("ref", timing_raw):
                batch.meta_info["output_attentions"] = False
                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                batch = batch.union(ref_log_probs)

        # compute values
        if self.use_critic:
            with timer("values", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        #breakpoint()
        with timer("adv", timing_raw):
            # get token level scores
            reward_tensor, reward_metrics_unreduced = ray.get(reward_ref)
            batch.batch["token_level_scores"] = reward_tensor
            bsz = int(reward_tensor.size(0))
            acc_list = [float(x) for x in (reward_metrics_unreduced.get("accuracy", [0.0] * bsz) or [])]
            if len(acc_list) < bsz:
                acc_list.extend([0.0] * (bsz - len(acc_list)))
            acc_list = acc_list[:bsz]
            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics_unreduced).items()}
            metrics.update(reward_metrics)

            if self.use_reference_policy:
                kl_for_dump = core_algos.compute_kl(
                    batch.batch["old_log_probs"],
                    batch.batch["ref_log_probs"],
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                kl_for_dump = kl_for_dump * batch.batch["response_mask"]

            # apply kl penalty if available
            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                # apply kl penalty to reward
                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
                for key, value in kl_metrics.items():
                    print(f"  [{self.trank}] [post_generate_update] kl_metrics key: {key}, shape: {getattr(value, 'shape', 'N/A')}, value: {value}")
            else:
                batch.batch.pop("token_level_kl", None)
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            print(f"  [{self.trank}] [post_generate_update] token_level_scores shape: {batch.batch['token_level_scores'].shape}, token_level_scores: {batch.batch['token_level_scores'].max(dim=-1)}")
            if r1_out is not None and r1_rows:
                from .latent_necessity import (
                    decode_response_text_monet,
                    is_correctness_positive,
                    latent_necessity_scalar,
                    maybe_print_latent_necessity_replay_debug,
                    r1_scalar_accuracy,
                )

                gts = batch.non_tensor_batch["ground_truth"]
                probs = batch.non_tensor_batch["problem"]
                use_rule_then_api = (
                    self.config.worker.rule_based_judge.judge_function_name
                    == "rule_then_api_batch_judge"
                    and getattr(self, "client", None) is not None
                )
                a0_list: List[bool] = []
                a1_list: List[bool] = []
                tlr_before_rn = batch.batch["token_level_rewards"].clone()
                use_batch_a0 = False
                if use_rule_then_api:
                    from examples.reward_function.monet_reward_function import rule_then_api_batch_judge

                    bsz_roll = int(batch.batch["responses"].size(0))
                    corr_nt = batch.non_tensor_batch.get("correctness")
                    if corr_nt is not None:
                        try:
                            corr_flat = np.asarray(corr_nt).reshape(-1)
                            if corr_flat.size >= bsz_roll:
                                use_batch_a0 = True
                        except Exception:
                            use_batch_a0 = False
                    if use_batch_a0:
                        # Same signal as pre_generate_monet_online + compute_rule_based_judge (decode/judge path).
                        a0_list = [
                            is_correctness_positive(corr_flat[int(src)])
                            for src in r1_rows
                        ]
                    else:
                        p0 = [
                            decode_response_text_monet(
                                self.tokenizer,
                                batch.batch["responses"][src],
                                batch.batch["response_mask"][src],
                            )
                            for src in r1_rows
                        ]
                        qs_fb = [probs[i] for i in r1_rows]
                        gt_fb = [str(gts[i]) for i in r1_rows]
                        c0 = rule_then_api_batch_judge(
                            questions=qs_fb,
                            preds=p0,
                            gts=gt_fb,
                            api_name=self.config.worker.rule_based_judge.api_name,
                            api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                            client=self.client,
                            repetition_penalty=self.config.worker.reward.repetition_penalty,
                        )
                        a0_list = [is_correctness_positive(x) for x in c0]

                    p1 = [
                        decode_response_text_monet(
                            self.tokenizer,
                            r1_out.batch["responses"][j],
                            r1_out.batch["response_mask"][j],
                        )
                        for j in range(len(r1_rows))
                    ]
                    qs = [probs[i] for i in r1_rows]
                    gt_sl = [str(gts[i]) for i in r1_rows]
                    c1 = rule_then_api_batch_judge(
                        questions=qs,
                        preds=p1,
                        gts=gt_sl,
                        api_name=self.config.worker.rule_based_judge.api_name,
                        api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                        client=self.client,
                        repetition_penalty=self.config.worker.reward.repetition_penalty,
                    )
                    a1_list = [is_correctness_positive(x) for x in c1]
                else:
                    for j, src in enumerate(r1_rows):
                        a0_list.append(float(acc_list[src]) == 1.0)
                        a1_list.append(
                            r1_scalar_accuracy(
                                self.tokenizer,
                                r1_out.batch["responses"][j],
                                r1_out.batch["response_mask"][j],
                                str(gts[src]),
                            )
                            == 1.0
                        )

                rn_vals = []
                bsz_ln = int(batch.batch["responses"].size(0))
                ln_rn: List[Optional[float]] = [None] * bsz_ln
                ln_a0: List[Optional[int]] = [None] * bsz_ln
                ln_a1: List[Optional[int]] = [None] * bsz_ln
                for j, src in enumerate(r1_rows):
                    a0, a1 = a0_list[j], a1_list[j]
                    rn = latent_necessity_scalar(a0, a1, n=self.latent_necessity_reward_n)
                    ridx = int(batch.batch["response_mask"][src].sum().item()) - 1
                    ridx = max(0, ridx)
                    if self.latent_necessity_reward:
                        batch.batch["token_level_rewards"][src, ridx] = (
                            batch.batch["token_level_rewards"][src, ridx] + rn
                        )
                    rn_vals.append(rn)
                    ln_rn[src] = float(rn)
                    ln_a0[src] = int(1 if a0 else 0)
                    ln_a1[src] = int(1 if a1 else 0)
                if use_rule_then_api:
                    _ln_judge = (
                        "batch_correctness_a0_rule_then_api_r1"
                        if use_batch_a0
                        else "rule_then_api_batch_judge"
                    )
                else:
                    _ln_judge = "reward_accuracy_and_rule_r1"
                if rn_vals:
                    maybe_print_latent_necessity_replay_debug(
                        trank=self.trank,
                        tokenizer=self.tokenizer,
                        batch=batch,
                        r1_out=r1_out,
                        r1_rows=r1_rows,
                        a0_list=a0_list,
                        a1_list=a1_list,
                        acc_list=acc_list,
                        rn_vals=rn_vals,
                        tlr_before_rn=tlr_before_rn,
                        tlr_after_rn=batch.batch["token_level_rewards"],
                        token_level_scores=batch.batch["token_level_scores"],
                        judge_mode=_ln_judge,
                    )
                    if self.latent_necessity_reward:
                        metrics["reward/latent_necessity_mean"] = float(np.mean(rn_vals))
                ld = {
                    "latent_necessity_rn": ln_rn,
                    "latent_necessity_a0": ln_a0,
                    "latent_necessity_a1": ln_a1,
                    "latent_necessity_judge_mode": _ln_judge,
                    "latent_necessity_applied_to_rewards": bool(self.latent_necessity_reward),
                }
            else:
                ld = None
            # print(f"  [{self.trank}] [post_generate_update] token_level_rewards shape: {batch.batch['token_level_rewards'].shape}, token_level_rewards: {batch.batch['token_level_rewards']}")
            if self.add_latent_to_reward:
                has_valid_latent_start = batch.batch['input_ids'] == 151666
                has_valid_latent_end = batch.batch['input_ids'] == 151667
                has_valid_latent = has_valid_latent_start.cumsum(dim=1) - has_valid_latent_end.cumsum(dim=1) > 0
                has_valid_latent = has_valid_latent.sum(dim=1) > 0  # (batch_size, )
                z = torch.zeros_like(batch.batch["token_level_rewards"])
                z[:, 0] += has_valid_latent.float() * 0.5
                print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] token_level_rewards {batch.batch['token_level_rewards'].mean(dim=1).tolist()}, mean {batch.batch['token_level_rewards'].mean().tolist()} has_valid_latent: {has_valid_latent.float().item()}")
                batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + z
            elif self.add_token_attn_to_reward:
                z = torch.zeros_like(batch.batch["token_level_rewards"])
                # print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] token_level_rewards shape: {batch.batch['token_level_rewards'].shape}")
                # print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] token_level_rewards value: {batch.batch['token_level_rewards'].tolist()}")
                # print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] attn_token_weights_lst shape: {attn_token_weights_lst.shape if attn_token_weights_lst is not None else 'N/A'}")
                # print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] attn_token_weights_lst value: {attn_token_weights_lst.tolist() if attn_token_weights_lst is not None else 'N/A'}")
                z += attn_token_weights_lst
                batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + z
            elif self.add_attn_to_reward or self.add_relative_attn_to_reward:
                z = torch.zeros_like(batch.batch["token_level_rewards"])
                z[:, 0] += attn
                print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] token_level_rewards {batch.batch['token_level_rewards'].mean(dim=1).tolist()}, mean {batch.batch['token_level_rewards'].mean().tolist()} attn: {attn.tolist()}, mean {attn.mean().item()}")
                batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + z
            # elif self.add_relative_attn_to_reward: #already added in the actor/compute_log_prob
            #     z = torch.zeros_like(batch.batch["token_level_rewards"])
            #     relative_attn = (attn / (attn_image + 1e-8)) if attn_image is not None else attn
            #     z[:, 0] += relative_attn
            #     print(f"  [{self.trank}] [ray_trainer.py/post_generate_update] token_level_rewards {batch.batch['token_level_rewards'].mean(dim=1).tolist()}, mean {batch.batch['token_level_rewards'].mean().tolist()} attn: {attn.tolist()}, mean {attn.mean().item()}, relative_attn: {relative_attn.tolist()}, mean {relative_attn.mean().item()}")
            #     batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + z

            # Log attention/reward stats so they are visible in TensorBoard.
            tlr = batch.batch["token_level_rewards"]
            metrics["reward/token_level_rewards_mean"] = tlr.mean().item()
            metrics["reward/token_level_rewards_std"] = tlr.std(unbiased=False).item()
            metrics["reward/token_level_rewards_first_token_mean"] = tlr[:, 0].mean().item()
            if attn is not None:
                metrics["reward/attn_mean"] = attn.mean().item()
                metrics["reward/attn_std"] = attn.std(unbiased=False).item()
                metrics["reward/attn_min"] = attn.min().item()
                metrics["reward/attn_max"] = attn.max().item()
            if attn_image is not None:
                metrics["reward/attn_image_mean"] = attn_image.mean().item()
                metrics["reward/attn_image_std"] = attn_image.std(unbiased=False).item()
                metrics["reward/attn_image_min"] = attn_image.min().item()
                metrics["reward/attn_image_max"] = attn_image.max().item()

            # compute advantages, executed on the driver process
            batch = compute_advantage(
                self.config,
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                sampling_strategy=self.config.worker.rollout.sampling_strategy
            )

        # update critic
        if self.use_critic:
            with timer("update_critic", timing_raw):
                critic_output = self.critic_wg.update_critic(batch)

            critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
            metrics.update(critic_metrics)

        # update actor
        #breakpoint()
        actor_per_sample_pg_loss = None
        actor_per_sample_total_loss = None
        if self.config.trainer.critic_warmup <= self.global_step:
            with timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)

            actor_non_tensor_batch = dict(actor_output.non_tensor_batch)
            per_sample_pg_loss = actor_non_tensor_batch.pop("actor/per_sample_pg_loss", None)
            per_sample_total_loss = actor_non_tensor_batch.pop("actor/per_sample_total_loss", None)

            if per_sample_pg_loss is not None:
                pg_vals = self._to_object_list(per_sample_pg_loss, len(batch.batch))
                if pg_vals is not None:
                    actor_per_sample_pg_loss = [
                        None if v is None else float(v) for v in pg_vals
                    ]
            if per_sample_total_loss is not None:
                total_vals = self._to_object_list(per_sample_total_loss, len(batch.batch))
                if total_vals is not None:
                    actor_per_sample_total_loss = [
                        None if v is None else float(v) for v in total_vals
                    ]

            actor_metrics = reduce_metrics(actor_non_tensor_batch)
            metrics.update(actor_metrics)

        self._maybe_dump_post_generate_outputs(
            batch=batch,
            attn=attn,
            attn_image=attn_image,
            timing_raw=timing_raw,
            kl=kl_for_dump,
            actor_per_sample_pg_loss=actor_per_sample_pg_loss,
            actor_per_sample_total_loss=actor_per_sample_total_loss,
            ld=ld
        )

        # validate
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and self.global_step % self.config.trainer.val_freq == 0
        ):
            with timer("validation", timing_raw):
                val_metrics = self._validate()

            metrics.update(val_metrics)

        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
            with timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

    def _rebuild_train_dataloader_with_correct_pool(self, base_dataset):
        '''
        Rebuild the train_dataloader with samples that have correct answers and acc<select_acc_threshold.
        '''
        class CorrectAnswerDataset(Dataset):
            def __init__(self, base_dataset, correct_pool):
                self.base_dataset = base_dataset
                self.correct_pool = correct_pool
                self.qids = list(correct_pool.keys()) # ids of samples that have correct answers and acc<select_acc_threshold
            def __len__(self):
                return len(self.qids)

            def __getitem__(self, idx):
                qid = self.qids[idx]
                try:
                    sample = self.base_dataset[qid]
                    sample["mc_raw_prompt_ids"] = sample["raw_prompt_ids"]
                    sample["correct_solutions_text"] = random.choice(self.correct_pool[qid])
                except Exception as e:
                    n_pool = None
                    try:
                        n_pool = len(self.correct_pool[qid])
                    except Exception:
                        pass
                    pool_info = f" correct_pool_len={n_pool}" if n_pool is not None else ""
                    raise RuntimeError(
                        f"CorrectAnswerDataset(rebuild): failed dataloader_idx={idx} global_index(qid)={qid}.{pool_info} | {e!r}"
                    ) from e
                return sample

        correct_ds = CorrectAnswerDataset(base_dataset, self.correct_pool)

        self.train_dataloader = StatefulDataLoader(
            dataset=correct_ds,
            batch_size=self.config.data.rollout_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.data.dataloader_num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def build_train_dataloader_with_correct_gen_out_pool(self, base_dataset):
        '''
        Rebuild the train_dataloader with samples that have correct answers and acc<select_acc_threshold.
        '''
        class CorrectGenOutDataset(Dataset):
            def __init__(self, base_dataset, correct_pool, correct_gen_out_pool):
                self.base_dataset = base_dataset
                self.correct_pool = correct_pool
                self.correct_gen_out_pool = correct_gen_out_pool
                self.qids = list(correct_pool.keys()) # ids of samples that have correct answers and acc<select_acc_threshold
            def __len__(self):
                return len(self.qids)

            def __getitem__(self, idx):
                qid = self.qids[idx]
                group_list = self.correct_gen_out_pool[qid]
                sample = DataProto.concat(group_list) # concat the group
                return sample

        correct_ds = CorrectGenOutDataset(base_dataset, self.correct_pool, self.correct_gen_out_pool)

        def collate_fn_gen_out(features: List[DataProto]):
            return DataProto.concat(features)

        self.train_dataloader = StatefulDataLoader(
            dataset=correct_ds,
            batch_size=self.config.data.rollout_batch_size,
            shuffle=True,
            collate_fn=collate_fn_gen_out,
            num_workers=self.config.data.dataloader_num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def build_gen_batch(self, batch: DataProto) -> None:
        if "multi_modal_data" in batch.non_tensor_batch.keys():
            if self.config.worker.rollout.sampling_strategy == "greedy":
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"]
                )
                if batch.meta_info["mode"] == "train_pre_gen":
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index"]
                    )
                elif batch.meta_info["mode"] == "train_rl_gen":
                    gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "problem", "ground_truth", "prompt_before_processor", "global_index", "correct_solutions_text"]
                        )
            elif self.config.worker.rollout.sampling_strategy == "monet":
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index", "problem", "ground_truth"]
                )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )
        return gen_batch

    @staticmethod
    def split_solution_into_steps(solution: str, delim: str = "### Step") -> List[List[str]]:
        steps = solution.split(delim)
        steps = [re.sub(r"^ \d+(\.\d+)?: ", "", step).strip() for step in steps]
        return steps

