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
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple
from pathlib import Path

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    online_accum_size: int = 1024
    pr_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144
    filter_overlong_and_invalid_prompts: bool = True
    train_max_samples: Optional[int] = None
    val_max_samples: Optional[int] = None
    dataloader_num_workers: int = 4
    """auto keys"""
    online_difficulty_sampling: bool = False

    def post_init(self):
        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0
    # added by qixun
    sampling_method: str = "default"


@dataclass
class TrainerConfig:
    total_epochs: int = 10
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    save_freq: int = -1
    save_limit: int = -1
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        # if self.load_checkpoint_path is None:
        if self.load_checkpoint_path == 'auto':
            path = Path(self.save_checkpoint_path)
            paths = list(sorted(path.glob("global_step_*"), key=os.path.getmtime, reverse=True))
            num_shards = int(os.environ.get("RAY_NUM_GPUS", 4))
            for p in paths:
                # Common files expected in every checkpoint
                common_fs = [
                    p / "dataloader.pt",
                    p / "sample_hash_dict.pkl",
                    p / "sample_resp_len_stats.pkl",
                    p / "actor" / "huggingface" / "added_tokens.json",
                    # p / "actor" / "huggingface" / "chat_template.json", # jinja if transformers 4.54.0
                    p / "actor" / "huggingface" / "config.json",
                    p / "actor" / "huggingface" / "generation_config.json",
                    p / "actor" / "huggingface" / "merges.txt",
                    p / "actor" / "huggingface" / "preprocessor_config.json",
                    p / "actor" / "huggingface" / "special_tokens_map.json",
                    p / "actor" / "huggingface" / "tokenizer_config.json",
                    p / "actor" / "huggingface" / "tokenizer.json",
                    p / "actor" / "huggingface" / "vocab.json",
                ] + [
                    p / "actor" / f"extra_state_world_size_{num_shards}_rank_{i}.pt" for i in range(num_shards)
                ]

                # Detect DCP sharded format vs legacy per-rank .pt format
                is_dcp = 1 and (p / "actor" / "model").is_dir()

                if is_dcp:
                    # DCP format: model/ and optim/ are directories with shard files
                    model_optim_fs = [
                        p / "actor" / "model",
                        p / "actor" / "optim",
                    ]
                else:
                    # Legacy format: per-rank .pt files
                    model_optim_fs = [
                        p / "actor" / f"model_world_size_{num_shards}_rank_{i}.pt" for i in range(num_shards)
                    ] + [
                        p / "actor" / f"optim_world_size_{num_shards}_rank_{i}.pt" for i in range(num_shards)
                    ]

                fs = common_fs + model_optim_fs
                if not all(os.path.exists(f) for f in fs):
                    fmt = "DCP sharded" if is_dcp else "legacy"
                    print(f"Checkpoint path {p} is incomplete ({fmt} format), skip.")
                    continue
                self.load_checkpoint_path = str(p)
                fmt = "DCP sharded" if is_dcp else "legacy"
                print(f"Find latest checkpoint path ({fmt}): {self.load_checkpoint_path}")
                break
        if self.load_checkpoint_path and not os.path.exists(self.load_checkpoint_path):
            self.load_checkpoint_path = None
        if self.load_checkpoint_path is not None:
            self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)

@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.pr_batch_size = self.data.pr_batch_size
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.rollout.n_gpus_per_node = self.trainer.n_gpus_per_node
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.sampling_strategy = self.worker.rollout.sampling_strategy
        self.worker.ref.sampling_strategy = self.worker.rollout.sampling_strategy
        self.data.online_difficulty_sampling = self.worker.rollout.online_difficulty_sampling
    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
