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
Implement Actor
"""
print(f"[dp_actor.py]")
import monet_rl_patch
import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional
import pdb
import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import sys
def _transformers_version_tuple():
    """Return (major, minor) of the installed transformers package."""
    try:
        import transformers

        parts = transformers.__version__.split(".")
        return tuple(int(p) for p in parts[:2])
    except Exception as exc:
        print(f"[dp_actor.py] could not parse transformers version: {exc}", file=sys.stderr)
        return (0, 0)
version = _transformers_version_tuple()
print(f"[dp_actor.py] detected transformers version tuple: {version}", file=sys.stderr)
if version >= (4, 54):
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
else:
    from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input
import numpy as np
#from verl.workers.actor.fa_shim import index_first_axis, pad_input, unpad_input # implementation by AXZ
from transformers import AutoProcessor

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig
import traceback

__all__ = ["DataParallelPPOActor"]

def collect_varlen_segment_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_id: int,
    end_id: int,
    debug=False
) -> torch.Tensor:
    """
    Collect indices (on the varlen/unpadded sequence) for token positions strictly inside
    matched (start, end) segments, skipping the first matched segment per sequence.

    Args:
        input_ids: LongTensor of shape (B, S).
        attention_mask: Bool/Long/Byte Tensor of shape (B, S); 1=valid, 0=pad/ignored.
        start_id: int, the start marker token id.
        end_id: int, the end marker token id.

    Returns:
        LongTensor of shape (K,), where each element is the index on the unpadded varlen
        sequence (i.e., the [1, L] after unpad+transpose) corresponding to a kept position.
    """
    debugstring = ""
    assert input_ids.dim() == 2, "input_ids must be 2D (B, S)"
    assert attention_mask.shape == input_ids.shape, "attention_mask must match input_ids"

    device = input_ids.device
    B, S = input_ids.shape

    # Ensure mask is 0/1 long tensor
    mask = attention_mask.to(dtype=torch.long)

    # Flatten mask to compute varlen positions: prefix sum gives mapping to [0..T-1]
    # For any flat position p with mask[p]==1, its varlen index is prefix[p]-1.
    mask_flat = mask.reshape(-1)                                      # (B*S,)
    prefix = torch.cumsum(mask_flat, dim=0)                           # (B*S,)
    # We will only index prefix at places where mask==1.

    varlen_indices_per_batch = []
    varlen_indices_by_batch = []
    for b in range(B):
        varlen_indices = []
        row_ids = input_ids[b]                                        # (S,)
        row_mask = mask[b]                                            # (S,)

        # Find all start/end positions (on [0..S-1])
        starts = (row_ids == start_id).nonzero(as_tuple=False).squeeze(-1)  # (Ns,) or empty
        ends   = (row_ids == end_id).nonzero(as_tuple=False).squeeze(-1)    # (Ne,) or empty
        if debug:
            debugstring += f"[collect_varlen] batch {b}: starts={starts.tolist()}, ends={ends.tolist()}"
        if starts.numel() == 0 or ends.numel() == 0:
            varlen_indices_by_batch.append(varlen_indices)
            continue

        # Two-pointer greedy matching: for each start, find the nearest end to its right.
        i_ptr, j_ptr = 0, 0
        matched = []  # list of (s, e), with e > s
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = starts[i_ptr].item()
            # Move j_ptr until we find an end strictly to the right of s_pos
            while j_ptr < ends.numel() and ends[j_ptr].item() <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = ends[j_ptr].item()
            matched.append((s_pos, e_pos))
            i_ptr += 1
            j_ptr += 1
        if debug:
            debugstring += f", matched (s,e)={matched}"

        if len(matched) <= 0:
            # Nothing (or only the first segment which we must skip)
            varlen_indices_by_batch.append(varlen_indices)
            continue

        for (s_pos, e_pos) in matched[:]:
            # inner = torch.arange(s_pos, e_pos, device=device, dtype=torch.long)  # (Lseg,)
            inner = torch.arange(s_pos + 1, e_pos, device=device, dtype=torch.long)  # (Lseg,)

            # Filter by attention mask (positions not in varlen stream should be dropped)
            inner_valid = inner[row_mask[inner] == 1]
            if debug:
                debugstring += f", inner_valid={inner_valid.tolist()}"
            if inner_valid.numel() == 0:
                continue

            # Map (b, pos) -> flat index -> varlen index
            flat_pos = b * S + inner_valid                               # (Lkeep,)
            # mask_flat[flat_pos] must be 1 here; varlen idx = prefix - 1
            var_idx = prefix[flat_pos] - 1                                # still on device, Long
            varlen_indices_per_batch.append(var_idx)
            varlen_indices.append(var_idx)
        varlen_indices_by_batch.append(varlen_indices)
    if debug:
        return debugstring
    if len(varlen_indices_per_batch) == 0:
        return torch.empty(0, dtype=torch.long, device=device), varlen_indices_by_batch

    # Concatenate all batches; these indices correspond to positions on the
    # unpadded [1, total_nnz] sequence (i.e., after unpad + transpose).
    return varlen_indices_per_batch, varlen_indices_by_batch

def compute_latent_log_probs(latent_poss, latents, last_hidden_state, sigma=1.0):
    """
    Compute log-prob under a headless isotropic Gaussian:
        z ~ N(mu=last_hidden_state[..., latent_poss, :], sigma^2 I)
    where `latents` is the sampled z used in rollout.

    Args:
        latent_poss: 1D LongTensor/list of positions for latent tokens (length L).
        latents:     Tensor of shape [L, D], rollout latents (z) at those positions.
        last_hidden_state: Tensor of shape [B, T, D], hidden states; we use batch 0.

    Returns:
        logp_sum: scalar tensor, sum of log-probs over all latent positions.
                  This is the sample-level log-prob commonly used in PPO/GRPO.
    """
    # Shape: [L, D]
    latent_outputs = last_hidden_state[0, latent_poss, :]
    latents = latents.to(latent_outputs)

    # Per-position log-prob of N(mu=latent_outputs, sigma^2 I)
    # log N(z; mu, sigma^2 I) = -0.5 * ||z - mu||^2 / sigma^2 - (D/2)*log(2*pi*sigma^2), removed const
    diff2 = (latents - latent_outputs).pow(2).sum(dim=-1)           # [L]
    latent_log_probs = - 0.5 * diff2 / (sigma ** 2)               # [L]
    return latent_log_probs


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
        self.print = print if self.rank == 0 else lambda *args, **kwargs: None
        self.latent_count = {'true': 0, 'total': 0}
        self.K = 0

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float, output_attentions: bool=False, caller: str="") -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        latent_poss = None
        latents = None
        strip_latent = os.environ.get('STRIP_LATENT_IN_ROLLOUT', '0') == '1'
        if not strip_latent and self.config.sampling_strategy == 'monet' and not self.config.ablate_latent:
            try:
                latent_poss = []
                start_id = int(os.getenv("ABS_VIS_START_ID"))
                end_id = int(os.getenv("ABS_VIS_END_ID"))
                _, varlen_by_batch = collect_varlen_segment_indices(
                    input_ids=micro_batch["input_ids"], # (micro_batch["input_ids"][1]==151666).nonzero() (micro_batch["input_ids"][1]==151667).nonzero()
                    attention_mask=micro_batch["attention_mask"],
                    start_id=start_id, end_id=end_id,
                )

                latents_list, per_sample = [], []
                for i, lat in enumerate(micro_batch['latents']):
                    self.latent_count['total'] += 1
                    if lat is not None:
                        t = torch.tensor(lat)  # (steps, D)
                        latent_len = int((os.environ.get("LATENT_SIZE", "10")))
                        if t.shape[0] % latent_len != 0:
                            self.print(f"[WARNING] [rank {self.rank}] A latent segment [{i}/{len(micro_batch['latents'])}] in a sample has length {t.shape[0]} not divisible by the latent size {latent_len}.")
                        poss_cnt = sum(v.numel() for v in varlen_by_batch[i]) if i < len(varlen_by_batch) else 0
                        if t.shape[0]!=poss_cnt:
                            k = [f"{l.min().item()}~{l.max().item()}={l.shape[0]}" for l in varlen_by_batch[i]]
                            S = micro_batch["input_ids"].shape[1]
                            input_id_at_k = [micro_batch["input_ids"][i, l.min().item()%S-3: l.max().item()%S+3].tolist() for l in varlen_by_batch[i]]
                            self.print(f"[WARNING] [rank {self.rank}] A latent segment [{i}/{len(micro_batch['latents'])}] in a sample has different numbers of recorded latent {t.shape[0]} and latent pad ids {poss_cnt}. Skip this sample for latent policy gradient computing." + f'micro_batch["input_ids"] shape: {micro_batch["input_ids"].shape}, varlen_by_batch: {k}, input_ids at poss: {input_id_at_k}, caller: {caller}')
                            debug = collect_varlen_segment_indices(
                                input_ids=micro_batch["input_ids"], # (micro_batch["input_ids"][1]==151666).nonzero() (micro_batch["input_ids"][1]==151667).nonzero()
                                attention_mask=micro_batch["attention_mask"],
                                start_id=start_id, end_id=end_id, debug=True
                            )
                            self.print(f"[WARNING] [rank {self.rank}] Debug info: {debug}")
                            processor = AutoProcessor.from_pretrained(f'{os.environ["a_very_big_data_disk"]}/Monet-SFT-7B/stage3')
                            # inputtext = processor.batch_decode(micro_batch["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            # inputtext = [(t.replace('<|image_pad|>', '[IMG]x' + str(t.count('<|image_pad|>')))).replace('<|endoftext|>', '') for t in inputtext]
                            # self.print(f"[WARNING] [rank {self.rank}] inputtext: {inputtext}")
                            responsetext = processor.batch_decode(micro_batch["responses"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            responsetext = [(t.replace('<|image_pad|>', '[IMG]x' + str(t.count('<|image_pad|>')))).replace('<|endoftext|>', '') for t in responsetext]
                            for rids, rtext, r in zip(micro_batch["responses"], responsetext, range(len(micro_batch["responses"]))):
                                self.print(f"[WARNING] [rank {self.rank}] responsetext[{r}]: {rtext} response ids:", rids[rids != 151643])
                            continue
                        else:
                            # pass
                            self.latent_count['true'] += 1
                        latents_list.append(t)
                        latent_poss.extend(varlen_by_batch[i])
                        per_sample.append((i, t.shape[0], poss_cnt))

                if len(latents_list) > 0 and len(latent_poss) > 0:
                    latent_poss = torch.cat(latent_poss, dim=0)
                    latents = torch.cat(latents_list, dim=0).to(input_ids.device)

                    if latents.shape[0] != latent_poss.shape[0]:
                        self.print(f"[WARNING] latents.shape[0] != latent_poss.shape[0], per-sample (idx, lat, poss)={per_sample}, total lat={latents.shape[0]}, poss={int(latent_poss.numel())}. Skip this mirco batch for latent policy gradient computing", flush=True)
                        output_hidden_states = False
                        latent_poss = None
                        latents = None

                else:
                    latent_poss = None
                    latents = None

                if latents is not None and latent_poss is not None:
                    output_hidden_states = True
                else:
                    output_hidden_states = False
            except Exception as e:
                self.print(f"[WARNING] Unexpected error before the latent importance sampling. Fall back to vanilla prob computation for this mirco batch.")
                output_hidden_states = False
                latent_poss = None
                latents = None
                self.print('Exception:', e)
                traceback.print_exc()
                pass
        else:
            output_hidden_states = False

        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        #breakpoint()
        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            # breakpoint()
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                latent_poss=latent_poss,
                latents=latents,
                output_hidden_states=output_hidden_states, # AXZ
                output_attentions=output_attentions,
                return_dict=True # AXZ
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            #breakpoint()
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            if self.config.sampling_strategy == 'monet' and not self.config.ablate_latent:
                if latents is not None:
                    latent_log_probs = compute_latent_log_probs(latent_poss, latents, output.hidden_states[-1], sigma=self.config.monet_rl_sigma)
                    log_probs[latent_poss] = latent_log_probs.to(log_probs.dtype)
                    #pdb.set_trace()
                    # compute_latent_log_probs(latent_poss, latents, output.hidden_states[-1], sigma=10).mean()
            
            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            #breakpoint()
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
        if self.latent_count['total'] % 50 == 0 and self.latent_count['total'] > 0:
            self.print(f"[INFO] [rank {self.rank}] latent_count: {self.latent_count}")
        return log_probs, output.attentions, latent_poss

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto, output_attentions: bool = False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("multi_modal_inputs")
        if "latents" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("latents")

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            1 if output_attentions else self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        # if self.rank == 0:
        #     micro_batches = tqdm(micro_batches, desc="Compute log probs", position=5 if self.config.sampling_strategy == "mc" else 2)
        attn_weights_lst = []
        attn_image_weights_lst = []
        attn_token_weights_lst = []
        response_len_lst = []
        B = 0
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, attn, latent_poss = self._forward_micro_batch(model_inputs, temperature=temperature, output_attentions=output_attentions, caller=f"compute_log_prob output_attentions={output_attentions} micro_batch {B} * len(micro_batch)={len(micro_batch)} / len(micro_batches)={len(micro_batches)}")
            B += 1
            log_probs_lst.append(log_probs)
            # if B == 0:
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] log_probs:", log_probs.shape, log_probs.tolist())
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] response_mask:", micro_batch.batch['response_mask'].shape, micro_batch.batch['response_mask'])
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] input_ids:", model_inputs['input_ids'].shape, model_inputs['input_ids'])
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] latent_poss:", latent_poss.shape, latent_poss.tolist())
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] attn:", len(attn), attn[0].shape, attn[0].tolist())
            if output_attentions:
                # [Rank 0] [actor/compute_log_prob] attn: None 1 torch.Size([1, 28, 13, 10])
                # print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] attn: {len(attn) if isinstance(attn, (list, tuple)) else None} first layer:{attn[0].shape if isinstance(attn, (list, tuple)) and len(attn)>0 else 'N/A'}", "input_ids:", model_inputs['input_ids'].shape, 'log_probs:', log_probs.shape)
                # if K < 2:
                #     print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] attn sample head 0:", attn[0][0, :, :].float().cpu().numpy())
                #     print(f"[Rank {self.rank}] [DataParallelPPOActor/compute_log_prob {B}/{len(micro_batches)}] input_ids:", model_inputs['input_ids'][0, :].cpu().tolist())
                attn = attn[0] # take the attentions from the first layer
                # if attn is None:
                    # print(f"[WARNING] [rank {self.rank}] batch {B} attn is None for output_attentions={output_attentions}, self.actor_module: {self.actor_module}, inputs:")
                    # for k, v in model_inputs.items():
                    #     if isinstance(v, torch.Tensor):
                    #         print(f"[WARNING] [rank {self.rank}] batch {B}  {k}: shape={v.shape} value {v.tolist() if 'ids' in k or 'mask' in k else v}")
                    #     else:
                    #         print(f"[WARNING] [rank {self.rank}] batch {B}  {k}: {type(v)} value {v}")
                attn = attn.mean(dim=1)  # (batch_size, q_len, k_len)

                # Keep only: queries strictly after first <|lvr_end|>, keys strictly inside
                # latent spans (<|lvr_start|>, <|lvr_end|>). Attention tensors may omit BOS.
                bsz, q_len, k_len = attn.shape
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs.get("attention_mask", None)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                else:
                    attention_mask = attention_mask.bool()

                lvr_end_id = int(os.environ.get("ABS_VIS_END_ID", "151667"))
                lvr_start_id = int(os.environ.get("ABS_VIS_START_ID", "151666"))
                bos_token_id = int(os.environ.get("BOS_TOKEN_ID", "151643"))
                image_token_id = int(os.environ.get("IMAGE_TOKEN_ID", "151655"))
                relative_attn = os.environ.get("add_relative_attn_to_reward", "0") == "1"
                latents_obj = model_inputs.get("latents", None)

                query_mask = torch.zeros((bsz, q_len), dtype=torch.bool, device=attn.device)
                key_mask = torch.zeros((bsz, k_len), dtype=torch.bool, device=attn.device)
                key_mask_image = torch.zeros((bsz, k_len), dtype=torch.bool, device=attn.device)
                for i in range(bsz):
                    lvr_steps = None
                    if latents_obj is not None:
                        try:
                            cell = latents_obj[i]
                        except Exception:
                            cell = None
                        if cell is not None:
                            if isinstance(cell, torch.Tensor):
                                if cell.ndim == 1:
                                    lvr_steps = 1
                                elif cell.ndim >= 2:
                                    lvr_steps = int(cell.shape[0])
                            else:
                                arr = np.asarray(cell)
                                if arr.ndim == 1 and arr.size > 0:
                                    lvr_steps = 1
                                elif arr.ndim >= 2:
                                    lvr_steps = int(arr.shape[0])

                    seq_ids = input_ids[i][attention_mask[i]]
                    seq_ids = seq_ids[seq_ids != bos_token_id]

                    # Align tokens to q-axis (right aligned when lengths differ).
                    if seq_ids.numel() >= q_len:
                        q_tokens = seq_ids[-q_len:]
                    else:
                        q_tokens = torch.full((q_len,), -1, dtype=seq_ids.dtype, device=seq_ids.device)
                        q_tokens[-seq_ids.numel():] = seq_ids

                    # Align tokens to k-axis (right aligned when lengths differ).
                    if seq_ids.numel() >= k_len:
                        k_tokens = seq_ids[-k_len:]
                    else:
                        k_tokens = torch.full((k_len,), -1, dtype=seq_ids.dtype, device=seq_ids.device)
                        k_tokens[-seq_ids.numel():] = seq_ids

                    if lvr_end_id > 0:
                        q_end_hits = (q_tokens == lvr_end_id).nonzero(as_tuple=False).squeeze(-1)
                    else:
                        q_end_hits = torch.empty(0, dtype=torch.long, device=q_tokens.device)

                    q_start_hits = (q_tokens == lvr_start_id).nonzero(as_tuple=False).squeeze(-1) if lvr_start_id > 0 else torch.empty(0, dtype=torch.long, device=q_tokens.device)
                    q_from_candidates = []
                    if q_end_hits.numel() > 0:
                        q_from_candidates.append(int(q_end_hits[-1].item()))  # last lvr_end
                    if q_start_hits.numel() > 0 and lvr_steps is not None:
                        q_from_candidates.append(int(q_start_hits[0].item()) + int(lvr_steps))  # first lvr_start + lvr_steps
                    if q_from_candidates:
                        q_from = max(q_from_candidates) + 1
                        if q_from < q_len:
                            query_mask[i, q_from:] = True

                    if lvr_start_id > 0 and lvr_end_id > 0:
                        starts = (k_tokens == lvr_start_id).nonzero(as_tuple=False).squeeze(-1)
                        ends = (k_tokens == lvr_end_id).nonzero(as_tuple=False).squeeze(-1)
                        si, ei = 0, 0
                        remaining_steps = int(lvr_steps) if lvr_steps is not None else None
                        while si < starts.numel() and ei < ends.numel():
                            s_pos = int(starts[si].item())
                            while ei < ends.numel() and int(ends[ei].item()) <= s_pos:
                                ei += 1
                            if ei >= ends.numel():
                                break
                            e_pos = int(ends[ei].item())
                            span_start = s_pos + 1
                            span_cap = max(0, e_pos - span_start)
                            if remaining_steps is None:
                                take = span_cap
                            else:
                                take = min(span_cap, max(0, remaining_steps))
                            if take > 0:
                                key_mask[i, span_start : span_start + take] = True
                            if remaining_steps is not None:
                                remaining_steps -= take
                            si += 1
                            ei += 1

                    if image_token_id > 0:
                        key_mask_image[i] = (k_tokens == image_token_id)

                    # Without <|lvr_start|>/<|lvr_end|>, q_from_candidates is empty so query_mask
                    # would stay all-false and attn_image would be zero. Fall back to every query
                    # position that corresponds to response tokens (image attention from full reply).
                    n = int(seq_ids.numel())
                    response_mask = model_inputs.get("response_mask", None)
                    if response_mask is not None:
                        response_len = int(response_mask[i].sum().item())
                    else:
                        response_len = int(model_inputs["responses"].shape[1])
                    response_len_lst.append(response_len)
                    if not query_mask[i].any():
                        valid_from = max(0, q_len - n)
                        resp_from = max(valid_from, q_len - response_len)
                        if resp_from < q_len:
                            query_mask[i, resp_from:] = True

                # Aggregate attention to latent keys, then average across selected queries.
                key_mask_f = key_mask.to(dtype=attn.dtype)
                key_mask_image_f = key_mask_image.to(dtype=attn.dtype)
                query_mask_f = query_mask.to(dtype=attn.dtype)
                attn_query = (attn * key_mask_f.unsqueeze(1)).sum(dim=2)  # (batch_size, q_len)
                attn_image_query = (attn * key_mask_image_f.unsqueeze(1)).sum(dim=2)  # (batch_size, q_len)
                denom = query_mask_f.sum(dim=1).clamp(min=1.0)
                attn_latent = (attn_query * query_mask_f).sum(dim=1) / denom
                token_attn_latent = (attn_query * query_mask_f)
                attn_image = (attn_image_query * query_mask_f).sum(dim=1) / denom
                if self.K < 5:
                    self.K += 1
                    # processor = AutoProcessor.from_pretrained(f'{os.environ["a_very_big_data_disk"]}/Monet-SFT-7B/stage3')
                    # responsetext = processor.batch_decode(model_inputs["responses"][:1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    # responsetext = [(t.replace('<|image_pad|>', '[IMG]x' + str(t.count('<|image_pad|>')))).replace('<|endoftext|>', '') for t in responsetext]
                    # print(f'[Rank {self.rank}] responsetext:', responsetext[0])
                    # print(f'[Rank {self.rank}][K={self.K}] responses:', model_inputs["responses"][0].shape, model_inputs["responses"][0].tolist())
                    # print(f'[Rank {self.rank}] k_tokens:', k_tokens.shape, k_tokens.tolist())
                    # print(f'[Rank {self.rank}] response_mask:', model_inputs["response_mask"][0].shape, model_inputs["response_mask"][0].tolist())
                    # print(f'[Rank {self.rank}] attention_mask:', model_inputs["attention_mask"][0].tolist())
                    # print(f'[Rank {self.rank}] num latent start:', (model_inputs["input_ids"] == lvr_start_id).sum(-1).tolist())
                    # print(f'[Rank {self.rank}] num latent end:', (model_inputs["input_ids"] == lvr_end_id).sum(-1).tolist())
                    # print(f'[Rank {self.rank}] key_mask_f:', key_mask_f.shape, key_mask_f[0].tolist())
                    # print(f'[Rank {self.rank}] query_mask_f:', query_mask_f.shape, query_mask_f[0].tolist())
                    # print(f'[Rank {self.rank}][K={self.K}] token_attn_latent:', token_attn_latent.shape, token_attn_latent[0].tolist())
                    # print(f'[Rank {self.rank}] attn_latent:', attn_latent.shape, attn_latent[0].tolist())
                    # print(f'[Rank {self.rank}] key_mask_image_f:', key_mask_image_f.shape, key_mask_image_f[0].tolist())
                    # print(f'[Rank {self.rank}] attn_image:', attn_image.shape, attn_image[0].tolist())
                if relative_attn:
                    safe_denom = attn_image.clamp(min=1e-6)
                    attn = torch.where(attn_image > 0, attn_latent / safe_denom, attn_latent)
                else:
                    attn = attn_latent
                attn_weights_lst.append(attn)
                attn_image_weights_lst.append(attn_image)
                attn_token_weights_lst += [token_attn_latent[0]]

        if output_attentions:
            attn_weights_lst = torch.concat(attn_weights_lst, dim=0) # shape (total_batch_size,)
            attn_image_weights_lst = torch.concat(attn_image_weights_lst, dim=0) # shape (total_batch_size,)
            token_attn_token_weights_lst = torch.zeros((len(response_len_lst), response_mask.shape[-1]), dtype=token_attn_latent.dtype, device=token_attn_latent.device)
            for i in range(len(response_len_lst)):
                response_len = response_len_lst[i]
                token_attn_token_weights_lst[i][: response_len] = attn_token_weights_lst[i][-response_len:]

        log_probs = torch.concat(log_probs_lst, dim=0)
        if output_attentions:
            return log_probs, {
                "attn_weights_lst": attn_weights_lst,
                "attn_image_weights_lst": attn_image_weights_lst,
                "attn_token_weights_lst": token_attn_token_weights_lst,
            }
        return log_probs, {}


    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        if self.config.sampling_strategy == "monet":
            non_tensor_select_keys.append('latents')

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        #breakpoint()
        # [DataParallelPPOActor/update_policy] data: 4 split into mini-batches of size 1 chunks: 4
        self.print('[DataParallelPPOActor/update_policy] data:', len(data), 'split into mini-batches of size', self.config.global_batch_size_per_device, 'chunks:', len(data) // self.config.global_batch_size_per_device)
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        #breakpoint()
        metrics = defaultdict(list)
        num_local_samples = len(data)
        per_sample_pg_sum = np.zeros(num_local_samples, dtype=np.float32)
        per_sample_total_sum = np.zeros(num_local_samples, dtype=np.float32)
        per_sample_count = np.zeros(num_local_samples, dtype=np.int32)
        for _ in range(self.config.ppo_epochs):
            # if self.rank == 0:
            #     mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=6 if self.config.sampling_strategy == "mc" else 2)

            for bb1, mini_batch in enumerate(mini_batches):
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                # if self.rank == 0:
                #     micro_batches = tqdm(micro_batches, desc="Update policy", position=7 if self.config.sampling_strategy == "mc" else 3)

                for bb2, micro_batch in enumerate(micro_batches):
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    # advantages: torch.Size([1, 1024]) old_log_probs: torch.Size([1, 1024]) [repeated 3x across cluster]

                    # all return: (bsz, response_length)
                    log_probs, _, _ = self._forward_micro_batch(model_inputs, temperature=temperature, output_attentions=False, caller=f"update_policy [rank {self.rank}, {bb2} / {len(micro_batches)} micro_batches, {bb1} / {len(mini_batches)} mini_batches]")
                    try:
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss
                    except Exception as e:
                        self.print(f"[WARNING] Exception when computing entropy loss: {e} old_log_probs:", old_log_probs.shape, "log_probs", log_probs.shape, "response_mask", response_mask.shape)
                        raise e

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    with torch.no_grad():
                        per_sample_pg = core_algos.compute_policy_loss_per_sample(
                            old_log_probs=old_log_probs.detach(),
                            log_probs=log_probs.detach(),
                            advantages=advantages.detach(),
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                        per_sample_total = per_sample_pg.clone()
                        if "ref_log_probs" in model_inputs:
                            ref_log_probs = model_inputs["ref_log_probs"]
                            kld_detached = core_algos.compute_kl(
                                log_probs=log_probs.detach(),
                                ref_log_probs=ref_log_probs.detach(),
                                kl_penalty=self.config.kl_penalty,
                            )
                            per_sample_kl = VF.masked_mean(kld_detached, response_mask, dim=-1)
                            per_sample_total = per_sample_total + per_sample_kl * self.config.kl_coef

                    mini_start = bb1 * self.config.global_batch_size_per_device
                    micro_start = mini_start + bb2 * self.config.micro_batch_size_per_device_for_update
                    n = min(int(per_sample_pg.shape[0]), max(0, num_local_samples - micro_start))
                    if n > 0:
                        sl = slice(micro_start, micro_start + n)
                        per_sample_pg_sum[sl] += per_sample_pg[:n].detach().float().cpu().numpy()
                        per_sample_total_sum[sl] += per_sample_total[:n].detach().float().cpu().numpy()
                        per_sample_count[sl] += 1

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        valid = per_sample_count > 0
        if np.any(valid):
            pg = np.zeros_like(per_sample_pg_sum, dtype=np.float32)
            total = np.zeros_like(per_sample_total_sum, dtype=np.float32)
            pg[valid] = per_sample_pg_sum[valid] / per_sample_count[valid]
            total[valid] = per_sample_total_sum[valid] / per_sample_count[valid]
            metrics["actor/per_sample_pg_loss"] = pg
            metrics["actor/per_sample_total_loss"] = total

        return metrics

