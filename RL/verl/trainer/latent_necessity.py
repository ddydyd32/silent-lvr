# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Latent necessity reward: re-roll with question + (text1, <start><end>) without
# continuous tokens between <start> and <end>, then score Rn from (A0, A1).
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from ..protocol import DataProto
from ..utils import torch_functional as VF
from verl.workers.reward.function import replace_abs_vis_token_content

try:
    from examples.reward_function.monet_reward_function import extract_and_check as _extract_check_answer
except ImportError:  # pragma: no cover
    _extract_check_answer = None
import traceback

def first_latent_span_in_response(
    response_token_ids: torch.Tensor,
    valid_len: int,
    start_id: int,
    end_id: int,
) -> Optional[Tuple[int, int]]:
    """Return (start_pos, end_pos) in [0, valid_len) for first <start>...<end> span, or None."""
    v = int(valid_len)
    if v <= 0:
        return None
    seq = response_token_ids[:v]
    s_hits = (seq == start_id).nonzero(as_tuple=False)
    if s_hits.numel() == 0:
        return None
    s = int(s_hits[0].item())
    e_hits = (seq[s + 1 :] == end_id).nonzero(as_tuple=False)
    if e_hits.numel() == 0:
        return None
    e = s + 1 + int(e_hits[0].item())
    return s, e


def latent_necessity_scalar(a0: bool, a1: bool, n=1) -> float:
    # expected value assuming equal probability for each action is 0.
    if a0 and a1:
        return -0.5 * n
    if a0 and (not a1):
        return n
    if (not a0) and a1:
        return -0.5 * n
    return 0.0


def extend_position_ids(
    pos_valid: torch.Tensor, num_append: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Match vLLM rollout: new positions = last + 1, +2, ... (same for each mrope row)."""
    if num_append <= 0:
        return pos_valid
    t = torch.arange(1, num_append + 1, device=device, dtype=dtype)
    if pos_valid.dim() == 1:
        last = pos_valid[-1:]
        return torch.cat([pos_valid, last + t], dim=0)
    if pos_valid.dim() == 2 and pos_valid.size(0) == 3:
        last = pos_valid[:, -1:]
        return torch.cat([pos_valid, last + t.view(1, -1).expand(3, -1)], dim=-1)
    raise ValueError(f"Unsupported position_ids shape {pos_valid.shape}")


def build_latent_necessity_r1_subbatch(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_snapshots: np.ndarray,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
) -> Optional[Tuple[DataProto, List[int]]]:
    """
    Build a sub-batch for ablated re-rollout using per-row vLLM rollout snapshots
    (``monet_rollout_vllm_input``): exact ``prompt_token_ids`` / ``multi_modal_data`` from
    the first generate, plus ``prompt_position_ids_unpadded`` for consistent mrope/flat
    positions. Does not read padded ``prompts`` tensors from the training batch.

    vLLM is invoked via ``non_tensor_batch['precomputed_vllm_inputs']`` on the worker
    (see ``vLLMRollout.generate_sequences``), so the second pass matches the first on
    the question prefix and multimodal payload.
    """
    bsz = int(responses.size(0))
    if rollout_snapshots is None or len(rollout_snapshots) != bsz:
        return None

    r1_max_seq_len = int(max_prompt_length) + int(max_response_length)
    start_id = int(os.environ.get("ABS_VIS_START_ID", "151666"))
    end_id = int(os.environ.get("ABS_VIS_END_ID", "151667"))

    device = responses.device
    source_rows: List[int] = []
    input_rows: List[torch.Tensor] = []
    att_rows: List[torch.Tensor] = []
    pos_rows: List[torch.Tensor] = []
    precomputed: List[Dict[str, Any]] = []

    for i in range(bsz):
        snap = rollout_snapshots[i]
        if snap is None:
            print(f"[build_latent_necessity_r1_subbatch] missing rollout snapshot for row {i}; skipping")
            continue
        vlen = int(response_mask[i].sum().item())
        span = first_latent_span_in_response(responses[i], vlen, start_id, end_id)
        if span is None:
            # print(f"[build_latent_necessity_r1_subbatch] no valid latent span found for row {i}; skipping")
            continue
        s, e = span
        valid = responses[i, :vlen]
        ablate_suffix = valid[: s + 1].detach().cpu().tolist() + [int(valid[e].item())]
        base_prompt = list(snap["prompt_token_ids"])
        new_raw = base_prompt + ablate_suffix
        if len(new_raw) > r1_max_seq_len:
            print(f"[build_latent_necessity_r1_subbatch] new input length {len(new_raw)} exceeds max {r1_max_seq_len} for row {i}; skipping")
            continue

        pos_u = torch.from_numpy(snap["prompt_position_ids_unpadded"]).to(
            device=device, dtype=torch.long
        )
        pos_ext = extend_position_ids(
            pos_u, len(ablate_suffix), device, pos_u.dtype if pos_u.numel() else torch.long
        )
        if int(pos_ext.shape[-1]) != len(new_raw):
            print(f"[build_latent_necessity_r1_subbatch] position_ids length {pos_ext.shape[-1]} does not match input length {len(new_raw)} for row {i}; skipping")
            continue

        input_1d = torch.tensor(new_raw, dtype=torch.long, device=device)
        att_1d = torch.ones_like(input_1d, dtype=torch.long, device=device)
        try:
            pi, am, pids = VF.postprocess_data(
                input_1d,
                att_1d,
                pos_ext,
                max_length=r1_max_seq_len,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation="error",
            )
        except RuntimeError:
            print(f"[build_latent_necessity_r1_subbatch] error occurred while postprocessing row {i}; skipping")
            traceback.print_exc()
            continue

        mm = snap.get("multi_modal_data")
        precomputed.append({"prompt_token_ids": new_raw, "multi_modal_data": mm})

        input_rows.append(pi.unsqueeze(0))
        att_rows.append(am.unsqueeze(0))
        pos_rows.append(pids.unsqueeze(0))
        source_rows.append(i)

    if not input_rows:
        print("[build_latent_necessity_r1_subbatch] no valid rows found for re-rollout, returning None")
        return None

    n_r = len(source_rows)
    pc_arr = np.empty(n_r, dtype=object)
    for j in range(n_r):
        pc_arr[j] = precomputed[j]

    out = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.cat(input_rows, dim=0),
                "attention_mask": torch.cat(att_rows, dim=0),
                "position_ids": torch.cat(pos_rows, dim=0),
            },
            batch_size=n_r,
        ),
        non_tensor_batch={"precomputed_vllm_inputs": pc_arr},
    )
    return out, source_rows


def decode_response_text_monet(
    tokenizer,
    responses_row: torch.Tensor,
    response_mask_row: torch.Tensor,
) -> str:
    """
    Match BatchFunctionRewardManager (monet reward): how `predict` strings are built for
    `compute_score` / `extract_and_check`.
    """
    vlen = int(response_mask_row.sum().item())
    valid_response_ids = responses_row[:vlen]
    s = replace_abs_vis_token_content(
        tokenizer.decode(valid_response_ids, skip_special_tokens=False)
    ).replace("<|endoftext|>", "").replace("<|im_end|>", "")
    return re.sub(r"\s*(<|>|/)\s*", r"\1", s)


def r1_scalar_accuracy(
    tokenizer,
    responses_row: torch.Tensor,
    response_mask_row: torch.Tensor,
    ground_truth: str,
) -> float:
    """Rule-based correctness only: same as `compute_score` + `extract_and_check` (no API)."""
    s = decode_response_text_monet(tokenizer, responses_row, response_mask_row)
    if _extract_check_answer is not None:
        return 1.0 if _extract_check_answer(s, ground_truth) else 0.0
    from verl.workers.rollout.utils.util import extract_and_check as ucheck

    return 1.0 if ucheck(s, ground_truth) else 0.0


def is_correctness_positive(x) -> bool:
    """`rule_then_api_batch_judge` can yield float, bool, or 0/1; training uses 1.0 = correct."""
    if x is None:
        return False
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x) == 1.0
    if isinstance(x, bool):
        return x
    return bool(x)


def _trunc_one_line(s: str, max_chars: int) -> str:
    s = s.replace("\n", "[nl]").replace("\r", "[cr]")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def maybe_print_latent_necessity_replay_debug(
    *,
    trank: int,
    tokenizer,
    batch: DataProto,
    r1_out: DataProto,
    r1_rows: List[int],
    a0_list: List[bool],
    a1_list: List[bool],
    acc_list: List[float],
    rn_vals: List[float],
    tlr_before_rn: torch.Tensor,
    tlr_after_rn: torch.Tensor,
    token_level_scores: torch.Tensor,
    judge_mode: str,
) -> None:
    """
    Visual inspection of ablation replay: set ``LATENT_NECESSITY_REPLAY_DEBUG=1``.

    Optional env:

    - ``LATENT_NECESSITY_REPLAY_DEBUG_ALL_RANKS=1`` — print on every rank (default: rank 0 only).
    - ``LATENT_NECESSITY_REPLAY_DEBUG_MAX`` — max replay rows to print (default 8).
    - ``LATENT_NECESSITY_REPLAY_DEBUG_CHARS`` — max chars per text field (default 600).
    """
    flag = os.environ.get("LATENT_NECESSITY_REPLAY_DEBUG", "").strip().lower()
    if flag not in ("1", "true", "yes", "y"):
        return
    all_ranks = os.environ.get("LATENT_NECESSITY_REPLAY_DEBUG_ALL_RANKS", "0").strip() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if not all_ranks and int(trank) != 0:
        return

    max_rows = int(os.environ.get("LATENT_NECESSITY_REPLAY_DEBUG_MAX", "32"))
    max_c = int(os.environ.get("LATENT_NECESSITY_REPLAY_DEBUG_CHARS", "1024"))
    bsz = int(batch.batch["responses"].size(0))
    n = min(len(r1_rows), max_rows)

    probs = batch.non_tensor_batch.get("problem", None)
    gts = batch.non_tensor_batch.get("ground_truth", None)

    lines: List[str] = []
    lines.append(
        f"[LN_REPLAY_DEBUG] rank={trank} judge={judge_mode} "
        f"replay_subbatch={len(r1_rows)} full_batch={bsz} r1_row_indices={r1_rows}"
    )
    for k in range(n):
        j = k
        src = r1_rows[j]
        a0, a1 = a0_list[j], a1_list[j]
        rn = float(rn_vals[j])
        ridx = int(batch.batch["response_mask"][src].sum().item()) - 1
        ridx = max(0, ridx)
        main_last = float(tlr_before_rn[src, ridx].item())
        after_last = float(tlr_after_rn[src, ridx].item())
        mean_score = float(
            (
                token_level_scores[src]
                * batch.batch["response_mask"][src].to(token_level_scores.dtype)
            )
            .sum()
            .item()
        )
        acc_b = float(acc_list[src]) if src < len(acc_list) else float("nan")

        ptxt = ""
        if probs is not None and src < len(probs):
            ptxt = _trunc_one_line(str(probs[src]), max_c)
        gtxt = ""
        if gts is not None and src < len(gts):
            gtxt = _trunc_one_line(str(gts[src]), max_c)

        orig = decode_response_text_monet(
            tokenizer,
            batch.batch["responses"][src],
            batch.batch["response_mask"][src],
        )
        rep = decode_response_text_monet(
            tokenizer,
            r1_out.batch["responses"][j],
            r1_out.batch["response_mask"][j],
        )

        lines.append(
            f"[LN_REPLAY_DEBUG] --- sample {k + 1}/{n}  batch_row={src}  r1_subrow={j} ---"
        )
        lines.append(
            f"  acc(from_reward_metrics)={acc_b:.4g}  a0={int(a0)}  a1={int(a1)}  "
            f"Rn={rn:+.3f}  (main_last={main_last:.4g}  last_after_Rn={after_last:.4g}  sum_scores_on_resp={mean_score:.4g})"
        )
        if ptxt:
            lines.append(f"  problem: {ptxt}")
        if gtxt:
            lines.append(f"  ground_truth: {gtxt}")
        lines.append(f"  response_orig: {_trunc_one_line(orig, max_c)}")
        lines.append(f"  response_replay: {_trunc_one_line(rep, max_c)}")

    if len(r1_rows) > n:
        lines.append(
            f"[LN_REPLAY_DEBUG] ... truncated; {len(r1_rows) - n} more rows (raise LATENT_NECESSITY_REPLAY_DEBUG_MAX)"
        )
    print("\n".join(lines), flush=True)
