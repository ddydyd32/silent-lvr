# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Monet RL forward overrides for transformers >= 4.54.0.

The 4.54 Qwen2.5-VL implementation collapses Qwen2_5_VLAttention/FA2/SDPA into
a single class that dispatches via ALL_ATTENTION_FUNCTIONS, and inserts a new
Qwen2_5_VLTextModel between Qwen2_5_VLModel and the decoder layers. The big
self-contained modeling file under monet_models/transformers/monet_modeling_qwen2_5_vl.py
mirrors the 4.51.3 layout and cannot be wired into 4.54 without major surgery.

Instead, under 4.54 we apply two minimal overrides:

  * Qwen2_5_VLForConditionalGeneration.forward
      Adds optional `latent_poss` / `latents` kwargs and, when both are
      provided, splices `latents` into `inputs_embeds[0, latent_poss]` before
      forwarding to `self.model(...)`. This mirrors what the 4.51.3 monet
      forward does in monet_modeling_qwen2_5_vl.py.

  * Qwen2_5_VLDecoderLayer.forward
      Gates `output_attentions` so it only fires for `layer_idx == 0`. This
      reproduces the layer-0-only attention output behavior the 4.51.3 monet
      implementation uses for the attention-based reward features in
      verl/workers/actor/dp_actor.py. The override is written so it works on
      pristine upstream 4.54 code (no prior hand-edit to the venv required).
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput  # noqa: F401  (re-exported for parity)

print(f"[Monet/RL/monet_models/transformers/monet_modeling_qwen2_5_vl_4_54.py] imported.")
def qwen2_5_vl_for_cond_gen_forward_4_54(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional["Cache"] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    latent_poss: Optional[torch.LongTensor] = None,
    latents: Optional[torch.Tensor] = None,
    **kwargs,
):
    """4.54.0 replacement for Qwen2_5_VLForConditionalGeneration.forward with latent injection."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLCausalLMOutputWithPast,
    )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # Build inputs_embeds ourselves if we need to splice latents into them.
    # Visual (pixel_values / pixel_values_videos) merging is left to
    # Qwen2_5_VLModel.forward, which uses input_ids to identify image/video
    # token positions when both inputs_embeds and input_ids are provided.
    if inputs_embeds is None and input_ids is not None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if (
        inputs_embeds is not None
        and latent_poss is not None
        and latents is not None
        and latent_poss.numel() > 0
    ):
        inputs_embeds[0, latent_poss] = latents.to(inputs_embeds.dtype)
    kwargs['return_dict'] = True
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


def qwen2_5_vl_decoder_layer_forward_4_54(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """4.54.0 replacement for Qwen2_5_VLDecoderLayer.forward.

    Functionally identical to the upstream forward, except that
    `output_attentions` is gated to `self.self_attn.layer_idx == 0` so the
    attention-based reward path in dp_actor only sees layer-0 attentions, as
    in the 4.51.3 monet implementation.
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    layer0_only_output_attentions = bool(output_attentions) and (self.self_attn.layer_idx == 0)

    K = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=layer0_only_output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states, self_attn_weights = K[:2]
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs
