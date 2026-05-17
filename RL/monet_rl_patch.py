# /Monet/sitecustomize.py
# -----------------------------------------------------------------------
# This patch is only for RL codes (vllm==0.8.5), not for SFT and inference.
# So use `MONET_RL_PATCH=1` only in the RL script.
#
# It dispatches to two implementations depending on the installed
# `transformers` version, so the same Monet/RL repo can train:
#   * Monet under transformers 4.51.3 (legacy class-replacement path)
#   * LVR    under transformers 4.54.0 (minimal forward-injection path)
# -----------------------------------------------------------------------

import os, sys, importlib, inspect

print(f"[sitecustomize] imported from {__file__}", file=sys.stderr)


def _transformers_version_tuple():
    """Return (major, minor) of the installed transformers package."""
    try:
        import transformers

        parts = transformers.__version__.split(".")
        return tuple(int(p) for p in parts[:2])
    except Exception as exc:
        print(f"[Monet RL patch] could not parse transformers version: {exc}", file=sys.stderr)
        return (0, 0)


def patch_qwen_monet_4_51():
    """Original Monet patch path for transformers 4.51.3.

    Replaces Qwen2.5-VL FA2/SDPA/DecoderLayer/Model/ForCondGen forwards (and
    the QWEN2_5_VL_ATTENTION_CLASSES registry) with the monet-customized
    versions in monet_models/transformers/monet_modeling_qwen2_5_vl.py.
    """
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q_official
    import monet_models.transformers.monet_modeling_qwen2_5_vl as q_monet
    print("[Monet RL patch] replacing... (4.51.3 path)")
    off_cls = q_official.Qwen2_5_VLForConditionalGeneration
    mon_cls = q_monet.Qwen2_5_VLForConditionalGeneration
    q_official.QWEN2_5_VL_ATTENTION_CLASSES["flash_attention_2"] = q_monet.QWEN2_5_VL_ATTENTION_CLASSES["flash_attention_2"]
    q_official.QWEN2_5_VL_ATTENTION_CLASSES["sdpa"] = q_monet.QWEN2_5_VL_ATTENTION_CLASSES["sdpa"]

    try:
        print(
            "[Monet RL patch] official forward sig before:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] monet    forward sig:",
            inspect.signature(mon_cls.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    off_cls.forward = mon_cls.forward
    q_official.Qwen2_5_VLModel.forward = q_monet.Qwen2_5_VLModel.forward
    q_official.Qwen2_5_VLFlashAttention2.forward = q_monet.Qwen2_5_VLFlashAttention2.forward
    q_official.Qwen2_5_VLSdpaAttention.forward = q_monet.Qwen2_5_VLSdpaAttention.forward
    q_official.Qwen2_5_VLDecoderLayer.forward = q_monet.Qwen2_5_VLDecoderLayer.forward
    #off_cls.__init__ = mon_cls.__init__

    try:
        print(
            "[Monet RL patch] official forward sig after:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] q_official.Qwen2_5_VLModel.forward sig after:",
            inspect.signature(q_official.Qwen2_5_VLModel.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] q_official.Qwen2_5_VLFlashAttention2.forward sig after:",
            inspect.signature(q_official.Qwen2_5_VLFlashAttention2.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    print(
        "[Monet RL patch] Patched methods of Qwen2_5_VLForConditionalGeneration in-place (4.51.3)",
        file=sys.stderr,
    )


# Backwards-compat alias: older code paths and external scripts still call
# patch_qwen_monet() directly.
def patch_qwen_monet():
    return patch_qwen_monet_4_51()


def patch_qwen_monet_4_54():
    """Minimal Monet patch path for transformers >= 4.54.0.

    In 4.54 the per-impl FA2/SDPA classes and the QWEN2_5_VL_ATTENTION_CLASSES
    registry are gone, the text decoder is wrapped in Qwen2_5_VLTextModel under
    Qwen2_5_VLModel.language_model, and the single Qwen2_5_VLAttention.forward
    already recomputes attn_weights when output_attentions=True. So we only
    override:

      * Qwen2_5_VLForConditionalGeneration.forward — to thread `latent_poss`
        and `latents` into inputs_embeds before the model forward.
      * Qwen2_5_VLDecoderLayer.forward — to gate `output_attentions` to
        layer 0 only (matches the 4.51 monet behavior on a pristine 4.54
        install).
    """
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q_official
    from monet_models.transformers.monet_modeling_qwen2_5_vl_4_54 import (
        qwen2_5_vl_decoder_layer_forward_4_54,
        qwen2_5_vl_for_cond_gen_forward_4_54,
    )

    print("[Monet RL patch] replacing... (4.54.0 path)")

    off_cls = q_official.Qwen2_5_VLForConditionalGeneration
    try:
        print(
            "[Monet RL patch] official forward sig before:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] monet 4.54 forward sig:",
            inspect.signature(qwen2_5_vl_for_cond_gen_forward_4_54),
            file=sys.stderr,
        )
    except Exception:
        pass

    off_cls.forward = qwen2_5_vl_for_cond_gen_forward_4_54
    q_official.Qwen2_5_VLDecoderLayer.forward = qwen2_5_vl_decoder_layer_forward_4_54

    try:
        print(
            "[Monet RL patch] official forward sig after:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] q_official.Qwen2_5_VLDecoderLayer.forward sig after:",
            inspect.signature(q_official.Qwen2_5_VLDecoderLayer.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    # vLLM-side weight name remap: in HF >= 4.54, FSDP state-dict keys exposed
    # by Qwen2_5_VLModel are prefixed with `model.language_model.`. The default
    # `model.` -> `language_model.model.` rule would double the segment, so we
    # install a more specific prefix mapping first. Best-effort: skip if vLLM
    # isn't importable in this process (e.g. trainer-only workers).
    try:
        from vllm.model_executor.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as _VllmQwen25VL,
        )
        from vllm.model_executor.models.utils import WeightsMapper

        _VllmQwen25VL.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
                "model.": "language_model.model.",
            }
        )
        print("[Monet RL patch] patched vLLM qwen2_5_vl weights mapper", file=sys.stderr)
    except Exception as exc:
        print(f"[Monet RL patch] qwen2_5_vl vllm mapper patch skipped: {exc}", file=sys.stderr)

    print(
        "[Monet RL patch] Patched methods of Qwen2_5_VLForConditionalGeneration in-place (4.54.0)",
        file=sys.stderr,
    )


def patch():
    print("[sitecustomize] patch() called", file=sys.stderr)

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    # Defaults match the original Monet model. LVR users override these in the
    # launch script (typically 151665 / 151668).
    os.environ["LATENT_START_ID"] = os.getenv("LATENT_START_ID")
    os.environ["LATENT_END_ID"] = os.getenv("LATENT_END_ID")
    os.environ["AVT_LATENT_HOOK_BIN"] = "1"
    print("can I import")
    sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
        "monet_models.vllm.monet_gpu_model_runner"
    )
    print("can I patch")

    version = _transformers_version_tuple()
    print(f"[Monet RL patch] detected transformers version tuple: {version}", file=sys.stderr)
    if version >= (4, 54):
        patch_qwen_monet_4_54()
    else:
        patch_qwen_monet_4_51()
    '''sys.modules[
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    ] = importlib.import_module(
        "monet_models.transformers.monet_modeling_qwen2_5_vl"
    )'''

    print("[Monet RL patch] vllm & transformers patched", file=sys.stderr)

patch()


