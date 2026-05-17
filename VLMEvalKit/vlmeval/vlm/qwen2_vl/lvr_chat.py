"""Hugging Face Transformers inference for LVR checkpoints using QwenWithLVR (lvr_official)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

from ...smp import get_gpu_memory
from ..base import BaseModel
from .model import Qwen2VLChat, VLLM_MAX_IMAGE_INPUT_NUM
from .prompt import Qwen2VLPromptMixin

# Image bounds from lvr_official/src/params.py DataArguments (used by train_lvr / datasets).
LVR_DEFAULT_MIN_PIXELS = 3136
LVR_DEFAULT_MAX_PIXELS = 12845056  # == 16384 * 28 * 28


def _resolve_lvr_official_root(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("LVR_OFFICIAL_ROOT")
    if env:
        return os.path.abspath(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "lvr_official"
        marker = cand / "src" / "model" / "qwen_lvr_model.py"
        if cand.is_dir() and marker.is_file():
            return str(cand)
    raise FileNotFoundError(
        "Could not find lvr_official (expected a sibling folder of the repo root containing "
        "src/model/qwen_lvr_model.py). Set environment variable LVR_OFFICIAL_ROOT to the "
        "absolute path of your lvr_official directory."
    )


def _strip_checkpoint_tags(path: str) -> str:
    out = path
    for tags in ("-gt", "-zerobias", "-b", "-random", "-noimage", "-0token", "-0bias"):
        if tags in out:
            out = out.split(tags)[0]
    return out


class LVRQwen2VLChat(Qwen2VLChat):
    """Qwen2.5-VL + LVR (QwenWithLVR) on Transformers.

    When ``min_pixels`` / ``max_pixels`` are omitted, uses ``lvr_official`` training defaults
    (``DataArguments.image_min_pixels`` / ``image_max_pixels`` in ``src/params.py``), same as
    ``AutoProcessor.from_pretrained(..., min_pixels=..., max_pixels=...)`` in ``train_lvr.py``.
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = False,
        lvr_steps: int = 8,
        decoding_strategy: str = "steps",
        lvr_official_root: str | None = None,
        trust_remote_code: bool = True,
        lvr_generate_max_new: int | None = None,
        ablate_random_latent=False,
        **kwargs,
    ):
        kwargs.pop("use_vllm", None)
        kwargs.pop("use_lmdeploy", None)
        kwargs.pop("gpu_utils", None)

        Qwen2VLPromptMixin.__init__(self, use_custom_prompt=use_custom_prompt)
        BaseModel.__init__(self)

        self.K = 1
        self.min_pixels = LVR_DEFAULT_MIN_PIXELS if min_pixels is None else min_pixels
        self.max_pixels = LVR_DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.ablate_random_latent = ablate_random_latent
        print("[LVRQwen2VLChat] ablate_random_latent: ", self.ablate_random_latent, 'lvr_steps: ', lvr_steps)
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print(
                "The total number of video tokens might become too large, resulting in an overly long input sequence. "  # noqa: E501
                "We recommend lowering **total_pixels** to below **24576 × 28 × 28**."
            )
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop("fps", 2)
        self.nframe = kwargs.pop("nframe", 128)
        if self.fps is None and self.nframe is None:
            print(
                "Warning: fps and nframe are both None, using default nframe/fps setting in qwen-vl-utils, "
                "the fps/nframe setting in video dataset is omitted"
            )
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        self.lvr_steps = int(lvr_steps)
        self.lvr_decoding_strategy = decoding_strategy
        self.lvr_generate_max_new = lvr_generate_max_new if lvr_generate_max_new is not None else 64
        self._lvr_print_once = True

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0

        self.use_vllm = False
        self.use_lmdeploy = False
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM

        lvr_root = _resolve_lvr_official_root(lvr_official_root)
        if lvr_root not in sys.path:
            sys.path.insert(0, lvr_root)

        from transformers import AutoProcessor, Qwen2_5_VLConfig

        chkpt = _strip_checkpoint_tags(model_path)
        config = Qwen2_5_VLConfig.from_pretrained(chkpt, trust_remote_code=trust_remote_code)

        use_lvr_branch = (
            "vanilla_sft" not in chkpt.lower()
            and ("_checkpoints" in chkpt or "lvr" in chkpt.lower() or "global_step" in chkpt)
        )
        if not use_lvr_branch:
            raise ValueError(
                f"LVRQwen2VLChat: checkpoint path {chkpt!r} does not match the LVR load rule "
                "(need 'lvr', '_checkpoints', or 'global_step' in path, and not 'vanilla_sft'). "
                "Use Qwen2VLChat for plain Qwen2.5-VL checkpoints."
            )

        from src.train.monkey_patch_forward_lvr_rl import (  # type: ignore  # noqa: E402
            replace_qwen2_5_with_mixed_modality_forward_lvr_rl_4_54_0,
        )
        from src.model.qwen_lvr_model import QwenWithLVR  # type: ignore  # noqa: E402

        replace_qwen2_5_with_mixed_modality_forward_lvr_rl_4_54_0()

        load_kw = dict(
            config=config,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
        )
        try:
            self.model = QwenWithLVR.from_pretrained(
                chkpt, attn_implementation="flash_attention_2", **load_kw
            )
        except Exception as err:
            logging.warning("LVR load with flash_attention_2 failed (%s); retrying with sdpa.", err)
            self.model = QwenWithLVR.from_pretrained(chkpt, attn_implementation="sdpa", **load_kw)

        if "vis-layer" in chkpt.lower():
            self.model.vis_layer = True
        else:
            self.model.vis_layer = False

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            chkpt,
            trust_remote_code=trust_remote_code,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        from qwen_vl_utils import process_vision_info

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        if videos:
            raise NotImplementedError(
                "LVRQwen2VLChat currently supports image inputs only (video in message is not supported)."
            )

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        decoding_strategy = self.lvr_decoding_strategy
        steps = self.lvr_steps
        lvr_steps = [steps]
        kwargs = {"use_cache": True}

        start = getattr(self.model.config, "lvr_start_id", 151665)
        latent_end = getattr(self.model.config, "lvr_latent_end_id", 151667)
        end = getattr(self.model.config, "lvr_end_id", 151668)
        newline = self.processor.tokenizer("\n<answer>", add_special_tokens=False).input_ids

        with torch.no_grad():
            if "-gt" in decoding_strategy:
                raise NotImplementedError(
                    "LVRQwen2VLChat: decoding_strategy contains '-gt' but VLMEvalKit does not wire gt_bbox. "
                    "Use decoding_strategy without '-gt', or extend this class with bbox support."
                )
            if "monet" in decoding_strategy.lower():
                raise NotImplementedError("LVRQwen2VLChat: monet decoding_strategy is not supported here.")
            if "steps" in decoding_strategy:
                kwargs["decoding_strategy"] = decoding_strategy.split("-gt")[0].split("-zerobias")[0]
                kwargs["lvr_steps"] = lvr_steps
                kwargs["output_hidden_states"] = False
                kwargs["output_attentions"] = False
                kwargs["return_dict_in_generate"] = True

                if lvr_steps[0] == 0:
                    if self._lvr_print_once:
                        print(
                            "### Warning: lvr_steps is 0: no LVR refinement steps; model inserts "
                            "<lvr_start>, <lvr_latent_end>, <lvr_end> placeholders (same as lvr_official). ###"
                        )
                        print(newline)
                        self._lvr_print_once = False
                    inputs["input_ids"] = torch.cat(
                        [
                            inputs["input_ids"],
                            torch.tensor([[start]], device=self.model.device),
                            torch.tensor([[latent_end]], device=self.model.device),
                            torch.tensor([[end]], device=self.model.device),
                        ],
                        dim=1,
                    )
                    inputs["attention_mask"] = torch.cat(
                        [
                            inputs["attention_mask"],
                            torch.ones(
                                [1, inputs["input_ids"].shape[1] - inputs["attention_mask"].shape[1]],
                                device=self.model.device,
                            ),
                        ],
                        dim=1,
                    )

                gen_kw = {**self.generate_kwargs, **kwargs, "max_new_tokens": self.lvr_generate_max_new}
                outputs = self.model.generate(**inputs, **gen_kw)
                generated_ids = outputs.sequences
            else:
                gen_kw = {**self.generate_kwargs, **kwargs}
                generated_ids = self.model.generate(**inputs, **gen_kw)

            if "gt" in decoding_strategy:
                generated_ids_trimmed = generated_ids
            else:
                input_item = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs_embeds"]
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_item, generated_ids)
                ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end_idx = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end_idx = i
                    break
                if i == lt - 1:
                    end_idx = lt
                    break
            if end_idx is not None:
                response = resp[:end_idx]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
