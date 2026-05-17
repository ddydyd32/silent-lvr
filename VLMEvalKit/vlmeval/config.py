from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial
from pathlib import Path
import os
HOME = os.environ.get("HOME")
monet_system_prompt="You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}."

monet_series = {
    "LVR-7B": partial(
        LVRQwen2VLChat,
        model_path=f"{HOME}/LVR-7B",
        min_pixels=3136,
        max_pixels=12845056,
        use_custom_prompt=False,
        use_vllm=False,
        gpu_utils=0.4,
        lvr_steps=8,
        do_sample=False,
        decoding_strategy="steps",
    ),
    "2gpu-LVR-7B-SFT2.0.4-200": partial(
        Qwen2VLChat,
        model_path=f"{HOME}/checkpoints/2gpu-LVR-7B-SFT2.0.4/checkpoint-200",
        min_pixels=1280 * 28 * 28,
        max_pixels=8192 * 28 * 28,
        use_vllm=True,
        gpu_utils=0.5,
    )
}
for exp in Path(f"{HOME}/checkpoints/").iterdir():
    if not exp.is_dir():
        continue
    for gs in exp.glob("global_step_*"):
        model_path = gs / "actor" / "huggingface"
        if model_path.exists():
            step_num = gs.name.replace("global_step_", "")
            modelname = f"{exp.name}-{step_num}"
            if 'lvr' in modelname.lower():
                monet_series[modelname] = partial(
                    Qwen2VLChat,
                    model_path=str(model_path),
                    min_pixels=1280 * 28 * 28,
                    max_pixels=8192 * 28 * 28,
                    use_vllm=True,
                    gpu_utils=0.5,
                )
            else:
                monet_series[modelname] = partial(
                    Qwen2VLChat,
                    model_path=str(model_path),
                    min_pixels=1280 * 28 * 28,
                    max_pixels=8192 * 28 * 28,
                    system_prompt=monet_system_prompt,
                    use_vllm=True,
                    gpu_utils=0.5,
                )
            # print(f"Monet/VLMEvalKit/vlmeval/config.py Adding {modelname} from {model_path}")

supported_VLM = {}

model_groups = [
    monet_series
]

for grp in model_groups:
    supported_VLM.update(grp)
