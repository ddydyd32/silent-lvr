# sitecustomize.py (top-level)
# Runs in every Python process (parent + spawned workers)

import os, sys, importlib
os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module("Monet_models.monet_gpu_model_runner")
print('[sitecustomize] vLLM runner patched via sitecustomize:', __file__)