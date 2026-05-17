#!/bin/bash

# curl -LsSf https://astral.sh/uv/install.sh | sh

export HOME=xxxxxxxxxxxxxxxxxxxxxxx
export a_very_big_data_disk=xxxxxxxxxxxxxxxxxxxxxxx
export UV_HOME=xxxxxxxxxxxxxxxxxxxxxxx
export UV_CACHE_DIR=$HOME/uv_cache
export HF_DATASETS_CACHE=$HOME/huggingface/datasets
export HF_HOME=$HOME/huggingface
export UV_VENV_CLEAR=0
ml cuda/12.6
set -x

cd $HOME
mkdir -p $HOME/Monet/RL
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# set up evaluation env
cd $HOME/Monet
uv venv monet --python=3.10 --allow-existing
source monet/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); import vllm; import transformers;"
deactivate

# set up training env
cd $HOME/Monet/RL
uv venv easyr1 --python=3.11 --allow-existing
source easyr1/bin/activate
uv pip install -r requirements.txt
uv pip install $HOME/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -c "import torch; print(torch.cuda.is_available()); import vllm; import transformers; import flash_attn"

# download datasets and models
hf download --repo-type dataset yifanzhang114/Thyme-RL --local-dir ${a_very_big_data_disk}/Thyme-RL
hf download NOVAglow646/Monet-SFT-7B --local-dir ${a_very_big_data_disk}/Monet-SFT-7B --include "stage3/*"

# set up lvr training env
cd $HOME/Monet/RL
uv venv lvr --python=3.11 --allow-existing
source lvr/bin/activate
uv pip install -r requirements_lvr.txt
uv pip install $HOME/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -c "import torch; print(torch.cuda.is_available()); import vllm; import transformers; import flash_attn"
