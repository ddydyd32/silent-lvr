#!/bin/bash -l
#SBATCH --job-name=lvr-rl-thymerl-necessity0-attn1-latent0-includelatentsample1-kl0.01-temp0.5-rollout8-numgpu4
#SBATCH --output=lvr-rl-thymerl-necessity0-attn1-latent0-includelatentsample1-kl0.01-temp0.5-rollout8-numgpu4.-%j.log
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=72:15:00
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=256G
[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

# ---- configuration ----
TOTAL_TIME=$((72*60*60-45*60))     # total allowed runtime in seconds (example: 05:45:00)
# TOTAL_TIME=$((330))     # total allowed runtime in seconds (example: 05:45:00)
WARNING=300              # send signal this many seconds before end
TIMEOUT=$((TOTAL_TIME - WARNING))

# ---- signal handler ----
handler() {
    echo "[BASH] Signal received at $(date)"
    echo "[BASH] Forwarding SIGUSR1 to Python process ${PID}"

    if [[ -n "${PID}" ]]; then
        kill -USR1 "${PID}" 2>/dev/null
        wait "${PID}"
    fi

    echo "[BASH] Cleanup complete"
    exit 0
}

trap handler USR1 SIGINT SIGTERM


set -x

export CUDA_VISIBLE_DEVICES=0,1

export GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxx
export a_very_big_data_disk=xxxxxxxxxxxxxxxxx
export home=xxxxxxxxxxxxxxxxx
export UV_HOME=${a_very_big_data_disk}/uv_home
export UV_CACHE_DIR=${a_very_big_data_disk}/uv_cache
export HF_DATASETS_CACHE=${a_very_big_data_disk}/huggingface/datasets
export HF_HOME=${a_very_big_data_disk}/huggingface
export MPLCONFIGDIR=${a_very_big_data_disk}/matplotlib
export MPLCONFIGDIR=$a_very_big_data_disk/matplotlib
export TRITON_HOME=$a_very_big_data_disk/triton
export UV_VENV_CLEAR=0

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=ignore
export monet_DEBUG=0

unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=120
export VLLM_NO_USAGE_STATS=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=16
export RAY_NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
export RAY_METRICS_EXPORT_PORT=3333
export MASTER_PORT=33303
export RAY_LOG_TO_STDERR=0
export RAY_LOCAL_MODE=0
export RAY_task_exit_on_oom=1
export RAY_SPILL_DIR=${a_very_big_data_disk}/ray_spill
export RAY_TMPDIR=${a_very_big_data_disk}/ray_tmp
mkdir -p ${RAY_SPILL_DIR}
mkdir -p ${RAY_TMPDIR}
export RAY_local_fs_capacity_threshold=0.98
export RAY_OBJECT_STORE_MEMORY=$((8 * 1024 * 1024 * 1024)) # 64GB
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300
# conda activate easyr1
cd $HOME/Monet/RL
rm quit
source lvr/bin/activate
which python
MONET_RL_PATCH=1 # overwrite the transformers and vllm forward module
MODEL_PATH=${a_very_big_data_disk}/checkpoints/2gpu-LVR-7B-SFT2.0.4/checkpoint-200
latent_size=8
export LATENT_SIZE=${latent_size}
ROLLOUT_N=8
TEMPERATURE=0.5
GPU_UTILIZATION=0.92
SELECT_ACC_THRESHOLD=0.6
KL_COEF=0.01
ORI_BSZ=64
ONLINE_ACCUM_SIZE=256
ROLLOUT_BATCH_SIZE=4
ONLINE_ACCUM_SIZE=64
global_batch_size=1
micro_batch_size_per_device_for_update=1
micro_batch_size_per_device_for_experience=2
max_steps=400
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1
N_GPUS_PER_NODE=${RAY_NUM_GPUS}
echo "N_GPUS_PER_NODE: ${N_GPUS_PER_NODE}"
TENSOR_PARALLEL_SIZE=1
MONET_RL_SIGMA=10.0
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=1024
export ABS_VIS_END_ID=151668
export ABS_VIS_START_ID=151665
export LATENT_START_ID=${ABS_VIS_START_ID}
export LATENT_END_ID=${ABS_VIS_END_ID}
export latent_necessity_reward=0
export LATENT_NECESSITY_REPLAY_DEBUG=1
export LATENT_NECESSITY_REWARD_N=0.0
export add_attn_to_reward=1
export add_token_attn_to_reward=0
export add_relative_attn_to_reward=0
export add_latent_to_reward=0
export include_latent_sample=1
export STRIP_LATENT_IN_ROLLOUT=0
export RAY_memory_monitor_refresh_ms=1
export RAY_gcs_rpc_server_reconnect_timeout_s=20

# datasetname=virl39k
datasetname=thymerl
# datasetname=thymerl-latentwins
# datasetname=thymerl-nolatentwins
if [ "$datasetname" = "virl39k" ]; then
    train_files=${a_very_big_data_disk}/virl39k_train.parquet
    val_files=${a_very_big_data_disk}/virl39k_val.parquet
elif [ "$datasetname" = "thymerl-latentwins" ]; then
    train_files=${a_very_big_data_disk}/generated_responses_thyme-rl_Monet-SFT_temp0.5_p0.99_max1024_siglip_clusters/generated_responses_thyme-rl_Monet-SFT_temp0.5_p0.99_max1024_latent_wins.parquet
    val_files=${a_very_big_data_disk}/Thyme-RL/data@val
elif [ "$datasetname" = "thymerl-nolatentwins" ]; then
    train_files=${a_very_big_data_disk}/generated_responses_thyme-rl_Monet-SFT_temp0.5_p0.99_max1024_siglip_clusters/generated_responses_thyme-rl_Monet-SFT_temp0.5_p0.99_max1024_no_latent_wins.parquet
    val_files=${a_very_big_data_disk}/Thyme-RL/data@val
elif [ "$datasetname" = "thymerl" ]; then
    train_files=${a_very_big_data_disk}/Thyme-RL/data
    val_files=${a_very_big_data_disk}/Thyme-RL/data@val
else
    echo "Unknown dataset: ${datasetname}"
    exit 1
fi

experiment_name=lvr-rl-${datasetname}-necessity${LATENT_NECESSITY_REWARD_N}-attn${add_attn_to_reward}-latent${add_latent_to_reward}-includelatentsample${include_latent_sample}-kl${KL_COEF}-temp${TEMPERATURE}-rolloutn${ROLLOUT_N}-numgpu${N_GPUS_PER_NODE}
export save_checkpoint_path=${a_very_big_data_disk}/checkpoints/${experiment_name}

mkdir -p ./training_logs
export RAY_memory_usage_threshold=0.95
nvidia-smi
export HOME=$home
ml cuda/12.6
export TMPDIR=${a_very_big_data_disk}/tmp
which python

python -m verl.trainer.main \
    trainer.save_checkpoint_path=${save_checkpoint_path} \
    trainer.load_checkpoint_path=auto \
    trainer.val_before_train=false \
    trainer.max_steps=${max_steps} \
    trainer.save_freq=20 \
    trainer.save_limit=-1 \
    config=examples/config_lvr.yaml \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.global_batch_size=${global_batch_size} \
    worker.actor.micro_batch_size_per_device_for_update=${micro_batch_size_per_device_for_update} \
    worker.actor.micro_batch_size_per_device_for_experience=${micro_batch_size_per_device_for_experience} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.sampling_strategy=monet \
    worker.rollout.max_num_seqs=128 \
    worker.rollout.monet.select_acc_threshold=${SELECT_ACC_THRESHOLD} \
    worker.rollout.online_difficulty_sampling=true \
    worker.reward.reward_function=./examples/reward_function/monet_reward_function.py:compute_score_w_prev_correctness \
    worker.reward.repetition_penalty=true \
    worker.rule_based_judge.judge_function=./examples/reward_function/monet_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name="gemini-2.5-pro" \
    worker.actor.monet_rl_sigma=${MONET_RL_SIGMA} \
    worker.ref.monet_rl_sigma=${MONET_RL_SIGMA} \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.online_accum_size=${ONLINE_ACCUM_SIZE} \
    data.dataloader_num_workers=8 \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH}  &
PID=$!


echo "[BASH] Python PID: ${PID}"

# ---- watchdog timer ----
(
    sleep "${TIMEOUT}"
    echo "[WATCHDOG] Sending SIGUSR1 to Python (${PID}) at $(date)"
    kill -USR1 "${PID}" 2>/dev/null
) &
WATCHDOG_PID=$!

echo "[BASH] Watchdog PID: ${WATCHDOG_PID}"
echo "[BASH] Timeout set for ${TIMEOUT} seconds, with a warning signal ${WARNING} seconds before the end."

# ---- wait for python ----
wait "${PID}"

# ---- cleanup watchdog if python finished early ----
kill "${WATCHDOG_PID}" 2>/dev/null

echo "[BASH] Training finished at $(date)"
echo "[BASH] Finished running on node $(hostname)"

exit 0

