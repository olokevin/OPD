#!/usr/bin/env bash
#SBATCH --job-name=nersc_opd_qwen4b_1p7b
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/pscratch/sd/y/yequan/opd/logs/slurm-%x-%j.out

# OPD: teacher Qwen3-4B-Non-Thinking-RL-Math (Step500) -> student Qwen3-1.7B,
# both NON-THINKING. One Perlmutter GPU node (4x A100 80GB). Hyperparameters per
# the user-provided table (= on_policy_distillation.sh DAPO-Math-17k defaults).
# wandb project: nersc_opd_qwen4b_1p7b.
#
# Smoke (interactive):  srun --jobid=<ID> --ntasks=1 --gpus=4 bash slurm/opd_qwen4b_1p7b.sbatch
# Full run (queue):     sbatch slurm/opd_qwen4b_1p7b.sbatch

# ---- paths ----
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/huggingface}
export PROJECT_PATH=${PROJECT_PATH:-${DATA_ROOT}/checkpoints}
mkdir -p "${DATA_ROOT}/logs" "${PROJECT_PATH}" "${HF_HOME}"

# ---- NERSC modules + conda ----
module load gpu              2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate "${DATA_ROOT}/envs/verl"

cd "${OPD_REPO}"

# ---- offline / no-telemetry: compute nodes have NO outbound internet ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export SWANLAB_MODE=local
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0

# ---- wandb (offline; sync after job on a login node) ----
export WANDB_PROJECT=nersc_opd_qwen4b_1p7b
export WANDB_DIR=${DATA_ROOT}/wandb
mkdir -p "${WANDB_DIR}"

# ---- NCCL / Slingshot ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export OMP_NUM_THREADS=8

# ---- per-job cache isolation on node-local scratch ----
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
JOBTAG="${SLURM_JOB_ID:-local$$}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${JOBTAG}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/torchinductor_${JOBTAG}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${JOBTAG}"
export OUTLINES_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/outlines_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$OUTLINES_CACHE_DIR"

# ---- vLLM stability ----
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "================================================================"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID:-<none>}"
echo "OPD_REPO:    ${OPD_REPO}"
echo "DATA_ROOT:   ${DATA_ROOT}"
echo "================================================================"
nvidia-smi || true
echo "================================================================"

# ================================================================
# Hyperparameters — from the user-provided table
# ================================================================
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}

# Dataset: DAPO-Math-17k (the default these hyperparameters correspond to).
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}

export TEMPERATURE=${TEMPERATURE:-1.0}          # Training temperature
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}   # Global = Mini batch size = 64 (PARALLEL_SIZE=1)
export N_RESPONSES=${N_RESPONSES:-4}            # Rollout number
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}     # LogProb top-K
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}   # Student Top-K
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export LR=${LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}          # Epoch
export USE_KL=${USE_KL:-False}                  # KL coefficient 0.0
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
# Top-p 1.0: vLLM training-rollout default top_p is 1.0 (no override needed).

# wandb project name (verl uses trainer.project_name as the wandb project).
export PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}

# Both models are NON-THINKING -> disable the Qwen3 thinking chat template.
export EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False"

# ---- operational knobs (not in the table; single-node-sane defaults) ----
# Cap val response length so max_model_len stays 8192 (the launcher default
# MAX_VAL_RESP_LENGTH=31744 would bloat vLLM KV cache to 32768 and slow rollout).
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export VAL_N=${VAL_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export IS_PLOT=${IS_PLOT:-False}

# ---- 4x A100 80GB sizing (single node) ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-12}

export PROJECT_PATH

bash "${OPD_REPO}/on_policy_distillation.sh"

echo "================================================================"
echo "End time: $(date)"
echo "================================================================"
