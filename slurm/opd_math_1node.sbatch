#!/usr/bin/env bash
#SBATCH --job-name=opd_math
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/pscratch/sd/y/yequan/opd/logs/slurm-%x-%j.out

# On-policy distillation (OPD) — MATH (SimpleRL-Zoo-aligned) on ONE Perlmutter
# GPU node (4x A100 80GB). Combines the NERSC plumbing of opd_train.sbatch with
# the MATH train/eval knobs of scripts/opd/math/full.sh, scaled to 4 GPUs.
#
# Run interactively (recommended for first smoke test):
#   salloc --nodes 1 --qos interactive --time 2:00:00 \
#          --constraint 'gpu&hbm80g' --gpus 4 --account m4788_g --no-shell
#   srun --jobid=<ID> bash slurm/opd_math_1node.sbatch
# Or submit:  sbatch slurm/opd_math_1node.sbatch

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
# Any library that phones home will hang on a dead socket until it times out
# (the "silent stuck in web access" failure mode). Force everything offline.
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
export WANDB_PROJECT=OnPolicyDistillation
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
echo "HF_HOME:     ${HF_HOME}"
echo "================================================================"
nvidia-smi || true
echo "================================================================"

# ================================================================
# MATH (SimpleRL-Zoo-aligned) knobs — from scripts/opd/math/full.sh
# ================================================================
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}

export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-MATH}   # -> math-lv3to5 train + MATH-500 eval
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3072}
export TEMPERATURE=${TEMPERATURE:-0.6}
export N_RESPONSES=${N_RESPONSES:-8}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_N=${VAL_N:-4}
export PROJECT_NAME=${PROJECT_NAME:-opd-qwen-math}
export LR=${LR:-1e-6}
export IS_PLOT=${IS_PLOT:-False}

# ---- 4x A100 80GB sizing (single node) ----
# Pin visible GPUs to the node's 4 (Slurm's cgroup exposes them as 0-3).
# Without this, on_policy_distillation.sh defaults CUDA_VISIBLE_DEVICES=6,7.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-12}

# Smoke-run defaults: pure training steps, no early validation/checkpointing so
# the first several steps land fast. Override SAVE_FREQ/TEST_FREQ for real runs.
export SAVE_FREQ=${SAVE_FREQ:-1000}
export TEST_FREQ=${TEST_FREQ:-1000}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

export PROJECT_PATH

bash "${OPD_REPO}/on_policy_distillation.sh"

echo "================================================================"
echo "End time: $(date)"
echo "================================================================"
