#!/usr/bin/env bash
#SBATCH --job-name=opd_train
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/pscratch/sd/l/liyantan/opd/logs/slurm-%x-%j.out

# On-policy distillation (OPD) training on a single Perlmutter GPU node.
# 4x A100 80GB; teacher (~4B) + student (~1.7B) + FSDP + vLLM rollout.
# Submit:  sbatch ref/slurm/opd_train.sbatch
#
# Prerequisites (must be in place BEFORE submitting — compute nodes have no
# outbound internet):
#   1. Conda env `verl` installed on $PSCRATCH (see CLAUDE.md §Environment setup).
#   2. HF_HOME primed on login node:
#        huggingface-cli download Qwen/Qwen3-1.7B
#        huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
#   3. Datasets present at $OPD_REPO/datasets/dapo-math-17k.parquet and
#      $OPD_REPO/datasets/test_data/{AIME24,AIME25,AMC23}/test.parquet.

# ---- paths (edit these once for your NERSC account) ----
export OPD_REPO=${OPD_REPO:-/global/homes/l/liyantan/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/l/liyantan/opd}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/huggingface}
export PROJECT_PATH=${PROJECT_PATH:-${DATA_ROOT}/checkpoints}
mkdir -p "${DATA_ROOT}/logs" "${PROJECT_PATH}" "${HF_HOME}"

# ---- NERSC modules + conda ----
module load gpu          2>/dev/null || true
module load nccl         2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true

# Activate the `verl` env (py3.12). Adjust the prefix to wherever you installed it.
source "${DATA_ROOT}/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "${DATA_ROOT}/envs/verl"

cd "${OPD_REPO}"

# ---- offline mode: compute nodes have no outbound internet ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- wandb / swanlab: log offline, sync after job ----
export WANDB_MODE=offline
export WANDB_PROJECT=OnPolicyDistillation
export WANDB_DIR=${DATA_ROOT}/wandb
export WANDB_NAME=opd-${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"

# ---- NCCL: Slingshot-friendly settings ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072

# ---- torch / NCCL timeouts: long PPO rollout phases ----
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200

# ---- CPU threading ----
export OMP_NUM_THREADS=8

# ---- Per-job cache isolation on node-local scratch ----
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/torchinductor_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${SLURM_JOB_ID}"
export OUTLINES_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/outlines_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$OUTLINES_CACHE_DIR"

# ---- vLLM stability knobs (carried over from OPD_2gpu.sh) ----
# Disable FlashInfer sampler — its JIT hits a gcc/nvcc include_next bug on some
# toolchains. Fall back to PyTorch-native sampling.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "================================================================"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node:   ${SLURM_GPUS_PER_NODE}"
echo "OPD_REPO:    ${OPD_REPO}"
echo "DATA_ROOT:   ${DATA_ROOT}"
echo "HF_HOME:     ${HF_HOME}"
echo "================================================================"
nvidia-smi || true
echo "================================================================"

# ---- OPD hyperparameters (overridable via env on `sbatch --export=...`) ----
# Models: use HF repo IDs (resolved from $HF_HOME cache in offline mode).
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}

# 4× A100 80GB sizing. Smaller than the 8× A800 80GB defaults in
# on_policy_distillation.sh, but larger than OPD_2gpu.sh's 2-GPU settings.
export N_GPUS_PER_NODE=4
export NNODES=1
export MODEL_DTYPE=bfloat16
export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIM_OFFLOAD=True       # 4B teacher + 1.7B actor → keep optim on CPU
export REF_PARAM_OFFLOAD=True
export REWARD_PARAM_OFFLOAD=False
export GPU_MEMORY_UTILIZATION=0.75
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-4}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-15360}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-12}
export VAL_N=${VAL_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# Where checkpoints/logs land — PROJECT_PATH is consumed by on_policy_distillation.sh.
export PROJECT_PATH

# Hand off to the existing launcher. It auto-detects $SLURM_JOB_ID and skips the
# local-run log-tee block, but still starts ray and invokes verl.trainer.main_ppo.
bash "${OPD_REPO}/on_policy_distillation.sh"

echo "================================================================"
echo "End time:    $(date)"
echo "To sync wandb on login node:"
echo "  wandb sync ${WANDB_DIR}/offline-run-*"
echo "================================================================"
