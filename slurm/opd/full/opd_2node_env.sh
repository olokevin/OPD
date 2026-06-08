#!/bin/bash
# opd_2node_env.sh — OPD DRIVER env for the 2-node (8x A100 80GB) run. Runs as
# the Ray driver step on the head node (RAY_EXTERNAL=1: cluster already up). Sets
# the user's hyperparameters, a FIXED checkpoint dir (keep only latest, resume
# auto), then hands off to on_policy_distillation.sh. Runtime/system env (HF,
# wandb, vLLM, NCCL, caches) lives in opd_2node_rayenv.sh — sourced here for
# the driver and by every ray-start step so the actors inherit it.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ================================================================
# Models + hyperparameters (user table) — teacher Qwen3-4B-Non-Thinking-RL-Math
# (Step500), student Qwen3-1.7B, both non-thinking.
# ================================================================
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}

export TEMPERATURE=${TEMPERATURE:-1.0}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export LR=${LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export USE_KL=${USE_KL:-False}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}

# Non-thinking + keep ONLY the latest checkpoint + auto-resume.
export EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode=auto"

# ---- FIXED checkpoint dir + experiment name (stable across relaunches) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/nersc_opd_qwen4b_1p7b_2node}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nersc_opd_qwen4b_1p7b_2node}
mkdir -p "${CKPT_PATH}"

# ---- operational knobs: save often so a 4h expiry loses little ----
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export VAL_N=${VAL_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-20}
export IS_PLOT=${IS_PLOT:-False}

# ---- 2 nodes x 4 A100 80GB = 8 GPUs ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-12}

# RAY_EXTERNAL=1 + RAY_ADDRESS are exported by the caller (opd_2node_inside.sh).
echo "DRIVER attaching to RAY_ADDRESS=${RAY_ADDRESS} (NNODES=${NNODES}) CKPT_PATH=${CKPT_PATH}"

bash "${OPD_REPO}/on_policy_distillation.sh"
