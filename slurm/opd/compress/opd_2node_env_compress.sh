#!/bin/bash
# opd_2node_env_compress.sh — DRIVER env for compress->OPD (Stage 2) on NERSC.
# The student is a reasoning-aware-COMPRESSED Qwen3-4B (SVD-V2 attn + Nystrom mlp
# @ retain 0.7) built by slurm/opd/compress/nersc_build_students.sh; it is a dense
# 4B-shaped HF checkpoint, trained FULL (PEFT none) — no structural zeros.
#
# Same teacher / dataset / hyperparameters as slurm/opd/full (opd_2node_env.sh):
#   teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500, DAPO-Math-17k,
#   T=1.0, mbs=64, n=4, top-k=16/only_stu, rw=student_p, no-KL, lr=1e-6.
# Only difference vs full: the student is the 4B compressed model (not Qwen3-1.7B),
# so the memory-fit knobs are tuned for a 4B actor + 4B teacher (optim offload on,
# ppo_max_token_len/max_num_batched_tokens capped at 16384, gpu util 0.6) — the
# same budget the single-GPU compress->OPD pipeline used.
#
# Runtime/system env (HF, wandb, vLLM, NCCL, caches) lives in opd_2node_rayenv.sh.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- models + data (teacher/dataset identical to full; student = compressed 4B) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${DATA_ROOT}/compress_opd/students/svd_nystrom_r07_forward}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}

# ---- hyperparameters: identical to slurm/opd/full ----
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

# ---- full model (no PEFT) ----
export PEFT_MODE=none

# wandb/project: own project for the compress thread (rayenv pins WANDB_PROJECT;
# override it here so these runs don't land in the 1.7B project).
export PROJECT_NAME=${PROJECT_NAME:-opd_compress_svd_nystrom}
export WANDB_PROJECT=${PROJECT_NAME}

# Non-thinking + keep ONLY the latest checkpoint + auto-resume + 4B memory-fit
# (cap ppo_max_token_len/max_num_batched_tokens at 16384; the launcher otherwise
# recomputes them to 32768 from MAX_RESP_LENGTH=7168, which OOMs a 4B actor).
export EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode=auto \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384"

# ---- FIXED checkpoint dir + experiment name (overridable by the controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/opd_compress_svd_nystrom_forward}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_compress_svd_nystrom_forward}
export WANDB_RUN_ID=${WANDB_RUN_ID:-opd_compress_svd_nystrom_forward}
mkdir -p "${CKPT_PATH}"

# ---- operational knobs: save often so a 4h expiry loses little ----
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export VAL_N=${VAL_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-20}
export IS_PLOT=${IS_PLOT:-False}

# ---- 2 nodes x 4 A100 80GB = 8 GPUs; 4B actor + 4B teacher memory budget ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-True}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-8}

echo "COMPRESS-OPD driver: actor=${ACTOR_MODEL_PATH} teacher=${REWARD_MODEL_PATH}"
echo "  RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT_PATH=${CKPT_PATH} LR=${LR}"
bash "${OPD_REPO}/on_policy_distillation.sh"
