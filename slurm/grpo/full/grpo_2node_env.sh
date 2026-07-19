#!/bin/bash
# grpo_2node_env.sh — DRIVER env for the FULL-model GRPO run on Qwen2.5-7B.
# Zero-RL GRPO (rule-based math reward, NO teacher/reward LLM) on 2 nodes
# (8x A100 80GB). Runs as the Ray driver step on the head node (RAY_EXTERNAL=1:
# cluster already up). Train on MATH lv3-5, eval MATH-500 during training.
# Runtime/system env (HF, wandb, vLLM, NCCL, caches) lives in opd_2node_rayenv.sh.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

# wandb project for both GRPO runs (overrides the rayenv default).
export WANDB_PROJECT=${WANDB_PROJECT:-nersc_grpo_qwen2p5_7b}
source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- GRPO estimator + wandb logger ----
export ADV_ESTIMATOR=grpo
export TRAINER_LOGGER="[console,wandb]"
export PROJECT_NAME=${PROJECT_NAME:-nersc_grpo_qwen2p5_7b}

# ---- model + data: Qwen2.5-7B base, MATH lv3-5 train / MATH-500 val ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-MATH}   # -> math-lv3to5 train, MATH-500 val

# ---- GRPO hyperparameters ----
export TEMPERATURE=${TEMPERATURE:-1.0}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
export LR=${LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export USE_KL=${USE_KL:-False}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0}
export LOG_PROB_TOP_K=0        # GRPO: no teacher top-k reward

# keep ONLY the latest checkpoint + auto-resume. Qwen2.5 base takes no thinking arg.
export EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-"trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 trainer.resume_mode=auto"}

# ---- FIXED checkpoint dir + experiment name (overridable by controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/grpo_full_qwen2p5_7b_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_full_qwen2p5_7b_math}
export WANDB_RUN_ID=${WANDB_RUN_ID:-grpo_full_qwen2p5_7b_math}
mkdir -p "${CKPT_PATH}"

# ---- eval during training on MATH-500 ----
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export VAL_N=${VAL_N:-4}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-10}
export IS_PLOT=${IS_PLOT:-False}

# ---- 2 nodes x 4 A100 80GB = 8 GPUs; memory-safe knobs for a 7B actor ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

echo "GRPO full driver: RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT=${CKPT_PATH} LR=${LR}"
bash "${OPD_REPO}/grpo.sh"
