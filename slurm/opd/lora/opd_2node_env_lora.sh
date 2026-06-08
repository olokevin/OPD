#!/bin/bash
# opd_2node_env_lora.sh — DRIVER env for the LoRA OPD run (rank 128, alpha 256),
# mirroring scripts/opd/math/lora.sh on the multi-node DAPO setup. Runtime/system
# env lives in rayenv.sh.
#
# Checkpoint/resume: the FSDP checkpoint saves the LoRA adapter params; resume re-
# applies LoRA (run-root peft_meta absent) then load_checkpoint restores the saved
# adapter weights. Eval/rollout uses vLLM's native LoRA path (max_lora_rank + lora
# weight sync), no manual merge needed. LoRA needs no FSDP2 / use_orig_params.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- models + data (same as full/fura) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}

# ---- hyperparameters; LR defaults to job1's 1e-5 ----
export TEMPERATURE=${TEMPERATURE:-1.0}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export LR=${LR:-1e-5}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export USE_KL=${USE_KL:-False}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}

# ---- LoRA (from scripts/opd/math/lora.sh) ----
export PEFT_MODE=lora
export PEFT_TARGET_MODULES=all
export LORA_RANK=${LORA_RANK:-128}
export LORA_ALPHA=${LORA_ALPHA:-256}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}

export EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode=auto"

# ---- FIXED checkpoint dir + experiment name (overridable by the controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/job1_lora_dapo_lr1e-5}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_lora_dapo_lr1e-5}
export WANDB_RUN_ID=${WANDB_RUN_ID:-opd_lora_dapo_lr1e-5}
mkdir -p "${CKPT_PATH}"

# ---- operational knobs ----
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export VAL_N=${VAL_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-20}
export IS_PLOT=${IS_PLOT:-False}

# ---- sizing (LoRA: base frozen, tiny trainable + optim) ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-False}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-12}

echo "LoRA driver attaching RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT_PATH=${CKPT_PATH} LR=${LR} rank=${LORA_RANK}"
bash "${OPD_REPO}/on_policy_distillation.sh"
