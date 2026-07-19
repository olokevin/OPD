#!/bin/bash
# grpo_2node_env_fura.sh — DRIVER env for the FURA (BlockTT) GRPO run on
# Qwen2.5-7B. Same GRPO setup as grpo_2node_env.sh, but PEFT_MODE=blocktt (FurA:
# bf16 frozen core, only the small BTT side trains) on FSDP2. Train MATH lv3-5,
# eval MATH-500 during training. Runtime/system env lives in opd_2node_rayenv.sh.
#
# Checkpoint/resume: the FSDP checkpoint manager saves the actual BTT core params
# + optimizer state (NOT merged to nn.Linear); resume_mode=auto restores them.
# Eval/rollout uses adapter.export_for_vllm which materializes each BTTLinear ->
# dense weight for vLLM.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

export WANDB_PROJECT=${WANDB_PROJECT:-nersc_grpo_qwen2p5_7b}
source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- GRPO estimator + wandb logger ----
export ADV_ESTIMATOR=grpo
export TRAINER_LOGGER="[console,wandb]"
export PROJECT_NAME=${PROJECT_NAME:-nersc_grpo_qwen2p5_7b}

# ---- model + data (same as full) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-MATH}

# ---- GRPO hyperparameters; FURA typically wants a larger LR than full ----
export TEMPERATURE=${TEMPERATURE:-1.0}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
export LR=${LR:-1e-5}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export USE_KL=${USE_KL:-False}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0}
export LOG_PROB_TOP_K=0

# ---- FURA = BlockTT (non-quantized frozen core) ----
export PEFT_MODE=blocktt
export PEFT_TARGET_MODULES=all
export BTT_DECOMP_MODE=output_one_block
export BTT_RANK=full
export BTT_TRAIN_POSITION=small
export BTT_S_MERGED_TO=keep_trainable
export BTT_CONVERT_MODE=svd
export BTT_FACTORIZE_BY_HEAD=True
export BTT_NORMALIZE_AFTER_UPDATE=False
export BTT_QFURA=False

# keep-1 ckpt + resume + FSDP2 (FSDP1 fails BlockTT's frozen-embedding writeback).
export EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-"trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 trainer.resume_mode=auto \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2"}

# ---- FIXED checkpoint dir + experiment name (overridable by controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/grpo_fura_qwen2p5_7b_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_fura_qwen2p5_7b_math}
export WANDB_RUN_ID=${WANDB_RUN_ID:-grpo_fura_qwen2p5_7b_math}
mkdir -p "${CKPT_PATH}"

# ---- eval during training on MATH-500 ----
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export VAL_N=${VAL_N:-4}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-10}
export IS_PLOT=${IS_PLOT:-False}

# ---- FURA REQUIRES ACTOR_PARAM_OFFLOAD=False (Linear->BTT conversion needs the
# target weights on CUDA). Optim offload off (BTT trainable set is tiny). ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIM_OFFLOAD=False
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

echo "GRPO fura driver: RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT=${CKPT_PATH} LR=${LR}"
bash "${OPD_REPO}/grpo.sh"
