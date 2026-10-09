#!/bin/bash
# fura_env.sh — DRIVER env for the SimpleRL-aligned FURA (BlockTT) GRPO run on
# Qwen2.5-7B, on 2 NERSC nodes (2x4 = 8x A100 80GB). Same SimpleRL-Zoo recipe as
# full_env.sh but PEFT_MODE=blocktt (FurA: bf16 frozen core, only the small BTT
# side trains) on FSDP2.
#
# ALGORITHM/HYPERPARAMETERS + the FurA/BlockTT config mirror the single-node
# source of truth scripts/grpo/fura.sh (decomp output_one_block, rank=full, train
# small side, convert svd, factorize-by-head). The HARDWARE/MEMORY/offload knobs +
# the multi-node bootstrap differ, AND s_merged_to is keep_trainable here (NOT the
# single-node script's `frozen`): under 2-node FSDP2, `frozen` folds btt_s into the
# frozen core and export_for_vllm -> materialize_dense_weight() then reshapes a
# DTensor whose non-leftmost dim is sharded ("Attempted to flatten sharded
# dimension 1"). keep_trainable keeps btt_s a separate trainable param and is the
# validated 2-node FURA setting (slurm/opd/fura, slurm/grpo/fura all use it).
#
# Checkpoint/resume: the FSDP checkpoint manager saves the BTT core params +
# optimizer state (NOT merged to nn.Linear); resume_mode=auto restores them.
# Eval/rollout materializes each BTTLinear -> dense weight for vLLM.
# Default LR here is 1e-5; the controller overrides per sweep point (5e-6/1e-6).
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

export WANDB_PROJECT=${WANDB_PROJECT:-nersc_simplerl_qwen2p5_7b}
source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- GRPO estimator + wandb logger ----
export ADV_ESTIMATOR=grpo
export TRAINER_LOGGER="[console,wandb]"
export PROJECT_NAME=${PROJECT_NAME:-nersc_simplerl_qwen2p5_7b}

# ---- model + SimpleRL-structured MATH data (same as full) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-MATH}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/train_data/math-lv3to5_simplerl/train.parquet}
export TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500_simplerl/test.parquet"]'}

# ---- SimpleRL-Zoo hyperparameters (match scripts/grpo/fura.sh) ----
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export TEMPERATURE=${TEMPERATURE:-1.0}
export N_RESPONSES=${N_RESPONSES:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-256}
export LR=${LR:-1e-5}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
export USE_KL=${USE_KL:-True}
export KL_LOSS_COEF=${KL_LOSS_COEF:-1e-4}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
export SHUFFLE=${SHUFFLE:-True}
export ROLLOUT_IS=${ROLLOUT_IS:-none}
export LOG_PROB_TOP_K=0

# ---- FURA = BlockTT (non-quantized bf16 frozen core) ----
export PEFT_MODE=${PEFT_MODE:-blocktt}
export PEFT_TARGET_MODULES=${PEFT_TARGET_MODULES:-all}
export BTT_DECOMP_MODE=${BTT_DECOMP_MODE:-output_one_block}
export BTT_RANK=${BTT_RANK:-full}
export BTT_TRAIN_POSITION=${BTT_TRAIN_POSITION:-small}
export BTT_S_MERGED_TO=${BTT_S_MERGED_TO:-keep_trainable}
export BTT_CONVERT_MODE=${BTT_CONVERT_MODE:-svd}
export BTT_FACTORIZE_BY_HEAD=${BTT_FACTORIZE_BY_HEAD:-True}
export BTT_NORMALIZE_AFTER_UPDATE=${BTT_NORMALIZE_AFTER_UPDATE:-False}
export BTT_QFURA=${BTT_QFURA:-False}

# ---- eval during training on MATH-500 (SimpleRL n=8, temp1.0, top_p0.95) ----
export VAL_N=${VAL_N:-8}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export IS_PLOT=${IS_PLOT:-False}

# keep-1 ckpt + auto-resume + FSDP2 (FSDP1 fails BlockTT's frozen-embedding
# writeback; FSDP2 per-parameter DTensor sharding trains cleanly).
export EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-"trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 trainer.resume_mode=auto \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2"}

# ---- FIXED checkpoint dir + experiment / wandb run id (overridable by controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/simplerl_fura_qwen2p5_7b_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-simplerl_fura_qwen2p5_7b_math}
export WANDB_RUN_ID=${WANDB_RUN_ID:-simplerl_fura_qwen2p5_7b_math}
mkdir -p "${CKPT_PATH}"

# ---- FURA REQUIRES ACTOR_PARAM_OFFLOAD=False (Linear->BTT conversion at init
# needs all target Linear weights on CUDA). Optim offload off (BTT trainable set
# is tiny). 2 nodes x 4 GPUs, no tensor parallel. ----
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIM_OFFLOAD=False
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

echo "SimpleRL fura driver: RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT=${CKPT_PATH} LR=${LR} PROJECT=${PROJECT_NAME} RUN=${WANDB_RUN_ID}"
bash "${OPD_REPO}/grpo.sh"
