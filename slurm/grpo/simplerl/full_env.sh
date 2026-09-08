#!/bin/bash
# full_env.sh — DRIVER env for the SimpleRL-aligned FULL-model GRPO run on
# Qwen2.5-7B, on 2 NERSC nodes (2x4 = 8x A100 80GB). Runs as the Ray driver step
# on the head node (RAY_EXTERNAL=1; cluster already up via opd_2node_inside.sh).
#
# ALGORITHM/HYPERPARAMETERS are the SimpleRL-Zoo Qwen2.5-7B recipe and are kept
# byte-for-byte in sync with the single-node source of truth scripts/grpo/full.sh
# (max_resp 8192, mbs 256, train_batch 1024, n 8, LR 5e-7, KL_LOSS_COEF 1e-4,
# entropy 0.001, low_var_kl, SHUFFLE=True, ROLLOUT_IS=none, SimpleRL prompts).
# Only the HARDWARE/MEMORY/offload knobs + the multi-node bootstrap (RAY_EXTERNAL,
# resume, fixed ckpt) differ from that file — a 2-node FSDP shard needs more
# offloading than a single 8-GPU node with NVLink.
#
# Runtime/system env (HF, wandb, vLLM, NCCL, JIT caches) lives in
# opd_2node_rayenv.sh (sourced below; every ray-start srun step also sources it).
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

# wandb project for the SimpleRL-aligned campaign (verl uses PROJECT_NAME as the
# wandb project via trainer.project_name; WANDB_PROJECT set for consistency).
export WANDB_PROJECT=${WANDB_PROJECT:-nersc_simplerl_qwen2p5_7b}
source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
cd "${OPD_REPO}"

# ---- GRPO estimator + wandb logger ----
export ADV_ESTIMATOR=grpo
export TRAINER_LOGGER="[console,wandb]"
export PROJECT_NAME=${PROJECT_NAME:-nersc_simplerl_qwen2p5_7b}

# ---- model + SimpleRL-structured MATH data (qwen25-math-cot prompt copies) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-7B}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-MATH}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/train_data/math-lv3to5_simplerl/train.parquet}
export TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500_simplerl/test.parquet"]'}

# ---- SimpleRL-Zoo Qwen2.5-7B hyperparameters (match scripts/grpo/full.sh) ----
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export TEMPERATURE=${TEMPERATURE:-1.0}
export N_RESPONSES=${N_RESPONSES:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-256}
export LR=${LR:-5e-7}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
export USE_KL=${USE_KL:-True}
export KL_LOSS_COEF=${KL_LOSS_COEF:-1e-4}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
export SHUFFLE=${SHUFFLE:-True}
export ROLLOUT_IS=${ROLLOUT_IS:-none}
export LOG_PROB_TOP_K=0        # GRPO: no teacher top-k reward

# ---- eval during training on MATH-500 (SimpleRL n=8, temp1.0, top_p0.95) ----
export VAL_N=${VAL_N:-8}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-5}
export IS_PLOT=${IS_PLOT:-False}

# keep-1 ckpt + auto-resume across the 4h interactive cap.
export EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-"trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 trainer.resume_mode=auto"}

# ---- FIXED checkpoint dir + experiment / wandb run id (overridable by controller) ----
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/simplerl_full_qwen2p5_7b_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-simplerl_full_qwen2p5_7b_math}
export WANDB_RUN_ID=${WANDB_RUN_ID:-simplerl_full_qwen2p5_7b_math}
mkdir -p "${CKPT_PATH}"

# ---- 2 nodes x 4 A100 80GB = 8 GPUs; no tensor parallel; memory-safe for a 7B
# actor at 8192 response length (2-node FSDP shard -> optim offload on). ----
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-2}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# ---- full finetune (no PEFT) ----
export PEFT_MODE=${PEFT_MODE:-none}

echo "SimpleRL full driver: RAY_ADDRESS=${RAY_ADDRESS} NNODES=${NNODES} CKPT=${CKPT_PATH} LR=${LR} PROJECT=${PROJECT_NAME} RUN=${WANDB_RUN_ID}"
bash "${OPD_REPO}/grpo.sh"
