#!/bin/bash
# bp_opd_nersc.sh -- BP-OPD reproducing the known-good NERSC 2-node run on ONE GPU.
#
# Hyperparameters copied verbatim from slurm/opd/full/opd_2node_env.sh (the run
# behind wandb `nersc_opd_qwen4b_1p7b/opd_full_dapo_lr1e-6`, which took
# AMC23 mean@8 0.416 -> 0.538 and AIME24 0.175 -> 0.238 over 160 steps):
#   student  Qwen/Qwen3-1.7B          (non-thinking)
#   teacher  Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
#   DAPO-Math-17k, T=1.0, batch 64, n=4, top-K 16 only_stu, student_p,
#   lr 1e-6, KL 0, token-mean, MODEL_DTYPE=bfloat16, val n=8 @ T=1.0/top-p 0.95
#   on AIME25 + AMC23 + AIME24.
#
# `enable_thinking=False` is load-bearing: Qwen3-1.7B is a hybrid model and
# without it every rollout opens a <think> block and overruns the budget.
#
# Deltas forced by 1x H100 instead of 8x A100:
#   MAX_RESP_LENGTH 7168 -> 3072 (caller can override)
#   GPU_MEMORY_UTILIZATION 0.75 -> 0.45   (teacher is CO-LOCATED here)
#   REWARD_MICRO_BATCH_SIZE_PER_GPU 12 -> 4 (full-vocab logits on one card)
#   REWARD_PARAM_OFFLOAD False -> True
#   (SAVE_FREQ stays 10; max_*_ckpt_to_keep=1 added so disk holds)
#
#   TRAIN_GPU=7 bash scripts/zo_opd/nersc_align/bp_opd_nersc.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH

set -a

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}
HF_HOME=/data/yequan/huggingface

# ---- slurm/opd/full/opd_2node_env.sh, verbatim ----
ADV_ESTIMATOR=token_reward_direct
TEMPERATURE=${TEMPERATURE:-1.0}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
N_RESPONSES=${N_RESPONSES:-4}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
USE_KL=${USE_KL:-False}
# Validation is deliberately the SAME protocol the ZO/ES trainers run -- MATH-500,
# greedy, n=1, 500 prompts, 3072 tokens, graded by the same ttrl_math reward_func --
# so `eval/accuracy` in wandb means the identical thing for BP, es_token and ES-OPD
# and the curves can be overlaid directly. Set VAL_PROTOCOL=aime to get the old
# AIME25+AMC23+AIME24 n=8 T=1.0 sweep back.
VAL_PROTOCOL=${VAL_PROTOCOL:-math500}
if [ "$VAL_PROTOCOL" = "math500" ]; then
  export TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500/test.parquet"]'}
  VAL_N=${VAL_N:-1}
  VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
  VAL_TOP_P=${VAL_TOP_P:-1.0}
  export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
else
  VAL_N=${VAL_N:-8}
  VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
  VAL_TOP_P=${VAL_TOP_P:-0.95}
  export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
fi
TEST_FREQ=${TEST_FREQ:-20}
IS_PLOT=False
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

# max_*_ckpt_to_keep=1 is REQUIRED with SAVE_FREQ=10: verl otherwise keeps every
# checkpoint, and at ~20 GB each that is ~540 GB over a 279-step run.
EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1"

# ---- single-GPU deltas ----
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-4}
REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-True}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
SAVE_FREQ=${SAVE_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-7}
N_GPUS_PER_NODE=1
PARALLEL_SIZE=1
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN
RAY_ISOLATE=1

PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-singlegpu_bp_opd_r${MAX_RESP_LENGTH}_lr${LR}}
LOG_DIR=${LOG_DIR:-logs/nersc}

set +a
mkdir -p "$LOG_DIR"
echo "=== BP-OPD (NERSC-aligned, single GPU) ==="
echo "  student $ACTOR_MODEL_PATH   teacher $REWARD_MODEL_PATH"
echo "  gpu $CUDA_VISIBLE_DEVICES  resp<=$MAX_RESP_LENGTH  n=$N_RESPONSES lr=$LR dtype=$MODEL_DTYPE"
echo "  exp $EXPERIMENT_NAME"
bash on_policy_distillation.sh
