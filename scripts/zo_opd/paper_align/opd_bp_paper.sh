#!/bin/bash
# opd_bp_paper.sh -- BP-based OPD, aligned to the paper's Section 3.1 setting.
#
#   student  Qwen3-1.7B-Base          (paper: base student)
#   teacher  lllyx/Qwen3-4B-Base-GRPO (paper: zero-RL teacher, pattern-matched
#                                      to a base student -- Figure 2)
#   prompts  DAPO-Math-17k-Processed  (the \boxed{} template the teacher was
#                                      RL'd on -- Section 5.2 "prompt template")
#   OPD defaults from Table 2: T=1.0, batch 64, rollout n=4, top-K=16,
#   only_stu, LR 1e-6, no KL, token-mean.
#
# Two deltas vs the paper, both deliberate:
#   * MAX_RESP_LENGTH 3072 (paper 7168) -- single-GPU memory budget.
#   * validation on MATH-500 + AMC23 + AIME24 at n=4 (paper AIME/AMC avg@16 at
#     31744) -- MATH-500 is the only one of the three with enough problems to
#     resolve a 1.7B-scale change at this token budget.
#
# MODEL_DTYPE=fp32 is load-bearing: verl does `actor_module.to(model_dtype)`
# (fsdp_workers.py:449), so bfloat16 makes the OPTIMIZER MASTER WEIGHTS bf16 and
# an Adam-sized 1e-6 update lands on only ~1.4% of weights -- the model is
# numerically frozen. fp32 master + bf16 compute is verl's default for a reason.
#
#   TRAIN_GPU=6 bash scripts/zo_opd/paper_align/opd_bp_paper.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH

set -a  # auto-export to on_policy_distillation.sh

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-lllyx/Qwen3-4B-Base-GRPO}
HF_HOME=/data/yequan/huggingface

ADV_ESTIMATOR=token_reward_direct
TRAIN_DATASET_NAME=DAPO-Math-17k-Processed
TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k-processed.parquet}
TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500/test.parquet","datasets/test_data/AMC23/test.parquet","datasets/test_data/AIME24/test.parquet"]'}

# --- paper Table 2 ---
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=1.0
MINI_BATCH_SIZE=64          # train_batch_size = 64 * PARALLEL_SIZE(1)
N_RESPONSES=${N_RESPONSES:-4}
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
LR=${LR:-1e-6}
USE_KL=False
LOSS_AGG_MODE=token-mean
TOTAL_EPOCHS=1

MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3072}

# --- validation ---
VAL_N=${VAL_N:-4}
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
IS_PLOT=False

# --- precision / memory (single H100 NVL 95GB, actor + teacher co-located) ---
MODEL_DTYPE=${MODEL_DTYPE:-fp32}          # fp32 MASTER weights (see header)
REWARD_MODEL_DTYPE=${REWARD_MODEL_DTYPE:-bfloat16}   # teacher: inference only
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# --- hardware ---
CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-6}
N_GPUS_PER_NODE=1
PARALLEL_SIZE=1
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN
RAY_ISOLATE=1

PROJECT_NAME=${PROJECT_NAME:-opd-paper-align}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bp_opd_q34bgrpo_to_1p7bbase_r${MAX_RESP_LENGTH}_n${N_RESPONSES}_lr${LR}_$(date +%m%d_%H%M)}
LOG_DIR=${LOG_DIR:-logs/train}

set +a

mkdir -p "$LOG_DIR"
echo "=== BP-OPD (paper-aligned) ==="
echo "  student  $ACTOR_MODEL_PATH"
echo "  teacher  $REWARD_MODEL_PATH"
echo "  data     $TRAIN_DATASET"
echo "  gpu      $CUDA_VISIBLE_DEVICES   resp<=$MAX_RESP_LENGTH  n=$N_RESPONSES  lr=$LR  dtype=$MODEL_DTYPE"
echo "  exp      $EXPERIMENT_NAME"
bash on_policy_distillation.sh
