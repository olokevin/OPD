#!/bin/bash
# scripts/grpo/full.sh — full-finetune GRPO (zero-RL) on Qwen2.5-7B.
#
# GRPO (adv_estimator=grpo, rule-based math reward, no teacher/RM) via the shared
# grpo.sh base launcher. Defaults reproduce the SimpleRL-Zoo Qwen2.5-7B recipe
# (hkust-nlp/simpleRL-reason, arXiv:2503.18892):
#
#   train_grpo_math_tune_ray.sh --model_name Qwen-2.5-7B --max_response_length 8192 \
#     --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 \
#     --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 --rollout_tp 2 --save_freq 5
#
# with defaults LR=5e-7, ppo_mini_batch_size=256, kl_loss_type=low_var_kl,
# total_epochs=20, dataset=simplelr_math_35 (MATH lv3-5). ONE deliberate deviation:
# no tensor parallel (PARALLEL_SIZE=1) per the run requirement, vs their rollout_tp=2.
#
# Override per-run, e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS_PER_NODE=4 bash scripts/grpo/full.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set -a  # auto-export every assignment below to grpo.sh

# ---- models + data ----
ACTOR_MODEL_PATH=Qwen/Qwen2.5-7B
HF_HOME=${HF_HOME:-/data/yequan/huggingface}
# SimpleRL-Zoo trains on simplelr_math_35 (MATH level 3-5). In this repo the MATH
# branch routes to datasets/train_data/math-lv3to5/train.parquet + MATH-500 eval.
TRAIN_DATASET_NAME=MATH
PROJECT_NAME=grpo-qwen25-7b

# ---- hyperparameters (exact SimpleRL-Zoo Qwen2.5-7B recipe) ----
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=8192               # --max_response_length 8192
MAX_VAL_RESP_LENGTH=8192
TEMPERATURE=1.0
N_RESPONSES=8                      # --rollout_n 8
TRAIN_BATCH_SIZE=1024             # --train_batch_size 1024 (rollout batch)
MINI_BATCH_SIZE=256               # ppo_mini_batch_size (script default)
LR=5e-7                            # LEARNING_RATE default
TOTAL_EPOCHS=20                   # TOTAL_EPOCHS default
USE_KL=True                       # GRPO uses KL loss
KL_LOSS_COEF=1e-4                 # --kl_loss_coef 0.0001 (0.5B-14B models)
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0.001               # --entropy_coeffient 0.001
MODEL_DTYPE=bfloat16
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
SAVE_FREQ=5                       # --save_freq 5
TEST_FREQ=5                       # TEST_FREQ default
IS_PLOT=False

# ---- no tensor parallel ----
PARALLEL_SIZE=1

# ---- hardware / memory-fit (single 8-GPU node by default) ----
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.75        # --rollout_gpu_memory_util 0.75

# ---- full finetune (no PEFT) ----
PEFT_MODE=none

set +a

bash grpo.sh
