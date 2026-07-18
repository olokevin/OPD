#!/bin/bash
# scripts/grpo/full.sh — full-finetune GRPO (zero-RL) on Qwen2.5-7B.
#
# GRPO (adv_estimator=grpo, rule-based math reward, no teacher/RM) via the shared
# grpo.sh base launcher. Settings are aligned to the OPD SLURM FurA launch
# (slurm/opd/fura/opd_2node_env_fura.sh) and the SimpleRL-Zoo Qwen2.5-7B recipe
# (hkust-nlp/simpleRL-reason, arXiv:2503.18892): DAPO-Math-17k train,
# bf16, no tensor parallel (PARALLEL_SIZE=1).
#
# Override per-run, e.g. `LR=5e-7 bash scripts/grpo/full.sh` or
# `CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS_PER_NODE=4 bash scripts/grpo/full.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set -a  # auto-export every assignment below to grpo.sh

# ---- models + data ----
ACTOR_MODEL_PATH=Qwen/Qwen2.5-7B
HF_HOME=${HF_HOME:-/data/yequan/huggingface}
TRAIN_DATASET_NAME=DAPO-Math-17k
TRAIN_DATASET=datasets/dapo-math-17k.parquet   # same file as the slurm fura launch
PROJECT_NAME=grpo-qwen25-7b

# ---- hyperparameters (SimpleRL-Zoo / slurm fura aligned) ----
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=7168
MAX_VAL_RESP_LENGTH=7168
TEMPERATURE=1.0
N_RESPONSES=8
MINI_BATCH_SIZE=64
LR=1e-6
TOTAL_EPOCHS=1
USE_KL=False                       # slurm/opd/fura used USE_KL=False
MODEL_DTYPE=bfloat16
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
SAVE_FREQ=20
TEST_FREQ=20
IS_PLOT=False

# ---- no tensor parallel ----
PARALLEL_SIZE=1

# ---- hardware / memory-fit (single 8-GPU node by default) ----
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.7

# ---- full finetune (no PEFT) ----
PEFT_MODE=none

set +a

bash grpo.sh
