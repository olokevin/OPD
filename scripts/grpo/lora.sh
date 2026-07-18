#!/bin/bash
# scripts/grpo/lora.sh — LoRA GRPO (zero-RL) on Qwen2.5-7B.
#
# Same GRPO base + SimpleRL-Zoo / slurm-fura-aligned setting as full.sh, but with
# PEFT_MODE=lora (only the LoRA adapters train). No tensor parallel.
#
# Override per-run, e.g. `LORA_RANK=64 LORA_ALPHA=128 bash scripts/grpo/lora.sh`.
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
LR=1e-5                            # LoRA typically uses a higher LR than full-FT
TOTAL_EPOCHS=1
USE_KL=False
MODEL_DTYPE=bfloat16
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
SAVE_FREQ=20
TEST_FREQ=20
IS_PLOT=False

# ---- no tensor parallel ----
PARALLEL_SIZE=1

# ---- hardware / memory-fit ----
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.7

# ---- PEFT: LoRA ----
PEFT_MODE=lora
PEFT_TARGET_MODULES=all
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.0

set +a

bash grpo.sh
