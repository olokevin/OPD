#!/bin/bash
# scripts/grpo/lora.sh — LoRA GRPO (zero-RL) on Qwen2.5-7B.
#
# Same GRPO base + SimpleRL-Zoo Qwen2.5-7B RL recipe as full.sh (dataset, batch
# sizes, KL/entropy, epochs, response length), but with PEFT_MODE=lora (only the
# LoRA adapters train). Two deliberate deviations from full.sh: no tensor
# parallel (PARALLEL_SIZE=1) and a higher LR (PEFT adapters need a larger step
# than a 5e-7 full-FT LR).
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
TRAIN_DATASET_NAME=MATH            # simplelr_math_35 -> math-lv3to5 + MATH-500 eval
# SimpleRL-Zoo `qwen25-math-cot`-structured prompt copies (see full.sh / scripts/
# grpo/prep_simplerl_prompts.py). Kept identical to full.sh so the common training
# setting matches; OPD runs keep using the un-suffixed shared parquets.
TRAIN_DATASET=datasets/train_data/math-lv3to5_simplerl/train.parquet
TEST_FILE='["datasets/test_data/MATH-500_simplerl/test.parquet"]'
PROJECT_NAME=grpo-qwen25-7b

# ---- hyperparameters (SimpleRL-Zoo Qwen2.5-7B recipe; LR raised for LoRA) ----
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=8192
MAX_VAL_RESP_LENGTH=8192
TEMPERATURE=1.0
N_RESPONSES=8
TRAIN_BATCH_SIZE=1024
MINI_BATCH_SIZE=256
LR=1e-5                            # LoRA LR (NOT the 5e-7 full-FT LR)
TOTAL_EPOCHS=20
USE_KL=True
KL_LOSS_COEF=1e-4
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0.001
SHUFFLE=True                      # match full.sh: SimpleRL-Zoo shuffles the train set
ROLLOUT_IS=none                   # match full.sh: drop the verl-fork IS correction
MODEL_DTYPE=bfloat16
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
SAVE_FREQ=5
TEST_FREQ=5
IS_PLOT=False

# ---- no tensor parallel ----
PARALLEL_SIZE=1

# ---- hardware / memory-fit ----
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.75

# ---- PEFT: LoRA ----
PEFT_MODE=lora
PEFT_TARGET_MODULES=all
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.0

set +a

bash grpo.sh
