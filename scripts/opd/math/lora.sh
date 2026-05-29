#!/bin/bash
# lora.sh — OPD-math with LoRA on a single H100.
#
# Same SimpleRL-Zoo-aligned math train/eval setting as full.sh (MATH train +
# MATH-500 eval, train temp 0.6 / eval temp 1.0 / top_p 0.95, max resp 3072,
# 8 rollouts) plus PEFT_MODE=lora on top. Override any knob per-run:
# e.g. `LR=5e-5 bash lora.sh` (LR search) or `LORA_RANK=32 bash lora.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

set -a  # auto-export every assignment below to on_policy_distillation.sh

# Models (HF repo IDs; resolved from the HF_HOME cache).
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

# ---- SimpleRL-Zoo-aligned math train/eval setting ----
TRAIN_DATASET_NAME=MATH            # routes to MATH train + MATH-500 eval
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
TEMPERATURE=0.6
N_RESPONSES=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
PROJECT_NAME=opd-qwen-math
SAVE_FREQ=100
TEST_FREQ=5
IS_PLOT=False   # matplotlib not in verl env; wandb logs metrics. See full.sh.

# Default LR (overridden by the LR search: 1e-5 / 5e-5 / 1e-4).
LR=1e-5

# Hardware: single H100.
CUDA_VISIBLE_DEVICES=5
N_GPUS_PER_NODE=1

# vLLM workarounds (see full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs. On-GPU optimizer states (LoRA's are tiny) for speed; see full.sh.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MINI_BATCH_SIZE=64
VAL_N=4

# PEFT: LoRA.
PEFT_MODE=lora
PEFT_TARGET_MODULES=all
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.0

set +a

bash on_policy_distillation.sh
