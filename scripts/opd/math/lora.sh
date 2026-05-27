#!/bin/bash
# lora.sh — OPD-math with LoRA on a single 80GB GPU.
#
# Inherits everything from full.sh's memory-fit profile (offloading, bf16,
# trimmed reward micro-batch) and layers PEFT_MODE=lora on top. Override any
# knob per-run: e.g. `LORA_RANK=32 bash lora.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

set -a  # auto-export every assignment below to on_policy_distillation.sh

# Models (HF repo IDs; resolved from the HF_HOME cache).
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

TRAIN_DATASET_NAME=math-500k
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
PROJECT_NAME=opd-qwen-math

# Hardware.
CUDA_VISIBLE_DEVICES=3
N_GPUS_PER_NODE=1

# vLLM workarounds (see full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=True
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
