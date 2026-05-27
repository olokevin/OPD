#!/bin/bash
# qfura.sh — OPD-math with qFurA (BlockTT with an NF4-quantized frozen core)
# on a single 80GB GPU.
#
# qFurA = PEFT_MODE=blocktt + BTT_QFURA=True. The frozen side of each BTT
# factorization is replaced with a `bitsandbytes` 4-bit NF4 blob; only the
# small side trains. See src/compress/README.md "qfura" section for details
# and the streaming converter used for larger backbones.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

set -a

# Models.
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

# vLLM workarounds.
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

# PEFT: qFurA = BlockTT with NF4-quantized frozen core.
PEFT_MODE=blocktt
PEFT_TARGET_MODULES=all
BTT_DECOMP_MODE=output_one_block
BTT_RANK=full
BTT_TRAIN_POSITION=small
BTT_S_MERGED_TO=keep_trainable
BTT_CONVERT_MODE=svd
BTT_FACTORIZE_BY_HEAD=True
BTT_NORMALIZE_AFTER_UPDATE=False
BTT_QFURA=True                     # qFurA: quantize frozen core to NF4

set +a

bash on_policy_distillation.sh