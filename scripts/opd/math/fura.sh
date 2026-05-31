#!/bin/bash
# fura.sh — OPD-math with FurA (BlockTT with a non-quantized frozen core) on a
# single H100.
#
# FurA = PEFT_MODE=blocktt with the standard (bf16) frozen core. The frozen side
# of each BTT factorization is fixed; only the small side trains. See
# src/compress/README.md for the decomposition convention.
#
# Same SimpleRL-Zoo-aligned math train/eval setting as full.sh (MATH train +
# MATH-500 eval, train temp 0.6 / eval temp 1.0 / top_p 0.95, max resp 3072,
# 8 rollouts). Override per-run, e.g. `LR=1e-4 bash fura.sh` (LR search).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

set -a

# Models.
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

# Default LR (overridden by the LR search: 5e-5 / 1e-4 / 2e-4).
LR=1e-4

# Hardware: single H100. Defaults to GPU 3; honor an external override.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
N_GPUS_PER_NODE=1

# vLLM workarounds (see full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs. NOTE: ACTOR_PARAM_OFFLOAD MUST be False for FurA — the
# Linear->BlockTT conversion at init requires all target Linear weights on CUDA
# (param_offload=True leaves them on CPU -> "Linear->BTT conversion requires all
# target Linear weights on CUDA"). The 1.7B actor + 4B teacher fit on one free
# 95GB H100 without param offload.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MINI_BATCH_SIZE=64
VAL_N=4

# PEFT: FurA = BlockTT, non-quantized frozen core.
PEFT_MODE=blocktt
PEFT_TARGET_MODULES=all
BTT_DECOMP_MODE=output_one_block   # input_one_block | output_one_block
BTT_RANK=full                      # int | "full" | float in (0,1] with calib
BTT_TRAIN_POSITION=small           # small | large | both
BTT_S_MERGED_TO=keep_trainable     # frozen | trainable | keep_trainable
BTT_CONVERT_MODE=svd               # svd | random | calib
BTT_FACTORIZE_BY_HEAD=True
BTT_NORMALIZE_AFTER_UPDATE=False
BTT_QFURA=False                    # FurA: keep frozen core in bf16

# Use FSDP2 for the actor/ref. FSDP1 (use_orig_params=True, required for
# BlockTT's mixed trainable/frozen params) fails the per-step writeback on the
# frozen embedding: "Cannot writeback when the parameter shape changes
# [311164928] vs [151936,2048]". FSDP2's per-parameter DTensor sharding has no
# FlatParameter writeback constraint and wraps the (untied) embedding in its own
# unit, so BlockTT trains cleanly.
EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2"

set +a

bash on_policy_distillation.sh
