#!/bin/bash
# btt_v2_combined_opd.sh — single-GPU smoke run of the compress→OPD pipeline.
#
# Compresses Qwen3-4B-Non-Thinking into a BlockTT factorization with
# doubly-whitened v2_combined calibration, where the backward-pass gradient
# that drives covariance is the *same* loss OPD uses during training
# (teacher-student policy gradient surrogate, top_k=16, only_stu, student_p).
#
# Compression target: drop trainable Linear params by ~50% (BTT_RANK=0.5).
# Combined with embeddings/lm_head/norms being untouched, that lands the
# 4B base in the ~1.7B–2B effective range — enough to validate the compress
# path is doing real work without needing a precisely tuned rank ratio.
#
# Calibration is small (CALIB_NUM_SEQS=8) and runs on whatever the OPD
# training data is, mirroring the deployment setting. For a real run bump
# CALIB_NUM_SEQS to 128–256.
#
# Usage:
#   bash scripts/compress_opd/math/btt_v2_combined_opd.sh
# Override any knob via env:
#   CALIB_NUM_SEQS=32 BTT_RANK=0.3 bash scripts/compress_opd/math/btt_v2_combined_opd.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

set -a  # auto-export every assignment below to on_policy_distillation.sh

# Models — actor IS the 4B teacher pre-compression; reward model is the same
# 4B, used both as the calibration teacher (loss=opd) and as the OPD
# distillation teacher during training.
ACTOR_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

TRAIN_DATASET_NAME=math-500k
MAX_PROMPT_LENGTH=512
MAX_RESP_LENGTH=1024
MAX_VAL_RESP_LENGTH=1024
PROJECT_NAME=opd-compress-qwen-math

# Smoke-test cadence — keep checkpoints sparse, run validation rarely.
SAVE_FREQ=200
TEST_FREQ=200
TOTAL_EPOCHS=1
VAL_N=1

# Hardware: single GPU.
CUDA_VISIBLE_DEVICES=0
N_GPUS_PER_NODE=1

# vLLM workarounds (see scripts/opd/math/full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit for 4B actor + 4B teacher on one 96GB H100.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=True
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.45
REWARD_MICRO_BATCH_SIZE_PER_GPU=4
MINI_BATCH_SIZE=8
N_RESPONSES=2

# PEFT: BlockTT compression with v2_combined calibration + OPD loss.
PEFT_MODE=blocktt
PEFT_TARGET_MODULES=all
BTT_DECOMP_MODE=input_one_block   # smaller side comes from input dim
BTT_RANK=0.5                       # ratio in (0,1] — only valid with calibrated modes
BTT_TRAIN_POSITION=small           # train the small factor
BTT_S_MERGED_TO=frozen
BTT_CONVERT_MODE=svd
BTT_FACTORIZE_BY_HEAD=True
BTT_NORMALIZE_AFTER_UPDATE=False
BTT_QFURA=False

# Calibration: v2_combined (double whitening) + OPD-faithful gradient.
CALIB_MODE=v2_combined
CALIB_SOURCE=training_data         # reuse the OPD training data for calibration
CALIB_NUM_SEQS=8                   # smoke value; bump to 128+ for real runs
CALIB_MAX_LENGTH=512
CALIB_BATCH_SIZE=2
CALIB_SEED=0

# OPD-faithful calibration loss — gradient direction matches the OPD trainer.
# These four knobs must mirror the OPD training settings; bash launcher already
# exports LOG_PROB_TOP_K / TOP_K_STRATEGY / REWARD_WEIGHT_MODE / TEMPERATURE,
# so default them to those values where possible.
CALIB_LOSS=opd
CALIB_TOP_K=${LOG_PROB_TOP_K:-16}
CALIB_TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
CALIB_REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
CALIB_TEMPERATURE=${TEMPERATURE:-1.0}
CALIB_TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}

set +a

bash "$REPO_ROOT/on_policy_distillation.sh"
