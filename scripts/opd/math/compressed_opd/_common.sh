#!/bin/bash
# Shared env for compressed_opd math launchers.
#
# Setup: take Qwen3-4B-Non-Thinking-RL-Math-Step500 as *both* the teacher and
# the initialization for the student, then BTT-compress the student down to
# ~1.7B params (per-linear-layer compression ratio 0.36 over the ~3.63B
# compressible-linear footprint, embedding + lm_head pass through unchanged
# since they are tied for Qwen3-4B → 0.389B; ratio chosen so:
#   0.389B (embed/tied lm_head) + 0.36 × 3.633B (linears) ≈ 1.70B).
#
# Calibration uses MATH train as calibration data and the *actual* OPD
# teacher-student loss to drive backward-covariance collection — see
# src/compress/calibration_opd_loss.py and verl/workers/peft/blocktt.py.
# CALIB_LOSS=opd flows through every adapter mode but only v2_combined uses
# backward covariances; v2 (forward-only) ignores it gracefully.
#
# Training / eval knobs come from full.sh (SimpleRL-Zoo math: MATH train +
# MATH-500 eval, train temp 0.6 / eval temp 1.0 / top_p 0.95, max resp 3072,
# 8 rollouts). Per-script overrides (LR, GPU, CALIB_MODE) live in btt_v2.sh
# and btt_v2_combined.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

set -a  # auto-export everything below

# ---- Models ----
# Student is initialised from the same 4B checkpoint as the teacher, then
# BTT-compressed to ~1.7B params before training starts.
ACTOR_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

# ---- SimpleRL-Zoo-aligned math train/eval setting (mirrors full.sh) ----
TRAIN_DATASET_NAME=MATH
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
TEMPERATURE=0.6
N_RESPONSES=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
PROJECT_NAME=opd-qwen-math-compressed
SAVE_FREQ=100
TEST_FREQ=5
IS_PLOT=False

# Default LR. Compressed-model full-FT — 1e-5 (the user-requested LR).
LR=${LR:-1e-5}

# Hardware: single H100 per run. GPU pinned by the caller.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
N_GPUS_PER_NODE=1
# Make sure two concurrent runs on different GPUs get distinct Ray heads.
RAY_ISOLATE=${RAY_ISOLATE:-1}

# vLLM workarounds (see full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs. ACTOR_PARAM_OFFLOAD MUST be False for BTT — the
# Linear->BTT conversion at init requires the target Linear weights on CUDA
# (param_offload=True leaves them on CPU and the conversion errors out).
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MINI_BATCH_SIZE=64
VAL_N=4

# ---- BTT compression ----
# output_one_block + per-linear ratio = 0.36 → ~1.66B total params (target 1.7B):
# 0.389B (tied embed/lm_head) + 0.36 × 3.633B (compressible linears) ≈ 1.7B.
# 'square' would give a similar count but is rejected by verl's BlockTT adapter
# in non-calibration-free paths (only input/output_one_block are allowed).
# Both sides train (treat the compressed model as a fresh ~1.7B model to full-FT).
PEFT_MODE=blocktt
PEFT_TARGET_MODULES=all
BTT_DECOMP_MODE=output_one_block
BTT_RANK=0.36
BTT_TRAIN_POSITION=both
BTT_S_MERGED_TO=split
BTT_CONVERT_MODE=svd
BTT_FACTORIZE_BY_HEAD=True
BTT_NORMALIZE_AFTER_UPDATE=False
BTT_QFURA=False

# ---- Calibration: MATH train + OPD-faithful teacher-student gradient ----
# CALIB_MODE is set per-script (v2 or v2_combined).
CALIB_SOURCE=training_data
CALIB_NUM_SEQS=128
CALIB_BATCH_SIZE=4              # 4B teacher + 4B student under bf16 on one H100
CALIB_SEED=3
CALIB_LOSS=opd
CALIB_TOP_K=16
CALIB_TOP_K_STRATEGY=only_stu
CALIB_REWARD_WEIGHT_MODE=student_p
CALIB_TEMPERATURE=$TEMPERATURE
CALIB_TEACHER_TEMPERATURE=1.0

# ---- Non-thinking chat template (teacher + student are both Non-Thinking) ----
# Hydra override drives the trainer; ENABLE_THINKING also propagates into the
# calibration loader inside the fsdp worker (which doesn't see data.* config).
# Also force FSDP2 for actor + ref: FSDP1 with use_orig_params=True (required
# for BlockTT's mixed trainable/frozen params) fails the per-step writeback on
# the frozen embedding ("Cannot writeback when the parameter shape changes");
# FSDP2's per-parameter DTensor sharding has no such constraint. This matches
# fura.sh's verified-working config (commit 7a8a61c).
EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2"
ENABLE_THINKING=False

set +a
