#!/bin/bash
# opd_sparsegpt.sh — OPD MATH training for a SparseGPT-compressed Qwen3-4B
# student (~1.7B effective params, 64% unstructured zero) against the original
# uncompressed Qwen/Qwen3-4B (non-thinking) teacher.
#
# Required env:
#   ACTOR_MODEL_PATH   absolute path to the saved SparseGPT student HF dir
#   EXPERIMENT_TAG     short label for the run (e.g. v1_openthought3 / v2_4bgen)
# Optional:
#   CUDA_VISIBLE_DEVICES  default 0
#   LR                    default 5e-7 (SimpleRL-Zoo)
#   TOTAL_STEPS           default unset (one full epoch over math-lv3to5)
#
# Note: PEFT_MODE=none — the student is already pruned; we train the dense
# weights directly. The zero entries simply contribute 0 grad → if we were
# strict we'd mask grads, but for a pilot, gradient on near-zero weights is
# fine (the OPD loss should not re-densify in 138 steps; we measure deltas).
#
# Concurrent runs (one per GPU): set RAY_ISOLATE=1 + CUDA_VISIBLE_DEVICES=<n>.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Required.
: "${ACTOR_MODEL_PATH:?must set ACTOR_MODEL_PATH to a SparseGPT student dir}"
: "${EXPERIMENT_TAG:?must set EXPERIMENT_TAG for logging}"

set -a  # auto-export

# ---- Models ----
# Student = SparseGPT-compressed Qwen3-4B (path passed in via env).
# Teacher = ORIGINAL UNCOMPRESSED Qwen3-4B (non-thinking).
REWARD_MODEL_PATH=Qwen/Qwen3-4B
HF_HOME=/data/yequan/huggingface

# ---- SimpleRL-Zoo-aligned math train/eval setting ----
TRAIN_DATASET_NAME=MATH
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
TEMPERATURE=0.6
N_RESPONSES=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
PROJECT_NAME=opd-qwen-math-sparsegpt
SAVE_FREQ=100
TEST_FREQ=5
IS_PLOT=False

# LR — start from SimpleRL-Zoo default 5e-7. The compressed_opd BTT runs went
# 1e-6, but BTT had factor instability; on a SparseGPT dense student we use the
# canonical SimpleRL-Zoo lr.
LR=${LR:-5e-7}

# Hardware — single GPU, caller pins via CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_GPUS_PER_NODE=1
RAY_ISOLATE=${RAY_ISOLATE:-1}

VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Preserve the SparseGPT zero-mask across training: verl's actor optimizer is
# vanilla Adam (no sparsity awareness). Without this, every zero entry becomes
# nonzero after step 1 (∂L/∂W on a zero weight is generally nonzero), and the
# student silently re-densifies to a 4B-dense model. The patch
# (verl/verl/workers/sparsity_mask.py + dp_actor._optimizer_step) zeros grads
# at masked positions before clip/step AND re-zeros weights after step.
SPARSEGPT_PRESERVE_MASK=1
SPARSEGPT_PRESERVE_SKIP=lm_head,embed

# Memory-fit knobs. The SparseGPT student is on disk as a *dense* 4.02B model
# (with 64% zeros) — verl/HF still materialise it dense, so we budget for a
# 4B-class actor + 4B teacher on a single 95 GB H100. Use the working
# btt_v2_combined config: optim_offload=True + GPU_MEMORY_UTILIZATION=0.40 +
# halved PPO_MAX_TOKEN_LEN_PER_GPU + halved max_num_batched_tokens. Without
# these, step 2's update_actor OOMs (reservation 121 GB > 93 GB).
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=True
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.40
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MINI_BATCH_SIZE=64
VAL_N=4

# Non-thinking chat template (both teacher and student are Qwen3-4B → 1.7B
# non-thinking lineage; same template). Also override
# ppo_max_token_len_per_gpu/max_num_batched_tokens to 16384 (on_policy_dist.sh
# unconditionally recomputes them to 32768 otherwise) so update_actor fits.
EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384"
ENABLE_THINKING=False

# No PEFT — the student weights are already on disk (sparse), trained as dense.
PEFT_MODE=none

# Tag the experiment so we can find it in checkpoints / wandb easily.
PROJECT_PATH=/data/yequan/opd/compressed_opd_v2/${EXPERIMENT_TAG}

set +a

bash "$REPO_ROOT/on_policy_distillation.sh"
