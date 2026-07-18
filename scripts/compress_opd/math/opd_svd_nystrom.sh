#!/bin/bash
# opd_svd_nystrom.sh — Stage 2 of compress→OPD: OPD MATH training on top of the
# reasoning-aware-compressed Qwen3-4B student built by
# `build_svd_nystrom_student.py` (SVD-V2 attn + Nystrom mlp @ retain 0.7,
# last decoder layer dense, sequence/full OpenThought3 calibration).
#
# Teacher is the ORIGINAL uncompressed Qwen/Qwen3-4B (non-thinking).
#
# Required env:
#   ACTOR_MODEL_PATH   absolute path to the saved compressed student HF dir
#                      (output of `build_svd_nystrom_student.py --save-dir ...`)
#   EXPERIMENT_TAG     short label for the run
#                      (e.g. svd_nystrom_r0.7)
# Optional:
#   CUDA_VISIBLE_DEVICES  default 0
#   LR                    default 5e-7 (SimpleRL-Zoo)
#
# Concurrent runs (one per GPU): set RAY_ISOLATE=1 + CUDA_VISIBLE_DEVICES=<n>.
#
# NOTE: unlike opd_sparsegpt.sh we do NOT set SPARSEGPT_PRESERVE_MASK — SVD-V2
# and Nystrom reconstruct dense low-rank/subsampled weights, so there are no
# structural zeros to preserve. The compressed student is just a dense HF
# checkpoint with information-truncated weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

: "${ACTOR_MODEL_PATH:?must set ACTOR_MODEL_PATH to the compressed-student HF dir}"
: "${EXPERIMENT_TAG:?must set EXPERIMENT_TAG for logging}"

set -a  # auto-export

# ---- Models ----
# Student = SVD-V2 + Nystrom compressed Qwen3-4B (path passed in via env).
# Teacher = ORIGINAL UNCOMPRESSED Qwen3-4B (non-thinking).
REWARD_MODEL_PATH=Qwen/Qwen3-4B
HF_HOME=/data/yequan/huggingface

# ---- Train: OpenThoughts3 (30k math prompts, verl OPD schema) ----
# `datasets/OpenThoughts3_opd.parquet` ships ready-to-use: columns
# {data_source, prompt, ability, reward_model{ground_truth,style}, extra_info}.
# Passing TRAIN_DATASET directly skips the launcher's TRAIN_DATASET_NAME case
# (which only knows DAPO/MATH/gsm8k/math-500). TRAIN_DATASET_NAME stays as the
# run/ckpt tag.
TRAIN_DATASET=${TRAIN_DATASET:-datasets/OpenThoughts3_opd.parquet}
TRAIN_DATASET_NAME=OpenThoughts3
# ---- Eval: AIME24 + AIME25 + AMC23 + MATH-500 ----
# TEST_FILE is read by on_policy_distillation.sh as the val parquet list.
TEST_FILE='["datasets/test_data/AIME24/test.parquet", "datasets/test_data/AIME25/test.parquet", "datasets/test_data/AMC23/test.parquet", "datasets/test_data/MATH-500/test.parquet"]'
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
TEMPERATURE=0.6
N_RESPONSES=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
PROJECT_NAME=opd-qwen-math-svd-nystrom
SAVE_FREQ=100
TEST_FREQ=5
IS_PLOT=False

LR=${LR:-5e-7}

# Hardware — single GPU, caller pins via CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_GPUS_PER_NODE=1
RAY_ISOLATE=${RAY_ISOLATE:-1}

VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs. The compressed student is on disk as a dense 4B-shaped HF
# model (SVD/Nystrom mutate weights in place; shapes are unchanged), so we
# budget the same as the SparseGPT pipeline: 4B-class actor + 4B teacher on a
# single 95 GB H100 needs optim_offload=True + GPU util 0.40 + halved
# ppo_max_token_len_per_gpu/max_num_batched_tokens.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=True
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.40
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MINI_BATCH_SIZE=64
VAL_N=4

# Non-thinking chat template (teacher and student are Qwen3-4B non-thinking
# lineage). Also override ppo_max_token_len_per_gpu / max_num_batched_tokens
# to 16384 (on_policy_distillation.sh otherwise recomputes them to 32768) so
# update_actor fits on one H100.
EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384"
ENABLE_THINKING=False

# No PEFT — the student weights are already on disk (compressed), trained as dense.
PEFT_MODE=none

# Tag the experiment so we can find it in checkpoints / wandb easily.
PROJECT_PATH=/data/yequan/opd/compress_opd_svd_nystrom/${EXPERIMENT_TAG}

set +a

bash "$REPO_ROOT/on_policy_distillation.sh"
