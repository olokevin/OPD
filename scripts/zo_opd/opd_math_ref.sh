#!/bin/bash
# opd_math_ref.sh — REFERENCE standard OPD-math training (verl PPO, token_reward_direct).
#
# This is the established teacher-reward OPD pipeline (on_policy_distillation.sh),
# NOT the NP trainer. It is the baseline the NP run (opd_math_np.sh) is compared
# against. Mirrors scripts/opd/math/full.sh's model/teacher/data/memory settings
# but in the requested regime:
#   GREEDY decode (TEMPERATURE=0), 64 prompts -> 64 responses per update
#   (MINI_BATCH_SIZE=64 -> train_batch_size=64 at PARALLEL_SIZE=1; N_RESPONSES=1).
# Single GPU; wandb project opd-qwen-math.
#
#   CUDA_VISIBLE_DEVICES=7 bash scripts/zo_opd/opd_math_ref.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set -a  # auto-export to on_policy_distillation.sh

# Models (aligned to full.sh).
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

# SimpleRL-Zoo-aligned math setting (same as full.sh) ...
TRAIN_DATASET_NAME=MATH
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072
MAX_VAL_RESP_LENGTH=3072
PROJECT_NAME=opd-qwen-math
SAVE_FREQ=100
TEST_FREQ=5
IS_PLOT=False

# ... EXCEPT the requested greedy / 64-prompts-64-responses regime:
TEMPERATURE=0.0           # GREEDY train rollout
N_RESPONSES=1             # 1 response/prompt -> 64 prompts give 64 responses
MINI_BATCH_SIZE=64        # train_batch_size = MINI_BATCH_SIZE * PARALLEL_SIZE(=1) = 64
VAL_TEMPERATURE=0.0       # greedy eval too
VAL_N=1
# on_policy_distillation.sh hard-codes val_kwargs.do_sample=True, and verl's
# validate_config asserts rollout temperature>0 when do_sample=True. Greedy eval
# means do_sample=False; override it via the EXTRA_HYDRA_ARGS hook (appended last).
EXTRA_HYDRA_ARGS="actor_rollout_ref.rollout.val_kwargs.do_sample=False"

# Default LR (full.sh's smoke default; this is the reference, not the swept run).
LR=${LR:-1e-6}

# Hardware: single GPU. Caller pins CUDA_VISIBLE_DEVICES (=7 for the reference).
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
N_GPUS_PER_NODE=1

# vLLM (same as full.sh).
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit (1.7B actor + 4B teacher on one GPU; same as full.sh).
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8

# Private Ray head so this run coexists with the NP sweep (4/5/6) without a
# global `ray stop` tearing them down.
RAY_ISOLATE=1

set +a

bash on_policy_distillation.sh
