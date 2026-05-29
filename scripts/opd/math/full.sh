#!/bin/bash
# full.sh — full-finetune OPD-math on a single H100 (95GB).
#
# Math train/eval setting is aligned to SimpleRL-Zoo
# ("Investigating and Taming Zero RL for Open Base Models", arXiv:2503.18892,
# hkust-nlp/simpleRL-reason): train on the MATH train split, evaluate on
# MATH-500, with their generation knobs (train rollout temperature 0.6, eval
# temperature 1.0 / top_p 0.95, max response 3072, 8 rollouts/prompt).
#
# Only the knobs needed to fit one H100 plus actor/teacher paths live here.
# Everything else is inherited from on_policy_distillation.sh. Override per-run,
# e.g. `LR=5e-6 bash full.sh` (LR search) or `CUDA_VISIBLE_DEVICES=5 bash full.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a  # auto-export every assignment below to on_policy_distillation.sh

# Models (HF repo IDs; resolved from the HF_HOME cache).
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

# ---- SimpleRL-Zoo-aligned math train/eval setting ----
TRAIN_DATASET_NAME=MATH            # routes to MATH train + MATH-500 eval
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=3072               # SimpleRL-Zoo train max_response_length=3000
MAX_VAL_RESP_LENGTH=3072
TEMPERATURE=0.6                    # SimpleRL-Zoo train rollout temperature
N_RESPONSES=8                      # SimpleRL-Zoo n_samples_per_prompt
VAL_TEMPERATURE=1.0                # SimpleRL-Zoo eval temperature
VAL_TOP_P=0.95                     # SimpleRL-Zoo eval top_p
PROJECT_NAME=opd-qwen-math
SAVE_FREQ=100
TEST_FREQ=5
# In-trainer matplotlib/swanlab plotting is off (matplotlib isn't in the verl
# env; wandb already logs all metrics). Avoids the noisy "Error plotting" path.
IS_PLOT=False

# Default LR (overridden by the LR search: 1e-6 / 5e-6 / 1e-5).
LR=1e-6

# Hardware: single H100.
CUDA_VISIBLE_DEVICES=4
N_GPUS_PER_NODE=1

# vLLM: disable FlashInfer sampler (its JIT compile hits a gcc-11 + nvcc
# include_next bug — `-isystem /usr/include` from flashinfer's build script
# demotes /usr/include past /usr/include/c++/11/cmath, breaking
# `#include_next <math.h>`). Falls back to PyTorch-native top-p/top-k sampling.
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs (1.7B actor + 4B teacher resident on one 95GB H100).
# Keep param offload (cheap) but keep the optimizer states on-GPU: on a free
# 95GB H100 the 1.7B actor's Adam states fit, and on-GPU optim cuts update_actor
# from ~105s to ~40s/step. Drop GPU_MEMORY_UTILIZATION if this ever OOMs.
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
# Keep MINI_BATCH_SIZE=64 (the 8-GPU default) so the per-update sample count
# stays identical to on_policy_distillation.sh. Verl divides this across DP
# ranks and dynamic-bsz token packing under ppo_max_token_len_per_gpu drives
# gradient accumulation.
MINI_BATCH_SIZE=64
VAL_N=4

set +a

bash on_policy_distillation.sh
