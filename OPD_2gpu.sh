#!/bin/bash
# OPD_2gpu.sh — launch on_policy_distillation.sh on 2× 80GB GPUs.
#
# Only the knobs needed to fit the 2-GPU memory budget plus actor/teacher paths
# live here. Everything else is inherited from on_policy_distillation.sh.
# Override any of these per-run, e.g. `LR=5e-7 bash OPD_2gpu.sh` — the launcher
# already accepts arbitrary `${VAR:-default}` overrides.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a  # auto-export every assignment below to on_policy_distillation.sh

# Models (HF repo IDs; resolved from the HF_HOME cache).
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface

# Hardware.
CUDA_VISIBLE_DEVICES=3,6
N_GPUS_PER_NODE=2

# vLLM: disable FlashInfer sampler (its JIT compile hits a gcc-11 + nvcc
# include_next bug — `-isystem /usr/include` from flashinfer's build script
# demotes /usr/include past /usr/include/c++/11/cmath, breaking
# `#include_next <math.h>`). Falls back to PyTorch-native top-p/top-k sampling.
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory-fit knobs (the reason this wrapper exists).
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=True
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
MAX_RESP_LENGTH=4096
MAX_VAL_RESP_LENGTH=8192
# Keep MINI_BATCH_SIZE=64 (the 8-GPU default) so the per-update sample count
# stays identical to on_policy_distillation.sh: 64 prompts × N_RESPONSES per
# update. Verl divides this across DP ranks automatically — on 2 GPUs each rank
# sees 32 seqs/minibatch vs 8/rank on 8 GPUs, and dynamic-bsz token packing
# under ppo_max_token_len_per_gpu (=32768) drives gradient accumulation. Net:
# same optimizer step, just ~4× more micro-steps per update.
MINI_BATCH_SIZE=64
VAL_N=4

set +a

bash "$SCRIPT_DIR/on_policy_distillation.sh"
