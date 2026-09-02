#!/bin/bash
# SGD-GRPO on the ES fixed 64-problem batch, to find WHERE plain SGD moves the weights.
#
# "Do We Need Adam?" (arXiv:2602.07729) trains RLVR with vanilla SGD (lr=1e-1, no momentum,
# bf16) and finds it touches < 0.02% of the parameters -- the update is below one bf16 ULP
# almost everywhere.  This runs that recipe for a few steps on exactly the ES thread's task
# (Qwen2.5-Math-7B, the first 64 MATH lvl 3-5 problems of run_es_math.sh, 1536-token
# rollouts, greedy MATH-500 at 3000 tokens) and saves bf16 HF weights every SAVE_FREQ steps;
# scripts/es/build_sgd_mask.py then diffs them against the base to get the mask that
# PERTURB_MODE=sgdmask perturbs.  Section 18 of docs/results/ES/es_results.md.
#
# Usage:  DEVICES=0 STEPS=10 bash scripts/es/run_sgd_mask.sh

set -x
set -euo pipefail

REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH

export CUDA_VISIBLE_DEVICES=${DEVICES:-0}
export N_GPUS_PER_NODE=1
export NNODES=1
export RAY_ISOLATE=1                       # private Ray head; never touch other runs
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_LAUNCH_BLOCKING=0

# ---- task: identical to run_es_math.sh ----
export ADV_ESTIMATOR=grpo
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
export TRAIN_DATASET_NAME=MATH
export TRAIN_DATASET=${TRAIN_DATASET:-${REPO}/datasets/es_math/math_lv3to5_qwenmath_train_b64.parquet}
export TEST_FILE="[\"${REPO}/datasets/es_math/math500_qwenmath_test.parquet\"]"
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1536}      # ES train budget
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3000}  # ES eval budget
export N_RESPONSES=${N_RESPONSES:-8}
export TEMPERATURE=${TEMPERATURE:-1.0}     # GRPO needs sampling; ES is greedy
export TRAIN_BATCH_SIZE=64
export MINI_BATCH_SIZE=64                  # one optimizer step per 64-problem batch
export SHUFFLE=False
export ROLLOUT_IS=none
export USE_KL=False
STEPS=${STEPS:-10}
export TOTAL_EPOCHS=$STEPS                 # 64 rows / batch 64 = exactly one step per epoch
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VAL_N=1
export VAL_TEMPERATURE=0                   # greedy MATH-500, comparable to the ES evals
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# ---- the paper's optimizer: vanilla SGD, lr 1e-1, no momentum, no weight decay, bf16 ----
# bf16 module params (not the fp32 master the BP leg used): the update sparsity IS the
# bf16 rounding of sub-ULP steps, so the weights must be bf16 when SGD writes them.
export MODEL_DTYPE=bfloat16
export LR=${LR:-0.1}
export PEFT_MODE=none
export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIM_OFFLOAD=False           # SGD has no state
export REF_PARAM_OFFLOAD=True
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
# hf_model only: /data has <100 GB free, a bf16 HF dump is 15 GB, FSDP shards would double it.
export EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.optim.optimizer=SGD \
actor_rollout_ref.actor.optim.weight_decay=0 \
actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]"

# Reap an orphaned Ray head from a crashed run on this GPU (see run_bp_math.sh).
_gpu0=${CUDA_VISIBLE_DEVICES%%,*}
_raytmp=/tmp/ray_grpo_gpu${_gpu0:-0}
for _p in $(pgrep -f "$_raytmp" 2>/dev/null || true); do
  if tr '\0' ' ' < "/proc/$_p/cmdline" 2>/dev/null | grep -q -- "$_raytmp"; then
    echo "[sgd] reaping orphaned ray pid $_p ($_raytmp)"; kill -9 "$_p" 2>/dev/null || true
  fi
done
rm -rf "$_raytmp"
for _i in $(seq 1 30); do
  _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | paste -sd+ | bc)
  [ "${_used:-0}" -lt 2000 ] && break
  echo "[sgd] waiting for GPU $CUDA_VISIBLE_DEVICES to drain (${_used} MiB used)"; sleep 10
done

export PROJECT_NAME=${PROJECT_NAME:-BP-q2p5-7b}
export TRAINER_LOGGER=${TRAINER_LOGGER:-['console','wandb']}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sgd-dense_math-lv3to5-b64_lr${LR}_n${N_RESPONSES}_bf16_st${STEPS}}
export PROJECT_PATH=${PROJECT_PATH:-/data/yequan/bp/${PROJECT_NAME}}
export CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}
mkdir -p "$CKPT_PATH"

bash grpo.sh
