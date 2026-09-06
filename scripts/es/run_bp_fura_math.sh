#!/bin/bash
# FuRA (BlockTT small core) trained with BACKPROP on the ES thread's task -- the BP
# counterpart of the ES `fura` arm, on exactly the protocol of the dense SGD-GRPO
# reference (scripts/es/run_sgd_mask.sh, docs/results/ES/es_results.md section 18.3):
# Qwen2.5-Math-7B, the fixed 64-problem MATH lvl 3-5 batch, GRPO n=8 T=1.0,
# 1536-token rollouts, vanilla SGD (no momentum, no weight decay, clip 1.0), 10 steps,
# greedy MATH-500 at 3000 tokens.  Dense reference: lr 0.1 -> 52.4 / 72.4 @5 / 72.2 @10.
#
# What differs from the dense reference, and why:
#   * PEFT_MODE=blocktt, output_one_block, rank=full, train_position=small,
#     s_merged_to=frozen, factorize_by_head=False.  Per input block j,
#     W[:, blk_j] = A_j R_j with A_j = U_j S_j frozen and R_j = Vh_j (b x b) trained;
#     block shapes from _closest_factor_pair(in) (3584 -> 56x64, 18944 -> 128x148),
#     i.e. the same factorisation the ES `fura` arm uses.  train_bias=False so only
#     the 2-D cores move (ES perturbs 2-D weights only).  HF keeps q/k/v and gate/up
#     as separate Linears, so the core count is 117.0M (1.54%) vs ES fura's 97.8M on
#     vLLM's fused weights -- same discrepancy section 13.1 accepted for isobtt.
#   * MODEL_DTYPE=fp32: fp32 master for the cores, bf16 compute via FSDP mixed
#     precision.  The dense run trained bf16 weights on purpose (section 18 studies the
#     bf16 update sparsity); the cores here have entries ~0.1 whose bf16 half-ULP
#     (~5e-4) is ~40x the per-entry SGD step at lr=1, so bf16 cores would silently
#     drop the update.
#   * FSDP2: blocktt's mixed trainable/frozen params break FSDP1's writeback.
#   * TEST_FREQ=1 (dense evaluated at 0/5/10) for a per-step convergence curve.
#
# Usage:  DEVICES=4 LR=1.0 bash scripts/es/run_bp_fura_math.sh

set -x
set -euo pipefail

REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH

export CUDA_VISIBLE_DEVICES=${DEVICES:-4}
export N_GPUS_PER_NODE=1
export NNODES=1
export RAY_ISOLATE=1                       # private Ray head; never touch other runs
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_LAUNCH_BLOCKING=0

# ---- task: identical to run_sgd_mask.sh / run_es_math.sh ----
export ADV_ESTIMATOR=grpo
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
export TRAIN_DATASET_NAME=MATH
export TRAIN_DATASET=${TRAIN_DATASET:-${REPO}/datasets/es_math/math_lv3to5_qwenmath_train_b64.parquet}
export TEST_FILE="[\"${REPO}/datasets/es_math/math500_qwenmath_test.parquet\"]"
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1536}      # ES train budget
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3000}  # ES eval budget
export N_RESPONSES=${N_RESPONSES:-8}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TRAIN_BATCH_SIZE=64
export MINI_BATCH_SIZE=64                  # one optimizer step per 64-problem batch
export SHUFFLE=False
export ROLLOUT_IS=none
export USE_KL=False
STEPS=${STEPS:-10}
export TOTAL_EPOCHS=$STEPS                 # 64 rows / batch 64 = exactly one step per epoch
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=0                         # no checkpoints (/data is nearly full)
export VAL_N=1
export VAL_TEMPERATURE=0                   # greedy MATH-500, comparable to the ES evals
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# ---- optimizer: the dense reference's SGD, LR searched (dense used 0.1) ----
export LR=${LR:-1.0}
export MODEL_DTYPE=fp32                    # fp32 master for the cores (see header)
export ACTOR_PARAM_OFFLOAD=False           # BTT conversion needs CUDA weights
export ACTOR_OPTIM_OFFLOAD=False           # SGD has no state
export REF_PARAM_OFFLOAD=True
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# ---- PEFT: FuRA = BlockTT, frozen large core, trainable small core ----
export PEFT_MODE=blocktt
export PEFT_TARGET_MODULES=all
export BTT_DECOMP_MODE=output_one_block    # input dim blocked; R_j (b x b) is the small core
export BTT_RANK=full
export BTT_TRAIN_POSITION=small
export BTT_S_MERGED_TO=frozen              # A_j = U_j S_j frozen
export BTT_CONVERT_MODE=svd
export BTT_FACTORIZE_BY_HEAD=False         # closest-factor-pair blocks everywhere, as ES fura
export BTT_NORMALIZE_AFTER_UPDATE=False
export BTT_QFURA=False

export EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.optim.optimizer=SGD \
actor_rollout_ref.actor.optim.weight_decay=0 \
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.ref.strategy=fsdp2 \
++actor_rollout_ref.peft.blocktt.train_bias=False"

# Reap an orphaned Ray head from a crashed run on this GPU (see run_bp_math.sh).
_gpu0=${CUDA_VISIBLE_DEVICES%%,*}
_raytmp=/tmp/ray_grpo_gpu${_gpu0:-0}
for _p in $(pgrep -f "$_raytmp" 2>/dev/null || true); do
  if tr '\0' ' ' < "/proc/$_p/cmdline" 2>/dev/null | grep -q -- "$_raytmp"; then
    echo "[bp-fura] reaping orphaned ray pid $_p ($_raytmp)"; kill -9 "$_p" 2>/dev/null || true
  fi
done
rm -rf "$_raytmp"
# Other users' small jobs may sit on the GPU; only wait for OUR previous run to drain.
DRAIN_MIB=${DRAIN_MIB:-15000}
for _i in $(seq 1 30); do
  _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | paste -sd+ | bc)
  [ "${_used:-0}" -lt "$DRAIN_MIB" ] && break
  echo "[bp-fura] waiting for GPU $CUDA_VISIBLE_DEVICES to drain (${_used} MiB used)"; sleep 10
done

export PROJECT_NAME=${PROJECT_NAME:-BP-q2p5-7b}
export TRAINER_LOGGER=${TRAINER_LOGGER:-['console','wandb']}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sgd-fura_math-lv3to5-b64_lr${LR}_n${N_RESPONSES}_fp32_st${STEPS}}
export PROJECT_PATH=${PROJECT_PATH:-/data/yequan/bp/${PROJECT_NAME}}
export CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}
mkdir -p "$CKPT_PATH"

bash grpo.sh
