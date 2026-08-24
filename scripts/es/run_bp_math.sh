#!/bin/bash
# BP (true-gradient) counterpart of scripts/es/run_es_math.sh.
#
# Same task as the ES thread -- Qwen2.5-Math-7B, MATH lvl 3-5 -> MATH-500, binary
# \boxed{} reward -- but optimised with GRPO + AdamW instead of forward-only ES, so
# ES-vs-BP is a controlled comparison of the *optimiser*, not the task.
#
# BP_MODE selects the arm (maps to PEFT_MODE):
#   dense       full-parameter fine-tuning (baseline)
#   iso         W = C_L W0 C_R^T, block-diagonal Cayley rotations, spectrum fixed
#   isobtt      block-wise SVD, R_j in O(b) trained, per-block spectrum fixed
#   isobtt_mix  isobtt + orthogonal input mixer M in O(n_blk)
#
# Usage:  DEVICES=6,7 BP_MODE=iso bash scripts/es/run_bp_math.sh

set -x
set -euo pipefail

REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"

BP_MODE=${BP_MODE:-dense}
export CUDA_VISIBLE_DEVICES=${DEVICES:-6,7}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}
export NNODES=1
export RAY_ISOLATE=1                       # private Ray head; never touch other runs
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_LAUNCH_BLOCKING=0

# ---- task: identical to the ES thread ----
export ADV_ESTIMATOR=grpo
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
export TRAIN_DATASET_NAME=MATH             # math-lv3to5 -> MATH-500
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}      # ES eval budget was 3000
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3072}
export N_RESPONSES=${N_RESPONSES:-8}
export TEMPERATURE=${TEMPERATURE:-1.0}     # GRPO needs sampling; ES was greedy
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export SHUFFLE=${SHUFFLE:-True}
export ROLLOUT_IS=${ROLLOUT_IS:-none}      # SimpleRL-aligned GRPO
export USE_KL=${USE_KL:-False}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-0}
export VAL_N=${VAL_N:-4}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# ---- memory fit: 7B actor on 2x H100 NVL ----
# fp32 module params (grpo.sh's own actor default) + FSDP MixedPrecision
# param_dtype=bf16: the optimizer master stays fp32 so lr~5e-6 updates are not
# rounded away, while compute is still bf16. It also keeps every tensor in an FSDP
# unit one dtype, which FSDP1's FlatParameter requires.
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
export REF_PARAM_OFFLOAD=True
# The ISO adapters SVD/copy the real weights at apply() time, so params must be on
# CUDA (same constraint as blocktt); their optimiser state is ~1-4% of the model so
# it does not need offloading. Full FT does.
export ACTOR_PARAM_OFFLOAD=False

case "$BP_MODE" in
  dense)
    export PEFT_MODE=none
    export LR=${LR:-1e-6}
    export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
    export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
    TAG=bp-dense ;;
  iso|isobtt|isobtt_mix)
    export PEFT_MODE=$BP_MODE
    # LR matched on per-step *relative weight motion*. AdamW moves each coordinate
    # by ~lr, so full FT moves ||dW||/||W|| ~ lr/0.02 = 50*lr (5e-5 at lr=1e-6).
    # For the ISO modes the step lands in the rotation generator: entries ~lr give
    # ||dW||/||W|| ~ sqrt(b)*lr (~11*lr at b=128, ~8*lr at b=64), so 5e-6 puts all
    # three arms within ~2x of the dense per-step motion. Analytic, not swept --
    # and the ES thread showed step size dominates, so this is the first knob to
    # revisit if an arm looks mis-scaled.
    export LR=${LR:-5e-6}
    export ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-False}
    export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
    TAG=bp-$BP_MODE ;;
  *) echo "unknown BP_MODE=$BP_MODE" >&2; exit 2 ;;
esac

# A crashed arm leaves an orphaned Ray head holding this GPU pair's port and
# temp dir; the next arm's `ray start --head` then dies on a session-name
# mismatch. Reap only processes whose cmdline names OUR temp dir, so concurrent
# runs on other GPUs are untouched.
_gpu0=${CUDA_VISIBLE_DEVICES%%,*}
_raytmp=/tmp/ray_grpo_gpu${_gpu0:-0}
for _p in $(pgrep -f "$_raytmp" 2>/dev/null || true); do
  if tr '\0' ' ' < "/proc/$_p/cmdline" 2>/dev/null | grep -q -- "$_raytmp"; then
    echo "[bp] reaping orphaned ray pid $_p ($_raytmp)"; kill -9 "$_p" 2>/dev/null || true
  fi
done
rm -rf "$_raytmp"

# A killed/crashed arm's vLLM engine workers can hold GPU memory for a while after
# the ray head dies; starting the next arm too early fails with "Free memory on
# device ... is less than desired GPU memory utilization". Wait for the pair to drain.
for _i in $(seq 1 30); do
  _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | paste -sd+ | bc)
  [ "${_used:-0}" -lt 2000 ] && break
  echo "[bp] waiting for GPUs $CUDA_VISIBLE_DEVICES to drain (${_used} MiB used)"; sleep 10
done

export PROJECT_NAME=${PROJECT_NAME:-BP-q2p5-7b}
export TRAINER_LOGGER=${TRAINER_LOGGER:-['console','wandb']}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${TAG}_math-lv3to5_lr${LR}_n${N_RESPONSES}_bs${TRAIN_BATCH_SIZE}}
export PROJECT_PATH=${PROJECT_PATH:-/data/yequan/bp/${PROJECT_NAME}}
export CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}
mkdir -p "$CKPT_PATH"

bash grpo.sh
