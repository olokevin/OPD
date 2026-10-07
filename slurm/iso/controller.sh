#!/bin/bash
# controller.sh — 2-node auto-resume controller for ONE ISO-paper run on NERSC
# interactive nodes (4h cap). Same logic as the SimpleRL controller (branch
# `simplerl`, slurm/grpo/simplerl/controller.sh): loop salloc, resume from the fixed
# ckpt dir, prune old ckpts, transient salloc failures don't count as stalls.
# Use launch.sh rather than calling this directly.
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
ACCOUNT=${ACCOUNT:-m4788_g}
ALLOC_NODES=${ALLOC_NODES:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-60}
STALL_LIMIT=${STALL_LIMIT:-4}
SALLOC_FAIL_LIMIT=${SALLOC_FAIL_LIMIT:-40}
SALLOC_BACKOFF=${SALLOC_BACKOFF:-120}

export ENV_SCRIPT=iso/env.sh RAYENV_SCRIPT=iso/rayenv.sh
export PEFT_MODE=${PEFT_MODE:?} LR=${LR:?} EXP=${EXP:?} CKPT=${CKPT:?} WANDB_RUN_ID=${WANDB_RUN_ID:?}
export WANDB_PROJECT=${WANDB_PROJECT:-iso_rlvr_qwen3_4b}
JOB_TAG=${JOB_TAG:-$WANDB_RUN_ID}

ITERFILE=${CKPT}/latest_checkpointed_iteration.txt
LOGDIR=${DATA_ROOT}/logs/iso
mkdir -p "$CKPT" "$LOGDIR"
cur_iter() { cat "$ITERFILE" 2>/dev/null | tr -dc '0-9' || echo 0; }

# Keep ONLY the latest checkpoint (the resumed-from one lingers across relaunches).
prune_loop() {
  while true; do
    L=$(cat "$ITERFILE" 2>/dev/null | tr -dc '0-9')
    if [ -n "$L" ]; then
      for d in "$CKPT"/global_step_*; do
        [ -d "$d" ] || continue
        n=$(basename "$d" | tr -dc '0-9')
        [ -n "$n" ] && [ "$n" -lt "$L" ] && rm -rf "$d"
      done
    fi
    sleep 60
  done
}
prune_loop & PRUNE_PID=$!
trap 'kill $PRUNE_PID 2>/dev/null' EXIT

ATTEMPT=0; PREV_ITER=-1; STALL=0; SALLOC_FAIL=0
echo "=== [$JOB_TAG] controller start $(date) | ckpt=$CKPT | mode=$PEFT_MODE lr=$LR ==="
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START_ITER=$(cur_iter); START_ITER=${START_ITER:-0}
  RUNLOG=${LOGDIR}/${JOB_TAG}_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== [$JOB_TAG] attempt $ATTEMPT $(date) | start_iter=${START_ITER} | runlog=$RUNLOG ==="
  salloc --nodes "$ALLOC_NODES" --qos interactive --time 4:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh" > "$RUNLOG" 2>&1
  RC=$?
  END_ITER=$(cur_iter); END_ITER=${END_ITER:-0}
  TOTAL=$(grep -oE "Total training steps: [0-9]+" "$RUNLOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  echo "=== [$JOB_TAG] attempt $ATTEMPT done rc=$RC | iter ${START_ITER}->${END_ITER} | total=${TOTAL:-?} ==="
  if [ -n "$TOTAL" ] && [ "$END_ITER" -ge "$TOTAL" ] 2>/dev/null; then
    echo "=== [$JOB_TAG] TRAINING COMPLETE: iter ${END_ITER} >= total ${TOTAL} ==="; break; fi
  if [ "$RC" -eq 0 ]; then echo "=== [$JOB_TAG] driver exited 0 at iter ${END_ITER} ==="; break; fi
  if grep -qE "Unable to allocate resources|salloc: error" "$RUNLOG" 2>/dev/null \
     && ! grep -q "HEAD=" "$RUNLOG" 2>/dev/null; then
    SALLOC_FAIL=$((SALLOC_FAIL+1))
    echo "=== [$JOB_TAG] salloc did not allocate (${SALLOC_FAIL}/${SALLOC_FAIL_LIMIT}); backoff ${SALLOC_BACKOFF}s ==="
    [ "$SALLOC_FAIL" -ge "$SALLOC_FAIL_LIMIT" ] && { echo "=== [$JOB_TAG] STOPPING: scheduler down? ==="; break; }
    ATTEMPT=$((ATTEMPT-1)); sleep "$SALLOC_BACKOFF"; continue
  fi
  SALLOC_FAIL=0
  if [ "$END_ITER" -le "$PREV_ITER" ] && [ "$END_ITER" -le "$START_ITER" ]; then
    STALL=$((STALL+1)); echo "=== [$JOB_TAG] no progress (stall ${STALL}/${STALL_LIMIT}) ==="; else STALL=0; fi
  PREV_ITER=$END_ITER
  [ "$STALL" -ge "$STALL_LIMIT" ] && { echo "=== [$JOB_TAG] STOPPING: no progress across ${STALL_LIMIT} allocations ($RUNLOG) ==="; break; }
  sleep 20
done
echo "=== [$JOB_TAG] controller end $(date) | final_iter=$(cur_iter) ==="
