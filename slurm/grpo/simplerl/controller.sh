#!/bin/bash
# controller.sh — GENERIC 2-node auto-resume controller for the SimpleRL-aligned
# GRPO campaign on Qwen2.5-7B. One instance drives ONE run (full or a fura LR
# point); launch several concurrently (interactive QOS cap = 4 nodes -> 2 of
# these 2-node controllers at a time). Loops salloc across the 4h interactive cap
# and resumes from the fixed ckpt dir each segment. Reuses opd_2node_inside.sh
# (ENV_SCRIPT selects the SimpleRL full/fura driver).
#
# Drive it entirely by env vars, e.g.:
#   JOB_TAG=simplerl_full_lr5e-7 \
#   ENV_SCRIPT=grpo/simplerl/full_env.sh \
#   CKPT_PATH=/pscratch/sd/y/yequan/opd/checkpoints/simplerl_full_lr5e-7 \
#   EXPERIMENT_NAME=simplerl_full_lr5e-7 WANDB_RUN_ID=simplerl_full_lr5e-7 \
#   LR=5e-7 \
#   nohup bash slurm/grpo/simplerl/controller.sh \
#     > /pscratch/sd/$USER/opd/logs/ctl_simplerl_full_lr5e-7_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
ACCOUNT=${ACCOUNT:-m4788_g}
ALLOC_NODES=${ALLOC_NODES:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}
STALL_LIMIT=${STALL_LIMIT:-5}   # real training stalls (allocation ran, 0 progress)
# Transient scheduler failures (salloc never granted an allocation, e.g.
# "Unable to allocate resources: Connection timed out") are NOT training stalls
# and must not burn the stall budget — they get a separate, larger retry budget
# with backoff so a NERSC scheduler hiccup can't kill a half-done campaign.
SALLOC_FAIL_LIMIT=${SALLOC_FAIL_LIMIT:-40}
SALLOC_BACKOFF=${SALLOC_BACKOFF:-120}

# ---- required per-run config (exported so inside.sh -> rayenv -> env -> grpo.sh see it) ----
export ENV_SCRIPT=${ENV_SCRIPT:?set ENV_SCRIPT (e.g. grpo/simplerl/full_env.sh)}
export CKPT_PATH=${CKPT_PATH:?set CKPT_PATH}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
export WANDB_RUN_ID=${WANDB_RUN_ID:?set WANDB_RUN_ID}
export LR=${LR:?set LR}
export PROJECT_NAME=${PROJECT_NAME:-nersc_simplerl_qwen2p5_7b}
export WANDB_PROJECT=${WANDB_PROJECT:-nersc_simplerl_qwen2p5_7b}
JOB_TAG=${JOB_TAG:-${EXPERIMENT_NAME}}

CKPT="$CKPT_PATH"
ITERFILE=${CKPT}/latest_checkpointed_iteration.txt
mkdir -p "$CKPT" "${DATA_ROOT}/logs"
cur_iter() { cat "$ITERFILE" 2>/dev/null | tr -dc '0-9' || echo 0; }

# Keep ONLY the latest checkpoint (max_actor_ckpt_to_keep=1 only rotates in-process
# saves; the resumed-from ckpt lingers across relaunches).
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
echo "=== [$JOB_TAG] controller start $(date) | ckpt=$CKPT | nodes=$ALLOC_NODES | lr=$LR | env=$ENV_SCRIPT ==="
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START_ITER=$(cur_iter); START_ITER=${START_ITER:-0}
  RUNLOG=${DATA_ROOT}/logs/${JOB_TAG}_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
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
  # Transient scheduler failure? salloc never granted an allocation, so inside.sh
  # never ran (no "HEAD=" line). Don't burn the stall budget; back off and retry.
  if grep -qE "Unable to allocate resources|salloc: error" "$RUNLOG" 2>/dev/null \
     && ! grep -q "HEAD=" "$RUNLOG" 2>/dev/null; then
    SALLOC_FAIL=$((SALLOC_FAIL+1))
    echo "=== [$JOB_TAG] salloc did not allocate (transient scheduler failure ${SALLOC_FAIL}/${SALLOC_FAIL_LIMIT}); backoff ${SALLOC_BACKOFF}s ==="
    if [ "$SALLOC_FAIL" -ge "$SALLOC_FAIL_LIMIT" ]; then
      echo "=== [$JOB_TAG] STOPPING: salloc failed ${SALLOC_FAIL_LIMIT}x in a row (scheduler down?) ==="; break; fi
    ATTEMPT=$((ATTEMPT-1))   # infra failures don't count toward MAX_ATTEMPTS
    sleep "$SALLOC_BACKOFF"
    continue
  fi
  SALLOC_FAIL=0
  if [ "$END_ITER" -le "$PREV_ITER" ] && [ "$END_ITER" -le "$START_ITER" ]; then
    STALL=$((STALL+1)); echo "=== [$JOB_TAG] no progress (stall ${STALL}/${STALL_LIMIT}) ==="; else STALL=0; fi
  PREV_ITER=$END_ITER
  if [ "$STALL" -ge "$STALL_LIMIT" ]; then echo "=== [$JOB_TAG] STOPPING: no progress across ${STALL_LIMIT} allocations ($RUNLOG) ==="; break; fi
  sleep 20
done
echo "=== [$JOB_TAG] controller end $(date) | final_iter=$(cur_iter) ==="
