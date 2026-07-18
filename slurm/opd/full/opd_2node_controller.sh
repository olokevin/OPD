#!/bin/bash
# opd_2node_controller.sh — keep a 2-node (8 GPU) OPD run alive across the 4h
# interactive-QOS expiry. Each iteration grabs a fresh 2-node allocation and runs
# the Ray bootstrap + driver (which auto-resumes from the FIXED checkpoint dir).
# Stops when training completes, finishes early, or stalls (no checkpoint progress
# across several allocations -> likely a real failure, not just expiry).
#
# Launch detached from a login node:
#   nohup bash slurm/opd/full/opd_2node_controller.sh > /pscratch/sd/y/yequan/opd/logs/controller.log 2>&1 &
set -u
OPD_REPO=/global/u1/y/yequan/Project/OPD
DATA_ROOT=/pscratch/sd/y/yequan/opd
CKPT=${DATA_ROOT}/checkpoints/nersc_opd_qwen4b_1p7b_2node
ITERFILE=${CKPT}/latest_checkpointed_iteration.txt
ACCOUNT=m4788_g
MAX_ATTEMPTS=${MAX_ATTEMPTS:-24}
STALL_LIMIT=${STALL_LIMIT:-3}

mkdir -p "$CKPT" "${DATA_ROOT}/logs"
cur_iter() { cat "$ITERFILE" 2>/dev/null | tr -dc '0-9' || echo 0; }

# Keep ONLY the latest checkpoint. verl's max_actor_ckpt_to_keep=1 only rotates
# checkpoints saved within one process, so the checkpoint a segment RESUMES from
# lingers after the next save. This background pruner deletes any global_step_*
# dir older than latest_checkpointed_iteration.txt — it never touches the latest
# or an in-progress (higher-numbered) save, so it's safe to run continuously.
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

ATTEMPT=0; PREV_ITER=-1; STALL=0
echo "=== controller start $(date) | ckpt=$CKPT ==="
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START_ITER=$(cur_iter); START_ITER=${START_ITER:-0}
  RUNLOG=${DATA_ROOT}/logs/zo2node_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== attempt $ATTEMPT $(date) | start_iter=${START_ITER} | runlog=$RUNLOG ==="

  # Allocate 2 interactive nodes for up to 4h and run the bootstrap+driver under
  # the allocation. salloc returns when the driver exits OR the 4h limit revokes
  # the allocation.
  salloc --nodes 2 --qos interactive --time 4:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh" > "$RUNLOG" 2>&1
  RC=$?

  END_ITER=$(cur_iter); END_ITER=${END_ITER:-0}
  TOTAL=$(grep -oE "Total training steps: [0-9]+" "$RUNLOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  echo "=== attempt $ATTEMPT done rc=$RC | iter ${START_ITER}->${END_ITER} | total=${TOTAL:-?} ==="

  # Done conditions.
  if [ -n "$TOTAL" ] && [ "$END_ITER" -ge "$TOTAL" ] 2>/dev/null; then
    echo "=== TRAINING COMPLETE: iter ${END_ITER} >= total ${TOTAL} ==="; break
  fi
  if [ "$RC" -eq 0 ]; then
    echo "=== driver exited 0 (training finished) at iter ${END_ITER} ==="; break
  fi

  # Progress check: did the checkpoint advance this allocation?
  if [ "$END_ITER" -le "$PREV_ITER" ] && [ "$END_ITER" -le "$START_ITER" ]; then
    STALL=$((STALL+1))
    echo "=== no checkpoint progress (stall ${STALL}/${STALL_LIMIT}) ==="
  else
    STALL=0
  fi
  PREV_ITER=$END_ITER
  if [ "$STALL" -ge "$STALL_LIMIT" ]; then
    echo "=== STOPPING: no progress across ${STALL_LIMIT} allocations (check $RUNLOG) ==="; break
  fi
  sleep 20
done
echo "=== controller end $(date) | final_iter=$(cur_iter) ==="
