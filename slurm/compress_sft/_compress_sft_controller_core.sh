#!/bin/bash
# _compress_sft_controller_core.sh — shared auto-resume controller for the
# compress->SFT jobs. Sourced by compress_sft_{fwd,combined}_controller.sh after
# they export OBJECTIVE + the run config. Mirrors slurm/opd/compress/*_controller.sh:
# loop `salloc -N4 --qos interactive --time 4:00:00 bash inside.sh`; on 4h revoke
# (rc!=0, no completion) re-allocate and resume from the latest checkpoint. Stop
# when final-merged/ appears (rc=0) or no checkpoint progress across STALL_LIMIT allocs.
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
ACCOUNT=${ACCOUNT:-m4788_g}
ALLOC_NODES=${ALLOC_NODES:-4}   # 4 nodes/job; 2 jobs x 4 = 8 nodes IS allowed (node=4 is
                                # MaxTRESPerJob, not per-user; MaxJobsPU=2 — verified live)
RATIO=${RATIO:-0.7}             # retain ratio tag for paths/run names
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}
STALL_LIMIT=${STALL_LIMIT:-3}

: "${OBJECTIVE:?set OBJECTIVE=forward|combined}"
: "${YAML:?set YAML}"
export OPD_REPO DATA_ROOT
export ENV_SCRIPT=${ENV_SCRIPT:-compress_sft/compress_sft_env.sh}
export OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/compress_sft/sft/qwen3_4b/${OBJECTIVE}_r${RATIO}}
export RUN_NAME=${RUN_NAME:-qwen3_4b_nersc_compress_${OBJECTIVE}_r${RATIO}_sft}
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}

LOG_DIR=${DATA_ROOT}/compress_sft/logs
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

latest_step() {  # highest checkpoint-N step in OUTPUT_DIR, or 0
  ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' \
    | sed -E 's/.*checkpoint-//' | sort -n | tail -1 | tr -dc '0-9' || true
}
is_done() { [ -f "$OUTPUT_DIR/final-merged/config.json" ]; }

ATTEMPT=0; PREV=-1; STALL=0
echo "=== compress-sft [$OBJECTIVE] controller start $(date) | out=$OUTPUT_DIR | nodes=$ALLOC_NODES ==="
if is_done; then echo "=== already complete (final-merged exists) ==="; exit 0; fi

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START=$(latest_step); START=${START:-0}
  RUNLOG=${LOG_DIR}/${OBJECTIVE}_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== [$OBJECTIVE] attempt $ATTEMPT $(date) | start_step=${START} | runlog=$RUNLOG ==="
  salloc --nodes "$ALLOC_NODES" --qos interactive --time 4:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/compress_sft/compress_sft_inside.sh" > "$RUNLOG" 2>&1
  RC=$?
  END=$(latest_step); END=${END:-0}
  echo "=== [$OBJECTIVE] attempt $ATTEMPT done rc=$RC | step ${START}->${END} ==="
  if is_done; then echo "=== TRAINING COMPLETE: final-merged written ($OUTPUT_DIR) ==="; break; fi
  if [ "$RC" -eq 0 ]; then echo "=== driver exited 0 (no final-merged yet?) — check $RUNLOG ==="; break; fi
  if [ "$END" -le "$PREV" ] && [ "$END" -le "$START" ]; then
    STALL=$((STALL+1)); echo "=== no progress (stall ${STALL}/${STALL_LIMIT}) ==="; else STALL=0; fi
  PREV=$END
  if [ "$STALL" -ge "$STALL_LIMIT" ]; then
    echo "=== STOPPING: no progress across ${STALL_LIMIT} allocations ($RUNLOG) ==="; break; fi
  sleep 20
done
echo "=== compress-sft [$OBJECTIVE] controller end $(date) | final_step=$(latest_step) ==="
