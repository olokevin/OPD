#!/bin/bash
# Run the four BP arms back-to-back on one pair of GPUs (2-GPU FSDP each).
# Usage:  DEVICES=6,7 bash scripts/es/run_bp_all.sh [mode ...]
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
DEVICES=${DEVICES:-6,7}
MODES=${@:-"dense iso isobtt isobtt_mix"}
mkdir -p logs/bp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl
for m in $MODES; do
  LOG="logs/bp/${m}.log"
  echo "[bp-all] === $m starting $(date) -> $LOG ==="
  DEVICES="$DEVICES" BP_MODE="$m" bash scripts/es/run_bp_math.sh > "$LOG" 2>&1
  rc=$?   # capture BEFORE any other command: $(date) in the echo would clobber $?
  echo "[bp-all] === $m finished $(date) with exit $rc ==="
  [ "$rc" -ne 0 ] && echo "[bp-all] WARNING: $m FAILED (exit $rc) -- see $LOG"
  sleep 60   # let the GPUs drain before the next arm
done
echo "[bp-all] all done $(date)"
