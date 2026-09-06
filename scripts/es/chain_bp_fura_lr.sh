#!/bin/bash
# LR search for the BP fura arm (scripts/es/run_bp_fura_math.sh): one run per LR,
# sequentially on one GPU, each ~50 min (10 SGD steps + 11 greedy MATH-500 evals).
# Dense SGD reference used lr 0.1; the search starts at 10x that.
#
# Usage:  DEVICES=4 bash scripts/es/chain_bp_fura_lr.sh 1.0 0.3 3.0 0.1
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
DEVICES=${DEVICES:-4}
LRS=${@:-"1.0 0.3 3.0 0.1"}
mkdir -p logs/es
for lr in $LRS; do
  LOG="logs/es/bp_fura_lr${lr}_gpu${DEVICES}.log"
  echo "[chain] === lr=$lr starting $(date) -> $LOG ==="
  DEVICES="$DEVICES" LR="$lr" bash scripts/es/run_bp_fura_math.sh > "$LOG" 2>&1
  rc=$?   # capture BEFORE any other command
  echo "[chain] === lr=$lr finished $(date) with exit $rc ==="
  [ "$rc" -ne 0 ] && echo "[chain] WARNING: lr=$lr FAILED (exit $rc) -- see $LOG"
  python scripts/es/collect_bp_fura.py "$LOG" || true
  sleep 30   # let the GPU drain before the next run
done
echo "[chain] all done $(date)"
