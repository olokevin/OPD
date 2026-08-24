#!/bin/bash
# After the FuRA LR sweep finishes, run the winning alpha for the FULL 150 iterations.
# 20-iteration sweep points compare transients; dense needed ~40 steps to reach its 71.5
# plateau and 150 to reveal its -6.1pp train/held-out overfitting gap, so plateau-to-plateau
# is the only fair answer to "can FuRA match full ES".
set -u
WAIT_PID=${1:-}
REPO=${REPO:-/home/yequan/Project/compression/OPD}
DEVICE=${DEVICE:-2}
ALPHA_BEST=${ALPHA_BEST:-0.00625}   # 12.5x = footprint-matched; stable, best of the sweep
cd "$REPO"

if [ -n "$WAIT_PID" ]; then
  echo "[chain] waiting for sweep pid $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[chain] sweep exited at $(date); draining GPU for 90s"
  sleep 90
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl
DEVICE="$DEVICE" PERTURB_MODE=fura \
  SIGMA=0.001 ALPHA="$ALPHA_BEST" \
  NUM_ITERATIONS=150 EVAL_INTERVAL=10 \
  EXPERIMENT_NAME="fura-btt-smallcore-lr12.5x_sig0.001_a${ALPHA_BEST}_N30_long" \
  bash scripts/es/run_es_math.sh > logs/es/run4b_fura_lr12.5x_long.log 2>&1
echo "[chain] fura long run finished $(date) exit=$?"
