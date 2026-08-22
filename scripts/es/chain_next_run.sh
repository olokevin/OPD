#!/bin/bash
# Wait for a running ES job (by PID) to exit, then start the next mode on the same GPU.
# Usage: bash scripts/es/chain_next_run.sh <wait_pid> <device> <perturb_mode> <log_name>
set -u
WAIT_PID=$1; DEVICE=$2; MODE=$3; LOGNAME=$4
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"

echo "[chain] waiting for pid $WAIT_PID before starting '$MODE' on GPU $DEVICE ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain] pid $WAIT_PID exited at $(date). Waiting 90s for the GPU to drain."
sleep 90

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl
DEVICE="$DEVICE" PERTURB_MODE="$MODE" bash scripts/es/run_es_math.sh > "logs/es/${LOGNAME}.log" 2>&1
echo "[chain] '$MODE' finished at $(date) with exit $?"
