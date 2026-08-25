#!/bin/bash
# Run one or more Countdown ES arms back-to-back on a single GPU.
#   DEVICE=1 bash scripts/es/chain_countdown.sh dense fura
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
M15B=/data/yequan/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
for mode in "$@"; do
  echo "=== [$(date)] countdown arm: $mode on GPU ${DEVICE:-1}"
  DEVICE=${DEVICE:-1} PERTURB_MODE="$mode" MODEL=${MODEL:-$M15B} \
    VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-200} \
    PROJECT_NAME=${PROJECT_NAME:-ES-forget-cd-q1p5b} \
    EXPERIMENT_NAME="cd-${mode}_q1p5b_b200_N30" \
    bash scripts/es/run_countdown_es.sh
  echo "=== [$(date)] arm $mode exited with $?"
done
echo "CHAIN DONE: $*"
