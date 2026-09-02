#!/bin/bash
# stop_run.sh <experiment_name> <gpu> -- stop one ds15b run cleanly: kill its driver, then its
# private Ray head (RAY_ISOLATE) and workers, and wait until the card is empty.
#   bash scripts/zo_opd/ds15b/stop_run.sh ds15b_es_opd_k0_N32_sig1e-3_a5e-4 1
set -u
EXP=$1; GPU=$2
pkill -u "$(id -u)" -f "experiment_name=${EXP}[ ]" 2>/dev/null || true   # driver (+tee'd bash)
sleep 5
pkill -u "$(id -u)" -f "ray_opd_gpu${GPU}/sessio[n]_" 2>/dev/null || true # gcs/raylet/dashboard
sleep 5
# ray workers of that session hold the GPU; they die with the raylet, but make sure
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
  if [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$(id -un)" ]; then kill "$p" 2>/dev/null || true; fi
done
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ "$used" -lt 2000 ] && { echo "GPU $GPU free (${used} MiB)"; exit 0; }
  sleep 5
done
echo "GPU $GPU still has ${used} MiB in use"; exit 1
