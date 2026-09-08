#!/bin/bash
# The two es-decode arms of 2026-09-05, sequentially on ONE GPU:
#   1. full-rank (packed-bit +-1 matrix per rail)     2. rank-1 (a b^T per rail)
#   TRAIN_GPU=7 nohup bash scripts/zo_opd/ds15b/run_es_decode_arms.sh > logs/ds15b/es_decode/arms.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p logs/ds15b/es_decode
for rank in ${ARMS:-full 1}; do
  echo "[arms] rank=$rank start $(date '+%F %T')"
  TRAIN_GPU=${TRAIN_GPU:-7} NOISE_RANK=$rank bash scripts/zo_opd/ds15b/es_decode.sh
  echo "[arms] rank=$rank exit=$? $(date '+%F %T')"
  # reap anything the driver left on the GPU (teardown hang, see es_profile_results.md 14)
  sleep 20
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${TRAIN_GPU:-7}")
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}'); do kill -9 "$pid" 2>/dev/null; done
  sleep 30
done
echo "[arms] all done $(date '+%F %T')"
