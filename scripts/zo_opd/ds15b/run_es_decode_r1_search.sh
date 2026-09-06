#!/bin/bash
# es-decode rank-1 hyper-parameter search, sequential on ONE GPU (2026-09-06).
# Each arm: N=32 (16 antithetic pairs), sigma 1e-3 unless overridden, 64 seqs/step,
# eval (MATH-500 greedy) + checkpoint every 10 steps, post-update gain on the first wave.
# ARMS entries: "tag:ENV=val,ENV=val,..."
#   TRAIN_GPU=7 nohup bash scripts/zo_opd/ds15b/run_es_decode_r1_search.sh > logs/ds15b/es_decode/search.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mkdir -p logs/ds15b/es_decode
ARMS=${ARMS:-"a1.25e-3:ES_ALPHA=1.25e-3,ES_ITERS=40 raw_a2.8e-3:ES_ALPHA=2.8e-3,ES_NORMALIZE=raw,ES_ITERS=40 a2.5e-3:ES_ALPHA=2.5e-3,ES_ITERS=30 a5e-4:ES_ALPHA=5e-4,ES_ITERS=30"}
for arm in $ARMS; do
  tag=${arm%%:*}; envs=${arm#*:}
  echo "[search] arm=$tag envs=$envs start $(date '+%F %T')"
  ( export TRAIN_GPU=${TRAIN_GPU:-7} NOISE_RANK=1 EVAL_INTERVAL=10 SAVE_FREQ=10 CKPT_KEEP_LAST=5
    for kv in ${envs//,/ }; do export "$kv"; done
    bash scripts/zo_opd/ds15b/es_decode.sh )
  echo "[search] arm=$tag exit=$? $(date '+%F %T')"
  sleep 20
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${TRAIN_GPU:-7}")
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}'); do kill -9 "$pid" 2>/dev/null; done
  sleep 30
done
echo "[search] all done $(date '+%F %T')"
