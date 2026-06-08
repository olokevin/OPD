#!/usr/bin/env bash
# Block D launcher (GPU 5). Usage:
#   bash run_blockD.sh sanity     # D0 only, MATH/16 — pipeline check
#   bash run_blockD.sh full       # D0-D3, MATH/100 + C4 PPL
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # OPD repo root
REPO=$(pwd)
PY=/home/yequan/miniconda3/envs/verl/bin/python
export CUDA_VISIBLE_DEVICES=5
export HF_HOME=/data/yequan/huggingface
export PYTHONPATH="$REPO/src:$REPO/verl"
SD=scripts/reasoning_aware_compress
mode=${1:-full}
ts=$(date +%Y%m%d_%H%M%S)

if [ "$mode" = "sanity" ]; then
  $PY $SD/bi_whitened_svd.py --cells D0 --ratio 0.8 \
    --math-limit 16 --math-batch-size 8 \
    --calib-num-seqs 32 \
    --out $SD/results/blockD/_sanity.json \
    2>&1 | tee $SD/logs/blockD_sanity_${ts}.log
else
  $PY $SD/bi_whitened_svd.py --cells D0 D1 D2 D3 --ratio 0.8 \
    --math-limit 100 \
    --out $SD/results/blockD/bi_whitened_r0.8.json \
    2>&1 | tee $SD/logs/blockD_full_${ts}.log
fi
