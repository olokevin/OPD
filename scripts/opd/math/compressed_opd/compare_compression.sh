#!/bin/bash
# compare_compression.sh — run the 4B→~1.7B compression-strategy comparison.
#
# Compares the uncompressed Qwen3-4B (non-thinking) teacher against three
# ~1.7B-target compression strategies, under two calibration datasets:
#   strategies: sparsegpt (unstructured, ~64% per-linear), svd_v2_all,
#               svd_v2_attn_nystrom_mlp
#   calib:      c4, openthought3
# Metrics: C4 validation PPL + MATH-500 accuracy (HF generate + ttrl grader).
#
# Single GPU. Pin via CUDA_VISIBLE_DEVICES (default 4). Override anything:
#   CUDA_VISIBLE_DEVICES=5 MATH_LIMIT=500 bash compare_compression.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
# Keep peak memory low so the run survives a shared GPU.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PY=${PY:-python3}
MATH_LIMIT=${MATH_LIMIT:-200}
CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
CALIB_MAX_LENGTH=${CALIB_MAX_LENGTH:-2048}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-2}
MATH_BATCH_SIZE=${MATH_BATCH_SIZE:-16}
SPARSEGPT_MEM_GB=${SPARSEGPT_MEM_GB:-8}
OUT=${OUT:-"$SCRIPT_DIR/results/compare_results.json"}

mkdir -p "$(dirname "$OUT")"

cd "$REPO_ROOT"
$PY "$SCRIPT_DIR/compare_compression.py" \
  --calib c4 openthought3 \
  --strategies sparsegpt svd_v2_all svd_v2_attn_nystrom_mlp \
  --calib-num-seqs "$CALIB_NUM_SEQS" \
  --calib-max-length "$CALIB_MAX_LENGTH" \
  --calib-batch-size "$CALIB_BATCH_SIZE" \
  --ppl-seqlen 2048 \
  --math-limit "$MATH_LIMIT" \
  --math-max-new-tokens 2048 \
  --math-batch-size "$MATH_BATCH_SIZE" \
  --sparsegpt-mem-gb "$SPARSEGPT_MEM_GB" \
  --out "$OUT"
