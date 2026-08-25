#!/bin/bash
# Materialize every finished ES arm into an HF checkpoint + delta_stats.json.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
PY=/home/yequan/miniconda3/envs/verl/bin/python
ROOT=/data/yequan/es/ES-q2p5-7b
OUT=${OUT:-/data/yequan/es/materialized}
export HF_HOME=/data/yequan/huggingface
export CUDA_VISIBLE_DEVICES=${DEVICE:-1}
mkdir -p "$OUT" logs/es

run () {   # run <tag> <run-dir-glob>
  local tag=$1 dir=$2
  local coef
  coef=$(ls -d $ROOT/$dir/es_train_*/es_coef_best.pt 2>/dev/null | head -1)
  if [ -z "$coef" ]; then echo "[skip] $tag: no coef"; return; fi
  if [ -f "$OUT/$tag/model.safetensors" ]; then echo "[skip] $tag: exists"; return; fi
  echo "=== $tag  <- $coef"
  $PY scripts/es/materialize_es_ckpt.py --coef "$coef" --out "$OUT/$tag" \
      2>&1 | grep -v -E "FutureWarning|import pynvml|Automatically detected"
}

run dense        'es-dense-full_math-lv3to5-b64_sig0.001_a0.0005_N30'
run zoact        'zoact-r1_math-lv3to5-b64_sig0.001_a0.0005_N30'
run insparse     'insparse-d0.01_math-lv3to5-b64_sig0.001_a0.0005_N30'
run fura         'fura-btt-smallcore-sigmatched_sig0.0125_a0.00625_N30_long'
run iso          'iso-fixedspec-b128_math-lv3to5-b64_sig0.05_a0.025_N30'
run isobtt       'isobtt-fixedspec-smallcore_math-lv3to5-b64_sig0.05_a0.025_N30'
echo "ALL DONE"
