#!/bin/bash
# Evaluate a list of ES arms on prior-knowledge benchmarks + MATH-500.
#   DEVICE=2 bash scripts/es/run_forgetting_sweep.sh base dense zoact ...
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
BASE=/data/yequan/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e
MAT=/data/yequan/es/materialized

for tag in "$@"; do
  if [ "$tag" = "base" ]; then M=$BASE; else M=$MAT/$tag; fi
  if [ ! -e "$M" ]; then echo "[skip] $tag: $M missing"; continue; fi
  DEVICE=${DEVICE:-2} bash scripts/es/eval_forgetting.sh "$tag" "$M" 2>&1 \
    | grep -v -E "^(INFO|WARNING|Loading safetensors|\[Gloo\]|Processed prompts|Running loglikelihood|Adding requests)" \
    | grep -v -E "it/s\]$|it/s, est"
done
echo "SWEEP DONE: $*"
