#!/bin/bash
# layer_sensitivity.sh — per-layer compression-sensitivity sweep over GPUs 4-7.
#
# Two models (Qwen3-4B non-thinking base + the Non-Thinking-RL-Math Step500
# checkpoint), each with 36 decoder layers. For every layer we compress ONE
# module at a time — self_attn (4 linears, SVD-LLM-V2) or mlp (gate/up/down
# triplet, Nystrom) — at retain ratios 0.9/0.8/0.6/0.5, and grade MATH-500
# (first 200). Calibration = OpenThought3 math traces (same recipe as
# compare_compression.sh). Uncompressed baselines are graded once per model.
#
# Sharding: 2 shards, one model per GPU (all 36 layers, both modules, 4 ratios,
# + that model's uncompressed baseline). GPU6=base, GPU7=rlmath.
#
# Launch both in the background, tee logs, write per-shard JSON:
#   bash scripts/reasoning_aware_compress/layer_sensitivity.sh
# Override the GPU pair or MATH size:
#   GPUS="6 7" MATH_LIMIT=100 bash .../layer_sensitivity.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Keep each shard isolated to its own GPU; one Ray-free pure-HF process per GPU.
export TOKENIZERS_PARALLELISM=false

PY=${PY:-python3}
MATH_LIMIT=${MATH_LIMIT:-100}
MATH_BATCH_SIZE=${MATH_BATCH_SIZE:-16}
CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-2}
RATIOS=${RATIOS:-"0.9 0.8 0.6 0.5"}

BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B"}
RL_MODEL=${RL_MODEL:-"Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"}

RES_DIR="$SCRIPT_DIR/results"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$RES_DIR" "$LOG_DIR"

cd "$REPO_ROOT"

# run_shard <gpu> <model> <model_tag> <layers> <baseline_flag>
run_shard() {
  local gpu="$1" model="$2" tag="$3" layers="$4" baseline="$5"
  local lname; lname="$(echo "$layers" | tr ',-' '__')"
  local out="$RES_DIR/layer_sens_${tag}_l${lname}.json"
  local log="$LOG_DIR/layer_sens_${tag}_l${lname}.log"
  echo "[GPU $gpu] $tag layers=$layers baseline=$baseline -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$SCRIPT_DIR/layer_sensitivity.py" \
    --model "$model" --model-tag "$tag" \
    --layers "$layers" --modules self_attn mlp \
    --ratios $RATIOS \
    --calib-num-seqs "$CALIB_NUM_SEQS" --calib-batch-size "$CALIB_BATCH_SIZE" \
    --math-limit "$MATH_LIMIT" --math-batch-size "$MATH_BATCH_SIZE" \
    $baseline \
    --out "$out" > "$log" 2>&1 &
  echo "  pid $! log $log"
}

read -r G0 G1 <<< "${GPUS:-6 7}"

run_shard "$G0" "$BASE_MODEL" "qwen3-4b-base"   "all" "--eval-baseline"
run_shard "$G1" "$RL_MODEL"   "qwen3-4b-rlmath" "all" "--eval-baseline"

echo "Launched 2 shards on GPUs $G0 $G1 (MATH_LIMIT=$MATH_LIMIT). Waiting..."
wait
echo "All shards finished. Per-shard JSON under $RES_DIR/"
