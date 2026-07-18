#!/bin/bash
# layer_sensitivity_base_subset.sh — base-model-only sensitivity sweep over a
# chosen subset of decoder layers, sharded across GPUs 6 and 7.
#
# Model: Qwen/Qwen3-4B (non-thinking) ONLY. Layers swept in this order
# (late -> mid -> early), split across two GPUs (order preserved per shard):
#   GPU6: 35 34 33 32 31 30 20 19 18
#   GPU7: 17 16 15  5  4  3  2  1  0
# Both modules (self_attn svd_v2 / mlp nystrom), ratios 0.9/0.8/0.6/0.5,
# MATH-500 first 100, OpenThought3 calib. GPU6 also grades the baseline.
#
#   bash scripts/reasoning_aware_compress/layer_sensitivity_base_subset.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

PY=${PY:-python3}
MATH_LIMIT=${MATH_LIMIT:-100}
MATH_BATCH_SIZE=${MATH_BATCH_SIZE:-16}
CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-2}
RATIOS=${RATIOS:-"0.9 0.8 0.6 0.5"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B"}

RES_DIR="$SCRIPT_DIR/results"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$RES_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

# run_shard <gpu> <shard_id> <layers_csv> <baseline_flag>
run_shard() {
  local gpu="$1" sid="$2" layers="$3" baseline="$4"
  local out="$RES_DIR/layer_sens_qwen3-4b-base_${sid}.json"
  local log="$LOG_DIR/layer_sens_qwen3-4b-base_${sid}.log"
  echo "[GPU $gpu] base $sid layers=$layers baseline='$baseline' -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$SCRIPT_DIR/layer_sensitivity.py" \
    --model "$BASE_MODEL" --model-tag "qwen3-4b-base" \
    --layers "$layers" --modules self_attn mlp \
    --ratios $RATIOS \
    --calib-num-seqs "$CALIB_NUM_SEQS" --calib-batch-size "$CALIB_BATCH_SIZE" \
    --math-limit "$MATH_LIMIT" --math-batch-size "$MATH_BATCH_SIZE" \
    $baseline \
    --out "$out" > "$log" 2>&1 &
  echo "  pid $! log $log"
}

read -r G0 G1 <<< "${GPUS:-6 7}"

run_shard "$G0" "shardA" "35,34,33,32,31,30,20,19,18" "--eval-baseline"
run_shard "$G1" "shardB" "17,16,15,5,4,3,2,1,0"       ""

echo "Launched 2 base-only shards on GPUs $G0 $G1 (MATH_LIMIT=$MATH_LIMIT). Waiting..."
wait
echo "All shards finished. Per-shard JSON under $RES_DIR/"
