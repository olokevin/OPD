#!/usr/bin/env bash
# Reasoning-aware compress-then-train pipeline: in-process SVD(attn)+Nystrom(mlp/expert)
# compression at retain 0.7, then SFT on OpenThoughts3, then eval MATH-500@4096 + MMLU-Pro.
#
# 2x2 matrix: {qwen3_4b_base, olmoe_1b7b} x {forward, combined}, one run per GPU (1-4).
# TRAINING always runs in the `sft` conda env (compression is in-process at model-init,
# so the whole compress+train+save is ONE sft-env process). EVAL runs in the `verl` env
# (its ttrl_math grader imports verl/ray).
#
# Usage:
#   bash run_compress_sft.sh train all              # launch all 4 training runs (bg, GPUs 1-4)
#   bash run_compress_sft.sh train qwen3 forward 1  # one cell on a chosen GPU
#   bash run_compress_sft.sh eval  all              # eval every final-merged ckpt (serial, GPU 1)
#   bash run_compress_sft.sh eval  qwen3 forward 1  # eval one cell on a chosen GPU
set -euo pipefail

REPO=/home/yequan/Project/compression/OPD
LF=$REPO/LlamaFactory
SFT_CLI=/home/yequan/miniconda3/envs/sft/bin/llamafactory-cli
VERL_PY=/home/yequan/miniconda3/envs/verl/bin/python
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}

OUT_ROOT=/data/yequan/compress_sft
SFT_ROOT=$OUT_ROOT/sft
MET_ROOT=$OUT_ROOT/metrics

# model_tag -> wandb project + the example-yaml stem
declare -A YAML=(
  [qwen3:forward]=qwen3_4b_compressed_fwd_sft.yaml
  [qwen3:combined]=qwen3_4b_compressed_combined_sft.yaml
  [olmoe:forward]=olmoe_compressed_fwd_sft.yaml
  [olmoe:combined]=olmoe_compressed_combined_sft.yaml
)
declare -A TAG=( [qwen3]=qwen3_4b_base [olmoe]=olmoe_1b7b )

train_cell() {
  local model=$1 obj=$2 gpu=$3
  local tag=${TAG[$model]}
  local yaml=${YAML[$model:$obj]}
  local out=$SFT_ROOT/$tag/${obj}_r0.7
  echo ">>> [train] $tag $obj on GPU $gpu -> $out"
  cd "$LF"
  CUDA_VISIBLE_DEVICES=$gpu \
  WANDB_PROJECT=compress_sft_$tag \
  HF_HOME=$HF_HOME \
    "$SFT_CLI" train "examples/compress_train/$yaml" \
      output_dir="$out" \
      run_name="${tag}_compress_${obj}_r0.7_sft"
}

eval_cell() {
  local model=$1 obj=$2 gpu=$3
  local tag=${TAG[$model]}
  local ckpt=$SFT_ROOT/$tag/${obj}_r0.7/final-merged
  local met=$MET_ROOT/$tag/${obj}_r0.7
  mkdir -p "$met"
  if [[ ! -d $ckpt ]]; then echo "!!! missing $ckpt — train first"; return 1; fi
  echo ">>> [eval] $tag $obj on GPU $gpu  ($ckpt)"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:verl HF_HOME=$HF_HOME \
    "$VERL_PY" scripts/opd/math/compressed_opd/eval_opd_ckpt.py \
      --model-dir "$ckpt" --label "${tag}_${obj}_sft" \
      --metrics-json "$met/final_math500.json" \
      --math-limit 500 --math-max-new-tokens 4096 --math-batch-size 8
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:verl HF_HOME=$HF_HOME \
    "$VERL_PY" scripts/opd/math/compressed_opd/eval_mmlu_pro.py \
      --model-dir "$ckpt" --label "${tag}_${obj}_sft" \
      --metrics-json "$met/final_mmlu_pro.json" \
      --limit 1000 --mmlu-max-new-tokens 512 --mmlu-batch-size 16
}

PHASE=${1:?phase: train|eval}
shift

# Matrix launch. NOTE: OLMoE (fused MoE experts on transformers>=5.2) is a BLOCKED
# follow-up — `all` runs only the validated Qwen3 cells (fwd on GPU 1, combined on
# GPU 2). Launch OLMoE explicitly once fused-expert Nystrom lands:
#   bash run_compress_sft.sh train olmoe forward 3
run_all() {
  local fn=$1
  if [[ $fn == train_cell ]]; then
    $fn qwen3 forward  1 &
    $fn qwen3 combined 2 &
    wait
  else  # eval serially on one GPU to avoid contention
    for cell in "qwen3 forward" "qwen3 combined"; do
      $fn $cell 1 || true
    done
  fi
}

case "$PHASE" in
  train)
    if [[ "${1:-all}" == all ]]; then run_all train_cell
    else train_cell "$1" "$2" "${3:-0}"; fi ;;
  eval)
    if [[ "${1:-all}" == all ]]; then run_all eval_cell
    else eval_cell "$1" "$2" "${3:-0}"; fi ;;
  *) echo "unknown phase $PHASE (train|eval)"; exit 1 ;;
esac
