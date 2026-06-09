#!/bin/bash
# Training-free leg of the MoE expert-compression recovery atlas.
# Compress OLMoE-Instruct experts with each method x retain, save the reloadable
# HF checkpoint, and eval STEP-0 on the 4-task suite (MMLU/GSM8K/ARC-C/HellaSwag).
# NO recovery training here — that is the separate recovery phase.
#
# Schedules one job per free GPU from GPU_IDS (default "1 2 3" per user), round-robin.
#
# Usage:
#   bash scripts/moe_compress/run_trainfree_atlas.sh [GPU_IDS] [EVAL_LIMIT]
#   GPU_IDS="1 2 3"  EVAL_LIMIT=0(=full) bash scripts/moe_compress/run_trainfree_atlas.sh
set -u

REPO=/home/yequan/Project/compression/OPD
VERL_PY=/home/yequan/miniconda3/envs/verl/bin/python
CKPT=/data/yequan/moe_compress/ckpts
METRICS=/data/yequan/moe_compress/metrics
LOGS=$REPO/logs/moe_compress
mkdir -p "$CKPT" "$METRICS" "$LOGS"

GPU_IDS=(${1:-1 2 3})
EVAL_LIMIT=${2:-0}                       # 0 => full eval; >0 => cap examples/task
EVAL_FLAG="--eval --eval-limit ${EVAL_LIMIT}"
[ "$EVAL_LIMIT" = "0" ] && EVAL_FLAG="--eval --eval-limit 0"   # 0 passed through; compress_olmoe treats <=0 as full

METHODS=(random_drop reap_drop slimqwen_merge hcsmoe_merge nystrom nystrom_combined svd_llm_v2 sparsegpt mobe magnitude)
RETAINS=(0.75 0.50)
SEED=0                                    # step-0 training-free uses seed 0 (stochastic only for random_drop/data-order; full 3-seed matrix is the recovery phase)

# Build the job list
JOBS=()
for m in "${METHODS[@]}"; do
  for r in "${RETAINS[@]}"; do
    JOBS+=("$m:$r")
  done
done

echo "Atlas (training-free): ${#JOBS[@]} jobs over GPUs ${GPU_IDS[*]}, eval_limit=$EVAL_LIMIT"

# Round-robin assign jobs to GPUs; run NGPU at a time, wait, next wave.
NGPU=${#GPU_IDS[@]}
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for ((g=0; g<NGPU && i<${#JOBS[@]}; g++, i++)); do
    job="${JOBS[$i]}"; m="${job%%:*}"; r="${job##*:}"
    gpu="${GPU_IDS[$g]}"
    tag="${m}_r${r}_s${SEED}"
    log="$LOGS/${tag}.log"
    echo "  [GPU $gpu] $tag -> $log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface \
      HF_DATASETS_TRUST_REMOTE_CODE=1 \
      nohup "$VERL_PY" -m moe_compress.compress_olmoe \
        --method "$m" --retain "$r" --seed "$SEED" --calib-seqs 256 \
        --save-dir "$CKPT/$tag" --metrics-json "$METRICS/$tag.json" \
        $EVAL_FLAG > "$log" 2>&1 &
    pids+=($!)
  done
  echo "  ...waiting on wave (${pids[*]})"
  wait "${pids[@]}"
done

echo "Atlas training-free leg complete. Metrics in $METRICS/, logs in $LOGS/"
