#!/bin/bash
# final_eval.sh -- put base / BP / ES on ONE ruler.
#
# The in-run metrics are not comparable: BP logs val-core/{AMC23,AIME24,AIME25}
# mean@8 at T=1.0, while the es trainer logs eval/accuracy = MATH-500 GREEDY n=1.
# This re-scores all three checkpoints with an identical protocol.
#
# enable_thinking=false is mandatory -- these models were trained non-thinking.
set -u
R=/home/yequan/Project/compression/OPD-estoken
cd "$R"
export HF_HOME=/data/yequan/huggingface
PY=/home/yequan/miniconda3/envs/verl/bin/python
GPU=${EVAL_GPU:-5}
OUT=$R/logs/final_eval
mkdir -p "$OUT"

BP=/data/yequan/compress_train/OPD/merged/bp_opd_step279
ES=/data/yequan/compress_train/OPD/checkpoint/singlegpu_es_opd_r3072_lr1e-5_resume30/es_token_20260825_232122/step_169

run () {  # name  model  [tokenizer]
  local name="$1" model="$2" tokarg=""
  [ -n "${3:-}" ] && tokarg="--tokenizer $3"
  echo "=== $name  $(date +%H:%M:%S) ==="
  $PY scripts/zo_opd/paper_align/eval_math.py \
      --model "$model" $tokarg --gpu "$GPU" \
      --benches AMC23,AIME24,AIME25,MATH-500 \
      --n 8 --temperature 1.0 --top-p 0.95 --max-tokens 3072 \
      --enable-thinking false --tag "$name" --gpu-mem "${GPU_MEM:-0.80}" \
      --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  echo "  exit=$? -> $OUT/$name.json"
  grep -E "^\[$name\]" "$OUT/$name.log" || true
  # vLLM's EngineCore is a SUBPROCESS: killing the python parent leaves it
  # holding the whole card, which is what made the first attempt fail with
  # "Free memory on device (11.61/93.1 GiB) ... less than desired". Reap it and
  # wait for the memory to actually come back before the next model loads.
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  echo "  gpu $GPU now at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU")"
}

run base "Qwen/Qwen3-1.7B"
run bp_step279 "$BP" "Qwen/Qwen3-1.7B"
run es_step169 "$ES" "Qwen/Qwen3-1.7B"
echo "=== FINAL EVAL DONE $(date +%H:%M:%S) ==="
