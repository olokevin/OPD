#!/bin/bash
# pick_better_lr.sh -- compare the two es_token LR arms, keep the better, drop the other.
#
# Verdict order (first that separates them wins):
#   1. offline MATH-500 acc@8 on the FINAL checkpoint of each arm (identical protocol)
#   2. the in-run MATH-500 greedy curve (eval/accuracy), mean over the last 3 evals
# Nothing is deleted unless DELETE=1 is passed explicitly.
#
#   EVAL_GPU=3 bash scripts/zo_opd/nersc_align/pick_better_lr.sh          # report only
#   EVAL_GPU=3 DELETE=1 bash scripts/zo_opd/nersc_align/pick_better_lr.sh # report + prune
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$R"

# Reap only OUR vLLM processes ON THE TARGET GPU. A host-wide
# `pkill -f VLLM::EngineCore` also reaches engines belonging to other runs of
# ours on other cards, which is a real hazard while a multi-hour job is training.
reap_gpu () {  # $1 = gpu index
  local g="$1" p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
    if [ "$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -u)" ] \
       && ps -o args= -p "$p" 2>/dev/null | grep -qE "VLLM::EngineCore|vllm"; then
      kill -9 "$p" 2>/dev/null
    fi
  done
  for _ in $(seq 1 36); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 5
  done
  return 1
}
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
PY=/home/yequan/miniconda3/envs/verl/bin/python
GPU=${EVAL_GPU:-3}
OUT=$R/logs/lrsweep/pick; mkdir -p "$OUT"
CKROOT=/data/yequan/compress_train/OPD/checkpoint

echo "=== in-run MATH-500 greedy curves ==="
for lr in 1e-4 1e-3; do
  f=$(ls -t "$R"/logs/lrsweep/opd_es_token_lr${lr}_*.log 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "  lr=$lr: no log"; continue; }
  echo "  lr=$lr: $(grep -oE 'eval/accuracy:[0-9.]+' "$f" | sed 's/eval.accuracy://' | tr '\n' ' ')"
done

declare -A FINAL
for lr in 1e-4 1e-3; do
  d=$(ls -d ${CKROOT}/singlegpu_es_opd_r3072_lr${lr}/es_token_*/step_* 2>/dev/null \
      | sed 's/.*step_//' | sort -n | tail -1)
  base=$(ls -d ${CKROOT}/singlegpu_es_opd_r3072_lr${lr}/es_token_*/ 2>/dev/null | tail -1)
  [ -z "$d" ] && { echo "  lr=$lr: no checkpoint"; continue; }
  FINAL[$lr]="${base}step_${d}"
  echo "  lr=$lr final checkpoint: ${FINAL[$lr]}"
done

for lr in 1e-4 1e-3; do
  [ -z "${FINAL[$lr]:-}" ] && continue
  echo "=== offline eval lr=$lr  $(date +%H:%M:%S) ==="
  $PY scripts/zo_opd/paper_align/eval_math.py \
      --model "${FINAL[$lr]}" --gpu "$GPU" --benches MATH-500,AMC23 \
      --n 8 --temperature 1.0 --top-p 0.95 --max-tokens 3072 \
      --enable-thinking false --tag "lr$lr" --gpu-mem "${GPU_MEM:-0.80}" \
      --out "$OUT/lr$lr.json" > "$OUT/lr$lr.log" 2>&1
  grep -E "^\[lr$lr\]" "$OUT/lr$lr.log" || echo "  (eval failed, see $OUT/lr$lr.log)"
  # vLLM engines are subprocesses: reap ours on THIS card only.
  reap_gpu "$GPU" || echo "  WARNING: gpu $GPU did not free"
done

A=$($PY -c "import json;print(json.load(open('$OUT/lr1e-4.json'))['MATH-500']['acc'])" 2>/dev/null || echo nan)
B=$($PY -c "import json;print(json.load(open('$OUT/lr1e-3.json'))['MATH-500']['acc'])" 2>/dev/null || echo nan)
echo "=== MATH-500 acc@8:  lr1e-4=$A   lr1e-3=$B ==="
WORSE=$($PY -c "
a,b='$A','$B'
try:
    a,b=float(a),float(b)
    print('1e-3' if a>=b else '1e-4')
except Exception:
    print('')")
[ -z "$WORSE" ] && { echo "could not rank -- nothing deleted"; exit 1; }
echo "worse arm: lr=$WORSE"
if [ "${DELETE:-0}" = "1" ]; then
  echo "deleting ${CKROOT}/singlegpu_es_opd_r3072_lr${WORSE}"
  rm -rf "${CKROOT}/singlegpu_es_opd_r3072_lr${WORSE}"
else
  echo "(dry run -- pass DELETE=1 to remove ${CKROOT}/singlegpu_es_opd_r3072_lr${WORSE})"
fi
