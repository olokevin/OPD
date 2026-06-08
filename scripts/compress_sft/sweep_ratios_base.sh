#!/usr/bin/env bash
# Reproduce wiki §4 ratio-sweep (forward-only SVD-V2 attn + Nystrom MLP, WHOLE last
# layer dense) but for Qwen/Qwen3-4B-Base (non-thinking), ratios 0.8/0.7/0.6/0.5/0.4,
# MATH-500(100) + C4 PPL. Spreads the 5 ratios across GPUs 1,2,3. Run from repo root
# in the verl env. Results -> /data/yequan/compress_sft/metrics/ratio_sweep_base/.
set -uo pipefail

REPO=/home/yequan/Project/compression/OPD
VERL_PY=/home/yequan/miniconda3/envs/verl/bin/python
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
OUT=/data/yequan/compress_sft/metrics/ratio_sweep_base
LOGS=/data/yequan/compress_sft/logs/ratio_sweep_base
mkdir -p "$OUT" "$LOGS"

MODEL=Qwen/Qwen3-4B-Base

cell() {  # ratio gpu
  local ratio=$1 gpu=$2
  echo ">>> ratio=$ratio on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:verl HF_HOME=$HF_HOME \
    "$VERL_PY" "$REPO/scripts/compress_sft/build_svd_nystrom_student.py" \
      --model "$MODEL" --objective forward --ratio "$ratio" --skip-last-layers 1 \
      --calib-num-seqs 128 --skip-save --skip-mmlu \
      --metrics-json "$OUT/r${ratio}.json" \
      --math-limit 100 --math-max-new-tokens 4096 --batch-size 16 \
      > "$LOGS/r${ratio}.log" 2>&1
}

cd "$REPO"
# GPU 1: 0.8, 0.5   GPU 2: 0.7, 0.4   GPU 3: 0.6
( cell 0.8 1; cell 0.5 1 ) &
( cell 0.7 2; cell 0.4 2 ) &
( cell 0.6 3 ) &
wait
echo ""
echo "=== RATIO SWEEP TABLE (Qwen3-4B-Base, forward-only, last layer dense) ==="
printf "%-7s %-10s %-12s\n" "ratio" "C4_PPL" "MATH/100"
for r in 0.8 0.7 0.6 0.5 0.4; do
  f="$OUT/r${r}.json"
  if [[ -f $f ]]; then
    "$VERL_PY" -c "import json;d=json.load(open('$f'));print('%-7s %-10.1f %-12s'%('$r', d['c4_ppl'], str(round(d['math500_acc']*100,1))+'%'))"
  else
    printf "%-7s %-10s %-12s\n" "$r" "MISSING" "MISSING"
  fi
done
