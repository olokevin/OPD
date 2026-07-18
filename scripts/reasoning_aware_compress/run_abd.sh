#!/usr/bin/env bash
# A/B/D teacher-free deployment at the operating point (retain 0.8, last layer dense,
# MATH/100 + C4 PPL). D3/B2 (OPD claim cells) skipped — need a distinct teacher.
#
# Usage:
#   GPU=7 bash run_abd.sh D    # D0 D1 D2
#   GPU=7 bash run_abd.sh A    # A0 A1 A2
#   GPU=7 bash run_abd.sh B    # B0 B1
#   GPU=7 bash run_abd.sh all  # D then A then B, sequential
set -euo pipefail
cd "$(dirname "$0")/../../../.."
REPO=$(pwd)
PY=/home/yequan/miniconda3/envs/verl/bin/python
GPU=${GPU:-7}
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/data/yequan/huggingface
export PYTHONPATH="$REPO/src:$REPO/verl"
SD=scripts/reasoning_aware_compress
RATIO=${RATIO:-0.8}
MATH=${MATH:-100}
ts() { date +%Y%m%d_%H%M%S; }

run_D() {
  $PY $SD/bi_whitened_svd.py --cells D0 D1 D2 --ratio $RATIO --math-limit $MATH \
    --out $SD/results/blockD/bi_whitened_r${RATIO}.json \
    2>&1 | tee $SD/logs/blockD_$(ts).log
}
run_A() {
  $PY $SD/lr_sparse_residual.py --cells A0 A1 A2 --ratio $RATIO --math-limit $MATH \
    --sparse-frac 0.075 \
    --out $SD/results/blockA/lr_sparse_r${RATIO}.json \
    2>&1 | tee $SD/logs/blockA_$(ts).log
}
run_B() {
  $PY $SD/sequential_src.py --cells B0 B1 --ratio $RATIO --math-limit $MATH \
    --out $SD/results/blockB/src_r${RATIO}.json \
    2>&1 | tee $SD/logs/blockB_$(ts).log
}

case "${1:-all}" in
  D) run_D ;;
  A) run_A ;;
  B) run_B ;;
  all) run_D; run_A; run_B ;;
  *) echo "usage: GPU=N bash run_abd.sh [D|A|B|all]"; exit 1 ;;
esac
echo "DONE: $1"
