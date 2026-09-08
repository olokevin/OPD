#!/bin/bash
# eval_ckpts.sh -- the PAPER's ruler on OPSD checkpoints (arXiv:2601.18734 Table 2/8):
# AIME24 + AIME25 + HMMT25, Avg@12, T=1.0 / top-p 0.95 / top-k -1, 38912 new tokens,
# Qwen3 thinking mode ON.  The in-loop MATH-500 monitor is NOT this.
#
# verl's LoRA checkpoints need merging first (<ckpt>/actor/merged_hf holds the ADAPTER).
#
#   RUN_DIR=<...> GPU=5 STEPS="50 100" bash scripts/opsd/eval_ckpts.sh
#   GPU=5 STEPS="" bash scripts/opsd/eval_ckpts.sh          # base model only
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}

GPU=${GPU:-5}
STEPS=${STEPS:-"50 100"}
RUN_DIR=${RUN_DIR:-}
OUT=${OUT:-logs/opsd/eval}
SCRATCH=${SCRATCH:-/data/yequan/opd/opsd_merged}
EXTRA=${EXTRA:-}
mkdir -p "$OUT" "$SCRATCH"

if [ ! -f "$OUT/base.json" ]; then
  echo "=== [$(date)] base Qwen3-1.7B"
  python scripts/opsd/eval_opsd.py --model Qwen/Qwen3-1.7B --gpu "$GPU" --tag base \
    --out "$OUT/base.json" $EXTRA 2>&1 | grep -E "avg@|AVERAGE|Error"
fi

for st in $STEPS; do
  [ -n "$RUN_DIR" ] || { echo "RUN_DIR required for checkpoint evals"; exit 1; }
  ad="$RUN_DIR/global_step_$st/actor/merged_hf"
  [ -d "$ad" ] || { echo "missing adapter: $ad -- skipping"; continue; }
  m="$SCRATCH/step_$st"
  echo "=== [$(date)] step $st: merging"
  python scripts/opsd/merge_lora.py --adapter "$ad" --out "$m" || { echo "merge failed"; continue; }
  echo "=== [$(date)] step $st: eval"
  python scripts/opsd/eval_opsd.py --model "$m" --gpu "$GPU" --tag "step$st" \
    --out "$OUT/step$st.json" $EXTRA 2>&1 | grep -E "avg@|AVERAGE|Error"
  rm -rf "$m"
done
echo "=== [$(date)] EVAL_DONE  (json in $OUT)"
