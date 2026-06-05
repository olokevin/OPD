#!/usr/bin/env bash
# Post-hoc MATH-500 growth curve: eval every intermediate checkpoint-<step>-merged dir
# written during an svd_nystrom SFT run. Tier-2 of the mid-training growth signal
# (tier-1 = wandb eval_loss). Runs in the `verl` env (ttrl_math grader).
#
# Usage:
#   bash sweep_sft_ckpts.sh /data/yequan/compress_sft/sft/qwen3_4b_base/forward_r0.7 <gpu> [math_limit]
set -euo pipefail

REPO=/home/yequan/Project/compression/OPD
VERL_PY=/home/yequan/miniconda3/envs/verl/bin/python
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}

SFT_OUT=${1:?usage: sweep_sft_ckpts.sh <sft_output_dir> <gpu> [math_limit]}
GPU=${2:?need gpu id}
LIMIT=${3:-100}

OUT="$SFT_OUT/growth_math500.jsonl"
: > "$OUT"
echo ">>> growth sweep over $SFT_OUT (MATH-500 limit=$LIMIT @4096) -> $OUT"

# numeric-sorted by step; include final-merged last
mapfile -t CKPTS < <(ls -d "$SFT_OUT"/checkpoint-*-merged 2>/dev/null \
  | sed -E 's/.*checkpoint-([0-9]+)-merged/\1 &/' | sort -n | awk '{print $2}')
[[ -d "$SFT_OUT/final-merged" ]] && CKPTS+=("$SFT_OUT/final-merged")

if [[ ${#CKPTS[@]} -eq 0 ]]; then echo "no *-merged ckpts under $SFT_OUT"; exit 1; fi

cd "$REPO"
for ck in "${CKPTS[@]}"; do
  step=$(basename "$ck" | grep -oE '[0-9]+' || echo final)
  tmp=$(mktemp)
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src:verl HF_HOME=$HF_HOME \
    "$VERL_PY" scripts/opd/math/compressed_opd/eval_opd_ckpt.py \
      --model-dir "$ck" --label "step_${step}" --metrics-json "$tmp" \
      --skip-ppl --math-limit "$LIMIT" --math-max-new-tokens 4096 --math-batch-size 8
  # append {step, math500_acc} to the jsonl growth file
  "$VERL_PY" -c "import json,sys; m=json.load(open('$tmp')); print(json.dumps({'step':'$step','math500_acc':m.get('math500_acc')}))" >> "$OUT"
  rm -f "$tmp"
done
echo ">>> wrote growth curve: $OUT"
cat "$OUT"
