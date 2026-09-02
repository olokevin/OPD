#!/bin/bash
# `lora_zoact` at N=10 -- the matched control for `lora` (section 17).
#
#   DEVICE=7 POINTS="0.015385:0.0044412:44" bash scripts/es/chain_lorazoact_pop10.sh
#
# `lora_zoact` IS `lora` -- same parameterisation, same coefficient count, both factors
# ES-trained -- except A is initialised to the top-r calibrated activation directions
# instead of a random Gaussian.  Calibrated rows are unit-norm and random rows have
# E||row||^2 = 1, so the footprint per unit sigma is the same too.  Run each rank at the
# sigma/alpha that WON that rank's `lora` screen and nothing but the initialisation
# differs.
#
# Rank > 1 needs a rank-r calibration; CALIB_PATH selects it (set RECALIB_RANK to build
# one first, on this same GPU, so the chain stays sequential).
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
ITERS=${ITERS:-80}
RECALIB_RANK=${RECALIB_RANK:-0}
if [ "$RECALIB_RANK" != "0" ]; then
  OUT="${REPO}/datasets/es_math/calib_qwen2p5_math_7b_r${RECALIB_RANK}.pt"
  if [ ! -f "$OUT" ]; then
    echo "=== [$(date)] calibrating rank $RECALIB_RANK on GPU ${DEVICE:-7} -> $OUT"
    CUDA_VISIBLE_DEVICES=${DEVICE:-7} python3 scripts/es/calibrate_activations.py \
      --model Qwen/Qwen2.5-Math-7B \
      --train-file "${REPO}/datasets/es_math/math_lv3to5_qwenmath_train.parquet" \
      --out "$OUT" \
      --rollout-cache "${REPO}/datasets/es_math/calib_rollouts.jsonl" \
      --rank "$RECALIB_RANK" --sketch $((RECALIB_RANK + 20)) --passes 3
    echo "=== [$(date)] calibration exited with $?"
  fi
fi
for pt in ${POINTS:?set POINTS="sigma:alpha:rank ..."}; do
  IFS=: read -r sig alp rank <<<"$pt"
  CP="${REPO}/datasets/es_math/calib_qwen2p5_math_7b.pt"
  [ "$rank" -gt 1 ] && CP="${REPO}/datasets/es_math/calib_qwen2p5_math_7b_r${RECALIB_RANK}.pt"
  echo "=== [$(date)] lora_zoact r=$rank N=10 sigma=$sig alpha=$alp iters=$ITERS on GPU ${DEVICE:-7}"
  DEVICE=${DEVICE:-7} PERTURB_MODE=lora_zoact LORA_RANK="$rank" POPULATION_SIZE=10 \
    SIGMA="$sig" ALPHA="$alp" NUM_ITERATIONS="$ITERS" FORGET_TASKS= CALIB_PATH="$CP" \
    EXPERIMENT_NAME="lorazoact-r${rank}_math-lv3to5-b64_sig${sig}_a${alp}_N10_it${ITERS}" \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] lora_zoact r=$rank sigma=$sig alpha=$alp exited with $?"
done
echo "LORAZOACT POP10 CHAIN DONE"
