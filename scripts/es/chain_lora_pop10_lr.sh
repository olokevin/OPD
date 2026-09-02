#!/bin/bash
# sigma/alpha search for `lora` at N=10 on the section-7 MATH protocol (section 17).
#
#   DEVICE=7 bash scripts/es/chain_lora_pop10_lr.sh
#
# section 15.5 left LoRA-ES badly under-scaled: at the paper sigma=1e-3 its weight-space
# footprint ||dW||/||W|| is 3.25e-3 (r=44) and 3.84e-4 (r=1) against the dense reference
# 5.0e-2, and section 11.3 showed that for `fura` the operative knob was sigma, not alpha
# (13 pp on sigma alone).  So this is a footprint search, not an alpha-only one:
#
#   sigma_match = 5.0e-2 / (footprint per unit sigma)      alpha = sigma/2 * sqrt(10/30)
#   r=44 : footprint 3.25/sigma -> sigma 1.5385e-2, alpha 4.4412e-3
#   r=1  : footprint 0.384/sigma -> sigma 1.3021e-1, alpha 3.7588e-2
#
# The same recipe reproduces `fura`'s winning point exactly (sigma 1.25e-2 / alpha 3.6084e-3,
# section 16.4), so it is a prediction, not a fit.  alpha carries the sqrt(10/30) motion
# correction of section 16.4.
#
# Order is by informativeness -- an early stop still answers the two questions:
#   1. does footprint-matching rescue lora r=44 the way it rescued fura?
#   2. is the 15 pp zoact-vs-lora gap at rank 1 (section 15.6) real, or was it footprint?
#
# Each triple is  sigma:alpha:rank.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
ITERS=${ITERS:-80}
for pt in ${POINTS:-0.015385:0.0044412:44 0.13021:0.037588:1 0.004:0.0011547:44 0.001:0.0028868:44}; do
  IFS=: read -r sig alp rank <<<"$pt"
  echo "=== [$(date)] lora r=$rank N=10 sigma=$sig alpha=$alp iters=$ITERS on GPU ${DEVICE:-7}"
  DEVICE=${DEVICE:-7} PERTURB_MODE=lora LORA_RANK="$rank" POPULATION_SIZE=10 \
    SIGMA="$sig" ALPHA="$alp" NUM_ITERATIONS="$ITERS" FORGET_TASKS= \
    EXPERIMENT_NAME="lora-r${rank}_math-lv3to5-b64_sig${sig}_a${alp}_N10_it${ITERS}" \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] lora r=$rank sigma=$sig alpha=$alp exited with $?"
done
echo "LORA POP10 LR CHAIN DONE"
