#!/bin/bash
# sigma search for `fura_zoact` at N=10 on the section-7 MATH protocol (section 17).
#
#   DEVICE=6 bash scripts/es/chain_furazoact_pop10.sh
#
# `fura_zoact` composes the two calibrated frames: fura's frozen block-SVD output frame
# A_j and zoact's calibrated activation direction on the input side --
#     dW[:, blk_j] = A_j C_j V_j
# with only C_j (b x r) trained, i.e. in_features * r coefficients per layer
# (831,488 at r=1, 0.011% of 7.6 B -- the smallest arm on the page).
#
# It has no N=30 twin, so sigma has to be searched.  Its measured footprint per unit sigma
# (scripts/es/measure_es_footprint.py, real weights) is ~30x smaller than zoact's, so the
# paper sigma would put it far below the bf16 rollout floor.  alpha = sigma/2 * sqrt(10/30)
# throughout, the section-16.4 motion correction.
#
# Each triple is  sigma:alpha:rank.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
ITERS=${ITERS:-80}
for pt in ${POINTS:?set POINTS="sigma:alpha:rank ..."}; do
  IFS=: read -r sig alp rank <<<"$pt"
  echo "=== [$(date)] fura_zoact r=$rank N=10 sigma=$sig alpha=$alp iters=$ITERS on GPU ${DEVICE:-6}"
  DEVICE=${DEVICE:-6} PERTURB_MODE=fura_zoact SUBSPACE_RANK="$rank" POPULATION_SIZE=10 \
    SIGMA="$sig" ALPHA="$alp" NUM_ITERATIONS="$ITERS" FORGET_TASKS= \
    EXPERIMENT_NAME="furazoact-r${rank}_math-lv3to5-b64_sig${sig}_a${alp}_N10_it${ITERS}" \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] fura_zoact r=$rank sigma=$sig alpha=$alp exited with $?"
done
echo "FURAZOACT POP10 CHAIN DONE"
