#!/bin/bash
# Pick sigma by measuring `train/reward_std` over a handful of iterations (section 17.1).
#
#   DEVICE=6 MODE=fura_zoact SIGMAS="0.01 0.05 0.2 0.8" bash scripts/es/probe_reward_std.sh
#
# Every ES config that WORKS on this page sits in reward_std 0.040-0.055, and every one that
# fails sits outside it -- below ~0.035 the arm learns but crawls, above ~0.09 it degrades:
#
#   dense sigma 1e-3        0.0528   plateau 71.82   <- reference
#   fura  sigma 1.25e-2     0.0524   plateau 72.68   <- the winner
#   insparse sigma 1e-3     0.0419   plateau 72.07
#   zoact sigma 1e-3        0.0401   plateau 70.50
#   fura  sigma 1e-3        0.0343   60.2 @ 20, slow
#   lora r44 sigma 1e-3     0.0236   never plateaus (section 15.5)
#   zoact sigma 1.2e-2      0.1124   plateau 68.85, WORSE than its own paper sigma
#
# Weight-space footprint does NOT predict this (zoact works at 61x smaller footprint than
# dense), so sigma has to be measured per subspace -- but 3 iterations is enough, against
# ~2.7 h for a screen.  alpha is irrelevant to the probe; it is set to sigma/2*sqrt(10/30).
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
MODE=${MODE:?set MODE}
RANK=${RANK:-1}
ITERS=${PROBE_ITERS:-3}
for sig in ${SIGMAS:?set SIGMAS}; do
  alp=$(python3 -c "print('%.6g'%($sig/2*(1/3)**0.5))")
  echo "=== [$(date)] probe $MODE r=$RANK sigma=$sig alpha=$alp ($ITERS iters) on GPU ${DEVICE:-6}"
  DEVICE=${DEVICE:-6} PERTURB_MODE="$MODE" SUBSPACE_RANK="$RANK" LORA_RANK="$RANK" \
    POPULATION_SIZE=10 SIGMA="$sig" ALPHA="$alp" NUM_ITERATIONS="$ITERS" \
    EVAL_INTERVAL=100000 FORGET_TASKS= LOGGER='[console]' \
    EXPERIMENT_NAME="probe-${MODE}-r${RANK}_sig${sig}" \
    bash scripts/es/run_es_math.sh 2>&1 | grep -E 'reward_std|Eval @ step'
  echo "=== [$(date)] probe sigma=$sig done"
done
echo "PROBE CHAIN DONE"
