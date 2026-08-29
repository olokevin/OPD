#!/bin/bash
# Population-size ablation: the section-7 MATH protocol at N=10 instead of N=30.
#
#   DEVICE=5 bash scripts/es/chain_pop10.sh dense fura iso
#
# ONLY the population size changes.  Each arm keeps the sigma/alpha that tops the
# section-7 leaderboard, so the N=10 curve is directly comparable to its N=30 twin:
#   dense  sigma 1e-3    alpha 5e-4     (paper ES)
#   fura   sigma 1.25e-2 alpha 6.25e-3  (the sigma-matched winner, section 11.3)
#   iso    sigma 5e-2    alpha 2.5e-2   (footprint-matched, section 10.4)
#
# Note the ES update moves the coefficients by ~alpha/sqrt(N) per iteration, so holding
# alpha fixed makes an N=10 step sqrt(3) ~ 1.73x larger than an N=30 step.  That is the
# deliberate reading of "same hyperparameters, fewer probes"; a motion-matched variant
# would additionally scale alpha by sqrt(N/30).
#
# The prior-task probe stays OFF: it hard-kills a 7B ES worker (section 15.4) and its
# absence is also what keeps these runs comparable to section 7.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
POP=${POP:-10}
for mode in "$@"; do
  case "$mode" in
    dense) SIG=0.001  ALP=0.0005  ;;
    fura)  SIG=0.0125 ALP=0.00625 ;;
    iso)   SIG=0.05   ALP=0.025   ;;
    *)     echo "[skip] unknown mode $mode"; continue ;;
  esac
  echo "=== [$(date)] $mode  N=$POP  sigma=$SIG alpha=$ALP  on GPU ${DEVICE:-5}"
  DEVICE=${DEVICE:-5} PERTURB_MODE="$mode" POPULATION_SIZE="$POP" \
    SIGMA="$SIG" ALPHA="$ALP" FORGET_TASKS= \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] $mode N=$POP exited with $?"
done
echo "POP$POP CHAIN DONE: $*"
