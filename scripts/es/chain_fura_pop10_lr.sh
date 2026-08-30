#!/bin/bash
# Does fura's N=10 collapse come from fewer probes, or from the larger step that holding
# alpha fixed implies?  (section 16.3)
#
#   DEVICE=5 bash scripts/es/chain_fura_pop10_lr.sh
#
# The ES update moves coefficients by ~alpha/sqrt(N) per iteration, so the N=10 run at the
# leaderboard alpha=6.25e-3 took a step sqrt(3) = 1.73x larger than its N=30 twin:
#     N=30: 6.25e-3 / sqrt(30) = 1.1411e-3
#     N=10: 6.25e-3 / sqrt(10) = 1.9764e-3
# alpha = 6.25e-3 * sqrt(10/30) = 3.6084e-3 restores the N=30 motion EXACTLY, leaving the
# probe count as the only difference.  A second point at half that brackets the answer in
# case 10 probes also need a smaller step than 30 to be stable.
#
# sigma stays 1.25e-2 throughout -- this varies alpha only, unlike section 11.3.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
for alpha in ${ALPHAS:-0.0036084 0.0018042}; do
  echo "=== [$(date)] fura N=10 sigma=0.0125 alpha=$alpha (motion alpha/sqrt(10)=$(python3 -c "print('%.4e'%($alpha/10**0.5))")) on GPU ${DEVICE:-5}"
  DEVICE=${DEVICE:-5} PERTURB_MODE=fura POPULATION_SIZE=10 \
    SIGMA=0.0125 ALPHA="$alpha" FORGET_TASKS= \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] fura N=10 alpha=$alpha exited with $?"
done
echo "FURA POP10 LR CHAIN DONE"
