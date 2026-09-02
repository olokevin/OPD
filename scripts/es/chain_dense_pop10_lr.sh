#!/bin/bash
# alpha search for `dense` at N=10 on the section-7 MATH protocol (section 17).
#
#   DEVICE=6 bash scripts/es/chain_dense_pop10_lr.sh
#
# section 16.4 showed the rule is to hold the per-iteration motion alpha/sqrt(N) fixed, not
# alpha.  dense's N=30 leaderboard point is sigma=1e-3 / alpha=5e-4, motion 9.129e-5.  At
# N=10 the same alpha gives 1.581e-4 (1.73x) and cost -0.49 pp (ns) -- so unlike `fura`,
# dense tolerated the overshoot.  This brackets both sides of the motion-matched point:
#
#   alpha 2.8868e-4  motion 9.129e-5  1.00x  (matched)
#   alpha 5.0000e-4  motion 1.581e-4  1.73x  (already run, section 16.2)
#   alpha 1.0000e-3  motion 3.162e-4  3.46x
#   alpha 1.5000e-4  motion 4.743e-5  0.52x
#
# sigma stays 1e-3 throughout -- alpha only.  80 iterations is enough to rank (dense
# plateaus by step ~20); the winner is re-run at 150 for the headline curve.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
ITERS=${ITERS:-80}
for alpha in ${ALPHAS:-0.00028868 0.001 0.00015}; do
  echo "=== [$(date)] dense N=10 sigma=0.001 alpha=$alpha (motion $(python3 -c "print('%.4e'%($alpha/10**0.5))")) iters=$ITERS on GPU ${DEVICE:-6}"
  DEVICE=${DEVICE:-6} PERTURB_MODE=dense POPULATION_SIZE=10 \
    SIGMA=0.001 ALPHA="$alpha" NUM_ITERATIONS="$ITERS" FORGET_TASKS= \
    EXPERIMENT_NAME="es-dense-full_math-lv3to5-b64_sig0.001_a${alpha}_N10_it${ITERS}" \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] dense N=10 alpha=$alpha exited with $?"
done
echo "DENSE POP10 LR CHAIN DONE"
