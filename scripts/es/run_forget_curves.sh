#!/bin/bash
# ES arm with the prior-task (forgetting) probe on, so the run produces the forgetting
# curve of arXiv:2601.20861 next to the MATH-500 curve.
#
# Protocol is the one the six leaderboard arms used (fixed 64-problem batch, 1536 train
# tokens, 3000 eval tokens, 150 iterations, eval every 10) and each mode keeps the
# sigma that tops the section-7 leaderboard, so the new-task curves are directly
# comparable to docs/results/ES/es_results.md section 7.
#
#   DEVICE=1 PERTURB_MODE=dense bash scripts/es/run_forget_curves.sh
set -eu
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"

PERTURB_MODE=${PERTURB_MODE:-dense}
case "$PERTURB_MODE" in
  fura) export SIGMA=${SIGMA:-0.0125} ALPHA=${ALPHA:-0.00625} ;;
esac
export FORGET_TASKS=${FORGET_TASKS:-hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}
export FORGET_LIMIT=${FORGET_LIMIT:-1000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}
export NUM_ITERATIONS=${NUM_ITERATIONS:-150}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-forget-${PERTURB_MODE}_b64_N30}
export PERTURB_MODE
exec bash scripts/es/run_es_math.sh
