#!/bin/bash
# ES on Countdown with the prior-task (forgetting) probe -- the exact task pair
# arXiv:2601.20861 uses to show ES catastrophically forgets (Countdown -> HellaSwag).
#
#   DEVICE=1 PERTURB_MODE=dense bash scripts/es/run_countdown_es.sh
#
# Defaults follow the paper: 200 training problems, population 30, sigma 1e-3,
# alpha = sigma/2, and enough iterations to pass the ~200-iteration point where the
# paper reports the new task has converged but prior ability keeps falling.
set -eu
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"

PERTURB_MODE=${PERTURB_MODE:-dense}
case "$PERTURB_MODE" in
  fura) export SIGMA=${SIGMA:-0.0125} ALPHA=${ALPHA:-0.00625} ;;
esac
export TASK_TYPE=countdown
export TRAIN_FILE=${TRAIN_FILE:-${REPO}/datasets/es_countdown/countdown_train200.parquet}
export EVAL_FILE=${EVAL_FILE:-${REPO}/datasets/es_countdown/countdown_val500.parquet}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-200}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-200}
export MAX_TOKENS=${MAX_TOKENS:-512}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-512}
export NUM_ITERATIONS=${NUM_ITERATIONS:-300}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}
export FORGET_TASKS=${FORGET_TASKS:-hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq}
export FORGET_LIMIT=${FORGET_LIMIT:-1000}
export PROJECT_NAME=${PROJECT_NAME:-ES-forget-countdown}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cd-${PERTURB_MODE}_b200_N30}
export PERTURB_MODE
exec bash scripts/es/run_es_math.sh
