#!/bin/bash
# Phase 5 driver: one GPU, a list of prompt lengths, es paths + stock reference.
#   GPU=0 CTXS="0 2048" bash scripts/zo_opd/es_profile/run_phase5.sh
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$(realpath ../../../verl)"
PY=/home/yequan/miniconda3/envs/verl/bin/python
BS=${BS:-1,4,8,16,64}
NS=${NS:-0,1,4,8,16,32}
PATHS=${PATHS:-rows/full,shared/stream,fold/stream}
for L in ${CTXS:-0}; do
  echo "=== ctx L=$L es paths ($(date)) ===" 
  $PY phase5_decode_heatmap.py --Bs $BS --Ns $NS --paths $PATHS --prompt-len $L --gmu ${GMU:-0.5} --t-short ${TS:-64} --t-long ${TL:-384} \
     > results/phase5_L${L}.log 2>&1
  echo "=== ctx L=$L stock ($(date)) ==="
  $PY phase5_decode_heatmap.py --Bs $BS --prompt-len $L --stock-only --gmu ${GMU:-0.5} \
     > results/phase5_L${L}_stock.log 2>&1
done
echo "=== done ($(date)) ==="
