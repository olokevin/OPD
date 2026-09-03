#!/bin/bash
# run_gpu4_chain.sh -- the 0902 B=1 measurements, ALL on one GPU, sequentially:
#   1. decode sweep (5 paths x 14 N, prompt 512, min-of-2 slopes 64->512)
#   2. one-prompt ES step time, shipping path (rows/full/kernel/eager), N in NS
#   3. one-prompt ES step time, fused path (seq/stream/fused/graph), N in NS
#   PROF_GPU=4 bash scripts/zo_opd/es_profile/run_gpu4_chain.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PROF_GPU=${PROF_GPU:-4}
PY=/home/yequan/miniconda3/envs/verl/bin/python
R=scripts/zo_opd/es_profile/results
STEP_SCRIPT=scripts/zo_opd/es_profile/run_es_b1_step.sh
echo "[chain] sweep start $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=$PROF_GPU PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py \
  --Bs 1 --Ns 0,1,2,4,8,16,32,48,64,96,128,192,256,384 \
  --paths rows/full,fold/stream,rows/stream/fused/eager,seq/stream/fused/eager,seq/stream/fused/graph \
  --prompt-len 512 --t-short 64 --t-long 512 --repeats 2 --tag b1_fused > $R/b1_fused.log 2>&1
echo "[chain] sweep done $(date +%H:%M:%S)"
ES_GPU=$PROF_GPU NS="${NS:-1 8 32 128}" ES_PATH=rows/full/kernel/eager bash $STEP_SCRIPT > logs/es_b1/driver_old.log 2>&1
echo "[chain] es old done $(date +%H:%M:%S)"
ES_GPU=$PROF_GPU NS="${NS:-1 8 32 128}" ES_PATH=seq/stream/fused/graph bash $STEP_SCRIPT > logs/es_b1/driver_new.log 2>&1
echo "[chain] es new done $(date +%H:%M:%S)"
