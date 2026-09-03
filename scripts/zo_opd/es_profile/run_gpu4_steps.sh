#!/bin/bash
# the two one-prompt step series of run_gpu4_chain.sh, without the sweep
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PROF_GPU=${PROF_GPU:-4}
echo "[steps] old start $(date +%H:%M:%S)"
ES_GPU=$PROF_GPU NS="${NS:-1 8 32 128}" ES_PATH=rows/full/kernel/eager bash scripts/zo_opd/es_profile/run_es_b1_step.sh > logs/es_b1/driver_old.log 2>&1
echo "[steps] new start $(date +%H:%M:%S)"
ES_GPU=$PROF_GPU NS="${NS:-1 8 32 128}" ES_PATH=seq/stream/fused/graph bash scripts/zo_opd/es_profile/run_es_b1_step.sh > logs/es_b1/driver_new.log 2>&1
echo "[steps] done $(date +%H:%M:%S)"
