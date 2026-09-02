#!/bin/bash
# Phase 6 (2 GPUs): DP2 = the same single-GPU sweep on GPU0 and GPU1 at the
# SAME time (no decode-time communication; should reproduce the solo curve);
# TP2 = one engine with tensor_parallel_size=2 across the NVLink pair.
#   bash scripts/zo_opd/es_profile/run_phase6.sh
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/yequan/Project/compression/OPD-estoken/verl
PY=/home/yequan/miniconda3/envs/verl/bin/python
BS=${BS:-4,8,16}
NS=${NS:-0,4,8,16}
PATHS=${PATHS:-rows/full,shared/stream}
L=${L:-512}
echo "=== DP2: concurrent solo sweeps on GPU 0 and 1 ($(date)) ==="
CUDA_VISIBLE_DEVICES=0 $PY phase5_decode_heatmap.py --Bs $BS --Ns $NS --paths $PATHS --prompt-len $L --tag dp2_gpu0 > results/phase6_dp2_gpu0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY phase5_decode_heatmap.py --Bs $BS --Ns $NS --paths $PATHS --prompt-len $L --tag dp2_gpu1 > results/phase6_dp2_gpu1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "=== solo reference on GPU 0 alone ($(date)) ==="
CUDA_VISIBLE_DEVICES=0 $PY phase5_decode_heatmap.py --Bs $BS --Ns $NS --paths $PATHS --prompt-len $L --tag solo_gpu0 > results/phase6_solo_gpu0.log 2>&1
echo "=== TP2 on GPUs 0,1 ($(date)) ==="
CUDA_VISIBLE_DEVICES=0,1 VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY phase5_decode_heatmap.py --Bs $BS --Ns $NS --paths rows/full,shared/full --prompt-len $L --tp 2 --tag tp2 > results/phase6_tp2.log 2>&1
echo "=== TP2 stock ($(date)) ==="
CUDA_VISIBLE_DEVICES=0,1 $PY phase5_decode_heatmap.py --Bs $BS --prompt-len $L --tp 2 --stock-only --tag tp2_stock > results/phase6_tp2_stock.log 2>&1
echo "=== done ($(date)) ==="
