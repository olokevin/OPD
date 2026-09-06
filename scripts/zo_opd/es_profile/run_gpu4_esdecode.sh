#!/bin/bash
# GPU-4 chain (profiling only): es-decode gate, then B=1 decode-throughput sweeps on the
# ds15b student for es-token-decode (fresh rank-1) vs es-decode rank-1 / rank-4 / full.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY=/home/yequan/miniconda3/envs/verl/bin/python
R=scripts/zo_opd/es_profile/results
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export HF_HOME=/data/yequan/huggingface
echo "[chain] gate $(date +%T)"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_token_checks/check_es_decode.py > $R/es_decode_gate.log 2>&1
echo "[chain] gate exit=$? $(date +%T)"
common="--model $M --Bs 1 --prompt-len 512 --t-short 64 --t-long 512 --repeats 2 --gmu 0.5 --max-model-len 4096"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py --stock-only $common --tag ds15b_b1_stock > $R/ds15b_b1_stock.log 2>&1
echo "[chain] stock done $(date +%T)"
for spec in "token 1 esdecode_token" "seq 1 esdecode_r1" "seq 4 esdecode_r4" "seq full esdecode_full"; do
  set -- $spec
  Ns="0,1,4,8,16,32,64,128"; [ "$2" = "full" ] && Ns="0,1,4,8,16,32,64"
  echo "[chain] sweep rail_mode=$1 rank=$2 start $(date +%T)"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py $common \
    --Ns $Ns --paths seq/stream/fused/graph --rail-mode $1 --noise-rank $2 --tag ds15b_b1_$3 > $R/ds15b_b1_$3.log 2>&1
  echo "[chain] sweep $3 exit=$? $(date +%T)"
done
echo "[chain] all done $(date +%T)"
