#!/bin/bash
# GPU-4: (1) full-rank sweep re-run after the active-rails grid fix; (2) eager kernel
# audit token vs seq-r1 at N=0/8 to localize the seq-mode fixed per-token cost.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY=/home/yequan/miniconda3/envs/verl/bin/python
R=scripts/zo_opd/es_profile/results
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export HF_HOME=/data/yequan/huggingface
common="--model $M --Bs 1 --prompt-len 512 --t-short 64 --t-long 512 --repeats 2 --gmu 0.5 --max-model-len 4096"
echo "[chain2] full sweep $(date +%T)"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py $common \
  --Ns 0,1,4,8,16,32,64 --paths seq/stream/fused/graph --rail-mode seq --noise-rank full --tag ds15b_b1_esdecode_full > $R/ds15b_b1_esdecode_full.log 2>&1
echo "[chain2] full sweep exit=$? $(date +%T)"
for spec in "token 1 tok" "seq 1 r1"; do
  set -- $spec
  echo "[chain2] audit $3 $(date +%T)"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py $common --profile \
    --Ns 0,8 --paths seq/stream/fused/graph --rail-mode $1 --noise-rank $2 --tag ds15b_audit_$3 > $R/ds15b_audit_$3.log 2>&1
  echo "[chain2] audit $3 exit=$? $(date +%T)"
done
echo "[chain2] done $(date +%T)"
