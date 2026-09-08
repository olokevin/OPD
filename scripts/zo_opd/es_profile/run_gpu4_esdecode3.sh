#!/bin/bash
# GPU-4: after the fused-consumer apply-side fix: kernel gate, r1/r4 sweeps, r1 audit.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY=/home/yequan/miniconda3/envs/verl/bin/python
R=scripts/zo_opd/es_profile/results
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export HF_HOME=/data/yequan/huggingface
echo "[chain3] kernel gate $(date +%T)"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_token_checks/check_seq_kernels.py > $R/seq_kernels_gate.log 2>&1
echo "[chain3] kernel gate exit=$? $(grep -c PASS $R/seq_kernels_gate.log) PASS $(grep -c FAIL $R/seq_kernels_gate.log) FAIL $(date +%T)"
common="--model $M --Bs 1 --prompt-len 512 --t-short 64 --t-long 512 --repeats 2 --gmu 0.5 --max-model-len 4096"
for spec in "seq 1 esdecode_r1" "seq 4 esdecode_r4"; do
  set -- $spec
  echo "[chain3] sweep $3 $(date +%T)"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py $common \
    --Ns 0,1,4,8,16,32,64,128 --paths seq/stream/fused/graph --rail-mode $1 --noise-rank $2 --tag ds15b_b1_$3 > $R/ds15b_b1_$3.log 2>&1
  echo "[chain3] sweep $3 exit=$? $(date +%T)"
done
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/verl $PY scripts/zo_opd/es_profile/phase5_decode_heatmap.py $common --profile \
  --Ns 0,8 --paths seq/stream/fused/graph --rail-mode seq --noise-rank 1 --tag ds15b_audit_r1 > $R/ds15b_audit_r1.log 2>&1
echo "[chain3] audit r1 exit=$? $(date +%T)"
echo "[chain3] done $(date +%T)"
