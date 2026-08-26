#!/bin/bash
# LoRA-ES on the section-7 MATH task, one rank per job, run back-to-back on one GPU.
#
#   DEVICE=7 bash scripts/es/chain_lora_math.sh 44 1
#
# W = W_base + lora_scale * B @ A with BOTH factors ES-trained (B zero-init, so step 0 is
# the base model exactly).  Trainable coefficients = rank * sum(out+in) over the 112 fused
# linear weights = 2,222,080 per rank on Qwen2.5-Math-7B, so rank 44 reproduces `fura`'s
# 97,771,520 exactly and rank 1 is the minimal adapter.
#
# sigma = 1e-3 (the paper / dense-ES value); alpha = 5e-3 = 10x dense ES's 5e-4.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
for rank in "$@"; do
  echo "=== [$(date)] lora rank $rank on GPU ${DEVICE:-7}"
  DEVICE=${DEVICE:-7} PERTURB_MODE=lora LORA_RANK="$rank" \
    SIGMA=${SIGMA:-0.001} ALPHA=${ALPHA:-0.005} \
    FORGET_TASKS=${FORGET_TASKS:-hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq} \
    FORGET_LIMIT=${FORGET_LIMIT:-1000} \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] lora rank $rank exited with $?"
done
echo "LORA CHAIN DONE: $*"
