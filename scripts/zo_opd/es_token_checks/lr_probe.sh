#!/bin/bash
# lr_probe.sh -- short ZO-ES-token LR probe.
#
# LR=1e-3 (the shipped all-layer default) DIVERGED at temperature=1.0: the clean
# KL to the teacher went 0.227 -> ~2.6 at step 1 and stayed there, with
# dW_norm ~3x its step-0 value -- the self-amplifying pattern documented for NP.
# Sampling at T=1.0 puts the rails on a higher-entropy trajectory than the greedy
# benchmark the default was calibrated on, so the update is much larger.
#
# METRIC: eval/heldout_clean_loss -- the FIXED 16-prompt probe, logged inside the
# eval_interval branch (ray_trainer.py:333). Do NOT use train/L_clean_mean: it is
# computed on whatever 64 prompts that step drew and swings 0.23-3.4 batch to
# batch, which is far larger than any LR effect (LRs 1e-3, 1e-5 and 3e-5 produce
# indistinguishable L_clean curves). Only a fixed evaluation set can rank LRs.
#
# Reference: LR=1e-3 degrades the model monotonically --
#   probe 0.2244 -> 0.5565 (step 25) -> 1.1559 (step 50), MATH-500 5% -> 1.5% -> 0%.
#   LR_GPU=1 bash scripts/zo_opd/es_token_checks/lr_probe.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$(pwd)/verl
export HF_HOME=/data/yequan/huggingface
GPU=${LR_GPU:-1}
STEPS=${STEPS:-21}
EVAL_EVERY=${EVAL_EVERY:-10}
mkdir -p logs/lr_probe
for LR in ${LRS:-1e-4 1e-5 1e-6}; do
    echo "=== LR=$LR  $(date +%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=$GPU \
      PROJECT_NAME=zo-opd-lrprobe EXP=lrprobe_$LR \
      PACK_WIDTH=64 B_PACK_BUCKETS='[64]' \
      N_SAMPLE=8 SIGMA=0.01 SIGMA_MODE=absolute SAMPLE_METHOD=bernoulli \
      REWARD_WEIGHT_MODE=student_iw TOKEN_AGG=mean LR=$LR \
      BATCH_SIZE=64 MAX_RESP_LENGTH=1024 TEMPERATURE=1.0 \
      TRAIN_DATASET=datasets/train_data/math-lv3to5/train.parquet \
      EVAL_DATASET=datasets/test_data/MATH-500/test.parquet \
      NUM_ITERATIONS=$STEPS EVAL_INTERVAL=$EVAL_EVERY VAL_MAX_SAMPLES=50 \
      HELDOUT_PROBE_SIZE=16 \
      GPU_MEMORY_UTILIZATION=0.55 TEACHER_GPU_MEMORY_UTILIZATION=0.30 \
      ES_LOGGER='["console"]' LOG_DIR=logs/lr_probe \
      bash scripts/zo_opd/opd_es_token.sh > logs/lr_probe/lr_$LR.log 2>&1
    echo "  PROBE (fixed 16 prompts, lower=better): $(grep -oE '\[Probe @ step [0-9]+\] heldout_clean_loss=[0-9.]+' logs/lr_probe/lr_$LR.log | sed -E 's/.*step ([0-9]+).*=([0-9.]+)/s\1:\2/' | tr '\n' ' ')"
    echo "  MATH-500 (50 samples):                   $(grep -oE 'eval/avg_reward:[0-9.]+' logs/lr_probe/lr_$LR.log | sed 's/.*://' | tr '\n' ' ')"
    P=$(pgrep -f 'trainer\.main_es_token' | head -1)
    [ -n "$P" ] && { kill -TERM "$P" 2>/dev/null; sleep 20; kill -9 "$P" 2>/dev/null; }
    sleep 10
done
echo "=== LR PROBE DONE $(date +%H:%M:%S) ==="
