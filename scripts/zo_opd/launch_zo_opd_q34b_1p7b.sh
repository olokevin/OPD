#!/bin/bash
# launch_zo_opd_q34b_1p7b.sh -- OPD baseline then ZO-ES-token, SEQUENTIALLY on one GPU.
#
# Teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 -> student Qwen/Qwen3-1.7B,
# MATH lv3-5 train / MATH-500 eval, batch 64 x 1024, on-policy sampling (T=1.0).
# wandb project: zo-opd-q34b-1p7b
#
#   TRAIN_GPU=1 bash scripts/zo_opd/launch_zo_opd_q34b_1p7b.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$(pwd)/verl
export HF_HOME=/data/yequan/huggingface
GPU=${TRAIN_GPU:-1}
PROJ=zo-opd-q34b-1p7b
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/train

TRAIN=datasets/train_data/math-lv3to5/train.parquet
EVAL=datasets/test_data/MATH-500/test.parquet

# ---------------------------------------------------------------- 1) BP OPD --
# The established token_reward_direct pipeline, as the baseline the ZO run is
# read against. LR 1e-6 is the repo's OPD actor default.
echo "=== [1/2] BP-OPD  $(date) ==="
CUDA_VISIBLE_DEVICES=$GPU \
  PROJECT_NAME=$PROJ \
  TEMPERATURE=1.0 \
  MAX_RESP_LENGTH=1024 MAX_VAL_RESP_LENGTH=1024 \
  TRAIN_DATASET="$TRAIN" TEST_FILE="[\"$EVAL\"]" \
  LR=${BP_LR:-1e-6} TEST_FREQ=${BP_TEST_FREQ:-25} SAVE_FREQ=100 \
  RAY_ISOLATE=1 RAY_PORT=$((6800 + RANDOM % 150)) RAY_TMPDIR=/tmp/ray_bp_$TS \
  bash scripts/zo_opd/opd_math_ref.sh > logs/train/bp_opd_$TS.log 2>&1
echo "BP-OPD exit=$? ; log logs/train/bp_opd_$TS.log"
grep -oE 'perf/time_per_step:[0-9.]+' logs/train/bp_opd_$TS.log | tail -3
pkill -f "main_ppo" 2>/dev/null; sleep 20

# --------------------------------------------------------- 2) ZO-ES-token ----
# N=8 rails, pack_width=64 so the whole 64-prompt batch is ONE wave (the
# budget-sized scratch-KV reservation is what makes that fit).
# LR 1e-3 = the all-layer scale the config ships (NP lesson: ~30x below the
# single-layer LR, which diverges). No ES LR sweep exists yet -- the heldout
# probe is on so divergence shows up within a few steps.
echo "=== [2/2] ZO-ES-token  $(date) ==="
CUDA_VISIBLE_DEVICES=$GPU \
  PROJECT_NAME=$PROJ EXP=${ES_EXP:-N8_pw64_lr1e-3} \
  PACK_WIDTH=64 B_PACK_BUCKETS='[64]' \
  N_SAMPLE=8 SIGMA=0.01 SIGMA_MODE=absolute SAMPLE_METHOD=bernoulli \
  REWARD_WEIGHT_MODE=student_iw TOKEN_AGG=mean \
  LR=${ES_LR:-1e-3} \
  BATCH_SIZE=64 MAX_RESP_LENGTH=1024 TEMPERATURE=1.0 \
  TRAIN_DATASET="$TRAIN" EVAL_DATASET="$EVAL" \
  NUM_ITERATIONS=${ES_ITERS:-150} EVAL_INTERVAL=25 VAL_MAX_SAMPLES=200 \
  HELDOUT_PROBE_SIZE=16 \
  GPU_MEMORY_UTILIZATION=0.55 TEACHER_GPU_MEMORY_UTILIZATION=0.30 \
  ES_LOGGER='["console","wandb"]' LOG_DIR=logs/train \
  bash scripts/zo_opd/opd_es_token.sh > logs/train/es_token_$TS.log 2>&1
echo "ZO-ES-token exit=$? ; log logs/train/es_token_$TS.log"
grep -oE 'train/step_time:[0-9.]+' logs/train/es_token_$TS.log | tail -3
echo "=== BOTH DONE $(date) ==="
