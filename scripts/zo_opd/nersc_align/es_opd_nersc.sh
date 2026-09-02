#!/bin/bash
# es_opd_nersc.sh -- ZO-ES-token OPD on the SAME setting as bp_opd_nersc.sh, so
# the two are directly comparable.
#
#   student  Qwen/Qwen3-1.7B (non-thinking)   teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
#   DAPO-Math-17k, T=1.0 (student_iw is unbiased only when the clean token is
#   SAMPLED), max_tokens 3072, batch 64, N=8 rails.
#
# ENABLE_THINKING=false is load-bearing and NEW on the ES side: until this
# session the es task_utils prompt processor called apply_chat_template without
# the kwarg, so a hybrid Qwen3 student silently ran in thinking mode.
#
# Differences from the BP run that are inherent to the trainer:
#   * one rollout per prompt (BP: n=4) -> 64 sequences/step vs 256
#   * in-run eval is GREEDY n=1 on MATH-500 (BP: n=8 at T=1.0 on AIME/AMC).
#     MATH-500 at 500 prompts is the only metric with enough mass to move at
#     this step budget; compare the two via the offline eval, not in-run.
#
#   TRAIN_GPU=6 ES_LR=1e-5 bash scripts/zo_opd/nersc_align/es_opd_nersc.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$(pwd)/verl
export HF_HOME=/data/yequan/huggingface
export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-6}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/MATH-500/test.parquet}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-500}
export ENABLE_THINKING=${ENABLE_THINKING:-false}

export TEMPERATURE=${TEMPERATURE:-1.0}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}
export BATCH_SIZE=${BATCH_SIZE:-64}
export PACK_WIDTH=${PACK_WIDTH:-64}
export B_PACK_BUCKETS=${B_PACK_BUCKETS:-'[64]'}

export N_SAMPLE=${N_SAMPLE:-8}
export SIGMA=${SIGMA:-0.01}
export SIGMA_MODE=absolute
export SAMPLE_METHOD=bernoulli
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_iw}
export TOKEN_AGG=${TOKEN_AGG:-mean}
export LR=${ES_LR:-1e-5}
export FP32_MASTER=${FP32_MASTER:-true}
export ASSEMBLE_CHUNK=${ASSEMBLE_CHUNK:-256}

export NUM_ITERATIONS=${ES_ITERS:-200}
export EVAL_INTERVAL=${EVAL_INTERVAL:-20}
export HELDOUT_PROBE_SIZE=${HELDOUT_PROBE_SIZE:-64}
export SAVE_FREQ=${SAVE_FREQ:-10}   # HF checkpoint every 10 steps, newest 2 kept

# 0.42 x 93.1 - 3.4(weights) = 35.7 GiB KV = 19,470 blocks, vs the 16,384
# that pack_width=64 reserves (256 blocks/slot at prompt+3072). Fits with margin.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.42}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.16}
export TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-4}
# Prompts are filtered to MAX_PROMPT_LENGTH, so the teacher never sees more
# than that + MAX_RESP_LENGTH; +1024 of slack so a boundary case cannot kill
# a multi-hour run the way `decoder prompt (4098) > max_model_len (4096)` did.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$((1024 + MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))}

export PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}
export EXP=${EXP:-lr${LR}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-singlegpu_es_opd_r${MAX_RESP_LENGTH}_${EXP}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
export LOG_DIR=${LOG_DIR:-logs/nersc}

echo "=== ZO-ES-token OPD (NERSC-aligned) ==="
echo "  student $ACTOR_MODEL_PATH   teacher $TEACHER_MODEL_PATH"
echo "  gpu $CUDA_VISIBLE_DEVICES resp<=$MAX_RESP_LENGTH B=$BATCH_SIZE N=$N_SAMPLE lr=$LR thinking=$ENABLE_THINKING"
bash scripts/zo_opd/opd_es_token.sh
