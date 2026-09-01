#!/bin/bash
# opd_es_paper.sh -- ZO-ES-token OPD on the SAME paper-aligned setting as
# scripts/zo_opd/paper_align/opd_bp_paper.sh, so the two are read against
# each other:
#
#   student  Qwen3-1.7B-Base   teacher  lllyx/Qwen3-4B-Base-GRPO
#   prompts  dapo-math-17k-processed   eval  MATH-500 (greedy, full 500)
#   T=1.0 (student_iw is only unbiased when the clean token is SAMPLED),
#   max_tokens 3072.
#
# Differences from the BP run that are inherent to the trainer, not choices:
#   * one rollout per prompt (BP: n=4), so a step consumes 64 sequences vs 256.
#   * eval is greedy n=1 over MATH-500; BP's in-run val is n=4 at T=1.0.
#     Compare the offline eval, not these two numbers directly.
#
#   TRAIN_GPU=7 ES_LR=1e-4 ES_ITERS=200 bash scripts/zo_opd/paper_align/opd_es_paper.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$(pwd)/verl
export HF_HOME=/data/yequan/huggingface

export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-7}

# ---- paper-aligned models / data ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-lllyx/Qwen3-4B-Base-GRPO}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k-processed.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/MATH-500/test.parquet}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-500}

# ---- decode: on-policy sampling, 3072-token budget ----
export TEMPERATURE=${TEMPERATURE:-1.0}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-3072}
export BATCH_SIZE=${BATCH_SIZE:-64}
# 1024 prompt + 3072 response = 256 blocks/slot -> 96 slots fit (results 8.1),
# so the 64-prompt batch is still ONE wave.
export PACK_WIDTH=${PACK_WIDTH:-64}
export B_PACK_BUCKETS=${B_PACK_BUCKETS:-'[64]'}

# ---- estimator ----
export N_SAMPLE=${N_SAMPLE:-8}
export SIGMA=${SIGMA:-0.01}
export SIGMA_MODE=absolute
export SAMPLE_METHOD=bernoulli
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_iw}
export TOKEN_AGG=${TOKEN_AGG:-mean}
export LR=${ES_LR:-1e-4}
export FP32_MASTER=${FP32_MASTER:-true}

# ---- schedule / probe ----
export NUM_ITERATIONS=${ES_ITERS:-200}
export EVAL_INTERVAL=${EVAL_INTERVAL:-20}
# 64 fixed prompts, scored GREEDY (ray_trainer probe_sp) -> 4x the prompts and
# no sampling noise vs the +-8% floor of the old 16-prompt T=1.0 probe.
export HELDOUT_PROBE_SIZE=${HELDOUT_PROBE_SIZE:-64}

# ---- memory (student + 4B teacher co-located on one H100 NVL) ----
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
# The teacher only ever prefills TEACHER_BATCH_SIZE x 4096 = 32k tokens at a
# time; at 0.28 it was sitting on 92,848 tokens of KV. 0.16 leaves ~60k -- still
# ~2x what it needs -- and hands 12 GB back for the fp32 master + the assembly
# accumulator, which is what makes this fit on one 95 GB card.
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.16}
export TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-8}
# The teacher never scores more than prompt+max_tokens, but vLLM sizes its KV
# pool (and its profiling peak) for the model's full 32k context unless told
# otherwise -- at 0.16 utilisation that leaves 1.58 GiB against a 4.50 GiB
# requirement and the engine refuses to start.
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$((1024 + MAX_RESP_LENGTH))}

export PROJECT_NAME=${PROJECT_NAME:-opd-paper-align}
export EXP=${EXP:-lr${LR}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-es_opd_paper_1p7bbase_4bgrpo_r${MAX_RESP_LENGTH}_${EXP}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
export LOG_DIR=${LOG_DIR:-logs/train}

echo "=== ES-token OPD (paper-aligned) ==="
echo "  student $ACTOR_MODEL_PATH   teacher $TEACHER_MODEL_PATH"
echo "  gpu $CUDA_VISIBLE_DEVICES  resp<=$MAX_RESP_LENGTH  B=$BATCH_SIZE N=$N_SAMPLE lr=$LR fp32_master=$FP32_MASTER"
echo "  exp $EXPERIMENT_NAME"
bash scripts/zo_opd/opd_es_token.sh
