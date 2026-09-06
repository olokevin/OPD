#!/bin/bash
# es_decode.sh -- es-decode (ONE perturbation per rail, HELD for every generated
# token; decode rails on the clean KV) on the ds15b setting, the fourth rail
# algorithm of zo_opd_short.md, built 2026-09-05. Same pipeline as
# es_token_decode.sh (student/teacher/data/rollout/teacher scoring/eval), but:
#   * rail n carries dW_n = sigma * eps_n for the whole step (all tokens, all
#     rollouts), eps_n regenerated from a per-step seed; rail 2i+1 = -rail 2i
#   * NOISE_RANK=full: eps in {+-1}^{m x n} as packed bits (5.2 GB for 32 rails)
#     NOISE_RANK=r:    eps = sigma/sqrt(r) sum_k a_k b_k^T (Rademacher factors)
#   * fitness per rail = es-prefill's k1 objective on the sampled rollout
#     F_n = sum_t A_t (log pi_n(y_t) - log pi_0(y_t)) / sum_t 1,  A_t = log q - log pi_0
#   * update = es_update.py's OpenAI-ES step: antithetic pairs, d = (F+ - F-)/2,
#     W += alpha/n_pairs * sum d/RMS(d) * eps   (ES_ALPHA, ES_NORMALIZE)
# Mirror of es-prefill C: N=32 rails (16 pairs), sigma 1e-3, alpha 1.25e-3,
# 64 seqs/step, 7168 tokens, T=1.0 top-p 0.95, MATH-500 greedy eval every 20.
#
#   TRAIN_GPU=7 NOISE_RANK=full bash scripts/zo_opd/ds15b/es_decode.sh
#   TRAIN_GPU=7 NOISE_RANK=1    bash scripts/zo_opd/ds15b/es_decode.sh
#   SMOKE=1 TRAIN_GPU=7 NOISE_RANK=full bash scripts/zo_opd/ds15b/es_decode.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$REPO_ROOT/verl
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-7}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-hbx/JustRL-DeepSeek-1.5B}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/DAPO-Math-17k/DAPO-Math.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/MATH-500/test.parquet}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-500}
export ENABLE_THINKING=${ENABLE_THINKING:-false}

export TEMPERATURE=${TEMPERATURE:-1.0}
export ES_TOP_P=${ES_TOP_P:-0.95}
export ES_EOS_FROM_GENCFG=${ES_EOS_FROM_GENCFG:-false}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export BATCH_SIZE=${BATCH_SIZE:-64}
export PACK_WIDTH=${PACK_WIDTH:-8}
export B_PACK_BUCKETS=${B_PACK_BUCKETS:-'[8]'}

# es-decode knobs
export RAIL_MODE=seq
export NOISE_RANK=${NOISE_RANK:-full}          # full | 1 | r
export N_SAMPLE=${N_SAMPLE:-32}
export SIGMA=${SIGMA:-1e-3}                     # probe footprint sigma/RMS(W) = 1.9 % (es-prefill's)
export SIGMA_MODE=absolute
export SAMPLE_METHOD=bernoulli
export ES_ALPHA=${ES_ALPHA:-1.25e-3}            # es-prefill C
export ES_ANTITHETIC=${ES_ANTITHETIC:-true}
export ES_NORMALIZE=${ES_NORMALIZE:-zscore}
export FP32_MASTER=true
export LR=0                                      # unused in seq mode (alpha drives the step)

# kernels: fused consumers (Qwen2 port) carry rank<=8; full rank is the packed-bit GEMV;
# seq attention + streaming head; eager tail because top-p 0.95 is not in-graph.
export ATTN_IMPL=${ATTN_IMPL:-seq}
export LM_HEAD_IMPL=${LM_HEAD_IMPL:-stream}
export RAIL_IMPL=${RAIL_IMPL:-fused}
export STEP_IMPL=${STEP_IMPL:-eager}
export ES_LOSS_IMPL=sampled

export NUM_ITERATIONS=${ES_ITERS:-80}
export EVAL_INTERVAL=${EVAL_INTERVAL:-20}
export HELDOUT_PROBE_SIZE=${HELDOUT_PROBE_SIZE:-16}
export SAVE_FREQ=${SAVE_FREQ:-20}
export CKPT_KEEP_LAST=${CKPT_KEEP_LAST:-10}

export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.12}
export TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$((1024 + MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))}

export PROJECT_NAME=${PROJECT_NAME:-es_opd_JustRL_1p5b}
_RANK_TAG="r${NOISE_RANK}"; [ "$NOISE_RANK" = "full" ] && _RANK_TAG="full"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_es-decode_${_RANK_TAG}_N${N_SAMPLE}_sig${SIGMA}_a${ES_ALPHA}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
export LOG_DIR=${LOG_DIR:-logs/ds15b/es_decode}

if [ "${SMOKE:-0}" = "1" ]; then
  export BATCH_SIZE=8 MAX_RESP_LENGTH=256 N_SAMPLE=4 NUM_ITERATIONS=2 EVAL_INTERVAL=0
  export TEACHER_MAX_MODEL_LEN=$((1024 + 1024 + 256)) SAVE_FREQ=0 HELDOUT_PROBE_SIZE=0
  export WANDB_MODE=disabled EXPERIMENT_NAME=smoke_${EXPERIMENT_NAME} LOG_DIR=${LOG_DIR}/smoke
fi

_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
if [ "${_used:-0}" -gt 2000 ]; then echo "GPU $CUDA_VISIBLE_DEVICES busy (${_used} MiB) -- refusing"; exit 1; fi
mkdir -p "$LOG_DIR"

echo "=== es-decode (ds15b) ==="
echo "  student $ACTOR_MODEL_PATH  teacher $TEACHER_MODEL_PATH  gpu $CUDA_VISIBLE_DEVICES"
echo "  B=$BATCH_SIZE pw=$PACK_WIDTH N=$N_SAMPLE rank=$NOISE_RANK sigma=$SIGMA alpha=$ES_ALPHA antithetic=$ES_ANTITHETIC resp<=$MAX_RESP_LENGTH attn=$ATTN_IMPL lm_head=$LM_HEAD_IMPL rail=$RAIL_IMPL step=$STEP_IMPL"
bash scripts/zo_opd/opd_es_token.sh
