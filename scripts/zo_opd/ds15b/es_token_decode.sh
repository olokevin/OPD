#!/bin/bash
# es_token_decode.sh -- es-token-decode (fresh rank-1 perturbation per token, decode rails)
# on the ds15b setting, with the 2026-08-31 rail-aware kernels, for the efficiency +
# learning comparison against es-prefill (es_opd.sh) and BP (bp_opd.sh).
#
#   student  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   teacher hbx/JustRL-DeepSeek-1.5B
#   mirror of es-prefill: 64 sequences/step (64 prompts x n=1 here), N=32 rails,
#   probe footprint sigma/RMS(W) = 1e-3/0.0536 = 1.9 % (same as es-prefill sigma),
#   max_tokens 7168, T=1.0, top-p 0.95, MATH-500 greedy eval every 20 steps.
#   Kernels: attn_impl=shared, lm_head_impl=stream (es_profile_results.md).
#   LR is the es_token SGD scale (not es-prefill's alpha): pick by train/update_footprint,
#   target ~2e-3-6e-3 per step (es-prefill C ran 5.8e-3). Default 3e-4.
#
#   TRAIN_GPU=7 ES_LR=3e-4 bash scripts/zo_opd/ds15b/es_token_decode.sh
#   SMOKE=1 TRAIN_GPU=7 bash scripts/zo_opd/ds15b/es_token_decode.sh
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
export ENABLE_THINKING=${ENABLE_THINKING:-false}   # no-op for the DeepSeek template

export TEMPERATURE=${TEMPERATURE:-1.0}
export ES_TOP_P=${ES_TOP_P:-0.95}                  # parity with BP rollout + evals
export ES_EOS_FROM_GENCFG=${ES_EOS_FROM_GENCFG:-false}  # config.json eos IS 151643 here
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export BATCH_SIZE=${BATCH_SIZE:-64}                # 64 seqs/step, mirrors es-prefill
export PACK_WIDTH=${PACK_WIDTH:-8}                 # B(1+N)=264 rows/wave; ridge study says small B
export B_PACK_BUCKETS=${B_PACK_BUCKETS:-'[8]'}

export N_SAMPLE=${N_SAMPLE:-32}
export SIGMA=${SIGMA:-1e-3}
export SIGMA_MODE=absolute
export SAMPLE_METHOD=bernoulli
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_iw}
export TOKEN_AGG=${TOKEN_AGG:-mean}
export LR=${ES_LR:-3e-4}
export FP32_MASTER=true
export ASSEMBLE_CHUNK=${ASSEMBLE_CHUNK:-1024}

# rail-aware kernels (0902 generation: fused zero-launch rail, non-causal seq attention,
# fully in-graph token step -- es_profile_results.md + commit 9a4521e)
export ATTN_IMPL=${ATTN_IMPL:-seq}
export LM_HEAD_IMPL=${LM_HEAD_IMPL:-stream}
# rail_impl=fused binds to Qwen3's op chain (q/k-norm+RoPE); Qwen2/R1-Distill has no fused
# consumer (asserted in the smoke) -> kernel (the single-launch Triton rail op) on this model.
export RAIL_IMPL=${RAIL_IMPL:-kernel}
# step_impl=graph needs top_p=1.0 (top-p is not in-graph); we keep ES_TOP_P=0.95 for BP parity.
export STEP_IMPL=${STEP_IMPL:-eager}

# ES_LOSS_IMPL=topk: exact top-K truncated CE on the clean rail's top-K ids
# (HF teacher, no IW / no +1 score term) -- the arm built to make
# es-token-decode actually learn. Default keeps the sampled-token loss.
export ES_LOSS_IMPL=${ES_LOSS_IMPL:-sampled}
export ES_TOPK_K=${ES_TOPK_K:-16}

export NUM_ITERATIONS=${ES_ITERS:-150}
export EVAL_INTERVAL=${EVAL_INTERVAL:-20}
export HELDOUT_PROBE_SIZE=${HELDOUT_PROBE_SIZE:-16}
export SAVE_FREQ=${SAVE_FREQ:-20}

export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.12}
export TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$((1024 + MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))}

export PROJECT_NAME=${PROJECT_NAME:-es_opd_JustRL_1p5b}
_LOSS_TAG=""; [ "$ES_LOSS_IMPL" = "topk" ] && _LOSS_TAG="_topk${ES_TOPK_K}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_es-token-decode_N${N_SAMPLE}_sig${SIGMA}_lr${LR}${_LOSS_TAG}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
export LOG_DIR=${LOG_DIR:-logs/ds15b/es_token}

if [ "${SMOKE:-0}" = "1" ]; then
  export BATCH_SIZE=8 MAX_RESP_LENGTH=512 N_SAMPLE=8 NUM_ITERATIONS=2 EVAL_INTERVAL=0
  export TEACHER_MAX_MODEL_LEN=$((1024 + 1024 + 512)) SAVE_FREQ=0
  export WANDB_MODE=disabled EXPERIMENT_NAME=smoke_${EXPERIMENT_NAME} LOG_DIR=${LOG_DIR}/smoke
fi

_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
if [ "${_used:-0}" -gt 2000 ]; then echo "GPU $CUDA_VISIBLE_DEVICES busy (${_used} MiB) -- refusing"; exit 1; fi
mkdir -p "$LOG_DIR"

echo "=== es-token-decode (ds15b, rail-aware kernels) ==="
echo "  student $ACTOR_MODEL_PATH  teacher $TEACHER_MODEL_PATH  gpu $CUDA_VISIBLE_DEVICES"
echo "  B=$BATCH_SIZE pw=$PACK_WIDTH N=$N_SAMPLE sigma=$SIGMA lr=$LR resp<=$MAX_RESP_LENGTH attn=$ATTN_IMPL lm_head=$LM_HEAD_IMPL rail=$RAIL_IMPL step=$STEP_IMPL"
bash scripts/zo_opd/opd_es_token.sh
