#!/bin/bash
# Offline NP-vs-BP gradient check for the zeroth-order node-perturbation estimator.
#
# Computes, for ONE perturb layer on ONE frozen (prompt + greedy response):
#   - cosine similarity between the NP-estimated delta_W and the true BP dL/dW
#   - the grad norm of each, and the implied scale ratio (improper-scaling check)
#   - a learning-rate suggestion
#
# Driven by verl.trainer.zo_np.grad_check (eager HF + autograd, single GPU).
# Mirrors the env-var style of scripts/zo_opd/opd_np.sh; hyperparams default to
# the attached OPD-math figure (temp 1.0, LogProb top-K 16, Student top-K,
# top-p 1.0, max prompt 1024, max resp 7168, lr 1e-6, KL coef 0).
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/zo_np_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# One GPU only -- pick a free one. Override with CUDA_VISIBLE_DEVICES=N bash ...
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
# Deterministic backward (eager attention is set in-code).
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

# ---- models ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}

# ---- which layer to check (vLLM-real name; HF down_proj is unfused already) ----
export LAYER=${LAYER:-model.layers.0.mlp.down_proj}

# ---- OPD knobs (from the figure) ----
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}        # LogProb top-K = 16
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}  # Student Top-K
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export TEMPERATURE=${TEMPERATURE:-1.0}             # training temperature 1.0 (response is greedy-frozen for the check)
export MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
export MAX_RESP_LEN=${MAX_RESP_LEN:-7168}
export LR_REF=${LR_REF:-1e-6}                      # learning rate 1e-6

# ---- NP estimator knobs ----
export SIGMA=${SIGMA:-0.01}
export N_SAMPLE=${N_SAMPLE:-16}                    # matches "rollout/sample width"; figure LogProb top-K is separate
export N_ROLLOUT=${N_ROLLOUT:-4}                   # Rollout number = 4
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-grpo}
export NORMALIZE=${NORMALIZE:-true}               # ANP 1/||u||^2 normalization (as the trainer hardcodes)
export TOKEN_AGG=${TOKEN_AGG:-sum}
export GLOBAL_SEED=${GLOBAL_SEED:-42}

# ---- cost / scope ----
export MAX_STEPS=${MAX_STEPS:-64}                  # response steps scored (0 = all)
export DTYPE=${DTYPE:-bfloat16}
export ENABLE_THINKING=${ENABLE_THINKING:-false}

# ---- output ----
export OUT=${OUT:-scripts/zo_opd/results/grad_check_$(date +%Y%m%d_%H%M%S).json}
mkdir -p "$(dirname "$OUT")"

python3 -m verl.trainer.zo_np.grad_check \
    --student "${ACTOR_MODEL_PATH}" \
    --teacher "${REWARD_MODEL_PATH}" \
    --layer "${LAYER}" \
    --log-prob-top-k "${LOG_PROB_TOP_K}" \
    --teacher-temperature "${TEACHER_TEMPERATURE}" \
    --reward-weight-mode "${REWARD_WEIGHT_MODE}" \
    --max-prompt-len "${MAX_PROMPT_LEN}" \
    --max-resp-len "${MAX_RESP_LEN}" \
    --max-steps "${MAX_STEPS}" \
    --sigma "${SIGMA}" \
    --n-sample "${N_SAMPLE}" \
    --n-rollout "${N_ROLLOUT}" \
    --sample-method "${SAMPLE_METHOD}" \
    --grad-estimate-sample "${GRAD_ESTIMATE_SAMPLE}" \
    --normalize "${NORMALIZE}" \
    --token-agg "${TOKEN_AGG}" \
    --global-seed "${GLOBAL_SEED}" \
    --dtype "${DTYPE}" \
    --enable-thinking "${ENABLE_THINKING}" \
    --lr-ref "${LR_REF}" \
    --out "${OUT}"
