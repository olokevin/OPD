#!/bin/bash
# bp_opd.sh -- BP-OPD in the thunlp/OPD reference setting, on ONE GPU.
#
# Mirrors https://github.com/thunlp/OPD/blob/main/verl_example/opd.sh:
#   student  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   teacher  hbx/JustRL-DeepSeek-1.5B          (same architecture, RL'd)
#   data     datasets/DAPO-Math-17k/DAPO-Math.parquet   (= dapo-math-17k.parquet)
#   train_batch 64 prompts x n=4 = 256 seqs / step, ppo_mini_batch 16 prompts
#     -> 4 optimizer updates per step, each on 64 sequences (same data-per-update
#        as their 4-GPU run; on 1 GPU verl just accumulates more micro-batches)
#   prompt<=1024, response<=7168, lr 1e-6, 1 epoch, T=1.0, no KL, no task reward
#   loss  = their default DISTILLATION_LOSS_MODE=k1 + USE_POLICY_GRADIENT=True,
#           i.e. sampled-token reverse-KL as a per-token advantage in the PG loss.
#           In this repo that is LOG_PROB_TOP_K=0 (`token_reward_direct`).
#           Set LOG_PROB_TOP_K=16 for the top-K variant (only_stu / student_p).
#
# Single-GPU deltas: teacher is CO-LOCATED (bf16, param-offloaded), vLLM gets
# 0.5 of the card, actor/teacher micro-batches are token-budgeted.
# Validation: MATH-500 + AIME24, n=2 @ T=0.6 / top-p 0.95 (DeepSeek's recommended
# sampling for R1-Distill; greedy loops), 7168 tokens, every 20 steps.
#
#   TRAIN_GPU=4 bash scripts/zo_opd/ds15b/bp_opd.sh          # full run
#   SMOKE=1 TRAIN_GPU=4 bash scripts/zo_opd/ds15b/bp_opd.sh  # 2 tiny steps, no wandb
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
# Run THIS checkout's verl (the editable install points at the main checkout).
export PYTHONPATH="$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

set -a
HF_HOME=${HF_HOME:-/data/yequan/huggingface}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-hbx/JustRL-DeepSeek-1.5B}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/DAPO-Math-17k/DAPO-Math.parquet}

# ---- thunlp opd.sh, mirrored ----
ADV_ESTIMATOR=token_reward_direct
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}          # data.train_batch_size (prompts)
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}  # actor.ppo_mini_batch_size (prompts)
N_RESPONSES=${N_RESPONSES:-4}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}             # 0 = sampled-token k1 (their default)
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
TEMPERATURE=${TEMPERATURE:-1.0}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
USE_KL=${USE_KL:-False}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}                # verl default: fp32 master, bf16 compute
REWARD_MODEL_DTYPE=${REWARD_MODEL_DTYPE:-bfloat16}
IS_PLOT=False

# ---- validation ----
export TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500/test.parquet","datasets/test_data/AIME24/test.parquet"]'}
VAL_N=${VAL_N:-2}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
TEST_FREQ=${TEST_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
SAVE_FREQ=${SAVE_FREQ:-20}

# ---- single-GPU memory budget ----
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-True}
REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-9216}
EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  reward_model.use_dynamic_bsz=True \
  reward_model.forward_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  ${EXTRA_HYDRA_ARGS:-}"

CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-4}
N_GPUS_PER_NODE=1
PARALLEL_SIZE=1
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN
RAY_ISOLATE=1

PROJECT_NAME=${PROJECT_NAME:-es_opd_JustRL_1p5b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_bp_opd_k${LOG_PROB_TOP_K}_r${MAX_RESP_LENGTH}_lr${LR}}
LOG_DIR=${LOG_DIR:-logs/ds15b}

if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_DATASET=datasets/dapo-math-17k-1percent.parquet   # 179 prompts -> 2 steps
  MAX_RESP_LENGTH=${SMOKE_RESP_LEN:-1024}; MAX_VAL_RESP_LENGTH=$MAX_RESP_LENGTH
  TEST_FREQ=-1; VAL_BEFORE_TRAIN=False; SAVE_FREQ=-1
  WANDB_MODE=disabled
  EXPERIMENT_NAME=smoke_${EXPERIMENT_NAME}
  LOG_DIR=logs/ds15b/smoke
fi
set +a

mkdir -p "$LOG_DIR"
# A previous RAY_ISOLATE run on this GPU leaves its private Ray head (gcs/raylet/
# dashboard under /tmp/ray_opd_gpu<N>) alive; a fresh `ray start` on the same port then
# dies with "Session name ... does not match persisted value". Reap it -- but only if
# nothing of ours is still using the card.
_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
if [ "${_used:-0}" -gt 2000 ]; then
  echo "GPU $CUDA_VISIBLE_DEVICES already has ${_used} MiB in use -- refusing to launch"; exit 1
fi
pkill -u "$(id -u)" -f "ray_opd_gpu${CUDA_VISIBLE_DEVICES}/sessio[n]_" 2>/dev/null || true
sleep 2
echo "=== BP-OPD (thunlp-mirrored, single GPU) ==="
echo "  student $ACTOR_MODEL_PATH   teacher $REWARD_MODEL_PATH"
echo "  gpu $CUDA_VISIBLE_DEVICES  batch $MINI_BATCH_SIZE x n=$N_RESPONSES, mini $PPO_MINI_BATCH_SIZE"
echo "  resp<=$MAX_RESP_LENGTH  top_k=$LOG_PROB_TOP_K lr=$LR dtype=$MODEL_DTYPE"
echo "  exp $EXPERIMENT_NAME   log dir $LOG_DIR"
bash on_policy_distillation.sh
