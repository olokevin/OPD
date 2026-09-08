#!/bin/bash
# bp_opsd.sh -- BP On-Policy Self-Distillation (OPSD, arXiv:2601.18734) on ONE GPU.
#
# Mirrors https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_1b.sh:
#   model     Qwen/Qwen3-1.7B  (instruct) -- SAME model is student and teacher
#   teacher   the FROZEN initial policy (--fixed_teacher: base weights, LoRA disabled).
#             Here: a second frozen copy in verl's reward_model slot, at the same path.
#             It is given PRIVILEGED context (problem + reference solution + transition
#             prompt, thinking-mode ON); the student sees the problem only, thinking OFF.
#   data      datasets/opsd_openthoughts_math_30k.parquet
#             (= siyanzhao/Openthoughts_math_30k_opsd, built by build_opsd_dataset.py)
#   batch     32 prompts x n=1 = 32 seqs/step, ONE update/step (=> ratio 1, pure PG)
#   loss      full-vocabulary forward KL(p_T || p_S) with per-token pointwise clipping
#             tau=0.05  ->  in this repo: TOP_K_STRATEGY=only_tch + REWARD_WEIGHT_MODE=fkl_clip,
#             truncated to the teacher's top-LOG_PROB_TOP_K ids (see DEVIATIONS below).
#   lr 5e-6, grad-clip 0.1, LoRA r=64 alpha=128 on the 7 proj modules, T=1.1 top-p 0.95
#             top-k 20, response <= 1024, 100 steps.
#
# DEVIATIONS from the paper (all deliberate, all logged):
#  1. Forward KL is truncated to the teacher's top-K ids instead of the full 151k vocab
#     (verl scores the teacher in a separate worker and cannot carry full logits).  The
#     un-renormalised p_T weights are used, so this is the honest truncation of the same
#     sum; the paper's own --top_k_loss flag does the same thing (with renormalisation).
#  2. fp32 optimizer master (verl default) instead of the paper's pure bf16 -- a bf16
#     master silently freezes ~99% of weights at this lr (docs/results/ZO_OPD/opd_paper_align.md).
#  3. In-loop validation is MATH-500 n=1 non-thinking (cheap training monitor).  The
#     paper's Avg@12 AIME24/25 + HMMT25 @ 38912 tokens, thinking ON, is an OFFLINE eval
#     on the saved checkpoints.
#
#   TRAIN_GPU=5 bash scripts/opsd/bp_opsd.sh          # full run
#   SMOKE=1 TRAIN_GPU=5 bash scripts/opsd/bp_opsd.sh  # 2 tiny steps, no wandb
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.  verl runs vLLM
# in sleep mode, and its CuMemAllocator asserts outright:
#   "Expandable segments are not compatible with memory pool" (vllm/device_allocator/cumem.py).
# The card's <10 GB of slack is reclaimed instead by torch.cuda.empty_cache() at the end of
# es_update_actor, which is where the ES rails' full-vocab logits pile up.
export PYTHONPATH="$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

set -a
HF_HOME=${HF_HOME:-/data/yequan/huggingface}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$ACTOR_MODEL_PATH}   # self-teacher: same weights
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-OPSD-OpenThoughts-math}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/opsd_openthoughts_math_30k.parquet}

# ---- OPSD objective ----
ADV_ESTIMATOR=token_reward_direct
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-64}            # truncation of the full-vocab forward KL
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_tch}      # teacher's top-K = forward-KL support
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-fkl_clip}
OPSD_FKL_CLIP=${OPSD_FKL_CLIP:-0.05}            # paper's tau for Qwen3-1.7B

# ---- run_opsd_1b.sh, mirrored ----
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}          # data.train_batch_size (prompts)
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}  # == train batch -> 1 update/step, on-policy
N_RESPONSES=${N_RESPONSES:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1024}
TEMPERATURE=${TEMPERATURE:-1.1}
TOP_P=${TOP_P:-0.95}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-20}
LR=${LR:-5e-6}
GRAD_CLIP=${GRAD_CLIP:-0.1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
OPSD_STEPS=${OPSD_STEPS:-100}
USE_KL=${USE_KL:-False}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
REWARD_MODEL_DTYPE=${REWARD_MODEL_DTYPE:-bfloat16}
IS_PLOT=False

# ---- LoRA (paper: r=64, alpha=128, the 7 projection modules) ----
PEFT_MODE=${PEFT_MODE:-lora}
PEFT_TARGET_MODULES=${PEFT_TARGET_MODULES:-all}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

# ---- validation (cheap in-loop monitor; paper-faithful eval is offline) ----
export TEST_FILE=${TEST_FILE:-'["datasets/test_data/MATH-500/test.parquet"]'}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-3072}
VAL_N=${VAL_N:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
SAVE_FREQ=${SAVE_FREQ:-25}

# ---- single-GPU memory budget ----
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
# param_offload MUST stay False: offloading mid-init frees ~40 GiB while vLLM is
# profiling, which trips its "Error in memory profiling" assert.  At 1.7B + LoRA the
# whole thing fits anyway (actor fp32 ~7 GB + bf16 ~3.4 + teacher ~3.4 + vLLM ~33).
# It is also what algorithm.es_update requires, so BP and ES share the layout.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIM_OFFLOAD=${ACTOR_OPTIM_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}

EXTRA_HYDRA_ARGS="+data.apply_chat_template_kwargs.enable_thinking=False \
  data.shuffle=True \
  +reward_model.opsd_privileged=True \
  +reward_model.teacher_max_prompt_length=4096 \
  ++actor_rollout_ref.rollout.opsd_fkl_clip=${OPSD_FKL_CLIP} \
  ++actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
  ++actor_rollout_ref.rollout.top_p=${TOP_P} \
  ++actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
  trainer.total_training_steps=${OPSD_STEPS} \
  reward_model.use_dynamic_bsz=True \
  reward_model.forward_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
  trainer.max_actor_ckpt_to_keep=8 trainer.max_critic_ckpt_to_keep=8 \
  ${EXTRA_HYDRA_ARGS:-}"

CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-5}
N_GPUS_PER_NODE=1
PARALLEL_SIZE=1
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN
RAY_ISOLATE=1

PROJECT_NAME=${PROJECT_NAME:-opsd_qwen3_1p7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opsd_bp_k${LOG_PROB_TOP_K}_tau${OPSD_FKL_CLIP}_lora${LORA_RANK}_lr${LR}}
LOG_DIR=${LOG_DIR:-logs/opsd}

if [ "${SMOKE:-0}" = "1" ]; then
  MAX_RESP_LENGTH=${SMOKE_RESP_LEN:-256}; MAX_VAL_RESP_LENGTH=$MAX_RESP_LENGTH
  MINI_BATCH_SIZE=8; PPO_MINI_BATCH_SIZE=8
  OPSD_STEPS=2
  TEST_FREQ=-1; VAL_BEFORE_TRAIN=False; SAVE_FREQ=-1
  WANDB_MODE=disabled
  EXPERIMENT_NAME=smoke_${EXPERIMENT_NAME}
  LOG_DIR=logs/opsd/smoke
  EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS/trainer.total_training_steps=100/trainer.total_training_steps=2}"
fi
set +a

mkdir -p "$LOG_DIR"
_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
if [ "${_used:-0}" -gt 2000 ]; then
  echo "GPU $CUDA_VISIBLE_DEVICES already has ${_used} MiB in use -- refusing to launch"; exit 1
fi
pkill -u "$(id -u)" -f "ray_opd_gpu${CUDA_VISIBLE_DEVICES}/sessio[n]_" 2>/dev/null || true
sleep 2
echo "=== BP-OPSD (self-teacher w/ privileged context, single GPU) ==="
echo "  model $ACTOR_MODEL_PATH   teacher(frozen init) $REWARD_MODEL_PATH"
echo "  gpu $CUDA_VISIBLE_DEVICES  batch $MINI_BATCH_SIZE x n=$N_RESPONSES, mini $PPO_MINI_BATCH_SIZE"
echo "  resp<=$MAX_RESP_LENGTH  top_k=$LOG_PROB_TOP_K strat=$TOP_K_STRATEGY rw=$REWARD_WEIGHT_MODE tau=$OPSD_FKL_CLIP"
echo "  peft=$PEFT_MODE r=$LORA_RANK a=$LORA_ALPHA  lr=$LR clip=$GRAD_CLIP T=$TEMPERATURE"
echo "  exp $EXPERIMENT_NAME   log dir $LOG_DIR"
bash on_policy_distillation.sh
