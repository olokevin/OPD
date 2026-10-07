#!/bin/bash
# ISO paper (arXiv:2607.19331) Fig. 6 right: Qwen3-4B-Base math RLVR with DAPO.
# Recipe from App. H; only the optimizer arm (PEFT_MODE) and LR differ between runs.
# Prompt: stock verl Qwen3 chat template (enable_thinking=True = no empty think block);
# measured base avg@16-equivalent 21.4 vs 9.8 with enable_thinking=False (paper: 16.15).
#
#   PEFT_MODE=iso    LR=7.5e-7  -> ISO-AdamW (paper)
#   PEFT_MODE=isobtt LR=...     -> ISO on fura's block-wise SVD
#   PEFT_MODE=none   LR=2e-6    -> weight-space AdamW (paper's best)
#   PEFT_MODE=fura   LR=...     -> BlockTT small core only (output_one_block, S merged
#                                  into the frozen core); needs ACTOR_STRATEGY=fsdp2
#
# Multi-node (NERSC, slurm/iso/): RAY_EXTERNAL=1 attaches to a running Ray cluster;
# NNODES/DEVICES/REPO/CKPT_ROOT/RESUME_MODE are env-overridable.
#
# Usage:  DEVICES=5,7 PEFT_MODE=iso LR=7.5e-7 bash scripts/iso/dapo_math_4b.sh
#         SMOKE=1 ... for a 2-step end-to-end check with a tiny batch.
set -x
set -euo pipefail

REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
PY_ENV_BIN=${PY_ENV_BIN:-/home/yequan/miniconda3/envs/verl/bin}
[ -d "$PY_ENV_BIN" ] && export PATH=$PY_ENV_BIN:$PATH
export CUDA_VISIBLE_DEVICES=${DEVICES:-5,7}
N_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
NNODES=${NNODES:-1}
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export PYTHONPATH="$REPO/verl:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

PEFT_MODE=${PEFT_MODE:-iso}
LR=${LR:-7.5e-7}
SMOKE=${SMOKE:-0}
TOTAL_STEPS=${TOTAL_STEPS:-50}

# ---- paper App. H (math, 4B) ----
MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
TRAIN_FILE=$REPO/datasets/DeepMath-103K.parquet
TD=$REPO/datasets/test_data
VAL_FILES="['$TD/AIME24/test.parquet','$TD/AIME25/test.parquet','$TD/AMC23/test.parquet','$TD/Minerva/test.parquet','$TD/Olympiad-Bench/test.parquet']"
PROMPT_BSZ=256          # global prompt batch, kept after dynamic filtering
N_RESP=12               # rollouts per prompt (4B)
MAX_PROMPT=1024
MAX_RESP=8192
VAL_N=16; VAL_T=1.0; VAL_TOP_P=0.8; VAL_MAX=8192
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-$TEST_FREQ}
if [ "$SMOKE" = "1" ]; then
  PROMPT_BSZ=8; N_RESP=4; MAX_RESP=1024; VAL_N=1; VAL_MAX=1024; TOTAL_STEPS=${SMOKE_STEPS:-2}; TEST_FREQ=2; SAVE_FREQ=2
  VAL_FILES="['$TD/AIME24/test.parquet']"
fi

TAG=${TAG:-${PEFT_MODE}_lr${LR}}
[ "$SMOKE" = "1" ] && TAG=smoke_$TAG
EXP=${EXP:-dapo_q3-4b-base_deepmath_${TAG}}
PROJECT=${PROJECT:-iso_rlvr_qwen3_4b}
CKPT=${CKPT:-${CKPT_ROOT:-/data/yequan/opd/opd/DeepMath-103K}/$EXP}
mkdir -p "$CKPT" logs/iso
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp}

case "$PEFT_MODE" in
  none) PEFT_ARGS="" ;;
  iso|isobtt)
    PEFT_ARGS="++actor_rollout_ref.peft.mode=$PEFT_MODE \
      ++actor_rollout_ref.peft.target_modules=all \
      ++actor_rollout_ref.peft.iso.train_others=True \
      actor_rollout_ref.actor.fsdp_config.use_orig_params=True" ;;
  fura)
    # FuRA: per input block W[:, blk_j] = (U_j S_j) R_j, U_j S_j frozen, R_j (b x b)
    # trained; blocks from the closest factor pair, same as isobtt.
    PEFT_ARGS="++actor_rollout_ref.peft.mode=blocktt \
      ++actor_rollout_ref.peft.target_modules=all \
      ++actor_rollout_ref.peft.blocktt.decomp_mode=output_one_block \
      ++actor_rollout_ref.peft.blocktt.rank=full \
      ++actor_rollout_ref.peft.blocktt.train_position=small \
      ++actor_rollout_ref.peft.blocktt.s_merged_to=frozen \
      ++actor_rollout_ref.peft.blocktt.convert_mode=svd \
      ++actor_rollout_ref.peft.blocktt.factorize_by_head=False \
      ++actor_rollout_ref.peft.blocktt.normalize_after_update=False \
      ++actor_rollout_ref.peft.blocktt.qfura.enabled=False \
      actor_rollout_ref.actor.fsdp_config.use_orig_params=True" ;;
  *) echo "unknown PEFT_MODE=$PEFT_MODE" >&2; exit 2 ;;
esac

# ---- private Ray head on this GPU set (never touch other runs) ----
if [ "${RAY_EXTERNAL:-0}" != "1" ]; then
_gpu0=${CUDA_VISIBLE_DEVICES%%,*}
RAY_TMP=/tmp/ray_iso_gpu${_gpu0}
pkill -u "$(id -u)" -f "ray_iso_gpu${_gpu0}/sessio[n]_" || true
sleep 2
rm -rf "$RAY_TMP"; mkdir -p "$RAY_TMP"
RAY_PORT=$((6379 + _gpu0 * 100 + 37))
unset RAY_ADDRESS
ray start --head --port="$RAY_PORT" --temp-dir="$RAY_TMP" --dashboard-host=127.0.0.1 --num-gpus="$N_GPUS"
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
trap 'pkill -u "$(id -u)" -f "ray_iso_gpu${_gpu0}/sessio[n]_" || true' EXIT
fi

cd "$REPO/verl"
python3 -m recipe.dapo.main_dapo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILES" \
  data.prompt_key=prompt \
  data.shuffle=True \
  data.train_batch_size=$PROMPT_BSZ \
  data.gen_batch_size=$PROMPT_BSZ \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  ++data.apply_chat_template_kwargs.enable_thinking=${THINKING:-True} \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=seq_final_reward \
  algorithm.filter_groups.max_num_gen_batches=10 \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable=False \
  custom_reward_function.path="$REPO/verl/verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=$ACTOR_STRATEGY \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.optim.weight_decay=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PROMPT_BSZ \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_TOK:-16384} \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFFLOAD:-True} \
  actor_rollout_ref.ref.strategy=$ACTOR_STRATEGY \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((3 * ${PPO_TOK:-16384})) \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=$N_RESP \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.55} \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((3 * ${PPO_TOK:-16384})) \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_T \
  actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
  ++actor_rollout_ref.rollout.val_kwargs.max_tokens=$VAL_MAX \
  trainer.logger="['console','wandb']" \
  trainer.project_name=$PROJECT \
  trainer.experiment_name=$EXP \
  trainer.validation_data_dir=$CKPT/val_generations \
  trainer.log_val_generations=2 \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=$NNODES \
  trainer.val_before_train=${VAL_BEFORE:-True} \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.total_epochs=1 \
  trainer.default_local_dir="$CKPT" \
  trainer.resume_mode=${RESUME_MODE:-disable} \
  $PEFT_ARGS \
  ${EXTRA_HYDRA_ARGS:-}
