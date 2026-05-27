#!/bin/bash
#SBATCH --job-name=url
#SBATCH --output=logs/20251004/output_%j.log
#SBATCH --error=logs/20251004/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

# Configure logging when running outside SBATCH.
if [ -z "$SLURM_JOB_ID" ]; then
    # Create the log directory and file for local runs.
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    # Mirror output to both terminal and log file.
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# Make compress/ package importable when PEFT_MODE=blocktt|svd is enabled.
export PYTHONPATH="$(dirname "$(realpath "$0")")/src${PYTHONPATH:+:$PYTHONPATH}"

ray stop --force
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation' # TODO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
# export ADV_ESTIMATOR=token_reward_direct
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo
# export ADV_ESTIMATOR=token_grpo
export ADV_ESTIMATOR=grpo
export GRPO_OUTCOME_WEIGHT=1.0
# Swanlab setting used to continue exp  
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="jri5qia6iy67v7su0zjsv"


export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=7168  # TODO: 31744 /15360 / 7168 / 3072 / 5120
export MAX_VAL_RESP_LENGTH=31744 # TODO: 15360 / 7168 / 3072
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0} # Teacher logits temperature (default 1.0, no scaling)
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0} # TODO: 1.0 / 1.1 / 1.2 (default 1.0, no penalty)
export N_RESPONSES=8 # TODO: 4 / 8 / 16 / 32 (default: 8)
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0} # 0 represents no top-k sampling
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"union"} # "only_stu" or "only_tch" or "intersection" or "union" or "union-intersection"
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"} # "student_p" or "teacher_p" or "none"
# export LR=${LR:-1e-6}
# export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export USE_KL=${USE_KL:-False} # TODO: True / False (default False)
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False} # TODO: True / False (default False)
export MODEL_DTYPE=${MODEL_DTYPE:-fp32} # actor/ref/critic fsdp_config.model_dtype: fp32 or bfloat16
export IS_PLOT=${IS_PLOT:-False} # TODO: True / False (default False)
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"} # TODO: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" / "seq-mean-token-sum-norm" (default "token-mean")

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/train.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-1percent.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-1percent-processed.parquet
export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/sampled_5k.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/verl_format/train.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-Processed
# export TRAIN_DATASET_NAME=DeepMath-103K-sampled_5k
# export TRAIN_DATASET_NAME=DeepMath-103K

export TEST_DATA_DIR=datasets/test_data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# TODO:
# export ACTOR_MODEL_PATH=model/qwen3-1.7b-math-sft
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-sft/checkpoint-6000
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export ACTOR_MODEL_PATH=model/DS-1.5B-SFT
export ACTOR_MODEL_PATH=model/Qwen3-4B-Base
# export ACTOR_MODEL_NAME=model/Qwen3-4B-grpo
# export ACTOR_MODEL_NAME=model/Qwen3-4B
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=model/Qwen3-4B
# export REWARD_MODEL_PATH=model/Qwen3-1.7B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export REWARD_MODEL_PATH=model/Skywork-OR1-Math-7B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-14B
# export REWARD_MODEL_PATH=model/JustRL-DeepSeek-1.5B
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}_peft-${PEFT_MODE:-none}-$(date +%Y-%m-%d_%H-%M-%S)
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1


export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}_peft-${PEFT_MODE:-none}-$(date +%Y-%m-%d_%H-%M-%S)

# ---- PEFT ----
export PEFT_MODE=${PEFT_MODE:-none}
export PEFT_TARGET_MODULES=${PEFT_TARGET_MODULES:-all}

export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}

export QLORA_QUANT_TYPE=${QLORA_QUANT_TYPE:-nf4}
export QLORA_DOUBLE_QUANT=${QLORA_DOUBLE_QUANT:-True}
export QLORA_COMPUTE_DTYPE=${QLORA_COMPUTE_DTYPE:-bfloat16}

export BTT_DECOMP_MODE=${BTT_DECOMP_MODE:-input_one_block}
export BTT_RANK=${BTT_RANK:-full}
export BTT_TRAIN_POSITION=${BTT_TRAIN_POSITION:-small}
export BTT_S_MERGED_TO=${BTT_S_MERGED_TO:-frozen}
export BTT_CONVERT_MODE=${BTT_CONVERT_MODE:-svd}
export BTT_FACTORIZE_BY_HEAD=${BTT_FACTORIZE_BY_HEAD:-True}
export BTT_NORMALIZE_AFTER_UPDATE=${BTT_NORMALIZE_AFTER_UPDATE:-False}
export BTT_QFURA=${BTT_QFURA:-False}

export SVD_TRAIN_POSITION=${SVD_TRAIN_POSITION:-output}
export SVD_S_MERGED_TO=${SVD_S_MERGED_TO:-frozen}
export SVD_COMPRESSION_RATIO=${SVD_COMPRESSION_RATIO:-1.0}

export CALIB_MODE=${CALIB_MODE:-none}
export CALIB_SOURCE=${CALIB_SOURCE:-c4}
export CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
export CALIB_MAX_LENGTH=${CALIB_MAX_LENGTH:-2048}
export CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-8}
export CALIB_SEED=${CALIB_SEED:-3}
export CALIB_TRACES_PATH=${CALIB_TRACES_PATH:-}

PEFT_ARGS="+actor_rollout_ref.peft.mode=$PEFT_MODE \
+actor_rollout_ref.peft.target_modules=$PEFT_TARGET_MODULES"

case "$PEFT_MODE" in
  none) ;;
  lora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.lora.dropout=$LORA_DROPOUT \
      actor_rollout_ref.model.lora_rank=$LORA_RANK \
      actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" ;;
  qlora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.qlora.bnb_4bit_quant_type=$QLORA_QUANT_TYPE \
      +actor_rollout_ref.peft.qlora.bnb_4bit_use_double_quant=$QLORA_DOUBLE_QUANT \
      +actor_rollout_ref.peft.qlora.bnb_4bit_compute_dtype=$QLORA_COMPUTE_DTYPE \
      actor_rollout_ref.model.lora_rank=$LORA_RANK \
      actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" ;;
  blocktt)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.blocktt.decomp_mode=$BTT_DECOMP_MODE \
      +actor_rollout_ref.peft.blocktt.rank=$BTT_RANK \
      +actor_rollout_ref.peft.blocktt.train_position=$BTT_TRAIN_POSITION \
      +actor_rollout_ref.peft.blocktt.s_merged_to=$BTT_S_MERGED_TO \
      +actor_rollout_ref.peft.blocktt.convert_mode=$BTT_CONVERT_MODE \
      +actor_rollout_ref.peft.blocktt.factorize_by_head=$BTT_FACTORIZE_BY_HEAD \
      +actor_rollout_ref.peft.blocktt.normalize_after_update=$BTT_NORMALIZE_AFTER_UPDATE \
      +actor_rollout_ref.peft.blocktt.qfura.enabled=$BTT_QFURA" ;;
  svd)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.svd.train_position=$SVD_TRAIN_POSITION \
      +actor_rollout_ref.peft.svd.s_merged_to=$SVD_S_MERGED_TO \
      +actor_rollout_ref.peft.svd.compression_ratio=$SVD_COMPRESSION_RATIO" ;;
  *) echo "Unknown PEFT_MODE=$PEFT_MODE" >&2; exit 1 ;;
esac

if [ "$CALIB_MODE" != "none" ]; then
  PEFT_ARGS="$PEFT_ARGS \
    +actor_rollout_ref.peft.calib.mode=$CALIB_MODE \
    +actor_rollout_ref.peft.calib.source=$CALIB_SOURCE \
    +actor_rollout_ref.peft.calib.num_seqs=$CALIB_NUM_SEQS \
    +actor_rollout_ref.peft.calib.max_length=$CALIB_MAX_LENGTH \
    +actor_rollout_ref.peft.calib.batch_size=$CALIB_BATCH_SIZE \
    +actor_rollout_ref.peft.calib.seed=$CALIB_SEED"
  if [ -n "$CALIB_TRACES_PATH" ]; then
    PEFT_ARGS="$PEFT_ARGS +actor_rollout_ref.peft.calib.traces_path=$CALIB_TRACES_PATH"
  fi
fi
# ---- /PEFT ----

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"


ray start --head
sleep 5


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    +algorithm.rollout_correction.rollout_is=token \
    +algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=False \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT \
    $PEFT_ARGS

# Log the end time for local runs.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
