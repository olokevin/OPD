#!/bin/bash
# opd_es_token.sh — es_token (per-token weight-perturbation ES) OPD-math training.
#
# Trainer: verl.trainer.main_es_token (graphed packed 1+N rail decode, rank-1
#          weight perturbation with fixed Hadamard sign rails, sampled-token
#          teacher loss, chunked-GEMM assembly). docs/plans/es_token_trainer.md.
# Default = ALL decoder linears perturbed simultaneously, batch 64 x 1024,
#          N_SAMPLE=8 rails, greedy clean trajectory, student+teacher co-located.
#
#   CUDA_VISIBLE_DEVICES=2 LR=1e-3 EXP=lr1e-3 bash scripts/zo_opd/opd_es_token.sh
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opd_es_token_${EXP:-run}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

export TMPDIR=${ZO_TMPDIR:-/tmp}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export NP_KEEP_CUDA_VISIBLE=${NP_KEEP_CUDA_VISIBLE:-1}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

# ---- es_token knobs ----
export SIGMA=${SIGMA:-0.01}
export SIGMA_MODE=${SIGMA_MODE:-absolute}
export N_SAMPLE=${N_SAMPLE:-8}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-mean_baseline}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_iw}
export TOKEN_AGG=${TOKEN_AGG:-mean}
export LR=${LR:-1e-3}
export ASSEMBLE_CHUNK=${ASSEMBLE_CHUNK:-1024}
export USE_CUDA_GRAPH=${USE_CUDA_GRAPH:-true}
export PACK_WIDTH=${PACK_WIDTH:-4}
export B_PACK_BUCKETS=${B_PACK_BUCKETS:-'[2,4]'}
export PERTURB_RULES=${PERTURB_RULES:-'^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))$'}

# ---- teacher (sampled-token: one prefill per rollout) ----
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-16}

# ---- decode ----
export TEMPERATURE=${TEMPERATURE:-0.0}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1024}

# ---- hardware: single GPU, co-located teacher ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
export NUM_ENGINES=${NUM_ENGINES:-1}
export GPU_FRACTION=${GPU_FRACTION:-0.5}
export EXEC_BACKEND=${EXEC_BACKEND:-uni}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.30}

# ---- model & data ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/MATH-500/test.parquet}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-200}
export NUM_ITERATIONS=${NUM_ITERATIONS:-150}
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}
export HELDOUT_PROBE_SIZE=${HELDOUT_PROBE_SIZE:-16}

# ---- logging ----
export PROJECT_NAME=${PROJECT_NAME:-opd-qwen-math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-es_token_${ACTOR_MODEL_NAME}_${EXP:-lr${LR}}_$(date +%m%d_%H%M)}
export SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_es_token --config-name es_token_trainer \
    es_token.sigma=${SIGMA} es_token.sigma_mode=${SIGMA_MODE} \
    es_token.n_sample=${N_SAMPLE} es_token.batch_size=${BATCH_SIZE} \
    es_token.sample_method=${SAMPLE_METHOD} \
    es_token.grad_estimate_sample=${GRAD_ESTIMATE_SAMPLE} \
    es_token.reward_weight_mode=${REWARD_WEIGHT_MODE} \
    es_token.token_agg=${TOKEN_AGG} es_token.lr=${LR} \
    es_token.assemble_chunk=${ASSEMBLE_CHUNK} \
    es_token.use_cuda_graph=${USE_CUDA_GRAPH} \
    es_token.pack_width=${PACK_WIDTH} \
    "es_token.b_pack_buckets=${B_PACK_BUCKETS}" \
    'es_token.perturb_rules=["'"${PERTURB_RULES}"'"]' \
    es_token.teacher_model_path=${TEACHER_MODEL_PATH} \
    es_token.teacher_temperature=${TEACHER_TEMPERATURE} \
    es_token.teacher_batch_size=${TEACHER_BATCH_SIZE} \
    es_token.temperature=${TEMPERATURE} es_token.max_tokens=${MAX_RESP_LENGTH} \
    es_token.num_engines=${NUM_ENGINES} \
    es_token.num_iterations=${NUM_ITERATIONS} \
    es_token.eval_interval=${EVAL_INTERVAL} \
    es_token.heldout_probe_size=${HELDOUT_PROBE_SIZE} \
    es_token.gpu_fraction=${GPU_FRACTION} \
    es_token.distributed_executor_backend=${EXEC_BACKEND} \
    es_token.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    es_token.teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION} \
    model.path=${ACTOR_MODEL_PATH} \
    data.task_type=opd_math data.train_files=${TRAIN_DATASET} \
    data.val_files=${EVAL_DATASET} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=${ES_LOGGER} trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
