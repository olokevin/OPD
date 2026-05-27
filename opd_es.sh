#!/bin/bash
#SBATCH --job-name=opd_es
#SBATCH --output=logs/opd_es_output_%j.log
#SBATCH --error=logs/opd_es_error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

# Local-run fallback (matches on_policy_distillation.sh / grpo.sh style)
if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opd_es_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

ray stop --force
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ===================== ES hyperparameters =====================
export SIGMA=${SIGMA:-0.001}                          # noise scale
export ALPHA=${ALPHA:-0.0005}                         # ES learning rate
export POPULATION_SIZE=${POPULATION_SIZE:-30}         # perturbations per iteration
export NUM_ITERATIONS=${NUM_ITERATIONS:-200}          # total ES iterations
export ES_TEMPERATURE=${ES_TEMPERATURE:-0.0}          # rollout sampling temperature (0.0 = greedy)
export MAX_TOKENS=${MAX_TOKENS:-1024}                 # per-response token budget
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}             # eval every N iterations
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
export GLOBAL_SEED=${GLOBAL_SEED:-42}

# ===================== Hardware =====================
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
# num_engines defaults to one engine per GPU (TP=1 in RandOpt's ES code)
export NUM_ENGINES=${NUM_ENGINES:-${N_GPUS_PER_NODE}}

# ===================== Model & data =====================
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/AIME24/test.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-200}    # ES uses a small fixed train set per RandOpt convention
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}         # -1 = all

# ===================== Logging / checkpoints =====================
export PROJECT_PATH=${PROJECT_PATH:-/data/yequan/compress_train/OPD}
export PROJECT_NAME=${PROJECT_NAME:-OPD-ES}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-es_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_sigma_${SIGMA}_alpha_${ALPHA}_pop_${POPULATION_SIZE}_$(date +%Y-%m-%d_%H-%M-%S)}
export SAVE_DIR=${SAVE_DIR:-${PROJECT_PATH}/checkpoint/${EXPERIMENT_NAME}}
export ES_LOGGER=${ES_LOGGER:-'["console","wandb"]'}
export SAVE_FREQ=${SAVE_FREQ:-100}

mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_es \
    es.sigma=${SIGMA} \
    es.alpha=${ALPHA} \
    es.population_size=${POPULATION_SIZE} \
    es.num_engines=${NUM_ENGINES} \
    es.num_iterations=${NUM_ITERATIONS} \
    es.precision=bfloat16 \
    es.max_tokens=${MAX_TOKENS} \
    es.temperature=${ES_TEMPERATURE} \
    es.eval_interval=${EVAL_INTERVAL} \
    es.eval_batch_size=${EVAL_BATCH_SIZE} \
    es.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    es.global_seed=${GLOBAL_SEED} \
    es.verbose=false \
    es.worker_extension_cls='verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension' \
    model.path=${ACTOR_MODEL_PATH} \
    data.task_type=opd_math \
    data.train_files=${TRAIN_DATASET} \
    data.val_files=${EVAL_DATASET} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=${ES_LOGGER} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ}
