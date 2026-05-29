#!/bin/bash
#SBATCH --job-name=opd_np
#SBATCH --output=logs/opd_np_output_%j.log
#SBATCH --error=logs/opd_np_error_%j.log
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opd_np_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

ray stop --force
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# FlashAttention v1 verified for the shared-prefix multi-query decode; SDPA was not.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
# Some boxes' flashinfer JIT can't find math.h; the NP path doesn't need it.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# vLLM 0.11.0 V1 multiprocessing msgpack-serializes tensors across the EngineCore
# boundary, breaking collective_rpc tensor returns. Run single-process (matches ES).
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- NP knobs ----
export SIGMA=${SIGMA:-0.01}
export N_SAMPLE=${N_SAMPLE:-8}
export N_ROLLOUT=${N_ROLLOUT:-8}
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export PERTURB_GRANULARITY=${PERTURB_GRANULARITY:-token}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-grpo}
export GRAD_ESTIMATE_SEQUENCE=${GRAD_ESTIMATE_SEQUENCE:-grpo}
export EN_LAYERWISE=${EN_LAYERWISE:-true}
export LR=${LR:-1e-4}
export TOKEN_AGG=${TOKEN_AGG:-sum}
export LOSS_TYPE=${LOSS_TYPE:-opd}
# newline-separated regex list -> Hydra list; default = all decoder mlp.down_proj
export PERTURB_RULES=${PERTURB_RULES:-'^model\.layers\.\d+\.mlp\.down_proj$'}

# ---- teacher / OPD ----
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-model/Qwen3-4B-Non-Thinking-RL-Math}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-256}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}

# ---- hardware ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NUM_ENGINES=${NUM_ENGINES:-4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}

# ---- model & data ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/AIME24/test.parquet}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-200}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export NUM_ITERATIONS=${NUM_ITERATIONS:-200}

# ---- logging ----
export PROJECT_NAME=${PROJECT_NAME:-OPD-NP}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-np_${ACTOR_MODEL_NAME}_sigma_${SIGMA}_n_${N_SAMPLE}_$(date +%Y-%m-%d_%H-%M-%S)}
export SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
export NP_LOGGER=${NP_LOGGER:-'["console","wandb"]'}
mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_np --config-name np_trainer \
    np.sigma=${SIGMA} np.n_sample=${N_SAMPLE} np.n_rollout=${N_ROLLOUT} \
    np.sample_method=${SAMPLE_METHOD} np.perturb_granularity=${PERTURB_GRANULARITY} \
    np.grad_estimate_sample=${GRAD_ESTIMATE_SAMPLE} \
    np.grad_estimate_sequence=${GRAD_ESTIMATE_SEQUENCE} \
    np.en_layerwise_perturbation=${EN_LAYERWISE} np.lr=${LR} np.token_agg=${TOKEN_AGG} \
    np.loss_type=${LOSS_TYPE} 'np.perturb_rules=["'"${PERTURB_RULES}"'"]' \
    np.teacher_model_path=${TEACHER_MODEL_PATH} np.log_prob_top_k=${LOG_PROB_TOP_K} \
    np.top_k_strategy=${TOP_K_STRATEGY} np.teacher_temperature=${TEACHER_TEMPERATURE} \
    np.num_engines=${NUM_ENGINES} np.num_iterations=${NUM_ITERATIONS} \
    np.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    np.worker_extension_cls='verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension' \
    model.path=${ACTOR_MODEL_PATH} \
    data.task_type=opd_math data.train_files=${TRAIN_DATASET} data.val_files=${EVAL_DATASET} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=${NP_LOGGER} trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
