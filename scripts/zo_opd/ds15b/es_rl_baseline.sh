#!/bin/bash
# es_rl_baseline.sh -- sequence-level ES with a PLAIN RL-style reward (task accuracy,
# no teacher, no OPD) on the ds15b student, as the "what does reward-only ES buy" baseline.
#
#   model    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B      (no teacher involved)
#   data     DAPO-Math-17k (resampled batch per iteration), reward = ttrl_math correctness
#   ES       dense (every parameter), N=10 antithetic-free OpenAI-ES, greedy rollouts
#            sigma 1e-3 / alpha 2.89e-4 (the ES-math study's dense N=10 config;
#            watch train/reward_std -- the working band there was 0.040-0.055)
#   eval     MATH-500 greedy n=1 @ 7168 tokens every 10 iterations
#
#   TRAIN_GPU=6 bash scripts/zo_opd/ds15b/es_rl_baseline.sh
#   SMOKE=1 TRAIN_GPU=6 bash scripts/zo_opd/ds15b/es_rl_baseline.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$REPO_ROOT/verl
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-6}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export ES_KEEP_CUDA_VISIBLE=1

MODEL=${ACTOR_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TRAIN_FILE=${TRAIN_DATASET:-${REPO_ROOT}/datasets/DAPO-Math-17k/DAPO-Math.parquet}
EVAL_FILE=${EVAL_DATASET:-${REPO_ROOT}/datasets/test_data/MATH-500/test.parquet}

SIGMA=${SIGMA:-1e-3}
ALPHA=${ALPHA:-2.89e-4}
POPULATION_SIZE=${POPULATION_SIZE:-10}
NUM_ITERATIONS=${NUM_ITERATIONS:-150}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}      # resampled per iteration
TEMPERATURE=${TEMPERATURE:-0.0}               # greedy rollouts (ES-math protocol)
MAX_TOKENS=${MAX_TOKENS:-7168}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8448}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-7168}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-500}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

PROJECT_NAME=${PROJECT_NAME:-es_opd_JustRL_1p5b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_es-rl_dense_N${POPULATION_SIZE}_sig${SIGMA}_a${ALPHA}}
SAVE_DIR=${SAVE_DIR:-/data/yequan/opd/es_rl/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/ds15b/es_rl}
LOGGER=${LOGGER:-'[console,wandb]'}

if [ "${SMOKE:-0}" = "1" ]; then
  NUM_ITERATIONS=2; TRAIN_BATCH_SIZE=8; MAX_TOKENS=512; EVAL_INTERVAL=0
  export WANDB_MODE=disabled; EXPERIMENT_NAME=smoke_${EXPERIMENT_NAME}; LOG_DIR=${LOG_DIR}/smoke
fi

_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
if [ "${_used:-0}" -gt 2000 ]; then echo "GPU $CUDA_VISIBLE_DEVICES busy (${_used} MiB) -- refusing"; exit 1; fi

mkdir -p "$SAVE_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/es_rl_$(date +%Y%m%d_%H%M%S).log"
echo "=== ES-RL baseline (dense, reward fitness, no teacher) ==="
echo "  model $MODEL  gpu $CUDA_VISIBLE_DEVICES  N=$POPULATION_SIZE sigma=$SIGMA alpha=$ALPHA"
echo "  batch $TRAIN_BATCH_SIZE resampled, T=$TEMPERATURE, max_tokens $MAX_TOKENS  log $LOG_FILE"

python3 -m verl.trainer.main_es \
    es.fitness=reward \
    es.perturb_mode=dense \
    es.calib_path=null \
    es.sigma=${SIGMA} \
    es.alpha=${ALPHA} \
    es.population_size=${POPULATION_SIZE} \
    es.num_engines=1 \
    es.num_iterations=${NUM_ITERATIONS} \
    es.precision=bfloat16 \
    es.max_tokens=${MAX_TOKENS} \
    es.temperature=${TEMPERATURE} \
    es.train_batch_size=${TRAIN_BATCH_SIZE} \
    es.max_model_len=${MAX_MODEL_LEN} \
    es.eval_interval=${EVAL_INTERVAL} \
    es.eval_batch_size=${VAL_MAX_SAMPLES} \
    es.eval_max_tokens=${EVAL_MAX_TOKENS} \
    es.eval_before_train=true \
    es.save_best_coef=true \
    es.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    es.global_seed=${GLOBAL_SEED:-42} \
    es.verbose=false \
    es.worker_extension_cls='verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension' \
    model.path=${MODEL} \
    data.task_type=opd_math \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${EVAL_FILE} \
    data.train_max_samples=-1 \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger="${LOGGER}" \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.save_freq=0 2>&1 | tee "$LOG_FILE"
