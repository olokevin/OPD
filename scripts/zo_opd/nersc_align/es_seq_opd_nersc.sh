#!/bin/bash
# es_seq_opd_nersc.sh -- SEQUENCE-LEVEL ES for OPD, on the same student/teacher/data
# as bp_opd_nersc.sh and es_opd_nersc.sh so all three are directly comparable.
#
# How this differs from es_token (and why):
#   es_token perturbs the weights only at the CURRENT decode step and the rails read
#   the CLEAN row's KV, so each token's probe sees only that token's own gradient.
#   Measured consequence: cos(dW, target) ~ sqrt(N/D) ~ 7e-4, and the token count does
#   NOT accumulate (docs/results/zo_opd.md 12.5).
#   Here ONE fixed Gaussian perturbation of EVERY parameter is applied for the WHOLE
#   rollout, so the perturbation propagates through the trajectory exactly as a real
#   weight change would, and the fitness is the OPD loss itself:
#       fitness_n = -mean_t [ log pi_n(y_t) - log q(y_t) ],   y ~ pi_n
#   i.e. the reverse KL between the perturbed student's own rollout and the teacher.
#
# sigma / alpha are footprint-matched to the `dense` ES arm that gains +20 pp on
# MATH-500 (results/ES/es_results.md 6, 10.4):
#   perturbation ||dW||/||W|| = sigma / RMS(W).  Qwen3-1.7B RMS(W) = 6.15e-2, so
#   sigma = 3.0e-3 -> 4.9e-2, matching dense's measured 5.0e-2.
#   alpha = sigma/2 then reproduces dense's per-iteration motion alpha/sqrt(N)/RMS(W)
#   = 1.5e-3 * 16.3 / 5.48 = 4.5e-3, vs dense's 4.6e-3.
# MEASURED (es_seq_sigma_probe.sh, docs/results/zo_opd.md 13.2): unperturbed KL 0.2838;
#   sigma 1e-3/2e-3/3e-3/6e-3 -> KL 0.312/0.386/0.642/4.945, spread 0.011/0.040/0.088/7.44.
#   6e-3 is off a cliff; 3e-3 keeps 2x margin. Fallback if unstable: SIGMA=2e-3 ALPHA=1e-3.
#
#   TRAIN_GPUS=6 bash scripts/zo_opd/nersc_align/es_seq_opd_nersc.sh   (ONE gpu per job)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH=$(pwd)/verl
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-6}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# uni executor => no child ray worker, so vLLM must keep the launcher's pin or it
# lands on physical GPU0 (see ESNcclLLM.__init__).
export ES_KEEP_CUDA_VISIBLE=1

MODEL=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
TEACHER=${TEACHER_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
TRAIN_FILE=${TRAIN_DATASET:-${REPO_ROOT}/datasets/dapo-math-17k.parquet}
EVAL_FILE=${EVAL_DATASET:-${REPO_ROOT}/datasets/test_data/MATH-500/test.parquet}

SIGMA=${SIGMA:-3.0e-3}
ALPHA=${ALPHA:-1.5e-3}
POPULATION_SIZE=${POPULATION_SIZE:-30}
# NUM_ENGINES MUST BE 1. ES_KEEP_CUDA_VISIBLE=1 (needed so the `uni` executor puts
# vLLM on the pinned card rather than physical GPU0) makes every engine actor
# inherit the SAME device list, so with 2 engines both land on the first GPU and
# _init_inter_engine_group dies with "NCCL error: invalid usage". Multi-engine
# needs per-actor device assignment, which is not wired up. To use two cards, run
# two SEPARATE single-engine jobs (e.g. one sigma per GPU).
NUM_ENGINES=${NUM_ENGINES:-1}
NUM_ITERATIONS=${NUM_ITERATIONS:-150}
PERTURB_MODE=${PERTURB_MODE:-dense}       # all parameters, fp32 master

# The reverse-KL estimate is unbiased only for rollouts SAMPLED from pi_n.
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}                       # parity with BP's rollout and every eval
MAX_TOKENS=${MAX_TOKENS:-2048}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}   # resampled per iteration from the 17k pool
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-500}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-3072}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-500}
GLOBAL_SEED=${GLOBAL_SEED:-42}

# dense keeps an fp32 master (2x the bf16 model) OUTSIDE vLLM's budget, and the
# 4B teacher is co-located, so the student engine gets well under half the card.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.40}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.18}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-$((1024 + MAX_TOKENS))}
TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-24}
ENGINE_GPU_FRACTION=${ENGINE_GPU_FRACTION:-0.5}   # leaves bundle room for the teacher

PROJECT_NAME=${PROJECT_NAME:-nersc_opd_qwen4b_1p7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-singlegpu_esseq_opd_r${MAX_TOKENS}_sig${SIGMA}_a${ALPHA}_N${POPULATION_SIZE}}
SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/esseq}
mkdir -p "$SAVE_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/esseq_opd_$(date +%Y%m%d_%H%M%S).log"

echo "=== sequence-level ES-OPD (all parameters, reverse-KL fitness) ==="
echo "  student $MODEL   teacher $TEACHER"
echo "  gpus $CUDA_VISIBLE_DEVICES engines $NUM_ENGINES  N=$POPULATION_SIZE"
echo "  sigma=$SIGMA alpha=$ALPHA  T=$TEMPERATURE top_p=$TOP_P  max_tokens=$MAX_TOKENS"
echo "  log $LOG_FILE"

python3 -m verl.trainer.main_es \
    es.fitness=opd_kl \
    es.teacher_model_path="${TEACHER}" \
    es.teacher_temperature=1.0 \
    es.teacher_batch_size=${TEACHER_BATCH_SIZE} \
    es.teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION} \
    es.teacher_max_model_len=${TEACHER_MAX_MODEL_LEN} \
    es.teacher_gpu_fraction=0.01 \
    es.engine_gpu_fraction=${ENGINE_GPU_FRACTION} \
    es.distributed_executor_backend=uni \
    es.perturb_mode="${PERTURB_MODE}" \
    es.calib_path=null \
    es.sigma=${SIGMA} \
    es.alpha=${ALPHA} \
    es.population_size=${POPULATION_SIZE} \
    es.num_engines=${NUM_ENGINES} \
    es.num_iterations=${NUM_ITERATIONS} \
    es.precision=bfloat16 \
    es.max_tokens=${MAX_TOKENS} \
    es.temperature=${TEMPERATURE} \
    es.top_p=${TOP_P} \
    es.train_batch_size=${TRAIN_BATCH_SIZE} \
    es.eval_max_tokens=${EVAL_MAX_TOKENS} \
    es.max_model_len=${MAX_MODEL_LEN} \
    es.eval_interval=${EVAL_INTERVAL} \
    es.eval_batch_size=${EVAL_BATCH_SIZE} \
    es.eval_before_train=true \
    es.save_best_coef=true \
    es.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    es.global_seed=${GLOBAL_SEED} \
    es.verbose=false \
    es.worker_extension_cls='verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension' \
    model.path=${MODEL} \
    data.task_type=opd_math \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-false} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${EVAL_FILE} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES:--1} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger="${LOGGER:-[console,wandb]}" \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${NUM_ENGINES} \
    trainer.nnodes=1 \
    trainer.save_freq=0 2>&1 | tee "$LOG_FILE"
