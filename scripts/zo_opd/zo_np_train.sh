#!/bin/bash
# ZO-NP (zeroth-order node-perturbation) TRAINING — corrected scaling.
#
# Scaling fixes applied (vs the old opd_np.sh defaults):
#   - grpo mode now uses (L_q - mean)/sigma  (1/sigma finite-difference scale restored)
#   - ANP 1/||u||^2 normalization OFF  (NORMALIZE_ANP=false)
# so delta_W lands on the true-gradient scale and a sane lr (~1e-6) is meaningful.
#
# One run = ONE GPU. Student (1.7B) + teacher (4B) are CO-LOCATED on that GPU via
# fractional placement groups (GPU_FRACTION=0.5, mem-util ~0.35 each).
#
# Rollout: greedy clean-token decode (TEMPERATURE=0), ONE rollout per prompt
# (N_ROLLOUT=1), n_sample=64 perturbed copies per decode step. BATCH_SIZE=64
# distinct prompts accumulated into ONE delta_W per update.
#
# Drive one LR per GPU, e.g.:
#   CUDA_VISIBLE_DEVICES=1 LR=3e-7 EXP=lr3e-7 bash scripts/zo_opd/zo_np_train.sh
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/zo_np_train_${EXP:-run}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

# NOTE: do NOT `ray stop --force` here -- it is GLOBAL and would tear down the
# OTHER concurrent single-GPU runs' isolated Ray sessions. Each run uses its own
# address="local" session (unique temp dir in main_np.py), so no global stop is needed.
# Ray's plasma AF_UNIX socket path must stay < 107 bytes; the sandbox TMPDIR
# (/tmp/claude-*/...) is too long, so pin a short temp dir for the ray session.
export TMPDIR=${ZO_TMPDIR:-/tmp}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
# ONE visible GPU per run. Caller sets this (1, 2, or 3).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
# uni (in-process) executor must NOT drop the GPU pin, else vLLM lands on GPU0.
export NP_KEEP_CUDA_VISIBLE=${NP_KEEP_CUDA_VISIBLE:-1}
# Stop Ray from managing CUDA_VISIBLE_DEVICES (a fractional-GPU PG would set it
# empty); the launcher's CUDA_VISIBLE_DEVICES=N pin then reaches the uni worker.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

# ---- NP knobs (CORRECTED scaling) ----
export SIGMA=${SIGMA:-0.01}
export N_SAMPLE=${N_SAMPLE:-64}                 # perturbed copies per decode step
export N_ROLLOUT=${N_ROLLOUT:-1}                # one rollout per prompt
export BATCH_SIZE=${BATCH_SIZE:-64}             # distinct prompts per update
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export PERTURB_GRANULARITY=${PERTURB_GRANULARITY:-token}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-grpo}   # (L_q-mean)/sigma now
export GRAD_ESTIMATE_SEQUENCE=${GRAD_ESTIMATE_SEQUENCE:-grpo}
export EN_LAYERWISE=${EN_LAYERWISE:-true}
export NORMALIZE_ANP=${NORMALIZE_ANP:-false}    # ANP 1/||u||^2 OFF
export TOKEN_AGG=${TOKEN_AGG:-mean}             # mean over tokens -> lr stable across resp-len
export LR=${LR:-1e-6}
export LOSS_TYPE=${LOSS_TYPE:-opd}
export PERTURB_RULES=${PERTURB_RULES:-'^model\.layers\.\d+\.mlp\.down_proj$'}

# ---- teacher / OPD (figure: top-K 16, student top-K, temp 1.0) ----
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}

# ---- decode (greedy clean trajectory) ----
export TEMPERATURE=${TEMPERATURE:-0.0}          # greedy clean-token selection (NP needs one clean traj)
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1024} # cap response steps for tractable per-step cost

# ---- hardware: single GPU, co-located teacher ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
export NUM_ENGINES=${NUM_ENGINES:-1}
export GPU_FRACTION=${GPU_FRACTION:-0.5}
export EXEC_BACKEND=${EXEC_BACKEND:-uni}        # in-process worker -> fractional/co-located GPU OK
# Student + teacher co-located on ONE physical GPU. vLLM's gpu_memory_utilization
# is a fraction of TOTAL device memory and is reserved eagerly, so the two MUST
# sum well under 1.0 (0.30+0.30=0.60 leaves headroom; 0.45+0.45 fails -- the 2nd
# engine sees almost no free memory after the 1st reserves its fraction).
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
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

# ---- logging ----
export PROJECT_NAME=${PROJECT_NAME:-ZO-NP}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-zonp_${ACTOR_MODEL_NAME}_${EXP:-lr${LR}}_$(date +%m%d_%H%M)}
export SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
export NP_LOGGER=${NP_LOGGER:-'["console"]'}
mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_np --config-name np_trainer \
    np.sigma=${SIGMA} np.n_sample=${N_SAMPLE} np.n_rollout=${N_ROLLOUT} \
    np.batch_size=${BATCH_SIZE} \
    np.sample_method=${SAMPLE_METHOD} np.perturb_granularity=${PERTURB_GRANULARITY} \
    np.grad_estimate_sample=${GRAD_ESTIMATE_SAMPLE} \
    np.grad_estimate_sequence=${GRAD_ESTIMATE_SEQUENCE} \
    np.en_layerwise_perturbation=${EN_LAYERWISE} np.lr=${LR} np.token_agg=${TOKEN_AGG} \
    np.normalize_anp=${NORMALIZE_ANP} \
    np.loss_type=${LOSS_TYPE} 'np.perturb_rules=["'"${PERTURB_RULES}"'"]' \
    np.teacher_model_path=${TEACHER_MODEL_PATH} np.log_prob_top_k=${LOG_PROB_TOP_K} \
    np.top_k_strategy=${TOP_K_STRATEGY} np.teacher_temperature=${TEACHER_TEMPERATURE} \
    np.reward_weight_mode=${REWARD_WEIGHT_MODE} \
    np.temperature=${TEMPERATURE} np.max_tokens=${MAX_RESP_LENGTH} \
    np.num_engines=${NUM_ENGINES} np.num_iterations=${NUM_ITERATIONS} \
    np.eval_interval=${EVAL_INTERVAL} \
    np.gpu_fraction=${GPU_FRACTION} \
    np.distributed_executor_backend=${EXEC_BACKEND} \
    np.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    np.teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION} \
    np.worker_extension_cls='verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension' \
    model.path=${ACTOR_MODEL_PATH} \
    data.task_type=opd_math data.train_files=${TRAIN_DATASET} data.val_files=${EVAL_DATASET} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=${NP_LOGGER} trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
