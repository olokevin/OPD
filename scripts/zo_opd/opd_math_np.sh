#!/bin/bash
# opd_math_np.sh — NP (node-perturbation) OPD-math training, V2 graphed decode.
#
# Trainer:  verl.trainer.main_np  (the zeroth-order NP-OPD trainer, V2 buffer-in-graph).
# Decode:   decode_mode=graphed use_cuda_graph=true  (CUDA-graphed 1+N rails; §9 of
#           docs/wiki/zo_np_trainer.md — graphed ≡ V1, GPU-parity-verified).
# Regime:   GREEDY clean trajectory (TEMPERATURE=0), ONE rollout per prompt
#           (N_ROLLOUT=1), BATCH_SIZE=64 distinct prompts -> 64 clean responses
#           accumulated into ONE delta_W per update. n_sample=8 perturbation rails
#           per decode step (the §9 sweet spot; +19% wall-time, near-free).
# Model/teacher/data:  aligned to scripts/opd/math/full.sh (Qwen3-1.7B student,
#           Keven16 Qwen3-4B-Non-Thinking-RL-Math-Step500 teacher, MATH-500 eval).
# Estimator: grad_estimate_sample=grpo = (L_q - mean)/sigma  (the /std is dropped;
#           this form trained cleanly+monotonically at lr=3e-2, docs/results/zo_opd.md §6).
#
# One run = ONE GPU; student (1.7B) + teacher (4B) CO-LOCATED via fractional PGs.
# Drive one LR per GPU:
#   CUDA_VISIBLE_DEVICES=4 LR=1e-2 EXP=lr1e-2 bash scripts/zo_opd/opd_math_np.sh
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opd_math_np_${EXP:-run}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

# Per-run isolated Ray session (address="local", unique temp dir in main_np.py),
# so do NOT global `ray stop`. Keep the AF_UNIX plasma socket path short.
export TMPDIR=${ZO_TMPDIR:-/tmp}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
# ONE visible GPU per run (caller sets 4/5/6). uni executor must keep the pin.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export NP_KEEP_CUDA_VISIBLE=${NP_KEEP_CUDA_VISIBLE:-1}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

# ---- V2 graphed decode driver ----
export DECODE_MODE=${DECODE_MODE:-graphed}
export USE_CUDA_GRAPH=${USE_CUDA_GRAPH:-true}

# ---- NP knobs (corrected scaling; (L_q-mean)/sigma) ----
export SIGMA=${SIGMA:-0.01}
export N_SAMPLE=${N_SAMPLE:-8}                  # perturbation rails per decode step (§9 sweet spot)
export N_ROLLOUT=${N_ROLLOUT:-1}                # one clean response per prompt
export BATCH_SIZE=${BATCH_SIZE:-64}             # 64 prompts -> 64 responses per update
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export PERTURB_GRANULARITY=${PERTURB_GRANULARITY:-token}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-grpo}   # (L_q-mean)/sigma
export GRAD_ESTIMATE_SEQUENCE=${GRAD_ESTIMATE_SEQUENCE:-grpo}
export EN_LAYERWISE=${EN_LAYERWISE:-true}
export NORMALIZE_ANP=${NORMALIZE_ANP:-false}
export TOKEN_AGG=${TOKEN_AGG:-mean}
export LR=${LR:-3e-2}                           # default = the clean-training point (§6)
export LOSS_TYPE=${LOSS_TYPE:-opd}
export PERTURB_RULES=${PERTURB_RULES:-'^model\.layers\.\d+\.mlp\.down_proj$'}

# ---- teacher / OPD (aligned to full.sh teacher) ----
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-256}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}

# ---- decode (greedy clean trajectory) ----
export TEMPERATURE=${TEMPERATURE:-0.0}          # greedy
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-1024}

# ---- hardware: single GPU, co-located teacher (fractional PG) ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
export NUM_ENGINES=${NUM_ENGINES:-1}
export GPU_FRACTION=${GPU_FRACTION:-0.5}
export EXEC_BACKEND=${EXEC_BACKEND:-uni}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.30}

# ---- model & data (aligned to full.sh) ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/MATH-500/test.parquet}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-200}
export NUM_ITERATIONS=${NUM_ITERATIONS:-150}
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}

# ---- logging (wandb project opd-qwen-math) ----
export PROJECT_NAME=${PROJECT_NAME:-opd-qwen-math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-np_graphed_${ACTOR_MODEL_NAME}_${EXP:-lr${LR}}_$(date +%m%d_%H%M)}
export SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
export NP_LOGGER=${NP_LOGGER:-'["console","wandb"]'}
mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_np --config-name np_trainer \
    np.decode_mode=${DECODE_MODE} np.use_cuda_graph=${USE_CUDA_GRAPH} \
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
