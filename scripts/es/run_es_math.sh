#!/bin/bash
# ES fine-tuning of Qwen2.5-Math-7B on MATH (lvl 3-5), evaluated on MATH-500.
#
# Reproduces the setup of "Evolution Strategies at Scale" §4.3 / Appendix A.6:
#   Qwen-Math template, binary \boxed{} correctness reward (OatZero grader),
#   max 3000 response tokens, sigma=0.001, alpha=sigma/2, N=30, greedy decoding.
#
# PERTURB_MODE selects the run:
#   dense     run 1  paper ES baseline (all parameters)
#   zoact     run 2  ZO-Act r=1 activation-informed rank-1 subspace
#   insparse  run 3  top-magnitude input-channel sparsity
#   fura      run 4  full-rank BTT, small core only
#   iso       run 5  ISO fixed-spectrum ES (arXiv:2607.19331), full-matrix frames
#   isobtt    run 6  ISO on the block-wise SVD -- frozen per-block spectrum, the
#                    trained core is a small orthogonal R_j in O(b)
#
# Usage:  DEVICE=1 PERTURB_MODE=dense bash scripts/es/run_es_math.sh

set -x
set -euo pipefail

REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# FlashInfer's sampler JIT hits a gcc-11 + nvcc include_next bug on this host
# (`-isystem /usr/include` demotes /usr/include past /usr/include/c++/11/cmath,
# breaking `#include_next <math.h>`). Fall back to PyTorch-native sampling.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${DEVICE:-1}

# ---------------- ES hyperparameters (paper values) ----------------
PERTURB_MODE=${PERTURB_MODE:-dense}
case "$PERTURB_MODE" in
  iso|isobtt)
    # For the ISO modes the perturbation is multiplicative (W <- C_L W C_R^T with
    # C orthogonal), so sigma is defined as the *relative weight-space footprint*
    # ||dW||_F/||W||_F, not an absolute noise std.  Match the dense-ES footprint
    # (5.0e-2, measured); the paper's nominal sigma=1e-3 would put the ISO modes
    # below the 1.6e-3 bf16 rollout floor.  alpha = sigma/2 then reproduces the
    # dense per-iteration motion alpha/sqrt(N) exactly.
    SIGMA=${SIGMA:-0.05}
    ALPHA=${ALPHA:-0.025} ;;
  *)
    SIGMA=${SIGMA:-0.001}
    ALPHA=${ALPHA:-0.0005} ;;      # paper: alpha = sigma / 2
esac
POPULATION_SIZE=${POPULATION_SIZE:-30}
NUM_ITERATIONS=${NUM_ITERATIONS:-150}
MAX_TOKENS=${MAX_TOKENS:-1536}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-0}   # 0 = fixed batch (old); >0 = resample per iteration
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-3000}
TEMPERATURE=${TEMPERATURE:-0.0}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-500}
GLOBAL_SEED=${GLOBAL_SEED:-42}

# ---------------- structured-mode knobs ----------------
SUBSPACE_RANK=${SUBSPACE_RANK:-1}
INSPARSE_DENSITY=${INSPARSE_DENSITY:-0.01}
CALIB_PATH=${CALIB_PATH:-${REPO}/datasets/es_math/calib_qwen2p5_math_7b.pt}
ISO_BLOCK_SIZE=${ISO_BLOCK_SIZE:-128}   # skew-generator block size; cost is O(b*|W|)
ISO_PERM=${ISO_PERM:-true}              # re-draw the block basis every seed

# ---------------- hardware / model / data ----------------
NUM_ENGINES=${NUM_ENGINES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
# dense/iso keep an fp32 master (2x the bf16 model) resident outside vLLM's budget;
# the structured modes keep a bf16 base/basis instead (1x).
if [ "$PERTURB_MODE" = "dense" ] || [ "$PERTURB_MODE" = "iso" ]; then
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
else
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
fi

MODEL=${MODEL:-Qwen/Qwen2.5-Math-7B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
TRAIN_FILE=${TRAIN_FILE:-${REPO}/datasets/es_math/math_lv3to5_qwenmath_train.parquet}
EVAL_FILE=${EVAL_FILE:-${REPO}/datasets/es_math/math500_qwenmath_test.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-64}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

# ---------------- logging ----------------
PROJECT_NAME=${PROJECT_NAME:-ES-q2p5-7b}
case "$PERTURB_MODE" in
  dense)    TAG="es-dense-full" ;;
  zoact)    TAG="zoact-r${SUBSPACE_RANK}" ;;
  insparse) TAG="insparse-d${INSPARSE_DENSITY}" ;;
  fura)     TAG="fura-btt-smallcore" ;;
  iso)      TAG="iso-fixedspec-b${ISO_BLOCK_SIZE}" ;;
  isobtt)   TAG="isobtt-fixedspec-smallcore" ;;
  *)        TAG="$PERTURB_MODE" ;;
esac
if [ "${TRAIN_BATCH_SIZE}" != "0" ]; then _BTAG="rs${TRAIN_BATCH_SIZE}"; else _BTAG="b${TRAIN_MAX_SAMPLES}"; fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${TAG}_math-lv3to5-${_BTAG}_sig${SIGMA}_a${ALPHA}_N${POPULATION_SIZE}}
SAVE_DIR=${SAVE_DIR:-/data/yequan/es/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${REPO}/logs/es}
mkdir -p "$SAVE_DIR" "$LOG_DIR"

CALIB_ARG="null"
if [ "$PERTURB_MODE" = "zoact" ] || [ "$PERTURB_MODE" = "insparse" ]; then
  CALIB_ARG="$CALIB_PATH"
fi

# NOTE: do NOT `ray stop --force` here -- it is host-global and would kill a
# concurrent ES run on another GPU. Each job starts its own local Ray instance
# in a private temp dir (see main_es.run_es).

python3 -m verl.trainer.main_es \
    es.perturb_mode="${PERTURB_MODE}" \
    es.calib_path="${CALIB_ARG}" \
    es.subspace_rank=${SUBSPACE_RANK} \
    es.insparse_density=${INSPARSE_DENSITY} \
    es.iso_block_size=${ISO_BLOCK_SIZE} \
    es.iso_perm=${ISO_PERM} \
    es.sigma=${SIGMA} \
    es.alpha=${ALPHA} \
    es.population_size=${POPULATION_SIZE} \
    es.num_engines=${NUM_ENGINES} \
    es.num_iterations=${NUM_ITERATIONS} \
    es.precision=bfloat16 \
    es.max_tokens=${MAX_TOKENS} \
    es.train_batch_size=${TRAIN_BATCH_SIZE} \
    es.eval_max_tokens=${EVAL_MAX_TOKENS} \
    es.max_model_len=${MAX_MODEL_LEN} \
    es.temperature=${TEMPERATURE} \
    es.eval_interval=${EVAL_INTERVAL} \
    es.eval_batch_size=${EVAL_BATCH_SIZE} \
    es.eval_before_train=true \
    es.save_best_coef=true \
    es.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    es.global_seed=${GLOBAL_SEED} \
    es.verbose=false \
    es.worker_extension_cls='verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension' \
    model.path=${MODEL} \
    data.task_type=qwen_math \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${EVAL_FILE} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger="${LOGGER:-[console,wandb]}" \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=0
