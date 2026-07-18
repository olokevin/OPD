#!/bin/bash
# opd_2node_rayenv.sh — environment that every Ray daemon (and therefore every
# actor it spawns: TaskRunner + WorkerDict) must have. Ray actors inherit the
# RAYLET's env, NOT the driver's, so all runtime/system env has to live here and
# be sourced by each `ray start` srun step (and by the driver, for consistency).
#
# Two hazards on NERSC this guards against:
#   * HF / JIT file locks on GPFS/Lustre fail with OSError 524 (ENOTSUPP) -> keep
#     the HF cache on $PSCRATCH (flock OK) and all JIT caches on node-local /tmp.
#   * flashinfer's sampler JIT-compiles + flocks on first use -> disable it.
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate "${DATA_ROOT}/envs/verl"

# Make the `compress` package (src/compress, used by the BlockTT/FurA adapter)
# importable. on_policy_distillation.sh sets this for the driver, but Ray ACTORS
# inherit the raylet's env, so it must be set here too (else: blocktt apply ->
# "ModuleNotFoundError: No module named 'compress'").
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# HuggingFace: offline, cache on $PSCRATCH.
export HF_HOME=${DATA_ROOT}/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

# wandb: offline, ONE pinned run resumed across every relaunch segment. (The
# trainer loop + wandb logging run inside the TaskRunner ACTOR, so these must be
# on the raylet, not just the driver.)
export WANDB_MODE=offline
export WANDB_PROJECT=nersc_opd_qwen4b_1p7b
export WANDB_DIR=${DATA_ROOT}/wandb
export WANDB_RUN_ID=${WANDB_RUN_ID:-nersc_opd_qwen4b_1p7b_2node}
export WANDB_RESUME=allow
mkdir -p "${WANDB_DIR}" 2>/dev/null

# vLLM: disable flashinfer sampler (its JIT flocks on GPFS -> 524), use FLASH_ATTN.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1 RAY_USAGE_STATS_ENABLED=0 SWANLAB_MODE=local
export TOKENIZERS_PARALLELISM=true

# NCCL / Slingshot (the NCCL comms happen in the actors -> must be on the raylet).
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_TIMEOUT=7200 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export OMP_NUM_THREADS=8

# JIT / framework caches on node-local /tmp (flock works there, unlike GPFS).
_C=/tmp/${USER}/zo_cache
export TRITON_CACHE_DIR=${_C}/triton TORCHINDUCTOR_CACHE_DIR=${_C}/inductor
export XDG_CACHE_HOME=${_C}/xdg OUTLINES_CACHE_DIR=${_C}/outlines
export FLASHINFER_WORKSPACE_DIR=${_C}/flashinfer FLASHINFER_CACHE_DIR=${_C}/flashinfer
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$OUTLINES_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null
