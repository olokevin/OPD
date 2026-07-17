#!/bin/bash
# compress_sft_env.sh — runs on EACH node (one srun task/node) under the alloc.
# Activates the `sft` conda env, sets HF-offline + node-local JIT caches + NCCL
# (Slingshot), then runs `llamafactory-cli train`, which spawns torchrun with
# --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=4 to form the 16-way
# DDP group. Compression (svd_nystrom) runs in-process on every rank at model-init;
# DDP broadcasts rank-0's compressed weights, so all ranks align.
#
# Required env (exported by compress_sft_inside.sh):
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE FORCE_TORCHRUN
#   YAML OUTPUT_DIR RUN_NAME WANDB_RUN_ID
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
LF="$OPD_REPO/LlamaFactory"

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate "${DATA_ROOT}/envs/sft"

# compress package importable (LlamaFactory's compress_setup also self-adds it,
# but set it here for any subprocess/grader shellout).
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# HuggingFace: offline, cache on $PSCRATCH (flock OK; GPFS $HOME flock -> OSError 524).
export HF_HOME=${DATA_ROOT}/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=true
# Surface crash tracebacks: torchrun doesn't capture them (main isn't @record),
# and buffered stderr is lost when a worker exits — so flush unbuffered + dump on
# fatal signals. (Diagnostic for the silent exit-1 during svd compress.)
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

# wandb: offline, ONE pinned run resumed across every 4h segment.
export WANDB_MODE=offline
export WANDB_PROJECT=nersc_compress_sft_qwen4b
export WANDB_DIR=${DATA_ROOT}/compress_sft/wandb
export WANDB_RUN_ID=${WANDB_RUN_ID:-compress_sft_qwen3_4b}
export WANDB_RESUME=allow
mkdir -p "$WANDB_DIR" 2>/dev/null

# NCCL / Slingshot (raw torchrun across nodes).
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072
# In-process svd compress runs for minutes with NO collectives; the NCCL watchdog's
# async-error teardown was aborting a rank during compress on multi-node (exit 1, no
# traceback; 1-node smoke was fine). Disable blocking-wait + async-error teardown and
# raise the heartbeat/timeout so the long collective-free compress doesn't get killed.
export TORCH_NCCL_BLOCKING_WAIT=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 TORCH_NCCL_TIMEOUT=7200
export NCCL_TIMEOUT=7200 TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export OMP_NUM_THREADS=8

# JIT / framework caches on node-local /tmp (flock works there, unlike GPFS).
_C=/tmp/${USER}/compress_sft_cache
export TRITON_CACHE_DIR=${_C}/triton TORCHINDUCTOR_CACHE_DIR=${_C}/inductor
export XDG_CACHE_HOME=${_C}/xdg
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" 2>/dev/null

YAML=${YAML:?set YAML}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
RUN_NAME=${RUN_NAME:-compress_sft}
mkdir -p "$OUTPUT_DIR"

# Resume: if a factored checkpoint-N dir exists, resume from the latest one. All
# nodes see the same shared FS, so they compute the same path.
RESUME_ARGS=()
LATEST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' \
         | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST" ]; then
  echo "[env node=$NODE_RANK] resuming from $LATEST"
  RESUME_ARGS=(resume_from_checkpoint="$LATEST")
else
  echo "[env node=$NODE_RANK] fresh start (no checkpoint in $OUTPUT_DIR)"
fi

cd "$LF"
echo "[env node=$NODE_RANK] NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT yaml=$YAML"
exec llamafactory-cli train "examples/compress_train/$YAML" \
  output_dir="$OUTPUT_DIR" run_name="$RUN_NAME" "${RESUME_ARGS[@]}"
