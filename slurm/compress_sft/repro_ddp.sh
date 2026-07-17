#!/bin/bash
# repro_ddp.sh — 4-GPU DDP repro to test the rank-serialized compress fix. salloc 1 node.
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/compress_sft/repro_ddp.sh"
fi
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/sft
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=hsn TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
_C=/tmp/${USER}/compress_sft_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
cd "$OPD_REPO/LlamaFactory"
SM=${DATA_ROOT}/compress_sft/repro_ddp; rm -rf "$SM"; mkdir -p "$SM"
echo "########## 4-GPU DDP repro (serialized-compress fix test) ##########"
FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 \
  llamafactory-cli train examples/compress_train/qwen3_4b_nersc_fwd_r0.6_sft.yaml \
  output_dir="$SM" run_name=repro_ddp max_steps=3 max_samples=2000 \
  per_device_train_batch_size=2 gradient_accumulation_steps=1 gradient_checkpointing=false \
  logging_steps=1 save_steps=1000 eval_steps=1000 val_size=0.01 report_to=none plot_loss=false
echo "########## repro_ddp rc=$? ##########"
