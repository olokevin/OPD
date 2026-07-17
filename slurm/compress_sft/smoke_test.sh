#!/bin/bash
# smoke_test.sh — validate the compress->SFT pipeline under an interactive alloc.
# salloc this script (runs on login node; re-execs onto the compute node via srun).
#   salloc -N1 --qos interactive --time 1:00:00 -C 'gpu&hbm80g' --gpus-per-node=4 \
#          --account m4788_g bash slurm/compress_sft/smoke_test.sh
# Phases: (1) single-GPU compress init + train + save factored+merged,
#         (2) RESUME from the saved checkpoint, (3) 4-GPU DDP short run.
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/compress_sft/smoke_test.sh"
fi

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/sft
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DEFAULT_CQ_SIZE=131072
_C=/tmp/${USER}/compress_sft_cache
export TRITON_CACHE_DIR=${_C}/triton TORCHINDUCTOR_CACHE_DIR=${_C}/inductor XDG_CACHE_HOME=${_C}/xdg
mkdir -p "$_C"
cd "$OPD_REPO/LlamaFactory"
echo "node=$(hostname) gpus=$(nvidia-smi -L | wc -l) py=$(which python)"

YAML=examples/compress_train/qwen3_4b_nersc_fwd_r0.6_sft.yaml
# fast smoke overrides (small calib + tiny data + few steps)
COMMON="calib_num_seqs=16 max_samples=256 cutoff_len=4096 \
 per_device_train_batch_size=1 gradient_accumulation_steps=1 gradient_checkpointing=false \
 logging_steps=1 save_steps=2 eval_steps=2 val_size=0.02 report_to=none plot_loss=false"

SM=${DATA_ROOT}/compress_sft/smoke/fwd; rm -rf "$SM"; mkdir -p "$SM"
echo "########## PHASE 1: single-GPU compress+train+save (max_steps=4) ##########"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train "$YAML" \
  output_dir="$SM" run_name=smoke_fwd max_steps=4 $COMMON || { echo "PHASE1 FAILED rc=$?"; exit 1; }
echo "--- PHASE1 output ---"; ls -la "$SM"

LATEST=$(ls -d "$SM"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)
echo "########## PHASE 2: RESUME from $LATEST (max_steps=6) ##########"
[ -n "$LATEST" ] || { echo "PHASE2 FAILED: no checkpoint to resume"; exit 1; }
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train "$YAML" \
  output_dir="$SM" run_name=smoke_fwd_resume max_steps=6 resume_from_checkpoint="$LATEST" $COMMON \
  || { echo "PHASE2 FAILED rc=$?"; exit 1; }
echo "--- PHASE2 output (expect final-merged) ---"; ls -la "$SM"

echo "########## PHASE 3: 4-GPU DDP (max_steps=3, per_device=1) ##########"
SM2=${DATA_ROOT}/compress_sft/smoke/fwd_ddp; rm -rf "$SM2"; mkdir -p "$SM2"
FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29577 \
  llamafactory-cli train "$YAML" output_dir="$SM2" run_name=smoke_ddp max_steps=3 $COMMON \
  || { echo "PHASE3 FAILED rc=$?"; exit 1; }
echo "--- PHASE3 output ---"; ls -la "$SM2"
echo "########## SMOKE OK ##########"
