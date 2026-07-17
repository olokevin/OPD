#!/bin/bash
# repro_compress.sh — single-GPU (NO torchrun) repro of the FULL real config so the
# crash's Python traceback is visible (torchrun swallows it in the DDP runs). Uses the
# YAML as-is (cutoff 10240, calib_num_seqs 16, full dataset) + max_steps=2. salloc 1 node.
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/compress_sft/repro_compress.sh"
fi
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/sft
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
_C=/tmp/${USER}/compress_sft_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
cd "$OPD_REPO/LlamaFactory"
SM=${DATA_ROOT}/compress_sft/repro; rm -rf "$SM"; mkdir -p "$SM"
echo "########## single-GPU FULL-config repro (cutoff10240, calib16, full dataset) ##########"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/compress_train/qwen3_4b_nersc_fwd_r0.6_sft.yaml \
  output_dir="$SM" run_name=repro max_steps=2 max_samples=2000 \
  per_device_train_batch_size=1 gradient_accumulation_steps=1 gradient_checkpointing=false \
  logging_steps=1 save_steps=1000 eval_steps=1000 val_size=0.002 report_to=none plot_loss=false
echo "########## repro rc=$? ##########"
