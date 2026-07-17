#!/bin/bash
# probe_mem.sh — measure peak GPU memory at the REAL cutoff_len=10240 to pick the
# largest per_device_train_batch_size that fits 80GB with gradient_checkpointing=false.
# Single-GPU (per-rank memory ~= DDP per-rank memory). Runs a few steps per config.
# salloc this on 1 node. Tests per_device in {2,1}; OOM on a config -> too big.
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/compress_sft/probe_mem.sh"
fi
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/sft
export PYTHONPATH="${OPD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=true
_C=/tmp/${USER}/compress_sft_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
cd "$OPD_REPO/LlamaFactory"
YAML=examples/compress_train/qwen3_4b_nersc_fwd_r0.6_sft.yaml

run_cfg() {
  local pdb=$1
  local out=${DATA_ROOT}/compress_sft/probe/pdb${pdb}; rm -rf "$out"; mkdir -p "$out"
  echo "########## PROBE per_device=$pdb cutoff=10240 gc=false ##########"
  # background peak-mem sampler on GPU0
  ( max=0; for k in $(seq 1 600); do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
      [ -n "$m" ] && [ "$m" -gt "$max" ] && max=$m && echo "PEAK_MIB pdb=$pdb $max"
      grep -q "PROBE_DONE_$pdb" "$out/done" 2>/dev/null && break; sleep 3
    done ) &
  local SP=$!
  CUDA_VISIBLE_DEVICES=0 llamafactory-cli train "$YAML" \
    output_dir="$out" run_name=probe_pdb$pdb max_steps=3 \
    calib_num_seqs=8 max_samples=64 cutoff_len=10240 \
    per_device_train_batch_size=$pdb gradient_accumulation_steps=1 \
    gradient_checkpointing=false logging_steps=1 save_steps=1000 \
    eval_steps=1000 val_size=0.02 report_to=none plot_loss=false \
    && echo "RESULT pdb=$pdb FIT" || echo "RESULT pdb=$pdb OOM_OR_FAIL"
  echo "PROBE_DONE_$pdb" > "$out/done"; kill $SP 2>/dev/null; wait $SP 2>/dev/null
}

for pdb in ${PROBE_PDBS:-2 1}; do run_cfg "$pdb"; done
echo "########## PROBE COMPLETE ##########"
