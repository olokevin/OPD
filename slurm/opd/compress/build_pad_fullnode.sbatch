#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2:00:00
#SBATCH --account=m4788_g
#SBATCH --job-name=build_c128
# Build the combined-calibration compressed Qwen3-4B student for compress_opd on NERSC.
# stock-dense (skip_last_layers 0 + --save-stock-dense) so the verl actor can from_pretrained it.
set -uo pipefail
OPD_REPO=/global/u1/y/yequan/Project/OPD; DATA_ROOT=/pscratch/sd/y/yequan/opd; cd "$OPD_REPO"
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/verl
export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
_C=/tmp/${USER}/zo_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
export CUDA_VISIBLE_DEVICES=0
SDIR=${DATA_ROOT}/compress_opd/students/svd_nystrom_r07_combined_pad
echo "[build_c128] node=$(hostname) -> $SDIR start=$(date)"
python scripts/compress_opd/math/build_svd_nystrom_student.py \
  --model Qwen/Qwen3-4B --ratio 0.7 --objective combined \
  --skip-last-layers 1 --save-zero-padded --skip-ppl --skip-math \
  --save-dir "$SDIR" --metrics-json "${SDIR}_pre_opd.json"
echo "[build_c128] done rc=$? $(date) | config: $([ -f "$SDIR/config.json" ] && echo OK || echo MISSING)"
