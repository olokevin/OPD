#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --account=m4788_g
#SBATCH --job-name=bench_roll
set -uo pipefail
OPD_REPO=/global/u1/y/yequan/Project/OPD; DATA_ROOT=/pscratch/sd/y/yequan/opd; cd "$OPD_REPO"
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/verl
export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
_C=/tmp/${USER}/zo_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg OUTLINES_CACHE_DIR=${_C}/outlines FLASHINFER_CACHE_DIR=${_C}/fi; mkdir -p "$_C"
# bounded for the (slow) HF side; same params both -> tok/s ratio is the answer.
# NOTE: 2048 tokens UNDERSTATES vLLM's edge (its paged-attn/batching win grows with length+N).
export BENCH_PROMPTS=${BENCH_PROMPTS:-4} BENCH_N=${BENCH_N:-4} BENCH_MAX_NEW=${BENCH_MAX_NEW:-2048}
echo "[bench $BENCH_WHICH] node=$(hostname) start=$(date)"
python slurm/opd/compress/bench_rollout.py
echo "[bench $BENCH_WHICH] done rc=$? $(date)"
