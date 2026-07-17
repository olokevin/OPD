#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --account=m4788_g
#SBATCH --job-name=csft_evalv
# Fast vLLM eval (MATH-500 + MMLU, no AIME) of a compress_sft heterogeneous merged ckpt.
# Zero-pads -> vLLM -> ttrl grade. Writes math500.json + mmlu_pro.json + EVAL_DONE;
# the login-node logger posts eval/{math500,mmlu_pro}_acc to wandb. Env: CKPT,STEP,OBJECTIVE,RATIO.
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
CKPT=${CKPT:?}; STEP=${STEP:?}; OBJECTIVE=${OBJECTIVE:?}; RATIO=${RATIO:-0.7c}
MATH_LIMIT=${MATH_LIMIT:-500}; MMLU_LIMIT=${MMLU_LIMIT:-500}
MET=${DATA_ROOT}/compress_sft/metrics/${OBJECTIVE}_r${RATIO}/step${STEP}; mkdir -p "$MET"
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/verl
export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1
_C=/tmp/${USER}/zo_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg OUTLINES_CACHE_DIR=${_C}/outlines FLASHINFER_CACHE_DIR=${_C}/fi; mkdir -p "$_C"
cd "$OPD_REPO"
echo "[eval-vllm-job] ckpt=$CKPT step=$STEP obj=$OBJECTIVE ratio=$RATIO -> $MET"
python scripts/compress_sft/eval_vllm.py --model-dir "$CKPT" --tokenizer Qwen/Qwen3-4B \
  --label "${OBJECTIVE}_step${STEP}" --math-json "$MET/math500.json" --mmlu-json "$MET/mmlu_pro.json" \
  --math-limit "$MATH_LIMIT" --mmlu-limit "$MMLU_LIMIT" --tmp-dir "/tmp/${USER}/pad_step${STEP}" && touch "$MET/EVAL_DONE"
case "$CKPT" in *_evalstage/*) rm -rf "$CKPT";; esac
echo "[eval-vllm-job] done step=$STEP"
