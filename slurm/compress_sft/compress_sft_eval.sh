#!/bin/bash
# compress_sft_eval.sh — post-hoc benchmark eval of a compress->SFT checkpoint, in
# the VERL env (its ttrl_math grader needs verl/ray). Runs MATH-500 + MMLU-Pro on a
# DENSE -merged checkpoint via HF generate. Run under a GPU alloc (re-execs onto the
# compute node via srun when SLURM_JOB_ID is set), or directly on a GPU node.
#
#   CKPT=/pscratch/.../final-merged GPU=0 MATH_LIMIT=500 MMLU_LIMIT=1000 \
#     salloc -N1 --qos interactive --time 2:00:00 -C 'gpu&hbm80g' --gpus-per-node=4 \
#            --account m4788_g bash slurm/compress_sft/compress_sft_eval.sh
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}

if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/compress_sft/compress_sft_eval.sh"
fi

CKPT=${CKPT:?set CKPT=<final-merged dir>}
LABEL=${LABEL:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}
GPU=${GPU:-0}
MATH_LIMIT=${MATH_LIMIT:-500}
MATH_TOK=${MATH_TOK:-4096}
MATH_BS=${MATH_BS:-8}
MMLU_LIMIT=${MMLU_LIMIT:-1000}
MMLU_TOK=${MMLU_TOK:-512}
MMLU_BS=${MMLU_BS:-16}
MET=${MET:-${DATA_ROOT}/compress_sft/metrics/$LABEL}; mkdir -p "$MET"

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/verl
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
_C=/tmp/${USER}/compress_sft_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
cd "$OPD_REPO"
echo "[eval] ckpt=$CKPT label=$LABEL gpu=$GPU"

CUDA_VISIBLE_DEVICES=$GPU python scripts/compress_sft/eval_opd_ckpt.py \
  --model-dir "$CKPT" --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" --label "$LABEL" --metrics-json "$MET/math500.json" \
  --math-limit "$MATH_LIMIT" --math-max-new-tokens "$MATH_TOK" --math-batch-size "$MATH_BS" --skip-ppl
CUDA_VISIBLE_DEVICES=$GPU python scripts/compress_sft/eval_mmlu_pro.py \
  --model-dir "$CKPT" --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" --label "$LABEL" --metrics-json "$MET/mmlu_pro.json" \
  --limit "$MMLU_LIMIT" --mmlu-max-new-tokens "$MMLU_TOK" --mmlu-batch-size "$MMLU_BS"
echo "[eval] done -> $MET"
