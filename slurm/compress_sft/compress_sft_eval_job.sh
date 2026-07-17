#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --account=m4788_g
#SBATCH --job-name=csft_eval
# compress_sft_eval_job.sh — one-shot post-hoc eval of a (staged) merged checkpoint
# in the gpu_shared QOS (does NOT count against the interactive 2-job cap). Runs
# MATH-500 + AIME24 (eval_opd_ckpt) + MMLU-Pro (eval_mmlu_pro) in the verl env and
# writes metric JSONs under <metrics-root>/<obj>_r<ratio>/step<N>/, then touches an
# EVAL_DONE marker. The login-node log_train_to_wandb.py picks those up and logs
# eval/{math500,aime24,mmlu_pro}_acc into the SAME wandb run as the training curve.
#
# Submitted by compress_sft_eval_daemon.sh with: CKPT, STEP, OBJECTIVE, RATIO, [limits].
set -uo pipefail
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
CKPT=${CKPT:?set CKPT}; STEP=${STEP:?set STEP}; OBJECTIVE=${OBJECTIVE:?set OBJECTIVE}
RATIO=${RATIO:-0.7}
MATH_LIMIT=${MATH_LIMIT:-50}; MATH_TOK=${MATH_TOK:-2048}; MATH_BS=${MATH_BS:-8}
AIME_LIMIT=${AIME_LIMIT:-30}; AIME_TOK=${AIME_TOK:-4096}; AIME_BS=${AIME_BS:-8}
MMLU_LIMIT=${MMLU_LIMIT:-100}; MMLU_TOK=${MMLU_TOK:-512}; MMLU_BS=${MMLU_BS:-8}
MET=${DATA_ROOT}/compress_sft/metrics/${OBJECTIVE}_r${RATIO}/step${STEP}; mkdir -p "$MET"

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate ${DATA_ROOT}/envs/verl
export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
_C=/tmp/${USER}/compress_sft_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
cd "$OPD_REPO"
echo "[eval-job] ckpt=$CKPT step=$STEP obj=$OBJECTIVE ratio=$RATIO -> $MET"

# MATH-500 only (AIME24 dropped — too slow under HF-generate to finish in walltime).
python scripts/compress_sft/eval_opd_ckpt.py --model-dir "$CKPT" --tokenizer Qwen/Qwen3-4B \
  --label "${OBJECTIVE}_step${STEP}" --metrics-json "$MET/math500.json" --skip-ppl --skip-aime \
  --math-limit "$MATH_LIMIT" --math-max-new-tokens "$MATH_TOK" --math-batch-size "$MATH_BS" || true
python scripts/compress_sft/eval_mmlu_pro.py --model-dir "$CKPT" --tokenizer Qwen/Qwen3-4B \
  --label "${OBJECTIVE}_step${STEP}" --metrics-json "$MET/mmlu_pro.json" \
  --limit "$MMLU_LIMIT" --mmlu-max-new-tokens "$MMLU_TOK" --mmlu-batch-size "$MMLU_BS" || true

# marker: tells the login-node wandb logger this step's eval is complete to log.
touch "$MET/EVAL_DONE"

# drop the staged checkpoint copy (the daemon staged it to survive save_total_limit=1)
case "$CKPT" in *_evalstage/*) rm -rf "$CKPT"; echo "[eval-job] removed staged $CKPT";; esac
echo "[eval-job] done step=$STEP"
