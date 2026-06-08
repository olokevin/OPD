#!/bin/bash
# nersc_build_students.sh — Stage 1 of compress->OPD on NERSC (offline compute node).
# Regenerates the OpenThought3-Qwen3-4B calibration traces (if missing), then
# builds BOTH compressed Qwen3-4B students used by the two recipes:
#   svd_nystrom_r07_forward  (wiki D0: SVD-V2 attn + Nystrom mlp, forward-only)
#   svd_nystrom_r07_combined (wiki D2: bilateral fwd+CE-backward whitening)
# Saves them as dense HF dirs under $DATA_ROOT/compress_opd/students/.
#
# Single node (uses 2 GPUs: forward on GPU0, combined on GPU1, concurrently).
# Run under an interactive allocation, e.g.:
#   salloc -N1 --qos interactive --time 2:00:00 -C 'gpu&hbm80g' --gpus-per-node=4 \
#          --account m4788_g bash slurm/opd/compress/nersc_build_students.sh
# (salloc-with-command runs ON the login node under the alloc; this script srun's
#  itself onto the compute node when SLURM_JOB_ID is set.)
set -uo pipefail

OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
RATIO=${RATIO:-0.7}
NUM_CALIB_PROMPTS=${NUM_CALIB_PROMPTS:-512}
STUDENTS_DIR=${STUDENTS_DIR:-${DATA_ROOT}/compress_opd/students}
CALIB_JSONL=${CALIB_JSONL:-${OPD_REPO}/datasets/OpenThought3-Qwen3-4B/data/train.jsonl}

# If launched as the salloc command on the login node, re-exec onto the compute node.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${_ON_COMPUTE:-}" ]; then
  exec srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap \
       bash -c "export _ON_COMPUTE=1; exec bash $OPD_REPO/slurm/opd/compress/nersc_build_students.sh"
fi

# ---- offline runtime env (mirrors opd_2node_rayenv.sh) ----
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate "${DATA_ROOT}/envs/verl"
cd "$OPD_REPO"

export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${DATA_ROOT}/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=true
_C=/tmp/${USER}/zo_cache
export TRITON_CACHE_DIR=${_C}/triton TORCHINDUCTOR_CACHE_DIR=${_C}/inductor
export XDG_CACHE_HOME=${_C}/xdg OUTLINES_CACHE_DIR=${_C}/outlines
export FLASHINFER_WORKSPACE_DIR=${_C}/flashinfer FLASHINFER_CACHE_DIR=${_C}/flashinfer
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$OUTLINES_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null

LOG_DIR=${DATA_ROOT}/logs
mkdir -p "$LOG_DIR" "$STUDENTS_DIR"
PY=$(command -v python)
echo "[stage1] node=$(hostname) py=$PY ratio=$RATIO students=$STUDENTS_DIR"

# ---- 1. calibration traces (regenerate if missing) ----
if [ ! -s "$CALIB_JSONL" ]; then
  echo "[stage1] calib jsonl missing -> rolling out $NUM_CALIB_PROMPTS Qwen3-4B traces"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/compress_opd/math/build_ot3_calib_jsonl.py \
    --model Qwen/Qwen3-4B --num-prompts "$NUM_CALIB_PROMPTS" \
    --max-new-tokens 4096 --out "$CALIB_JSONL" \
    2>&1 | tee "$LOG_DIR/calib_rollout.log"
else
  echo "[stage1] calib jsonl present ($(wc -l < "$CALIB_JSONL") lines): $CALIB_JSONL"
fi
if [ ! -s "$CALIB_JSONL" ]; then
  echo "[stage1] ERROR: calib jsonl still missing/empty; aborting" >&2; exit 1
fi

# ---- 2. compress both objectives (concurrent: forward GPU0, combined GPU1) ----
build_one() {  # $1=objective $2=gpu $3=savedir
  local obj=$1 gpu=$2 sdir=$3
  echo "[stage1] compress objective=$obj on GPU$gpu -> $sdir"
  # --skip-last-layers 0 + --save-stock-dense: SVD-V2 attn is factorized and
  # Nystrom shrinks the MLP, so the model is NOT a stock Qwen3 unless we (a) keep
  # the MLP width uniform across ALL layers (compress the last one too) and (b)
  # merge the attn SVD factors into dense Linear + patch config.intermediate_size.
  # Without this the verl actor's from_pretrained dies on shape/key mismatch.
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/compress_opd/math/build_svd_nystrom_student.py \
    --model Qwen/Qwen3-4B --ratio "$RATIO" --objective "$obj" \
    --skip-last-layers 0 --save-stock-dense \
    --skip-ppl \
    --save-dir "$sdir" \
    --metrics-json "${sdir%/}_pre_opd.json" \
    2>&1 | tee "$LOG_DIR/compress_${obj}.log"
}

FWD_DIR="${STUDENTS_DIR}/svd_nystrom_r07_forward"
CMB_DIR="${STUDENTS_DIR}/svd_nystrom_r07_combined"

rc=0
if [ ! -f "${FWD_DIR}/config.json" ]; then
  build_one forward 0 "$FWD_DIR" & FWD_PID=$!
else echo "[stage1] forward student already built: $FWD_DIR"; FWD_PID=""; fi
if [ ! -f "${CMB_DIR}/config.json" ]; then
  build_one combined 1 "$CMB_DIR" & CMB_PID=$!
else echo "[stage1] combined student already built: $CMB_DIR"; CMB_PID=""; fi

[ -n "${FWD_PID:-}" ] && { wait "$FWD_PID" || rc=$?; }
[ -n "${CMB_PID:-}" ] && { wait "$CMB_PID" || rc=$?; }

echo "[stage1] done rc=$rc"
echo "  forward : $([ -f "${FWD_DIR}/config.json" ] && echo OK || echo MISSING)  $FWD_DIR"
echo "  combined: $([ -f "${CMB_DIR}/config.json" ] && echo OK || echo MISSING)  $CMB_DIR"
exit $rc
