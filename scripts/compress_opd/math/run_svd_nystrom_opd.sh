#!/bin/bash
# run_svd_nystrom_opd.sh — orchestrates the full compress→OPD pipeline:
#   Stage 1: reasoning-aware-compress Qwen3-4B (SVD-V2 attn + Nystrom mlp,
#            retain 0.7, sequence/full OpenThought3 calib, last layer dense)
#            → save HF dir + pre-OPD MATH-500 / C4-PPL metrics.
#   Stage 2: OPD on top of the compressed student
#            (teacher = uncompressed Qwen/Qwen3-4B non-thinking).
#
# Single H100, runs sequentially. Stage 1 must complete before Stage 2 starts.
#
# Override knobs via env vars:
#   GPU                CUDA device id (default 0)
#   RATIO              retain ratio (default 0.7 — the wiki best safe op point)
#   EXPERIMENT_TAG     run label (default derived from ratio)
#   STUDENT_DIR        where to save the compressed HF student
#                      (default /data/yequan/compress_opd/students/svd_nystrom_r${RATIO})
#   STAGE              stage1 | stage2 | both (default both)
#   PY                 python interpreter (default verl conda env)
#
# Usage:
#   bash scripts/compress_opd/math/run_svd_nystrom_opd.sh
#   GPU=1 RATIO=0.7 STAGE=stage1 bash scripts/compress_opd/math/run_svd_nystrom_opd.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

GPU=${GPU:-0}
RATIO=${RATIO:-0.7}
STAGE=${STAGE:-both}
PY=${PY:-/home/yequan/miniconda3/envs/verl/bin/python}

# Strip the leading '0.' for a filesystem-friendly tag (0.7 → r07).
TAG_RATIO=$(printf '%s' "$RATIO" | tr -d '.')
EXPERIMENT_TAG=${EXPERIMENT_TAG:-svd_nystrom_r${TAG_RATIO}}
STUDENT_DIR=${STUDENT_DIR:-/data/yequan/compress_opd/students/svd_nystrom_r${TAG_RATIO}}

METRICS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$REPO_ROOT/logs/compress_opd_svd_nystrom"
mkdir -p "$METRICS_DIR" "$LOG_DIR"
METRICS_JSON="$METRICS_DIR/${EXPERIMENT_TAG}_pre_opd.json"

echo "[run] GPU=$GPU RATIO=$RATIO TAG=$EXPERIMENT_TAG STAGE=$STAGE"
echo "[run] student dir: $STUDENT_DIR"
echo "[run] pre-OPD metrics: $METRICS_JSON"

if [[ "$STAGE" == "stage1" || "$STAGE" == "both" ]]; then
  echo "[stage1] reasoning-aware compress @ retain $RATIO"
  CUDA_VISIBLE_DEVICES=$GPU \
  HF_HOME=/data/yequan/huggingface \
  PYTHONPATH=src:verl \
    "$PY" "$SCRIPT_DIR/build_svd_nystrom_student.py" \
      --ratio "$RATIO" \
      --save-dir "$STUDENT_DIR" \
      --metrics-json "$METRICS_JSON" \
      2>&1 | tee "$LOG_DIR/${EXPERIMENT_TAG}_compress.log"
fi

if [[ "$STAGE" == "stage2" || "$STAGE" == "both" ]]; then
  if [[ ! -d "$STUDENT_DIR" ]]; then
    echo "[stage2] error: student dir $STUDENT_DIR does not exist (run stage1 first)" >&2
    exit 1
  fi
  echo "[stage2] OPD on $STUDENT_DIR"
  CUDA_VISIBLE_DEVICES=$GPU \
  RAY_ISOLATE=1 \
  LOG_DIR="$LOG_DIR" \
  ACTOR_MODEL_PATH="$STUDENT_DIR" \
  EXPERIMENT_TAG="$EXPERIMENT_TAG" \
    bash "$SCRIPT_DIR/opd_svd_nystrom.sh"
fi

echo "[run] done."
