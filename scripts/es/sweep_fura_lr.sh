#!/bin/bash
# FuRA learning-rate search: can a larger ES step let the small-core subspace
# match full-weight ES?
#
# WHY THESE VALUES.  The ES update is theta += (alpha/N) * sum_n Z_n eps_n, and the
# z-scores Z_n are scale-free, so the per-step motion is *linear in alpha*.  Measured
# footprints at the paper's sigma=1e-3 (scripts/es/test_es_perturb_modes.py):
#
#     dense  ||dW||/||W|| = 5.0e-2
#     fura   ||dW||/||W|| = 4.0e-3      -> 12.5x smaller
#
# so the alpha that reproduces dense's per-step weight motion is
#     alpha_matched = 12.5 * 5e-4 = 6.25e-3.
#
# FuRA's train reward spread at alpha=5e-4 was ~0.024-0.028, i.e. comparable to dense's
# ~0.027 -- the perturbation is perfectly resolvable, so the observed lag (60.2 @ step 20
# vs dense's 71.4) is a STEP-SIZE problem, not a bad gradient direction.  Hence: sweep
# alpha at fixed sigma, bracketing 12.5x geometrically, plus one control that scales
# sigma too (does a larger *exploration radius* buy anything beyond a larger step?).
#
# Order is by expected informativeness, so an early stop still answers the question.
#
# Usage: DEVICE=2 bash scripts/es/sweep_fura_lr.sh

set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
DEVICE=${DEVICE:-2}
ITERS=${ITERS:-20}          # 20 is enough to rank: dense/zoact/insparse/fura separated
EVAL_EVERY=${EVAL_EVERY:-5} # cleanly by step 10-20 in the completed runs
mkdir -p logs/es

#            label            SIGMA     ALPHA      note
CONFIGS=(
  "a12.5x       0.001     0.00625    footprint-matched step (theory point)"
  "a40x         0.001     0.02       aggressive"
  "match-sig    0.0125    0.00625    joint sigma+alpha scale-match (alpha=sigma/2)"
  "a4x          0.001     0.002      conservative guard"
)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  LABEL=$1; SIG=$2; ALP=$3
  NAME="fura-lr-${LABEL}_sig${SIG}_a${ALP}"
  LOG="logs/es/sweep_fura_${LABEL}.log"
  echo "=========================================================================="
  echo "[sweep] $NAME  (sigma=$SIG alpha=$ALP, $ITERS iters) on GPU $DEVICE"
  echo "[sweep] start $(date)  -> $LOG"
  echo "=========================================================================="
  DEVICE="$DEVICE" \
  PERTURB_MODE=fura \
  SIGMA="$SIG" ALPHA="$ALP" \
  NUM_ITERATIONS="$ITERS" EVAL_INTERVAL="$EVAL_EVERY" \
  EXPERIMENT_NAME="$NAME" \
  bash scripts/es/run_es_math.sh > "$LOG" 2>&1
  echo "[sweep] $NAME finished $(date) exit=$?"
  grep -E "Eval @ step" "$LOG" | sed "s/^/[sweep][$LABEL] /"
  sleep 60   # let the GPU drain before the next config
done
echo "[sweep] ALL CONFIGS DONE $(date)"
