#!/bin/bash
# calibrate_es_opsd.sh -- pick sigma for es-prefill OPSD by MEASUREMENT, not by analogy.
#
# The ds15b sigma (1e-3, probe footprint ~5% of RMS(W)) does NOT transfer: here the
# perturbed tensors are LoRA coefficients, whose B factor is ZERO at init, so
#   (a) RMS over the trainable tensors is ~0.007 (set by A's kaiming init), not ~0.02, and
#   (b) the induced dense-weight perturbation is  dW ~ 2*sigma*eps_B*A  -- one-sided in
#       eps_B at step 0, i.e. a different constant AND a different geometry.
#
# So sweep sigma with alpha ~ 0 (weights effectively frozen, every sigma probes the same
# model) and read, per step:
#   es/d_std              ES signal spread across rails -- too small => no signal
#   es/d_snr              |mean(d)|/std(d)
#   es/fitness_{plus,minus}_mean   F(W+-sigma*eps) - F(W_0) in the objective's own units
#   es/probe_footprint    sigma / RMS(W_trainable)
# Pick the largest sigma whose fitness perturbation is still in the linear regime
# (|F+| ~ |F-|, and both small vs the batch objective ~0.9), with 2x margin.
#
#   TRAIN_GPU=6 bash scripts/opsd/calibrate_es_opsd.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU=${TRAIN_GPU:-6}
SIGMAS=${SIGMAS:-"3e-4 1e-3 3e-3 1e-2"}
STEPS=${STEPS:-3}
RAILS=${ES_N_RAILS:-8}
OUT=${OUT:-logs/opsd/calib}
mkdir -p "$OUT"

for sig in $SIGMAS; do
  log="$OUT/sig${sig}.log"
  echo "=== sigma=$sig  (N=$RAILS, $STEPS steps) -> $log"
  TRAIN_GPU=$GPU ES_SIGMA=$sig ES_ALPHA=1e-9 ES_N_RAILS=$RAILS \
    OPSD_STEPS=$STEPS ES_STEPS=$STEPS \
    TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False WANDB_MODE=disabled \
    EXPERIMENT_NAME=calib_es_sig${sig} LOG_DIR="$OUT" \
    bash scripts/opsd/es_opsd.sh > "$log" 2>&1 || echo "  (sigma=$sig FAILED -- see $log)"
  # reap this GPU's ray head before the next sigma
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU"); do kill -9 "$p" 2>/dev/null || true; done
  pkill -u "$(id -u)" -f "ray_opd_gpu${GPU}/sessio[n]_" 2>/dev/null || true
  sleep 15
  echo "--- sigma=$sig ---"
  tr '\r' '\n' < "$log" | grep -oE "step:[0-9]+ .*" | tr ' ' '\n' \
    | grep -E "^es/(d_std|d_snr|d_mean|fitness_plus_mean|fitness_minus_mean|probe_footprint|post_update_gain|update_footprint)|^opsd/fwd_kl|^timing_s/(step|es_rails|es_per_rail)" || true
done
echo "=== calibration done; logs in $OUT"
