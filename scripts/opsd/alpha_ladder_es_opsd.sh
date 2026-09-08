#!/bin/bash
# alpha_ladder_es_opsd.sh -- pick the es-prefill step size for OPSD.
#
# sigma is fixed at the calibrated 1e-3 (see docs/results/OPSD/opsd_bp_vs_es.md §4).
# With z-score normalisation the per-step motion is exactly
#     update_rms = alpha / sqrt(n_pairs),   footprint phi = update_rms / RMS(W_train)
# and RMS(W_train) was measured at 7.75e-3, so at N=4 (n_pairs=2)
#     alpha = phi * sqrt(2) * 7.75e-3 = phi * 1.096e-2.
#
# Anchors:
#   phi = 6.5e-4  -- matches BP's own per-step motion (Adam at lr 5e-6 moves ~lr per coord)
#   phi = 2.0e-3  -- 3x BP
#   phi = 6.0e-3  -- the per-iteration motion the `dense`/`iso` ES arms used to gain +20 pp
#                    (docs/results/ES/es_results.md); cumulative walk 6e-2 over 100 steps,
#                    still inside the ~9-10% soft budget ds15b measured.
#
#   TRAIN_GPU=6 bash scripts/opsd/alpha_ladder_es_opsd.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU=${TRAIN_GPU:-6}
SIG=${ES_SIGMA:-1e-3}
RAILS=${ES_N_RAILS:-4}
ALPHAS=${ALPHAS:-"7.1e-6 2.19e-5 6.6e-5"}
STEPS=${STEPS:-30}
OUT=${OUT:-logs/opsd/alpha}
mkdir -p "$OUT"

wait_gpu_free () {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
    [ "${used:-99999}" -le 2000 ] && return 0
    sleep 10
  done
  echo "GPU $GPU never dropped below 2 GiB -- aborting"; return 1
}

for a in $ALPHAS; do
  log="$OUT/a${a}.log"
  wait_gpu_free
  echo "=== alpha=$a (sigma=$SIG, N=$RAILS, $STEPS steps, val@$STEPS) -> $log"
  TRAIN_GPU=$GPU ES_SIGMA=$SIG ES_ALPHA=$a ES_N_RAILS=$RAILS \
    OPSD_STEPS=$STEPS ES_STEPS=$STEPS \
    TEST_FREQ=$STEPS SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False WANDB_MODE=disabled \
    EXPERIMENT_NAME=ladder_es_a${a} LOG_DIR="$OUT" \
    bash scripts/opsd/es_opsd.sh > "$log" 2>&1 || echo "  (alpha=$a FAILED -- see $log)"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU"); do kill -9 "$p" 2>/dev/null || true; done
  pkill -u "$(id -u)" -f "ray_opd_gpu${GPU}/sessio[n]_" 2>/dev/null || true
  sleep 15
  echo "--- alpha=$a ---"
  tr '\r' '\n' < "$log" | grep -oE "val-core/MATH-500/acc/mean@1:[^ ]*" || echo "  (no val)"
  tr '\r' '\n' < "$log" | grep -oE "step:[0-9]+ .*" | while read -r l; do
    st=$(echo "$l"|grep -oE "^step:[0-9]+")
    fk=$(echo "$l"|grep -oE "opsd/fwd_kl_per_token:[^ ]*"|cut -d: -f2)
    sk=$(echo "$l"|grep -oE "critic/advantages/mean_sum_over_k:[^ ]*"|cut -d: -f2)
    pg=$(echo "$l"|grep -oE "es/post_update_gain:[^ ]*"|cut -d: -f2)
    cf=$(echo "$l"|grep -oE "es/cum_footprint:[^ ]*"|cut -d: -f2)
    ts=$(echo "$l"|grep -oE "timing_s/step:[^ ]*"|cut -d: -f2)
    printf "%-9s fkl=%-12s sumA=%-8s post_gain=%-13s cum_fp=%-11s t=%s\n" "$st" "$fk" "$sk" "$pg" "$cf" "$ts"
  done | tail -12 || true
done
echo "=== alpha ladder done; logs in $OUT"
