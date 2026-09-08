#!/bin/bash
# chain_after_bench.sh -- everything queued behind the parity benchmark.
#   GPU 5: BP-OPSD, 100 steps (the paper reproduction)
#   GPU 6: the two missing alpha-ladder points, then the full es-prefill arm at N=4
# Launch the two halves separately:
#   BP_GPU=5 bash scripts/opsd/chain_after_bench.sh bp
#   ES_GPU=6 bash scripts/opsd/chain_after_bench.sh es
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

wait_free () {
  for _ in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | head -1)
    [ "${u:-99999}" -le 1000 ] && return 0
    sleep 10
  done
  echo "GPU $1 never drained"; return 1
}
reap () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1"); do kill -9 "$p" 2>/dev/null; done
  pkill -u "$(id -u)" -f "ray_opd_gpu$1/sessio[n]_" 2>/dev/null; sleep 10
}

case "${1:-}" in
bp)
  G=${BP_GPU:-5}
  wait_free "$G"
  echo "=== [$(date)] BP-OPSD 100 steps on GPU $G"
  TRAIN_GPU=$G OPSD_STEPS=100 bash scripts/opsd/bp_opsd.sh > logs/opsd/bp_opsd_run2.log 2>&1
  echo "=== [$(date)] BP done"
  ;;
es)
  G=${ES_GPU:-6}
  # the two alpha-ladder points that never completed
  for a in 7.1e-6 6.6e-5; do
    wait_free "$G"
    echo "=== [$(date)] alpha ladder $a on GPU $G"
    TRAIN_GPU=$G ES_SIGMA=1e-3 ES_ALPHA=$a ES_N_RAILS=4 OPSD_STEPS=30 ES_STEPS=30 \
      TEST_FREQ=30 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False WANDB_MODE=disabled \
      EXPERIMENT_NAME=ladder2_es_a$a LOG_DIR=logs/opsd/alpha \
      bash scripts/opsd/es_opsd.sh > "logs/opsd/alpha/a${a}_v2.log" 2>&1
    reap "$G"
    echo "--- alpha=$a ---"
    tr '\r' '\n' < "logs/opsd/alpha/a${a}_v2.log" | grep -oE "val-core/MATH-500/acc/mean@1:[^ ]*" || echo "  (no val)"
    tr '\r' '\n' < "logs/opsd/alpha/a${a}_v2.log" | grep -oE "step:[0-9]+ .*" | tail -1 \
      | tr ' ' '\n' | grep -E "^es/(post_update_gain|cum_footprint|d_std|d_snr)|^opsd/fwd_kl" || true
  done
  echo "=== [$(date)] alpha ladder complete -- pick alpha, then launch the full ES arm"
  ;;
*) echo "usage: $0 bp|es"; exit 1;;
esac
