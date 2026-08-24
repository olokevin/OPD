#!/bin/bash
# sigma search at settings ALIGNED to the official implementation
# (github.com/VsonicV/es-at-scale, es_at_scale/train.py + trainer/es_trainer.py).
#
# Alignment audit -- what the official repo does vs what we now do:
#   template            qwen_math_template()  == ours, byte-identical      [ALIGNED]
#   reward shaping      z-score               == ours                      [ALIGNED]
#   alpha               sigma/2 when unset    == ours                      [ALIGNED]
#   population          30                    == ours                      [ALIGNED]
#   decoding            T=0.0, top_p=1.0, seed=global_seed+iter == ours     [ALIGNED]
#   precision           bfloat16              == ours (+fp32 master, strictly better)
#   max-tokens          3000 (train AND eval) -> now 3000                  [FIXED]
#   batch               1024, RESAMPLED every iteration from an 8.5k pool
#                       via DataLoader(shuffle=True)                       [FIXED, smaller B]
#   n-iterations        500                                                [not affordable]
#   eval-freq           5                     -> 5                         [ALIGNED]
#
# The batch was the big one: we had been holding ONE fixed 64-problem batch for all 150
# iterations, which is why train accuracy ran to 77.7 while held-out stalled at 71.5.
#
# Usage: DEVICE=1 SIGMAS="0.001 0.004" bash scripts/es/sweep_aligned_sigma.sh
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
DEVICE=${DEVICE:-1}
MODE=${MODE:-dense}
SIGMAS=${SIGMAS:-"0.001"}
ITERS=${ITERS:-20}
BATCH=${BATCH:-128}
mkdir -p logs/es

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

for SIG in $SIGMAS; do
  ALP=$(python3 -c "print(repr($SIG/2))")          # official: alpha = sigma/2
  NAME="aligned-${MODE}_rs${BATCH}_tok3000_sig${SIG}_a${ALP}"
  LOG="logs/es/aligned_${MODE}_sig${SIG}.log"
  echo "=================================================================="
  echo "[aligned] $NAME on GPU $DEVICE  ($(date))"
  echo "=================================================================="
  DEVICE="$DEVICE" PERTURB_MODE="$MODE" \
    SIGMA="$SIG" ALPHA="$ALP" \
    TRAIN_BATCH_SIZE="$BATCH" TRAIN_MAX_SAMPLES=-1 \
    MAX_TOKENS=3000 EVAL_MAX_TOKENS=3000 \
    NUM_ITERATIONS="$ITERS" EVAL_INTERVAL=5 \
    EXPERIMENT_NAME="$NAME" \
    bash scripts/es/run_es_math.sh > "$LOG" 2>&1
  echo "[aligned] $NAME done $(date) exit=$?"
  grep -E "Training batch:|Eval @ step" "$LOG" | sed "s/^/[aligned][sig=$SIG] /"
  sleep 60
done
echo "[aligned] ALL SIGMAS DONE $(date)"
