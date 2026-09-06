#!/bin/bash
# The section-17 N=10 leaderboard, re-run on the PAPER-ALIGNED data protocol (section 19).
#
#   DEVICE=6 ARMS="dense fura_zoact" bash scripts/es/chain_aligned_pop10.sh
#
# Everything on the section-17 leaderboard was measured on ONE FIXED 64-problem batch, never
# refreshed -- 16x smaller than the official implementation's and resampled never instead of
# every iteration (section 12).  That protocol lets ES memorise the batch: section 11.1
# measured dense's train-minus-heldout gap swinging +3.9 -> -6.1 pp, and every arm flattening
# at 71-73 by step ~40.  So the ranking may be "which method memorises 64 problems best".
#
# This re-runs each method at ITS OWN best (sigma, alpha) from section 17, changing only the
# data protocol:
#     train_batch_size = 1024, RESAMPLED every iteration from the full 8,890-problem pool
#     (train_max_samples=-1 -- without it the pool is truncated to 64 and the resample guard
#      `train_batch_size < len(train_data)` silently falls back to the fixed batch)
#
# Remaining deviation from the paper: train token budget stays 1,536 (paper 3,000).  Section 3
# measured that 1,536 keeps 98.4% of the base model's correct answers (p99 = 2,030) and nearly
# halves wall-clock; at batch 1024 that is the difference between ~12 h and ~25 h per arm.
# Held-out eval keeps the paper's 3,000.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH

ITERS=${ITERS:-100}
TB=${TB:-1024}

for arm in ${ARMS:?set ARMS="dense fura fura_zoact lora"}; do
  # best (sigma, alpha) per method from the section-17 N=10 leaderboard
  case "$arm" in
    dense)      MODE=dense      SIG=0.001    ALP=0.00028868 RANK=1  ;;
    fura)       MODE=fura       SIG=0.0125   ALP=0.0036084  RANK=1  ;;
    fura_zoact) MODE=fura_zoact SIG=0.05     ALP=0.014434   RANK=1  ;;
    lora)       MODE=lora       SIG=0.015385 ALP=0.0076925  RANK=44 ;;
    lora_r1)    MODE=lora       SIG=0.0022   ALP=0.0254040  RANK=1  ;;
    # `isobtt` IS "fura + ISO": fura's block-wise SVD with A_j = U_j diag(S_j) frozen and the
    # small core R_j constrained to O(b) by a Cayley step, so each block's spectrum is exactly
    # preserved.  sigma is a RELATIVE FOOTPRINT for the ISO modes, not a noise std (section
    # 10.4), hence 5e-2; alpha = sigma/2 * sqrt(10/30) is the section-16.4 motion correction.
    # Never LR-searched at N=10 -- this is the principled default, not a tuned point.
    isobtt)     MODE=isobtt     SIG=0.05     ALP=0.014434   RANK=1  ;;
    *) echo "[skip] unknown arm $arm"; continue ;;
  esac
  NAME="aligned-${arm}$([ "$arm" = lora ] && echo "-r${RANK}")_rs${TB}_sig${SIG}_a${ALP}_N10_it${ITERS}"
  echo "=== [$(date)] $arm  N=10  sigma=$SIG alpha=$ALP  batch=${TB} resampled  iters=$ITERS  on GPU ${DEVICE:-6}"
  echo "=== [$(date)] ckpt -> /data/yequan/es/ES-q2p5-7b/${NAME}"
  DEVICE=${DEVICE:-6} PERTURB_MODE="$MODE" SUBSPACE_RANK="$RANK" LORA_RANK="$RANK" \
    POPULATION_SIZE=10 SIGMA="$SIG" ALPHA="$ALP" \
    NUM_ITERATIONS="$ITERS" TRAIN_BATCH_SIZE="$TB" TRAIN_MAX_SAMPLES=-1 \
    FORGET_TASKS= \
    EXPERIMENT_NAME="$NAME" \
    SAVE_DIR="/data/yequan/es/ES-q2p5-7b/${NAME}" \
    bash scripts/es/run_es_math.sh
  echo "=== [$(date)] $arm exited with $?"
done
echo "ALIGNED POP10 CHAIN DONE: $ARMS"
