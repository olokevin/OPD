#!/bin/bash
# ruler_ckpt.sh -- score an HF checkpoint on the ds15b standard ruler (MATH-500 + AIME24,
# n=2 @ T=0.6, 7168 tokens) with the PPO trainer's validation, val_only. ONE GPU (default 4).
#   bash scripts/zo_opd/ds15b/ruler_ckpt.sh <ckpt_dir> <tag> [gpu]
# Prints "<tag> MATH-500 <acc> AIME24 <acc>" and appends it to results/ruler_scores.tsv.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
CK=$1; TAG=$2; GPU=${3:-4}
LOG_DIR=logs/ds15b/es_decode/ruler; mkdir -p "$LOG_DIR" scripts/zo_opd/es_profile/results
LOG=$LOG_DIR/${TAG}_driver.log
# reap a stale isolated Ray head from a previous ruler run (session-name mismatch otherwise)
for pid in $(pgrep -f "ray_ruler_gpu${GPU}"); do [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null; done
sleep 2
rm -rf /tmp/ray_ruler_gpu${GPU}
# val_only writes nothing worth keeping; steer default_local_dir to /tmp so ruler
# runs never accumulate on /data, and delete that scratch afterwards.
SCRATCH=/tmp/ruler_scratch_gpu${GPU}_$$
EXTRA_HYDRA_ARGS="trainer.val_only=True trainer.default_local_dir=$SCRATCH trainer.save_freq=-1" \
  TRAIN_GPU=$GPU ACTOR_MODEL_PATH=$CK EXPERIMENT_NAME=ruler_$TAG \
  LOG_DIR=$LOG_DIR RAY_TMPDIR=/tmp/ray_ruler_gpu${GPU} bash scripts/zo_opd/ds15b/bp_opd.sh > "$LOG" 2>&1
rm -rf "$SCRATCH"
m=$(grep -o "val-core/MATH-500/acc/mean@2:[^ ]*" "$LOG" | tail -1 | grep -o "[0-9.]*)" | tr -d ')')
a=$(grep -o "val-core/AIME24/acc/mean@2:[^ ]*" "$LOG" | tail -1 | grep -o "[0-9.]*)" | tr -d ')')
echo -e "$TAG\t${m:-NA}\t${a:-NA}\t$(date '+%F %T')" >> scripts/zo_opd/es_profile/results/ruler_scores.tsv
echo "$TAG MATH-500 ${m:-NA} AIME24 ${a:-NA}"
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}'); do kill -9 "$pid" 2>/dev/null; done
