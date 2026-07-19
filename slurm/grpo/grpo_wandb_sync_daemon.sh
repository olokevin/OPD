#!/bin/bash
# grpo_wandb_sync_daemon.sh — periodically push the live OFFLINE wandb runs for
# the Qwen2.5-7B GRPO jobs to wandb.ai from a login node (compute nodes have no
# internet). Syncs the LATEST offline-run dir for each pinned run id every
# INTERVAL seconds; a new 4h-segment creates a new dir and this switches to it
# automatically (pinned WANDB_RUN_ID + resume=allow appends to the same run).
#
#   nohup bash slurm/grpo/grpo_wandb_sync_daemon.sh > /pscratch/sd/$USER/opd/logs/grpo_wandb_sync_daemon.log 2>&1 &
set -u
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/verl
export WANDB_DIR=/pscratch/sd/y/yequan/opd/wandb
WB=/pscratch/sd/y/yequan/opd/wandb/wandb
RUN_IDS=("grpo_full_qwen2p5_7b_math" "grpo_fura_qwen2p5_7b_math")
INTERVAL=${INTERVAL:-900}

while true; do
  for rid in "${RUN_IDS[@]}"; do
    latest=$(ls -dt "$WB"/offline-run-*-"$rid" 2>/dev/null | head -1)
    [ -n "$latest" ] || continue
    echo "[$(date +%H:%M:%S)] sync $rid -> $(basename "$latest")"
    wandb sync "$latest" 2>&1 | grep -iE "Syncing|done|error" | tail -2
  done
  sleep "$INTERVAL"
done
