#!/bin/bash
# wandb_sync_daemon.sh — periodically sync the live OFFLINE wandb runs to
# wandb.ai from a login node (compute nodes have no internet, so runs log offline).
# Syncs the LATEST offline-run dir for each pinned run id every INTERVAL seconds;
# when a new 4h-segment creates a new dir, it switches to it automatically (the
# pinned WANDB_RUN_ID + resume=allow makes wandb append to the same run).
#
#   nohup bash slurm/wandb_sync_daemon.sh > /pscratch/sd/$USER/opd/logs/wandb_sync_daemon.log 2>&1 &
set -u
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/verl
export WANDB_DIR=/pscratch/sd/y/yequan/opd/wandb
WB=/pscratch/sd/y/yequan/opd/wandb/wandb
RUN_IDS=("opd_lora_dapo_lr1e-5" "opd_fura_dapo_lr7e-6")
INTERVAL=${INTERVAL:-1200}

while true; do
  for rid in "${RUN_IDS[@]}"; do
    latest=$(ls -dt "$WB"/offline-run-*-"$rid" 2>/dev/null | head -1)
    [ -n "$latest" ] || continue
    echo "[$(date +%H:%M:%S)] sync $rid -> $(basename "$latest")"
    wandb sync "$latest" 2>&1 | grep -iE "Syncing|done|error" | tail -2
  done
  sleep "$INTERVAL"
done
