#!/bin/bash
# compress_sft_wandb_sync.sh — periodically push the LIVE forward+combined offline
# wandb runs to project nersc_compress_sft_qwen4b. NERSC compute nodes have no
# outbound internet, so the jobs log offline and this daemon syncs from a login node.
# All per-attempt offline dirs share one run id per objective, so they consolidate.
#
# Launch detached on a login node:
#   nohup bash slurm/compress_sft/compress_sft_wandb_sync.sh \
#     > /pscratch/sd/$USER/opd/compress_sft/logs/wandb_sync_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Stop: pkill -f compress_sft_wandb_sync
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/sft
WB=/pscratch/sd/y/yequan/opd/compress_sft/wandb/wandb
PROJ=${WANDB_PROJECT:-nersc_compress_sft_qwen4b}
INT=${SYNC_INTERVAL:-300}
echo "[wandb-sync] project=$PROJ interval=${INT}s wbdir=$WB"
while true; do
  # Sync only the *_eval offline runs (single clean log each). The TRAIN runs are
  # driven directly from trainer_log.jsonl by log_train_to_wandb.py (online), because
  # HF's offline+resume across re-allocations doesn't consolidate cleanly.
  for d in $(ls -dt "$WB"/offline-run-*sft_eval* 2>/dev/null); do
    WANDB_PROJECT=$PROJ wandb sync --project "$PROJ" "$d" 2>&1 | grep -E "Syncing|done|error" | tail -1
  done
  sleep "$INT"
done
