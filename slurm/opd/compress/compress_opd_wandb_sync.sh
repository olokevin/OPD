#!/bin/bash
# compress_opd_wandb_sync.sh — push the compress_opd OFFLINE wandb run(s) to project
# nersc_compress_opd_qwen4b from a LOGIN node (NERSC compute nodes have no outbound
# internet, so the run logs offline; this syncs it). Each 4h re-allocation writes a
# new offline-run dir under the SAME run id, so they consolidate on the wandb side.
# `wandb sync --project` overrides whatever project name is embedded in the dir.
# Idempotent: already-synced dirs are skipped.
#
# Launch detached on a login node:
#   nohup bash slurm/opd/compress/compress_opd_wandb_sync.sh \
#     > /pscratch/sd/$USER/opd/logs/compress_opd_wandb_sync_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Stop: pkill -f compress_opd_wandb_sync
set -u
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/verl
WB=/pscratch/sd/y/yequan/opd/wandb/wandb
PROJ=${WANDB_PROJECT_OVERRIDE:-nersc_compress_opd_qwen4b}
RID=${RUN_ID:-opd_compress_svd_nystrom_combined_ot3}
INT=${SYNC_INTERVAL:-600}
export WANDB_DIR=/pscratch/sd/y/yequan/opd/wandb
echo "[compress-opd-wandb-sync] project=$PROJ run=$RID interval=${INT}s wbdir=$WB"
while true; do
  # sync every offline segment for this run id (ascending = oldest first so steps
  # land in order); already-synced dirs are no-ops. ".nofreeze_bad" dirs are excluded.
  for d in $(ls -dtr "$WB"/offline-run-*-"$RID" 2>/dev/null); do
    case "$d" in *.bad|*.nofreeze_bad) continue;; esac
    wandb sync --project "$PROJ" "$d" 2>&1 | grep -iE "Syncing|done|error|already synced" | tail -1
  done
  sleep "$INT"
done
