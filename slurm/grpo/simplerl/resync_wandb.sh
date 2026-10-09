#!/bin/bash
# resync_wandb.sh <run_id> [target_id] — rebuild ONE continuous wandb run from all
# of a run's offline segment dirs, appending them in step order into a single run.
#
# Hard-won facts baked into this script:
#   * wandb sync --append --id <t> <dir> accumulates history; first dir creates it.
#   * MUST export WANDB_ENTITY — a detached shell otherwise uploads to an
#     empty-entity limbo (wandb.ai//<project>/...).
#   * NEVER delete + recreate the SAME id: wandb tombstones a deleted id
#     ("previously created and deleted; try a new run id", 409). So we sync into a
#     NEW id "<run_id>_c" (the offline dirs keep the original DISPLAY NAME, so the
#     UI still shows "<run_id>"). No deletion anywhere.
#   * A per-target state file records appended dirs -> idempotent / resumable, no
#     duplicate steps on re-run.
#
#   bash slurm/grpo/simplerl/resync_wandb.sh simplerl_fura_lr1e-5
set -u
RID="${1:?usage: resync_wandb.sh <run_id> [target_id]}"
TARGET="${2:-${RID}_c}"
ENT=${WANDB_ENTITY:-yequan_zhao-university-of-california-santa-barbara}
PROJ=${PROJECT_NAME:-nersc_simplerl_qwen2p5_7b}
WB=/pscratch/sd/y/yequan/opd/wandb/wandb
STATE_DIR=/pscratch/sd/y/yequan/opd/logs/wandb_state; mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/$TARGET.done"; touch "$STATE"
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/verl
export WANDB_DIR=/pscratch/sd/y/yequan/opd/wandb
export WANDB_ENTITY="$ENT"

echo "=== resync $RID -> id=$TARGET (name stays $RID) @ $ENT/$PROJ ==="
mapfile -t DIRS < <(
  for d in "$WB"/offline-run-*-"$RID"; do
    [ -d "$d" ] || continue
    kb=$(du -s "$d" 2>/dev/null | cut -f1); [ "${kb:-0}" -ge 1024 ] && echo "$d"
  done | sort
)
echo "  ${#DIRS[@]} real segment dirs"
[ "${#DIRS[@]}" -eq 0 ] && { echo "  nothing to sync"; exit 1; }

for d in "${DIRS[@]}"; do
  b=$(basename "$d")
  grep -qxF "$b" "$STATE" && { echo "  [skip already-synced] $b"; continue; }
  if [ ! -s "$STATE" ]; then
    out=$(wandb sync --id "$TARGET" -p "$PROJ" "$d" 2>&1)
  else
    out=$(wandb sync --append --id "$TARGET" -p "$PROJ" "$d" 2>&1)
  fi
  if echo "$out" | grep -qi "done\."; then
    echo "$b" >> "$STATE"; echo "  [done] $b"
  else
    echo "  [FAILED] $b :: $(echo "$out" | grep -iE 'error' | tail -1)"
  fi
done
echo "=== resync $RID complete (target id $TARGET) ==="
