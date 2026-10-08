#!/bin/bash
# wandb_append_daemon.sh — keep the ISO-paper wandb runs current (copied from the
# SimpleRL one, branch `simplerl`) by INCREMENTALLY
# appending each new offline segment dir into one continuous run. Replaces the old
# wandb_sync.sh, which only pushed the latest segment and relied on wandb merging
# same-id offline runs (it doesn't) — so runs showed only the first segment.
#
# Correct approach (proven): wandb sync --append --id <run> <dir> accumulates history;
# the FIRST segment creates the run. Two hard-won requirements:
#   * export WANDB_ENTITY — else a detached shell uploads to an empty-entity limbo.
#   * per-run state file of already-appended dirs, so each segment is pushed exactly
#     once (no duplicate steps). The CURRENT (newest, still-growing) segment is held
#     back until a newer segment appears, so we never append a half-written dir.
#
#   setsid nohup bash slurm/iso/wandb_append_daemon.sh > /pscratch/sd/$USER/opd/logs/wandb_append_daemon.log 2>&1 &
set -u
source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/y/yequan/opd/envs/verl
export WANDB_DIR=/pscratch/sd/y/yequan/opd/wandb
export WANDB_ENTITY=${WANDB_ENTITY:-yequan_zhao-university-of-california-santa-barbara}
PROJ=${WANDB_PROJECT:-iso_rlvr_qwen3_4b}
WB=/pscratch/sd/y/yequan/opd/wandb/wandb
STATE_DIR=/pscratch/sd/y/yequan/opd/logs/iso/wandb_state; mkdir -p "$STATE_DIR"
RUN_IDS=(
  "q3-4b_full_lr2e-6"
  "q3-4b_fura_lr2e-5"
  "q3-4b_iso_lr7p5e-7"
  "q3-4b_isobtt_lr2p5e-6"
)
INTERVAL=${INTERVAL:-900}

# Is the controller for this run id still alive? (JOB_TAG lives in the process env,
# not argv, so match via /proc/<pid>/environ.) Uses return, NOT exit — an `exit`
# here would kill the whole daemon.
ctl_alive_for() {
  local p t
  for p in $(pgrep -f "slurm/iso/controller.sh" 2>/dev/null); do
    t=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep '^JOB_TAG=' | cut -d= -f2)
    [ "$t" = "$1" ] && return 0
  done
  # batch runs (submit_batch.sh): job name = run id, running or queued
  [ -n "$(squeue --me -h -n "$1" -o %i 2>/dev/null)" ] && return 0
  return 1
}

append_new() {
  # Sync into "<rid>_c" (the offline dirs keep the original display NAME, and the
  # bare id may be tombstoned from an earlier delete). State keyed on the target id.
  local rid=$1 target="${1}_c" state
  state="$STATE_DIR/${1}_c.done"
  touch "$state"
  # real segment dirs (skip aborted tiny <1MB), chronological (= step) order
  mapfile -t DIRS < <(
    for d in "$WB"/offline-run-*-"$rid"; do
      [ -d "$d" ] || continue
      kb=$(du -s "$d" 2>/dev/null | cut -f1); [ "${kb:-0}" -ge 1024 ] && echo "$d"
    done | sort
  )
  local n=${#DIRS[@]}
  [ "$n" -eq 0 ] && return
  # hold back the newest dir IF its run controller is still alive (segment may be
  # mid-write); once training ends, sync everything including the last.
  local hold=0
  ctl_alive_for "$rid" && hold=1
  local limit=$n; [ "$hold" = 1 ] && limit=$((n-1))
  local i=0
  for d in "${DIRS[@]}"; do
    i=$((i+1)); [ "$i" -gt "$limit" ] && break
    local b; b=$(basename "$d")
    grep -qxF "$b" "$state" && continue
    if [ ! -s "$state" ]; then
      out=$(wandb sync --id "$target" -p "$PROJ" "$d" 2>&1)
    else
      out=$(wandb sync --append --id "$target" -p "$PROJ" "$d" 2>&1)
    fi
    if echo "$out" | grep -qi "done\."; then
      echo "$b" >> "$state"; echo "[$(date +%H:%M:%S)] $target <- $b (done)"
    else
      echo "[$(date +%H:%M:%S)] $target <- $b FAILED: $(echo "$out" | tail -1)"
    fi
  done
}

echo "=== wandb append daemon start $(date) ==="
while true; do
  for rid in "${RUN_IDS[@]}"; do append_new "$rid"; done
  sleep "$INTERVAL"
done
