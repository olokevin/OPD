#!/bin/bash
# lr_search.sh — LR sweep for OPD-math (full / lora / fura) across GPU 4 and 5.
#
# Runs the 9-run grid one-per-GPU, two at a time, launching the next queued run
# as soon as a GPU frees. Each run uses RAY_ISOLATE=1 so concurrent single-GPU
# runs get a private Ray head (unique port + temp dir) and don't tear down each
# other via `ray stop --force`.
#
#   full : LR = 1e-6 5e-6 1e-5
#   lora : LR = 1e-5 5e-5 1e-4
#   fura : LR = 5e-5 1e-4 2e-4   (FurA = BlockTT, bf16 frozen core)
#
# Usage:
#   conda activate verl
#   bash scripts/opd/math/lr_search.sh                       # all 9 runs
#   SKIP_RUNS="full:1e-6" bash scripts/opd/math/lr_search.sh # skip smoke-tested run(s)
#   GPUS="4 5" bash scripts/opd/math/lr_search.sh            # choose GPU slots
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS:-4 5})                       # GPU slots, one run per GPU at a time
LOG_DIR=${LOG_DIR:-logs/lr_search}
mkdir -p "$LOG_DIR"
MANIFEST="$LOG_DIR/manifest.tsv"
POLL=${POLL:-30}                          # seconds between slot polls

# ---- the run grid: "config:LR" ----
RUNS=(
  "full:1e-6" "full:5e-6" "full:1e-5"
  "lora:1e-5" "lora:5e-5" "lora:1e-4"
  "fura:5e-5" "fura:1e-4" "fura:2e-4"
)
# Drop runs listed in SKIP_RUNS (space-separated "cfg:lr" tokens, e.g. the
# already smoke-tested run): SKIP_RUNS="full:1e-6 lora:5e-5".
SKIP_RUNS=${SKIP_RUNS:-}
filtered=()
for r in "${RUNS[@]}"; do
  skip=0
  for s in $SKIP_RUNS; do [ "$s" = "$r" ] && skip=1 && break; done
  if [ "$skip" = "1" ]; then
    echo "[lr_search] skipping ${r%%:*} LR=${r##*:} (in SKIP_RUNS)"
    continue
  fi
  filtered+=("$r")
done
RUNS=("${filtered[@]}")

echo "[lr_search] $(date) — ${#RUNS[@]} runs across GPUs: ${GPUS[*]}"
echo -e "config\tlr\tgpu\tpid\tlog\tstarted" >> "$MANIFEST"

# Track which run-PID occupies each GPU slot (parallel arrays).
declare -A SLOT_PID      # gpu -> pid of the run currently on it (or empty)

launch() {
  local cfg=$1 lr=$2 gpu=$3
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local log="$LOG_DIR/${cfg}_lr${lr}_gpu${gpu}_${ts}.log"
  # RAY_ISOLATE=1 -> private Ray head per run; CUDA_LAUNCH_BLOCKING=0 for speed.
  RAY_ISOLATE=1 CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=$gpu LR=$lr \
    bash "scripts/opd/math/${cfg}.sh" > "$log" 2>&1 &
  local pid=$!
  SLOT_PID[$gpu]=$pid
  echo -e "${cfg}\t${lr}\t${gpu}\t${pid}\t${log}\t${ts}" >> "$MANIFEST"
  echo "[lr_search] launched $cfg LR=$lr on GPU $gpu (pid $pid) -> $log"
}

# A GPU is "busy" if it already has >5GB allocated (e.g. a smoke-test run kept
# as the real full:1e-6). The queue leaves it alone and fills it once it frees.
gpu_busy() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] && [ "$used" -gt 5000 ] 2>/dev/null
}

idx=0
n=${#RUNS[@]}
# Initial fill: one run per free GPU slot (skip GPUs already in use).
for gpu in "${GPUS[@]}"; do
  [ $idx -lt $n ] || break
  if gpu_busy "$gpu"; then
    echo "[lr_search] GPU $gpu busy (>5GB used) — leaving free, will fill when it clears"
    continue
  fi
  r=${RUNS[$idx]}; launch "${r%%:*}" "${r##*:}" "$gpu"; idx=$((idx+1))
  sleep 20   # stagger so two Ray heads don't race on startup
done

# Queue loop: launch the next queued run on any GPU that is (a) a tracked slot
# whose run just exited, or (b) an untracked slot that has since gone idle (e.g.
# the smoke-test GPU finishing the run we kept outside the queue).
while [ $idx -lt $n ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
  sleep "$POLL"
  for gpu in "${GPUS[@]}"; do
    pid=${SLOT_PID[$gpu]:-}
    if [ -n "$pid" ]; then
      kill -0 "$pid" 2>/dev/null && continue   # still running
      wait "$pid" 2>/dev/null; rc=$?
      echo "[lr_search] $(date) — run on GPU $gpu (pid $pid) exited rc=$rc"
      unset 'SLOT_PID[$gpu]'
      # Best-effort cleanup of this GPU's isolated Ray head.
      RAY_TMPDIR="/tmp/ray_opd_gpu${gpu}" ray stop --force >/dev/null 2>&1 || true
    fi
    # Slot is now untracked. If work remains and the GPU is idle, fill it.
    if [ $idx -lt $n ] && ! gpu_busy "$gpu"; then
      r=${RUNS[$idx]}; launch "${r%%:*}" "${r##*:}" "$gpu"; idx=$((idx+1))
      sleep 20
    fi
  done
done

echo "[lr_search] $(date) — all ${n} runs finished. Manifest: $MANIFEST"
