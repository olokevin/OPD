#!/bin/bash
# compress_sft_eval_daemon.sh — login-node daemon: watches the forward+combined
# training output and submits gpu_shared eval jobs (compress_sft_eval_vllm_job.sh) that
# write MATH-500 + AIME24 + MMLU-Pro metric JSONs. The login-node wandb logger
# (log_train_to_wandb.py) then logs them into the SAME run as the training curve.
# Two eval points per the user's request:
#   (1) right after compression — checkpoint-0-merged (post-compress baseline,
#       preserved across save_total_limit rotation; evaluated once per objective).
#   (2) once checkpoints are saved — the latest trained checkpoint-N-merged,
#       throttled to ~every MIN_GAP steps (generation is slow).
# gpu_shared has no per-user job cap, so this never blocks the 2 interactive training jobs.
#
# Launch detached on a login node:
#   nohup bash slurm/compress_sft/compress_sft_eval_daemon.sh \
#     > /pscratch/sd/$USER/opd/compress_sft/logs/eval_daemon_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Stop: pkill -f compress_sft_eval_daemon
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
RATIO=${RATIO:-0.7}
SFT=${DATA_ROOT}/compress_sft/sft/qwen3_4b
STAGE=${DATA_ROOT}/compress_sft/_evalstage
LOGS=${DATA_ROOT}/compress_sft/logs
INT=${EVAL_INTERVAL:-900}           # poll every 15 min
MIN_GAP=${EVAL_MIN_GAP:-500}        # eval ~every 500 trained steps (gen is slow)
mkdir -p "$STAGE" "$LOGS"
declare -A LAST DONE0
echo "[eval-daemon] start $(date) | ratio=$RATIO poll=${INT}s min_gap=${MIN_GAP} | watching $SFT/{forward,combined}_r${RATIO}"

submit_eval() {  # obj step srcdir
  local obj=$1 n=$2 src=$3
  local stg="$STAGE/${obj}_r${RATIO}_step${n}"
  rm -rf "$stg" 2>/dev/null
  # snapshot before save_total_limit rotates it; skip if it vanished mid-copy
  if cp -r "$src" "$stg" 2>/dev/null && [ -f "$stg/config.json" ]; then
    local jid
    jid=$(CKPT="$stg" STEP="$n" OBJECTIVE="$obj" RATIO="$RATIO" \
          sbatch --parsable -o "${LOGS}/evaljob_${obj}_r${RATIO}_step${n}_%j.log" \
                 "$OPD_REPO/slurm/compress_sft/compress_sft_eval_vllm_job.sh" 2>/dev/null)
    echo "[eval-daemon] $(date) submitted eval $obj step=$n job=$jid (staged $stg)"
    return 0
  fi
  rm -rf "$stg" 2>/dev/null
  echo "[eval-daemon] $obj checkpoint-$n vanished before snapshot; will retry"
  return 1
}

while true; do
  for obj in forward combined; do
    base="$SFT/${obj}_r${RATIO}"
    # (1) post-compression baseline (step 0) — eval once; it never rotates away.
    z="$base/checkpoint-0-merged"
    if [ -d "$z" ] && [ -f "$z/config.json" ] && [ -z "${DONE0[$obj]:-}" ]; then
      submit_eval "$obj" 0 "$z" && DONE0[$obj]=1
    fi
    # (2) latest trained checkpoint, throttled to every MIN_GAP steps.
    d=$(ls -dt "$base"/checkpoint-*-merged 2>/dev/null | grep -v 'checkpoint-0-merged' | head -1)
    [ -n "$d" ] || continue
    n=$(basename "$d" | sed -E 's/checkpoint-([0-9]+)-merged/\1/')
    [ -n "$n" ] || continue
    prev=${LAST[$obj]:-0}
    if [ "$n" -ge $((prev + MIN_GAP)) ]; then
      submit_eval "$obj" "$n" "$d" && LAST[$obj]=$n
    fi
  done
  sleep "$INT"
done
