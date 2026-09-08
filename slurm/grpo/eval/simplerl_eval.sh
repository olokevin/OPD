#!/bin/bash
# simplerl_eval.sh — offline table-1 eval of a SimpleRL-aligned GRPO checkpoint via
# verl VAL_ONLY mode (the proven cross-full/fura path: resume the sharded FSDP ckpt
# and validate through the in-framework rollout, which materializes fura BlockTT ->
# dense for vLLM — the standalone eval_math500_simplerl.py can't, since the saved
# merged_hf/ is empty for fura). Greedy pass@1 (SimpleRL-Zoo Table 1 protocol) on the
# SimpleRL-structured benchmark parquets the models trained on.
#
# Usage: MODE=full|fura LR=5e-7 bash slurm/grpo/eval/simplerl_eval.sh
#   MODE picks the driver env; LR picks the run's checkpoint dir + wandb/exp name.
# One 2-node salloc with a retry loop (transient NERSC "Connection timed out" revokes
# ~1/4 sallocs). Results = MATH-500 / AIME24 / AMC23 / Minerva / Olympiad-Bench acc,
# logged to console + wandb project nersc_simplerl_qwen2p5_7b (run *_eval).
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
ACCOUNT=${ACCOUNT:-m4788_g}
MODE=${MODE:-${1:-full}}
LR=${LR:-${2:-5e-7}}
MAX_RETRY=${MAX_RETRY:-8}

case "$MODE" in
  full) export ENV_SCRIPT=grpo/simplerl/full_env.sh ;;
  fura) export ENV_SCRIPT=grpo/simplerl/fura_env.sh ;;
  *) echo "MODE must be full|fura" >&2; exit 1 ;;
esac

TAG=simplerl_${MODE}_lr${LR}
export CKPT_PATH=${CKPT_PATH:-${DATA_ROOT}/checkpoints/${TAG}}
export EXPERIMENT_NAME=${TAG}_eval
export WANDB_RUN_ID=${TAG}_eval
export PROJECT_NAME=nersc_simplerl_qwen2p5_7b
export WANDB_PROJECT=nersc_simplerl_qwen2p5_7b
export LR   # not used for training here, but the env scripts require it set

# ---- eval config: greedy pass@1 on the 5 SimpleRL benchmark parquets ----
export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export VAL_DO_SAMPLE=False          # greedy (SimpleRL Table 1)
export VAL_N=${VAL_N:-1}            # pass@1
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-16384}
export TEST_FILE='["datasets/test_data/MATH-500_simplerl/test.parquet","datasets/test_data/AIME24_simplerl/test.parquet","datasets/test_data/AMC23_simplerl/test.parquet","datasets/test_data/Minerva_simplerl/test.parquet","datasets/test_data/Olympiad-Bench_simplerl/test.parquet"]'
# resume_mode=auto loads the latest checkpoint; keep it (env scripts set it).

if [ ! -f "${CKPT_PATH}/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: no checkpoint at ${CKPT_PATH}" >&2; exit 1
fi
echo "=== simplerl eval MODE=$MODE LR=$LR ckpt=$CKPT_PATH (iter $(cat ${CKPT_PATH}/latest_checkpointed_iteration.txt 2>/dev/null)) ==="

mkdir -p "${DATA_ROOT}/logs"
ATTEMPT=0
while [ "$ATTEMPT" -lt "$MAX_RETRY" ]; do
  ATTEMPT=$((ATTEMPT+1))
  RUNLOG=${DATA_ROOT}/logs/eval_${TAG}_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== eval attempt $ATTEMPT -> $RUNLOG ==="
  salloc --nodes 2 --qos interactive --time 2:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh" > "$RUNLOG" 2>&1
  RC=$?
  # transient salloc failure (never allocated) -> retry without counting hard
  if grep -qE "Unable to allocate resources|salloc: error" "$RUNLOG" 2>/dev/null \
     && ! grep -q "HEAD=" "$RUNLOG" 2>/dev/null; then
    echo "=== eval attempt $ATTEMPT: salloc did not allocate; retrying ==="; sleep 60; ATTEMPT=$((ATTEMPT-1)); continue
  fi
  echo "=== eval attempt $ATTEMPT done rc=$RC ==="
  # verl prints val metrics as 'val-core/.../acc' lines; surface them
  grep -oE "val-core[^ ]*(acc|accuracy)[^ ]*[:/][0-9.]+|val/[^ ]*:[0-9.]+" "$RUNLOG" | sed 's/^/  RESULT: /' | tail -40
  [ "$RC" -eq 0 ] && { echo "=== eval OK ($RUNLOG) ==="; break; }
  echo "=== eval attempt $ATTEMPT failed rc=$RC; see $RUNLOG ==="
done
