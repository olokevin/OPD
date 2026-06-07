#!/bin/bash
# job1: FULL OPD finetune, 4 interactive nodes (16 GPUs), lr=1e-6, auto-resume.
# teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500, student Qwen/Qwen3-1.7B,
# dataset datasets/dapo-math-17k.parquet. Reuses the validated inside.sh + 2node
# env (PEFT none); only the checkpoint dir, run name, node count, LR, and dataset
# are overridden via env (all flow through to env.sh / on_policy_distillation.sh).
#
# Launch detached from a login node:
#   nohup bash slurm/zo_opd_job1_full_controller.sh > /pscratch/sd/$USER/opd/logs/job1_controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -u
OPD_REPO=/global/u1/y/yequan/Project/OPD
DATA_ROOT=/pscratch/sd/y/yequan/opd
ACCOUNT=m4788_g
ALLOC_NODES=4
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}
STALL_LIMIT=${STALL_LIMIT:-3}

# ---- job1 config (exported so inside.sh -> env.sh -> on_policy_distillation.sh see it) ----
export ENV_SCRIPT=opd/full/zo_opd_2node_env.sh                 # full (PEFT none)
export CKPT_PATH=${DATA_ROOT}/checkpoints/job1_full_dapo_lr1e-6
export EXPERIMENT_NAME=opd_full_dapo_lr1e-6
export WANDB_RUN_ID=opd_full_dapo_lr1e-6
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
export LR=1e-6
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-20}

CKPT="$CKPT_PATH"
ITERFILE=${CKPT}/latest_checkpointed_iteration.txt
mkdir -p "$CKPT" "${DATA_ROOT}/logs"
cur_iter() { cat "$ITERFILE" 2>/dev/null | tr -dc '0-9' || echo 0; }

# Keep ONLY the latest checkpoint (see zo_opd_2node_controller.sh for rationale).
prune_loop() {
  while true; do
    L=$(cat "$ITERFILE" 2>/dev/null | tr -dc '0-9')
    if [ -n "$L" ]; then
      for d in "$CKPT"/global_step_*; do
        [ -d "$d" ] || continue
        n=$(basename "$d" | tr -dc '0-9')
        [ -n "$n" ] && [ "$n" -lt "$L" ] && rm -rf "$d"
      done
    fi
    sleep 60
  done
}
prune_loop & PRUNE_PID=$!
trap 'kill $PRUNE_PID 2>/dev/null' EXIT

ATTEMPT=0; PREV_ITER=-1; STALL=0
echo "=== job1(full) controller start $(date) | ckpt=$CKPT | nodes=$ALLOC_NODES ==="
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START_ITER=$(cur_iter); START_ITER=${START_ITER:-0}
  RUNLOG=${DATA_ROOT}/logs/job1_full_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== attempt $ATTEMPT $(date) | start_iter=${START_ITER} | runlog=$RUNLOG ==="
  salloc --nodes "$ALLOC_NODES" --qos interactive --time 4:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/opd/zo_opd_2node_inside.sh" > "$RUNLOG" 2>&1
  RC=$?
  END_ITER=$(cur_iter); END_ITER=${END_ITER:-0}
  TOTAL=$(grep -oE "Total training steps: [0-9]+" "$RUNLOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  echo "=== attempt $ATTEMPT done rc=$RC | iter ${START_ITER}->${END_ITER} | total=${TOTAL:-?} ==="
  if [ -n "$TOTAL" ] && [ "$END_ITER" -ge "$TOTAL" ] 2>/dev/null; then
    echo "=== TRAINING COMPLETE: iter ${END_ITER} >= total ${TOTAL} ==="; break; fi
  if [ "$RC" -eq 0 ]; then echo "=== driver exited 0 at iter ${END_ITER} ==="; break; fi
  if [ "$END_ITER" -le "$PREV_ITER" ] && [ "$END_ITER" -le "$START_ITER" ]; then
    STALL=$((STALL+1)); echo "=== no progress (stall ${STALL}/${STALL_LIMIT}) ==="; else STALL=0; fi
  PREV_ITER=$END_ITER
  if [ "$STALL" -ge "$STALL_LIMIT" ]; then echo "=== STOPPING: no progress across ${STALL_LIMIT} allocations ($RUNLOG) ==="; break; fi
  sleep 20
done
echo "=== job1(full) controller end $(date) | final_iter=$(cur_iter) ==="
