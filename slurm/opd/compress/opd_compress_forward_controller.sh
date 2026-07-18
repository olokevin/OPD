#!/bin/bash
# compress->OPD, FORWARD-only student (wiki D0). 2 interactive nodes (8 GPUs),
# lr=1e-6, full model, auto-resume across the 4h interactive cap. Same teacher/
# dataset/hyperparameters as slurm/opd/full; student = compressed Qwen3-4B.
# Reuses the validated opd_2node_inside.sh + compress env (PEFT none).
#
# Launch detached from a login node (UNIQUE controller name -> safe to pkill):
#   nohup bash slurm/opd/compress/opd_compress_forward_controller.sh \
#     > /pscratch/sd/$USER/opd/logs/compress_fwd_controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -u
OPD_REPO=/global/u1/y/yequan/Project/OPD
DATA_ROOT=/pscratch/sd/y/yequan/opd
ACCOUNT=m4788_g
ALLOC_NODES=2
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}
STALL_LIMIT=${STALL_LIMIT:-3}

# ---- recipe config (exported -> inside.sh -> env.sh -> on_policy_distillation.sh) ----
export ENV_SCRIPT=opd/compress/opd_2node_env_compress.sh
export ACTOR_MODEL_PATH=${DATA_ROOT}/compress_opd/students/svd_nystrom_r07_forward
export CKPT_PATH=${DATA_ROOT}/checkpoints/opd_compress_svd_nystrom_forward
export EXPERIMENT_NAME=opd_compress_svd_nystrom_forward
export WANDB_RUN_ID=opd_compress_svd_nystrom_forward
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k
export LR=1e-6
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-20}

CKPT="$CKPT_PATH"
ITERFILE=${CKPT}/latest_checkpointed_iteration.txt
mkdir -p "$CKPT" "${DATA_ROOT}/logs"
cur_iter() { cat "$ITERFILE" 2>/dev/null | tr -dc '0-9' || echo 0; }

if [ ! -f "${ACTOR_MODEL_PATH}/config.json" ]; then
  echo "=== ERROR: compressed student missing: $ACTOR_MODEL_PATH (run nersc_build_students.sh first) ==="; exit 1; fi

# Keep ONLY the latest checkpoint (the resumed-from ckpt lingers; prune older).
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
echo "=== compress-fwd controller start $(date) | ckpt=$CKPT | nodes=$ALLOC_NODES | lr=$LR ==="
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  START_ITER=$(cur_iter); START_ITER=${START_ITER:-0}
  RUNLOG=${DATA_ROOT}/logs/compress_fwd_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).log
  echo "=== attempt $ATTEMPT $(date) | start_iter=${START_ITER} | runlog=$RUNLOG ==="
  salloc --nodes "$ALLOC_NODES" --qos interactive --time 4:00:00 \
         --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
         bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh" > "$RUNLOG" 2>&1
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
echo "=== compress-fwd controller end $(date) | final_iter=$(cur_iter) ==="
