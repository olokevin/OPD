#!/bin/bash
# launch.sh <arm> — start the detached controller for one ISO-paper run.
#   full    weight-space AdamW, LR 2e-6 (paper's best AdamW)
#   fura    BlockTT small core only, S merged into frozen core, LR 2e-5 (10x full), FSDP2
#   iso     ISO-AdamW, LR 7.5e-7 (paper)
#   isobtt  ISO on fura's block-wise SVD (U_j, V_j trained), LR 2.5e-6 (tuned, iso.md)
# Extra env (TOTAL_STEPS, TEST_FREQ, SMOKE, LR, ...) passes through.
#   bash slurm/iso/launch.sh full
set -eu
ARM=${1:?arm: full|fura|iso|isobtt}
OPD_REPO=/global/u1/y/yequan/Project/OPD
DATA_ROOT=/pscratch/sd/y/yequan/opd
export VAL_BEFORE=${VAL_BEFORE:-True}
case "$ARM" in
  full)   export PEFT_MODE=none   LR=${LR:-2e-6} ;;
  fura)   export PEFT_MODE=fura   LR=${LR:-2e-5} ACTOR_STRATEGY=fsdp2 OPTIM_OFFLOAD=False ;;
  iso)    export PEFT_MODE=iso    LR=${LR:-7.5e-7} ;;
  isobtt) export PEFT_MODE=isobtt LR=${LR:-2.5e-6} ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
SUFFIX=${SUFFIX:-}
[ "${SMOKE:-0}" = "1" ] && SUFFIX=_smoke$SUFFIX
export WANDB_RUN_ID=${WANDB_RUN_ID:-q3-4b_${ARM}_lr$(echo "$LR" | tr . p)${SUFFIX}}
export EXP=${EXP:-dapo_q3-4b-base_deepmath_${ARM}_lr${LR}_2node${SUFFIX}}
export CKPT=${CKPT:-${DATA_ROOT}/checkpoints/iso/${EXP}}
export JOB_TAG=$WANDB_RUN_ID
mkdir -p "${DATA_ROOT}/logs/iso"
LOG=${DATA_ROOT}/logs/iso/ctl_${JOB_TAG}_$(date +%Y%m%d_%H%M%S).log
setsid nohup bash "$OPD_REPO/slurm/iso/controller.sh" > "$LOG" 2>&1 < /dev/null &
echo "launched $ARM: pid $! | exp $EXP | wandb id $WANDB_RUN_ID | log $LOG"
