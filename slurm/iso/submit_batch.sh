#!/bin/bash
# submit_batch.sh <arm> [n_jobs] — submit an ISO-paper run as a chain of n_jobs (default 4)
# 24 h regular-QOS batch jobs on 2 nodes; job k+1 starts after job k ends (afterany) and
# resumes from the checkpoint. Same arm settings / ckpt dir / wandb id as launch.sh.
#   bash slurm/iso/submit_batch.sh iso 4
# Extend later by passing START_AFTER=<jobid>.  Cancel: scancel -n <wandb id>
set -eu
ARM=${1:?arm: full|fura|iso|isobtt}; NJOBS=${2:-4}
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
export WANDB_RUN_ID=${WANDB_RUN_ID:-q3-4b_${ARM}_lr$(echo "$LR" | tr . p)}
export EXP=${EXP:-dapo_q3-4b-base_deepmath_${ARM}_lr${LR}_2node}
export CKPT=${CKPT:-${DATA_ROOT}/checkpoints/iso/${EXP}}
export WANDB_PROJECT=${WANDB_PROJECT:-iso_rlvr_qwen3_4b}
mkdir -p "$CKPT" "$DATA_ROOT/logs/iso"
dep=${START_AFTER:-}
for i in $(seq 1 "$NJOBS"); do
  jid=$(sbatch --parsable --export=ALL -J "$WANDB_RUN_ID" ${dep:+--dependency=afterany:$dep} \
        -o "$DATA_ROOT/logs/iso/${WANDB_RUN_ID}_batch_%j.log" "$OPD_REPO/slurm/iso/batch.sbatch")
  echo "submitted $ARM job $i/$NJOBS: $jid ${dep:+(after $dep)}"
  dep=$jid
done
