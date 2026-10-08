#!/bin/bash
#SBATCH -A m4788_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
# batch.sbatch — one regular-QOS segment of an ISO-paper run (2 nodes x 4 A100).
# Submitted by submit_batch.sh as a chain of afterany-dependent jobs; each job resumes
# from the fixed ckpt dir (resume_mode=auto) and exits at once if the run is finished.
# Env (PEFT_MODE, LR, EXP, CKPT, WANDB_RUN_ID, ...) comes from submit_batch.sh via --export.
set -u
OPD_REPO=/global/u1/y/yequan/Project/OPD
export ENV_SCRIPT=iso/env.sh RAYENV_SCRIPT=iso/rayenv.sh
ITERFILE=$CKPT/latest_checkpointed_iteration.txt
TOTAL=${TOTAL_STEPS:-160}
L=$(tr -dc '0-9' < "$ITERFILE" 2>/dev/null)
echo "=== [$WANDB_RUN_ID] batch job $SLURM_JOB_ID start $(date) | iter=${L:-0}/$TOTAL ==="
if [ -n "$L" ] && [ "$L" -ge "$TOTAL" ]; then echo "=== already complete ==="; exit 0; fi
# keep only the latest ckpt (the resumed-from one lingers across jobs)
if [ -n "$L" ]; then
  for d in "$CKPT"/global_step_*; do
    n=$(basename "$d" | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt "$L" ] && rm -rf "$d"
  done
fi
bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh"
RC=$?
echo "=== [$WANDB_RUN_ID] batch job $SLURM_JOB_ID end $(date) rc=$RC iter=$(tr -dc '0-9' < "$ITERFILE")/$TOTAL ==="
exit $RC
