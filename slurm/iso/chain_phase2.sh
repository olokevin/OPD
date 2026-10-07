#!/bin/bash
# chain_phase2.sh — wait until the phase-1 controllers (full, fura) have exited,
# then launch phase 2 (iso, isobtt). The interactive QOS caps us at 4 nodes = two
# 2-node runs at a time.
#   setsid nohup bash slurm/iso/chain_phase2.sh > /pscratch/sd/$USER/opd/logs/iso/chain_phase2.log 2>&1 &
OPD_REPO=/global/u1/y/yequan/Project/OPD
alive() {   # any controller whose JOB_TAG is $1 still running?
  for p in $(pgrep -f "slurm/iso/controller.sh"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "JOB_TAG=$1" && return 0
  done; return 1
}
launched=""
while :; do
  for pair in "q3-4b_full_lr2e-6:iso" "q3-4b_fura_lr2e-5:isobtt"; do
    tag=${pair%%:*}; next=${pair##*:}
    case " $launched " in *" $next "*) continue ;; esac
    if ! alive "$tag"; then
      echo "$(date) $tag finished -> launching $next"
      VAL_BEFORE=False bash "$OPD_REPO/slurm/iso/launch.sh" "$next"
      launched="$launched $next"
    fi
  done
  [ "$(wc -w <<< "$launched")" -ge 2 ] && break
  sleep 300
done
echo "$(date) phase 2 launched:$launched"
