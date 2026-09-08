#!/bin/bash
# bench_parity.sh -- BP vs es-prefill step time on the OPSD setting, measured HONESTLY.
#
# Both arms run ALONE on one GPU, sequentially, with validation and checkpointing off,
# on the same data in the same order.  Median over steps 2..N (step 1 is cold: the
# reward module fires for the first time inside the timed phase).
#
# This has to be its own benchmark because co-tenancy distorts it badly: when another
# job landed on the same card, BP's step went 36 s -> 80 s with `update_actor` tripling
# (optimizer-offload traffic) -- so any parity number measured under sharing is void.
#
#   TRAIN_GPU=5 bash scripts/opsd/bench_parity.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU=${TRAIN_GPU:-5}
STEPS=${STEPS:-8}
OUT=${OUT:-logs/opsd/bench}
mkdir -p "$OUT"

wait_gpu_free () {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
    [ "${used:-99999}" -le 1000 ] && return 0
    sleep 10
  done
  echo "GPU $GPU never drained"; return 1
}

reap () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU"); do kill -9 "$p" 2>/dev/null; done
  pkill -u "$(id -u)" -f "ray_opd_gpu${GPU}/sessio[n]_" 2>/dev/null
  sleep 10
}

report () {  # $1=log $2=label
  tr '\r' '\n' < "$1" | grep -oE "step:[0-9]+ .*" | \
  /home/yequan/miniconda3/envs/verl/bin/python -c "
import sys, re, statistics as st
rows=[]
for l in sys.stdin:
    g=lambda k: (lambda m: float(m.group(1)) if m else None)(re.search(k+r':([-\d.e+]+)', l))
    s=re.match(r'step:(\d+)', l)
    if s and g('timing_s/step'): rows.append((int(s.group(1)), g('timing_s/step'), g('timing_s/gen'),
        g('timing_s/update_actor'), g('timing_s/es_rails'), g('timing_s/compute_rm_score')))
warm=[r for r in rows if r[0]>1]
if not warm: print('$2: no warm steps'); raise SystemExit
med=lambda i: st.median([r[i] for r in warm if r[i] is not None]) if any(r[i] is not None for r in warm) else float('nan')
print(f'$2  n={len(warm)}  step={med(1):6.1f}s  gen={med(2):5.1f}  update={med(3):5.1f}  rails={med(4):5.1f}  teacher={med(5):5.1f}')
"
}

COMMON="TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False WANDB_MODE=disabled"

echo "############ BP ############"
wait_gpu_free
env TRAIN_GPU=$GPU OPSD_STEPS=$STEPS $COMMON EXPERIMENT_NAME=bench_bp LOG_DIR="$OUT" \
  bash scripts/opsd/bp_opsd.sh > "$OUT/bp.log" 2>&1
reap; report "$OUT/bp.log" "BP        "

for N in 2 4 8; do
  echo "############ ES N=$N ############"
  wait_gpu_free
  env TRAIN_GPU=$GPU OPSD_STEPS=$STEPS ES_STEPS=$STEPS ES_SIGMA=1e-3 ES_ALPHA=2.19e-5 ES_N_RAILS=$N \
    $COMMON EXPERIMENT_NAME=bench_es_N$N LOG_DIR="$OUT" \
    bash scripts/opsd/es_opsd.sh > "$OUT/es_N$N.log" 2>&1
  reap; report "$OUT/es_N$N.log" "ES N=$N    "
done

echo; echo "===== SUMMARY (all measured alone on GPU $GPU) ====="
report "$OUT/bp.log" "BP        "
for N in 2 4 8; do report "$OUT/es_N$N.log" "ES N=$N    "; done
