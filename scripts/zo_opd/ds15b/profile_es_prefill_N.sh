#!/bin/bash
# profile_es_prefill_N.sh -- step-time profile of es-prefill at 256 seqs/step (64 x n=4)
# for varying rail count N, non-antithetic sampling. 2 training steps per point, no eval.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
GPU=${PROFILE_GPU:-5}
mkdir -p logs/ds15b/es/profile
for N in 2 4 8 16; do
  echo "=== profiling N=$N ==="
  TRAIN_GPU=$GPU MINI_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 \
  ES_N_RAILS=$N ES_SIGMA=1e-3 ES_ALPHA=6e-4 ES_ANTITHETIC=False ES_STEPS=2 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=-1 WANDB_MODE=disabled \
  EXPERIMENT_NAME=profile_b256_N$N LOG_DIR=logs/ds15b/es/profile \
  bash scripts/zo_opd/ds15b/es_opd.sh > logs/ds15b/es/profile/launcher_N$N.log 2>&1
  bash scripts/zo_opd/ds15b/stop_run.sh profile_b256_N$N $GPU >/dev/null 2>&1 || true
done
echo "=== profile table ==="
/home/yequan/miniconda3/envs/verl/bin/python - <<'PYEOF'
import re, glob
print(f"{'N':>3s} {'step1':>7s} {'step2':>7s} {'gen':>6s} {'teacher':>8s} {'logprob':>8s} {'rails':>7s} {'s/rail':>7s}")
for N in [2, 4, 8, 16]:
    logs = sorted(glob.glob("logs/ds15b/es/profile/run_*.log"))
    best = None
    for f in logs:
        head = open(f, errors="ignore").read(200000)
        if f"experiment_name=profile_b256_N{N} " in head or f"profile_b256_N{N}\n" in head or f"exp profile_b256_N{N}" in head:
            best = f
    if not best:
        # fall back: match by es/n_rails in step lines
        for f in logs:
            txt = open(f, errors="ignore").read()
            if f"es/n_rails:{N} " in txt or f"es/n_rails:{N}.0" in txt:
                best = f
    if not best: print(f"{N:3d}  (no log)"); continue
    steps = {}
    for line in open(best, errors="ignore"):
        m = re.search(r"step:(\d+) - actor", line)
        if not m: continue
        d = {}
        for tok in line.split(" - "):
            if ":" in tok:
                k, v = tok.split(":", 1)
                v = re.sub(r"np\.float64\(|\)", "", v.strip())
                try: d[k] = float(v)
                except: pass
        steps[int(m.group(1))] = d
    s1, s2 = steps.get(1, {}), steps.get(2, {})
    g = lambda d, k: d.get(k, float("nan"))
    print(f"{N:3d} {g(s1,'timing_s/step'):7.1f} {g(s2,'timing_s/step'):7.1f} {g(s2,'timing_s/gen'):6.1f} {g(s2,'timing_s/reward'):8.1f} {g(s2,'timing_s/compute_log_prob'):8.1f} {g(s2,'timing_s/es_rails'):7.1f} {g(s2,'timing_s/es_per_rail'):7.2f}")
PYEOF
echo "profile done"
