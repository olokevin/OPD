#!/bin/bash
# check_alllayer_e2e.sh — Task F2: NON-OPTIONAL end-to-end TRAINING smoke of the
# fully-CUDA-graphed packed all-layer decode (decode_mode=packed_graphed,
# en_layerwise=false) on the REAL Qwen3-1.7B student + Qwen3-4B teacher.
#
# Proves the all-layer graphed path actually LEARNS: runs ~20 update steps,
# probing a FIXED held-out set of prompts for clean teacher-KL on each eval step,
# then GREPS the log to assert the 5 acceptance criteria:
#   1. 20 steps completed.
#   2. train/dW_norm/<layer> > 0 for EVERY matched down_proj layer (28 for 1.7B).
#   3. train/weight_delta_mean > 0  (per-layer norm-delta IS reported in all-layer
#      mode; per-layer weight_changed_frac is OMITTED by the all-layer branch by
#      design, so we do NOT assert on it).
#   4. train/weight_sync_ok == 1.0 on every step.
#   5. eval/heldout_kl is logged at MULTIPLE steps and TRENDS DOWN (last < first).
#      This is the learning signal.
#
# Run:
#   bash scripts/zo_opd/np_checks/check_alllayer_e2e.sh
# Optional: CUDA_VISIBLE_DEVICES=<1|2|3> bash scripts/.../check_alllayer_e2e.sh
set -u

# ---------------------------------------------------------------- environment
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"   # worktree root (.../np-alllayer-graphed)
cd "$ROOT"

# Worktree verl MUST shadow the editable-installed (stale) main-checkout verl.
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}

# verl conda env python (has ray/torch/vllm; base env does NOT). Used both by the
# launcher (it calls python3 on PATH) and by our log-parsing helpers below.
PYBIN=${PYBIN:-/home/yequan/miniconda3/envs/verl/bin/python}

# GPU policy: 1/2/3 only. Default to 1; caller may override CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export NP_KEEP_CUDA_VISIBLE=1

# ---------------------------------------------------------------- run config
TS=$(date +%Y%m%d_%H%M%S)
TMP_DIR="${ROOT}/.claude/tmp/alllayer_e2e_${TS}"
mkdir -p "$TMP_DIR"
RUN_LOG="${TMP_DIR}/run.log"

# packed_graphed all-layer config, ~20 steps, frequent probes.
export DECODE_MODE=packed_graphed
export EN_LAYERWISE=false
export USE_CUDA_GRAPH=true
export BATCH_SIZE=8
# PACK_WIDTH=4 (not 8): the packed scratch-KV reserves ceil(max_model_len/block_size)
# = 2560 blocks PER prompt (the full context window, not just max_tokens), so the
# B_pack disjoint KV slices must fit num_gpu_blocks. With the teacher co-located
# (gpu_fraction=0.5 -> student GMU 0.30 -> ~11099 blocks), pack_width=8 needs 20480
# blocks and fails the fast-fail guard; pack_width=4 needs 10240 <= 11099. BATCH=8
# then waves into TWO pack_width=4 waves (exercises the multi-wave path too).
export PACK_WIDTH=4
export MAX_RESP_LENGTH=128
export N_SAMPLE=8
export N_ROLLOUT=1
export NUM_ITERATIONS=20
export EVAL_INTERVAL=5          # probes at steps 0,5,10,15,19 (last) => 5 probes
# LR for ALL-LAYER mode must be MUCH smaller than the single-layer default (3e-2):
# 28 down_proj layers update SIMULTANEOUSLY each step, so the aggregate parameter
# move is ~28x larger than a single-layer step. At 3e-2 the heldout teacher-KL
# DIVERGES (3.3->6.2->7.2->13.6 over 20 steps). 1e-3 keeps the per-step update in
# the stable regime so the KL can actually descend. Override LR=... to sweep.
export LR=${LR:-1e-3}
export TRAIN_MAX_SAMPLES=128    # 128 > 2*16 heldout => heldout_prompts non-empty
export VAL_MAX_SAMPLES=16       # small MATH-500 eval (we only gate on heldout_kl)
export NP_LOGGER='["console"]'

export LOG_DIR="$TMP_DIR"
export SAVE_DIR="$TMP_DIR/ckpt"
export EXP="alllayer_e2e"

EXPECTED_LAYERS=28             # Qwen3-1.7B down_proj count (num_hidden_layers)

echo "=========================================================================="
echo " Task F2 e2e smoke: packed_graphed ALL-LAYER, $NUM_ITERATIONS steps"
echo " GPU=$CUDA_VISIBLE_DEVICES  PYTHONPATH(head)=${PYTHONPATH%%:*}"
echo " run log -> $RUN_LOG"
echo "=========================================================================="

# ---------------------------------------------------------------- launch
# The launcher invokes `python3 -m verl.trainer.main_np`; ensure that python3 is the
# verl env (ray/torch/vllm live there). PYTHONPATH already shadows the stale
# editable-installed verl with the worktree's.
export PATH="$(dirname "$PYBIN"):$PATH"
# Tee the launcher's own stdout/stderr into RUN_LOG. The launcher ALSO tees to its
# own LOG_DIR file, but we parse RUN_LOG (captures everything incl. the probe prints).
bash scripts/zo_opd/opd_math_np.sh > "$RUN_LOG" 2>&1
LAUNCH_RC=$?

echo
echo "launcher exit code: $LAUNCH_RC"
echo "=========================================================================="
echo " ASSERTIONS (parsing $RUN_LOG)"
echo "=========================================================================="

FAIL=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAIL=1; }

# ---------------------------------------------------------------- C0: worktree module
# Confirm the worktree verl is the one that ran (the 'ALL-LAYER GRAPHED' debug
# string exists ONLY in the worktree ray_trainer.py). We also report ray_trainer
# module path if main_np logs it; the debug string requires NP_DEBUG_DECODE=1, so
# we instead verify the packed_graphed code path emitted its worktree-only logs by
# checking the assemble/decode markers. As a robust module check, locate the
# ray_trainer.py that PYTHONPATH resolves to and confirm it is in the worktree.
RESOLVED_RT=$("$PYBIN" - <<'PY' 2>/dev/null
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(":")[0])
import verl.trainer.np.ray_trainer as rt
print(rt.__file__)
PY
)
echo "  ray_trainer module resolved to: $RESOLVED_RT"
case "$RESOLVED_RT" in
  *"/np-alllayer-graphed/verl/"*) pass "C0 worktree verl module in use" ;;
  *) fail "C0 ray_trainer is NOT the worktree module (got: $RESOLVED_RT)" ;;
esac

# ---------------------------------------------------------------- C1: 20 steps completed
if grep -q "NP training completed" "$RUN_LOG"; then
  pass "C1 run reached 'NP training completed'"
elif grep -q "step:19" "$RUN_LOG"; then
  pass "C1 step 19 (last of 20) logged"
else
  fail "C1 run did NOT complete 20 steps (no 'NP training completed' and no step:19)"
fi

# ---------------------------------------------------------------- C2: 28/28 layers dW>0
# Collect the SET of distinct down_proj layers that showed dW_norm>0 on ANY step.
# Console line: 'step:N - train/dW_norm/model.layers.K.mlp.down_proj:VALUE - ...'
DW_LAYERS=$("$PYBIN" - "$RUN_LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
# match 'train/dW_norm/<layer>:<value>' tokens
pat = re.compile(r"train/dW_norm/(model\.layers\.\d+\.mlp\.down_proj):([0-9.eE+-]+)")
pos = set()
for layer, val in pat.findall(log):
    try:
        if float(val) > 0.0:
            pos.add(layer)
    except ValueError:
        pass
print(len(pos))
for l in sorted(pos, key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1))):
    print(l)
PY
)
N_DW=$(echo "$DW_LAYERS" | head -1)
echo "  dW>0 on $N_DW distinct down_proj layers (expected $EXPECTED_LAYERS)"
if [ "$N_DW" = "$EXPECTED_LAYERS" ]; then
  pass "C2 $N_DW/$EXPECTED_LAYERS layers had dW_norm>0"
else
  fail "C2 only $N_DW/$EXPECTED_LAYERS layers had dW_norm>0"
  echo "$DW_LAYERS" | tail -n +2 | sed 's/^/      seen: /'
fi

# ---------------------------------------------------------------- C3: weight_delta_mean>0
# train/weight_delta_mean is emitted in all-layer mode (per-layer norm-delta avg).
WDM=$("$PYBIN" - "$RUN_LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
vals = [float(v) for v in re.findall(r"train/weight_delta_mean:([0-9.eE+-]+)", log)]
pos = [v for v in vals if v > 0.0]
print(f"{len(vals)} {len(pos)} {max(vals) if vals else 0.0}")
PY
)
WDM_TOTAL=$(echo "$WDM" | awk '{print $1}')
WDM_POS=$(echo "$WDM" | awk '{print $2}')
WDM_MAX=$(echo "$WDM" | awk '{print $3}')
echo "  weight_delta_mean: $WDM_TOTAL logged, $WDM_POS >0, max=$WDM_MAX"
if [ "$WDM_TOTAL" -gt 0 ] && [ "$WDM_POS" -gt 0 ]; then
  pass "C3 weight_delta_mean>0 on $WDM_POS/$WDM_TOTAL steps (max=$WDM_MAX)"
else
  fail "C3 weight_delta_mean never >0 (logged=$WDM_TOTAL)"
fi

# ---------------------------------------------------------------- C4: weight_sync_ok==1.0
SYNC=$("$PYBIN" - "$RUN_LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
vals = [float(v) for v in re.findall(r"train/weight_sync_ok:([0-9.eE+-]+)", log)]
bad = [v for v in vals if v != 1.0]
print(f"{len(vals)} {len(bad)}")
PY
)
SYNC_TOTAL=$(echo "$SYNC" | awk '{print $1}')
SYNC_BAD=$(echo "$SYNC" | awk '{print $2}')
echo "  weight_sync_ok: $SYNC_TOTAL steps logged, $SYNC_BAD != 1.0"
if [ "$SYNC_TOTAL" -gt 0 ] && [ "$SYNC_BAD" -eq 0 ]; then
  pass "C4 weight_sync_ok == 1.0 on all $SYNC_TOTAL steps"
else
  fail "C4 weight_sync_ok had $SYNC_BAD non-1.0 steps (logged=$SYNC_TOTAL)"
fi

# ---------------------------------------------------------------- C5: heldout_kl trends down
# Probe line: '[Probe @ step N] heldout_kl=X.XXXX (fixed ... prompts; lower=better)'
echo
echo "  --- heldout_kl trajectory (step -> value) ---"
grep -oE "\[Probe @ step [0-9]+\] heldout_kl=[0-9.eE+-]+" "$RUN_LOG" \
  | sed -E 's/\[Probe @ step ([0-9]+)\] heldout_kl=(.*)/      step \1 -> \2/'

KL_VERDICT=$("$PYBIN" - "$RUN_LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read()
pairs = re.findall(r"\[Probe @ step (\d+)\] heldout_kl=([0-9.eE+-]+)", log)
pts = [(int(s), float(v)) for s, v in pairs]
pts.sort()
n = len(pts)
if n < 2:
    print(f"FAIL {n} probes (need >=2)")
    sys.exit(0)
first = pts[0][1]
last = pts[-1][1]
# monotone-ish: count strictly-down adjacent steps
downs = sum(1 for i in range(1, n) if pts[i][1] < pts[i-1][1])
trend = "DOWN" if last < first else ("FLAT" if last == first else "UP")
verdict = "PASS" if last < first else "FAIL"
print(f"{verdict} {n} probes first={first:.4f} last={last:.4f} trend={trend} "
      f"down_steps={downs}/{n-1}")
PY
)
echo "  heldout_kl verdict: $KL_VERDICT"
case "$KL_VERDICT" in
  PASS*) pass "C5 heldout_kl logged at multiple steps and TRENDS DOWN (last<first)" ;;
  *)     fail "C5 heldout_kl did NOT trend down -> $KL_VERDICT" ;;
esac

# ---------------------------------------------------------------- summary
echo
echo "=========================================================================="
if [ "$FAIL" -eq 0 ] && [ "$LAUNCH_RC" -eq 0 ]; then
  echo " RESULT: PASS (all 5 criteria met; launcher rc=0)"
  echo " full log: $RUN_LOG"
  exit 0
else
  echo " RESULT: FAIL (criteria FAIL=$FAIL, launcher rc=$LAUNCH_RC)"
  echo " full log: $RUN_LOG"
  exit 1
fi
