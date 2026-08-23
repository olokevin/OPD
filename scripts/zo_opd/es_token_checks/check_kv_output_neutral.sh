#!/bin/bash
# check_kv_output_neutral.sh -- the budget-sized scratch-KV reservation must be
# byte-identical to the old full-max_model_len one.
#
# Runs each setting in its OWN process: flipping ES_KV_FULL_RESERVE inside a
# single process reuses the CUDA graph already captured for that bucket, which
# would make the comparison vacuous.
#   KV_GPU=6 bash scripts/zo_opd/es_token_checks/check_kv_output_neutral.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${KV_GPU:-6}; PY=/home/yequan/miniconda3/envs/verl/bin/python
WT=$(pwd); D=$(mktemp -d); T=${T:-256}
E="CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT/verl HF_HOME=/data/yequan/huggingface"
for B in ${WIDTHS:-4 8}; do
    env $E ES_KV_FULL_RESERVE=1 $PY scripts/zo_opd/es_token_checks/dump_packed_wave.py \
        "$B" "$T" "$D/old_$B.pkl" >/dev/null 2>&1
    env $E $PY scripts/zo_opd/es_token_checks/dump_packed_wave.py \
        "$B" "$T" "$D/new_$B.pkl" >/dev/null 2>&1
done
$PY - "$D" ${WIDTHS:-4 8} <<'PYEOF'
import pickle, sys
D, widths = sys.argv[1], [int(x) for x in sys.argv[2:]]
ok = True
for B in widths:
    a = pickle.load(open(f"{D}/old_{B}.pkl", "rb"))
    b = pickle.load(open(f"{D}/new_{B}.pkl", "rb"))
    same = a["tok"] == b["tok"]
    md = max(float((x - y).abs().max()) for x, y in zip(a["pay"], b["pay"]))
    good = same and md == 0.0
    ok &= good
    print(f"  pack_width={B:<3d} tokens identical={same}  payload max|diff|={md:.3e}  "
          f"{'PASS' if good else 'FAIL'}")
print("\nOUTPUT-NEUTRAL" if ok else "\nRESERVATION CHANGED OUTPUT")
raise SystemExit(0 if ok else 1)
PYEOF
rm -rf "$D"
