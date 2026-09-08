#!/bin/bash
# ruler_watch.sh -- score every new es-decode search checkpoint on the standard ruler
# (GPU 4, sequential), appending to results/ruler_scores.tsv. Polls every 5 min.
#   nohup bash scripts/zo_opd/ds15b/ruler_watch.sh > logs/ds15b/es_decode/ruler/watch.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
ROOT=/data/yequan/compress_train/OPD/checkpoint
TSV=scripts/zo_opd/es_profile/results/ruler_scores.tsv
while true; do
  for ck in $(ls -d $ROOT/ds15b_es-decode_r1_*/es_token_*/step_* 2>/dev/null | sort); do
    exp=$(basename $(dirname $(dirname $ck))); step=$(basename $ck)
    tag="${exp#ds15b_}_${step}"
    grep -q "^$tag	" $TSV 2>/dev/null && continue
    [ -f "$ck/model.safetensors" ] || continue
    sleep 60   # let the save finish
    echo "[watch] scoring $tag $(date '+%F %T')"
    bash scripts/zo_opd/ds15b/ruler_ckpt.sh "$ck" "$tag" 4
  done
  sleep 300
done
