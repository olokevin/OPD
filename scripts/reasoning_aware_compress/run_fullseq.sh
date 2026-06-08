#!/usr/bin/env bash
# Full-sequence calibration experiment.
#   stage1 : 4 settings (token/sequence × full/lt2048) at retain 0.7, split GPU 2 & 3
#   stage2 : GPU=N SET=token:full bash run_fullseq.sh stage2   (best setting, 0.6/0.5/0.4)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
REPO=$(pwd); PY=/home/yequan/miniconda3/envs/verl/bin/python
SD=scripts/reasoning_aware_compress
export HF_HOME=/data/yequan/huggingface PYTHONPATH="$REPO/src:$REPO/verl"
mkdir -p $SD/results/fullseq $SD/logs
ts() { date +%Y%m%d_%H%M%S; }

case "${1:?usage: run_fullseq.sh stage1|stage2}" in
  stage1)
    CUDA_VISIBLE_DEVICES=2 $PY $SD/fullseq_calib_sweep.py --stage tune \
      --settings token:full sequence:full --ratio 0.7 \
      --out $SD/results/fullseq/tune_full.json \
      > $SD/logs/fullseq_tune_full_$(ts).log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 $PY $SD/fullseq_calib_sweep.py --stage tune \
      --settings token:lt2048 sequence:lt2048 --ratio 0.7 \
      --out $SD/results/fullseq/tune_lt2048.json \
      > $SD/logs/fullseq_tune_lt2048_$(ts).log 2>&1 &
    wait
    echo "stage1 done"
    ;;
  stage2)
    : "${GPU:?set GPU=N}"; : "${SET:?set SET=reweight:length}"
    CUDA_VISIBLE_DEVICES=$GPU $PY $SD/fullseq_calib_sweep.py --stage sweep \
      --setting "$SET" --ratios 0.6 0.5 0.4 \
      --out $SD/results/fullseq/sweep_${SET/:/_}.json \
      > $SD/logs/fullseq_sweep_$(ts).log 2>&1
    echo "stage2 done"
    ;;
esac
