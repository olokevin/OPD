#!/bin/bash
# Recovery-SFT leg: fine-tune compressed OLMoE experts on OpenThoughts3.
# One method per GPU. Freezes attention, trains mlp.* (experts+router).
# Logs to wandb project olmoe_compress_sft, evals at steps {0,100,500,2000,final}.
#
# Usage: bash scripts/moe_compress/run_recovery.sh
set -u

REPO=/home/yequan/Project/compression/OPD
VERL_PY=/home/yequan/miniconda3/envs/verl/bin/python
CKPT=/data/yequan/moe_compress/ckpts
LOGS=$REPO/logs/moe_compress
mkdir -p "$LOGS"

NUM_SAMPLES=${NUM_SAMPLES:-10000}
RETAIN=${RETAIN:-0.50}
CKPT_SUFFIX=${CKPT_SUFFIX:-_s0}     # use _native_s0 for OLMoE-native-calibrated ckpts
TAG_SUFFIX=${TAG_SUFFIX:-_sft}      # use _native_sft to keep wandb runs distinct

# method:gpu
JOBS=("nystrom:1" "nystrom_combined:2" "reap_drop:3")

for j in "${JOBS[@]}"; do
  m="${j%%:*}"; gpu="${j##*:}"
  tag="${m}_r${RETAIN}${TAG_SUFFIX}"
  ckpt="$CKPT/${m}_r${RETAIN}${CKPT_SUFFIX}"
  log="$LOGS/recover_${tag}.log"
  echo "[GPU $gpu] recover $m (retain $RETAIN, $NUM_SAMPLES samples) -> $log"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface \
    HF_DATASETS_TRUST_REMOTE_CODE=1 WANDB_PROJECT=olmoe_compress_sft \
    nohup "$VERL_PY" -m moe_compress.recover_sft \
      --ckpt "$ckpt" --tag "$tag" --num-samples "$NUM_SAMPLES" \
      > "$log" 2>&1 &
  echo "  PID $!"
done
echo "3 recovery runs launched on GPU 1,2,3. wandb: olmoe_compress_sft. Logs: $LOGS/recover_*.log"
