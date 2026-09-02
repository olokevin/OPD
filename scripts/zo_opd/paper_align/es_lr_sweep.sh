#!/bin/bash
# es_lr_sweep.sh -- short ES-token LR probe on the PAPER-ALIGNED setting.
#
# The previous sweep (es_token_checks/lr_probe.sh) ran on a setup where every
# rollout hit a 1024-token cap and the update landed on bf16 master weights, so
# it could only bound the LR (1e-3 destroys, <=1e-4 does nothing). Both are
# fixed here: Qwen3-1.7B-Base stops on its own inside 3072 tokens, and
# es_token.fp32_master keeps an fp32 copy so a sub-ulp step is not rounded away.
#
# METRIC: eval/heldout_clean_loss on 64 FIXED prompts, decoded GREEDILY
# (ray_trainer probe_sp) -- deterministic, so a difference between LRs is real.
# Do NOT rank LRs on train/L_clean_mean; it is per-batch data noise.
#
#   LR_GPU=7 LRS="3e-5 1e-4 3e-4" bash scripts/zo_opd/paper_align/es_lr_sweep.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${LR_GPU:-7}
STEPS=${STEPS:-20}
EVAL_EVERY=${EVAL_EVERY:-5}
OUT=${OUT:-logs/es_lr_sweep}
mkdir -p "$OUT"
for LR in ${LRS:-3e-5 1e-4 3e-4}; do
    echo "=== LR=$LR  $(date +%H:%M:%S) ==="
    TRAIN_GPU=$GPU ES_LR=$LR EXP=sweep_$LR \
      PROJECT_NAME=opd-paper-align \
      EXPERIMENT_NAME=es_sweep_lr${LR} \
      ES_ITERS=$STEPS EVAL_INTERVAL=$EVAL_EVERY VAL_MAX_SAMPLES=200 \
      HELDOUT_PROBE_SIZE=64 \
      ES_LOGGER='["console","wandb"]' LOG_DIR="$OUT" \
      bash scripts/zo_opd/paper_align/opd_es_paper.sh > "$OUT/lr_$LR.log" 2>&1
    echo "  probe (64 fixed, greedy, lower=better): $(grep -oE '\[Probe @ step [0-9]+\] heldout_clean_loss=[0-9.]+' "$OUT/lr_$LR.log" | sed -E 's/.*step ([0-9]+).*=([0-9.]+)/s\1:\2/' | tr '\n' ' ')"
    echo "  MATH-500@200 greedy acc:                $(grep -oE 'eval/accuracy:[0-9.]+' "$OUT/lr_$LR.log" | sed 's/.*://' | tr '\n' ' ')"
    # Teardown can hang (wiki es_token_trainer), and killing only the driver
    # LEAKS the ray-spawned vLLM engines: they keep their ~35 GB and the next
    # arm OOMs inside es_assemble_and_apply. Kill the driver, then anything
    # still holding this GPU, and wait for the memory to actually come back.
    pkill -TERM -f 'trainer\.main_es_token' 2>/dev/null; sleep 15
    pkill -9 -f 'trainer\.main_es_token' 2>/dev/null
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU"); do
        kill -9 "$pid" 2>/dev/null
    done
    for _ in $(seq 1 24); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
        [ "${used:-99999}" -lt 2000 ] && break
        sleep 5
    done
    echo "  GPU $GPU now at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU")"
done
echo "=== ES LR SWEEP DONE $(date +%H:%M:%S) ==="
