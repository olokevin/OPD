#!/bin/bash
# Wait for the ISO run (launcher pid $1) to exit, let GPUs 5,7 drain, then start isobtt.
cd /home/yequan/Project/compression/OPD
while kill -0 $1 2>/dev/null; do sleep 60; done
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,7 | paste -sd+ | bc)
  [ "$used" -lt 2000 ] && break; sleep 10
done
PPO_TOK=9216 VAL_BEFORE=${VAL_BEFORE:-True} DEVICES=5,7 PEFT_MODE=isobtt LR=${LR:-7.5e-6} TOTAL_STEPS=50 \
  bash scripts/iso/dapo_math_4b.sh > logs/iso/isobtt_lr${LR:-7.5e-6}.log 2>&1
