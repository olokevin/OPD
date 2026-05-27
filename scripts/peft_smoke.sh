#!/bin/bash
# Smoke loop over PEFT modes. 5-step run + save + resume on a small train slice.
# Requires: 1 GPU, the `verl` conda env active.
#
# NOTE: The plan originally referenced
#   datasets/DAPO-Math-17k/data/dapo-math-17k-1percent-processed.parquet
# which does not exist in this checkout. We fall back to the smallest available
# parquet (datasets/dapo-math-17k-processed.parquet, ~3.3MB). If you stage a
# real 1% slice in the future, override TRAIN_DATASET on the command line for
# faster smoke runs.
set -euo pipefail

export TRAIN_DATASET=datasets/dapo-math-17k-processed.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-1pct
export N_RESPONSES=2
export MINI_BATCH_SIZE=4
export MAX_RESP_LENGTH=512
export MAX_VAL_RESP_LENGTH=512
export SAVE_FREQ=3
export TEST_FREQ=1000
export TOTAL_EPOCHS=1
export N_GPUS_PER_NODE=1
export ACTOR_MODEL_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM
export REWARD_MODEL_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM

run_mode() {
  local mode="$1"; shift
  echo "=== smoke: PEFT_MODE=$mode ==="
  PEFT_MODE=$mode "$@" bash on_policy_distillation.sh 2>&1 | tee logs/smoke_$mode.log
  echo "=== smoke OK: $mode ==="
}

mkdir -p logs
run_mode none
run_mode lora LORA_RANK=4 LORA_ALPHA=8
run_mode qlora LORA_RANK=4 LORA_ALPHA=8
run_mode blocktt BTT_TRAIN_POSITION=small
run_mode blocktt BTT_QFURA=True BTT_TRAIN_POSITION=small
run_mode svd SVD_TRAIN_POSITION=output
