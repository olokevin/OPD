#!/bin/bash
# Smoke launcher: compression + 1 train step + val on AMC23 (small bench).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

source "$SCRIPT_DIR/_common.sh"

export CALIB_MODE=v2
export VAL_N=1
export TEST_FREQ=1
export TEST_FILE='["datasets/test_data/AMC23/test.parquet"]'
export TOTAL_EPOCHS=1
# Append smoke overrides to whatever _common.sh set.
export EXTRA_HYDRA_ARGS="$EXTRA_HYDRA_ARGS trainer.total_training_steps=1 trainer.val_before_train=True"

bash on_policy_distillation.sh
