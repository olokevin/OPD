#!/bin/bash
# grpo_eval.sh — post-training val_only eval of a finished GRPO run on the 4
# held-out benchmarks (AMC23, AIME24, Minerva, Olympiad-Bench). Loads the final
# checkpoint (resume_mode=auto from the training CKPT_PATH), runs verl's
# _validate() once, logs val-core/<bench>/acc to wandb, and exits.
#
# One 2-node interactive allocation, reuses the SAME env driver as training so
# the model + PEFT topology match (fura rebuilds BlockTT then loads trained cores).
#
#   MODE=full bash slurm/grpo/eval/grpo_eval.sh
#   MODE=fura bash slurm/grpo/eval/grpo_eval.sh
set -u
OPD_REPO=/global/u1/y/yequan/Project/OPD
DATA_ROOT=/pscratch/sd/y/yequan/opd
ACCOUNT=m4788_g
MODE=${MODE:-${1:-full}}

case "$MODE" in
  full) export ENV_SCRIPT=grpo/full/grpo_2node_env.sh ;;
  fura) export ENV_SCRIPT=grpo/fura/grpo_2node_env_fura.sh ;;
  *) echo "MODE must be full|fura" >&2; exit 1 ;;
esac

export WANDB_PROJECT=nersc_grpo_qwen2p5_7b
export TRAIN_DATASET_NAME=MATH
export CKPT_PATH=${DATA_ROOT}/checkpoints/grpo_${MODE}_qwen2p5_7b_math
export EXPERIMENT_NAME=grpo_${MODE}_qwen2p5_7b_eval
export WANDB_RUN_ID=grpo_${MODE}_qwen2p5_7b_eval

# val_only pass over the 4 benchmarks.
export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export TEST_FILE='["datasets/test_data/AMC23/test.parquet","datasets/test_data/AIME24/test.parquet","datasets/test_data/Minerva/test.parquet","datasets/test_data/Olympiad-Bench/test.parquet"]'
export VAL_N=${VAL_N:-16}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
export VAL_TOP_P=${VAL_TOP_P:-0.95}

mkdir -p "${DATA_ROOT}/logs"
RUNLOG=${DATA_ROOT}/logs/grpo_${MODE}_eval_$(date +%Y%m%d_%H%M%S).log
echo "=== grpo eval MODE=$MODE ckpt=$CKPT_PATH -> $RUNLOG ==="
salloc --nodes 2 --qos interactive --time 4:00:00 \
       --constraint 'gpu&hbm80g' --gpus-per-node=4 --account "$ACCOUNT" \
       bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh" > "$RUNLOG" 2>&1
echo "=== eval done rc=$? ; grep results: ==="
grep -iE "val-core/.*/acc|val-aux" "$RUNLOG" | tail -40
