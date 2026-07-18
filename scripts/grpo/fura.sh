#!/bin/bash
# scripts/grpo/fura.sh — FurA (BlockTT) GRPO (zero-RL) on Qwen2.5-7B.
#
# FurA = PEFT_MODE=blocktt with a non-quantized (bf16) frozen core: each target
# Linear is factorized into BlockTT cores; only the small side trains, the large
# side stays frozen. See src/compress/README.md for the decomposition convention.
#
# GRPO base (adv_estimator=grpo, rule-based math reward) via the shared grpo.sh.
# Settings mirror the OPD SLURM FurA launch (slurm/opd/fura/opd_2node_env_fura.sh)
# and the SimpleRL-Zoo Qwen2.5-7B recipe: DAPO-Math-17k train, bf16, FSDP2,
# no tensor parallel (PARALLEL_SIZE=1).
#
# Default FurA config per the run goal: decomp_mode=output_one_block (R core is
# small -> R trains), s_merged_to=frozen, train_position=small.
#
# Override per-run, e.g. `LR=1e-4 bash scripts/grpo/fura.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

set -a  # auto-export every assignment below to grpo.sh

# ---- models + data ----
ACTOR_MODEL_PATH=Qwen/Qwen2.5-7B
HF_HOME=${HF_HOME:-/data/yequan/huggingface}
TRAIN_DATASET_NAME=MATH            # simplelr_math_35 -> math-lv3to5 + MATH-500 eval
PROJECT_NAME=grpo-qwen25-7b

# ---- hyperparameters (SimpleRL-Zoo Qwen2.5-7B recipe; LR raised for FurA) ----
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=8192
MAX_VAL_RESP_LENGTH=8192
TEMPERATURE=1.0
N_RESPONSES=8
TRAIN_BATCH_SIZE=1024
MINI_BATCH_SIZE=256
LR=1e-5                            # FurA (BlockTT) LR (NOT the 5e-7 full-FT LR)
TOTAL_EPOCHS=20
USE_KL=True
KL_LOSS_COEF=1e-4
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0.001
MODEL_DTYPE=bfloat16
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
SAVE_FREQ=5
TEST_FREQ=5
IS_PLOT=False

# ---- no tensor parallel ----
PARALLEL_SIZE=1

# ---- hardware / memory-fit. NOTE: FurA REQUIRES ACTOR_PARAM_OFFLOAD=False --
# the Linear->BlockTT conversion at init needs all target Linear weights on CUDA
# (param_offload=True leaves them on CPU -> conversion error). ----
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.75

# ---- PEFT: FurA = BlockTT, non-quantized frozen core ----
PEFT_MODE=blocktt
PEFT_TARGET_MODULES=all
BTT_DECOMP_MODE=output_one_block   # input_one_block | output_one_block
BTT_RANK=full                      # int | "full" | float in (0,1] with calib
BTT_TRAIN_POSITION=small           # small | large | both
BTT_S_MERGED_TO=frozen             # frozen | trainable | keep_trainable
BTT_CONVERT_MODE=svd               # svd | random | calib
BTT_FACTORIZE_BY_HEAD=True
BTT_NORMALIZE_AFTER_UPDATE=False
BTT_QFURA=False                    # FurA: keep frozen core in bf16 (not NF4)

# Use FSDP2 for the actor/ref. FSDP1 (use_orig_params=True, required for
# BlockTT's mixed trainable/frozen params) fails the per-step writeback on the
# frozen embedding; FSDP2's per-parameter DTensor sharding trains cleanly.
# See scripts/opd/math/fura.sh for the full rationale.
EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2"

set +a

bash grpo.sh
