#!/bin/bash
# run_bp_b1.sh -- BP-OPD (verl PPO token_reward_direct) ONE-PROMPT step timing,
# the "BP" reference line of the 0902 B=1 rail plot (es_profile_results.md).
# Same student/teacher/data as opd_math_ref.sh, but train_batch_size=1,
# max_response_length=512 (greedy, always clips), 3 steps, no eval.
#
#   BP_GPU=4 bash scripts/zo_opd/es_profile/run_bp_b1.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH   # verl env (ray, python3); base conda has neither
mkdir -p logs/es_b1
BP_GPU=${BP_GPU:-4}
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/es_b1/bp_b1_${TS}.log

set -a
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
HF_HOME=/data/yequan/huggingface
TRAIN_DATASET_NAME=MATH
TRAIN_DATASET=datasets/train_data/math-lv3to5/train.parquet
TEST_FILE='["datasets/test_data/MATH-500/test.parquet"]'
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-512}
MAX_VAL_RESP_LENGTH=512
PROJECT_NAME=opd-qwen-math
SAVE_FREQ=9999
TEST_FREQ=9999
IS_PLOT=False
TEMPERATURE=0.0
N_RESPONSES=1
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
VAL_TEMPERATURE=0.0
VAL_N=1
EXTRA_HYDRA_ARGS="actor_rollout_ref.rollout.val_kwargs.do_sample=False trainer.total_training_steps=${BP_STEPS:-3} trainer.logger=['console']"
LR=1e-6
CUDA_VISIBLE_DEVICES=$BP_GPU
N_GPUS_PER_NODE=1
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ATTENTION_BACKEND=FLASH_ATTN
MODEL_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIM_OFFLOAD=False
REWARD_PARAM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.55
REWARD_MICRO_BATCH_SIZE_PER_GPU=8
RAY_ISOLATE=1
RAY_PORT=$(( 6800 + BP_GPU * 10 + (RANDOM % 9) ))
RAY_TMPDIR=/tmp/ray_bp_b1_gpu${BP_GPU}_${TS}
EXPERIMENT_NAME=bp_b1_${TS}
set +a

echo "log: $LOG"
bash on_policy_distillation.sh > "$LOG" 2>&1
echo "BP B=1 done; per-step timing:"
grep -oE "step:[0-9]+ .*timing_s/step:[0-9.]+" "$LOG" | sed -E 's/ - /\n/g' | grep -E "^step:|timing_s/(gen|generate_sequences|compute_log_prob|compute_rm_score|reward|update_actor|step):|response_length/mean" 
