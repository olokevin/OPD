#!/bin/bash
# run_openthoughts3_rollout.sh — generate an SFT dataset (same {"messages":[user,assistant]}
# JSONL format as the released lllyx/OpenThought3-Qwen3-4B) from OpenThoughts3 math prompts,
# but with a configurable generating MODEL and a configurable amount of prompts.
#
# Apple-to-apple with the released dataset:
#   - same prompt source  : /data/yequan/datasets/OpenThoughts3-1.2M-math.parquet (100% of the
#                            prompts that produced the released Qwen3-4B dataset)
#   - same strategy       : non-thinking, temperature=1.0, top_p=0.95, rejection sampling on
#                            (drop no-boxed / repetitive outputs), 1 response per prompt
#   - only difference     : the MODEL doing the generation.
#
# NOTE on max_tokens: the released dataset used max_tokens=7168, but the default model here
# (Qwen3-4B-Non-Thinking-RL-Math-Step500) emits much longer reasoning and gets truncated
# before \boxed{} at 7168 (→ rejected). We raise MAX_TOKENS to 32768 so generations complete.
#
# Overridable env vars: MODEL_PATH, INPUT_PARQUET, NUM_PROMPTS, GENERATE_RATIO, GPU_IDS,
#                       MAX_TOKENS, MAX_MODEL_LEN, OUTPUT_JSONL, ENABLE_THINKING.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The verl conda env supplies vllm + ninja; prepend its bin so flashinfer's JIT can find ninja.
VERL_ENV=/home/yequan/miniconda3/envs/verl
export PATH="$VERL_ENV/bin:$PATH"
PY="$VERL_ENV/bin/python"

MODEL_PATH=${MODEL_PATH:-/data/yequan/huggingface/hub/models--Keven16--Qwen3-4B-Non-Thinking-RL-Math-Step500/snapshots/05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa}
INPUT_PARQUET=${INPUT_PARQUET:-/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet}
NUM_PROMPTS=${NUM_PROMPTS:-100000}
GPU_IDS=${GPU_IDS:-0}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
ENABLE_THINKING=${ENABLE_THINKING:-false}
OUTPUT_JSONL=${OUTPUT_JSONL:-$REPO_ROOT/datasets/OpenThought3-Qwen3-4B-NonThinking-RL/data/train.jsonl}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUTPUT_JSONL")"
LOG="logs/openthoughts3_rollout_$(date +%Y%m%d_%H%M%S).log"

echo "model=$MODEL_PATH"
echo "input=$INPUT_PARQUET  num_prompts=$NUM_PROMPTS  gpu=$GPU_IDS"
echo "max_tokens=$MAX_TOKENS  max_model_len=$MAX_MODEL_LEN  thinking=$ENABLE_THINKING"
echo "output=$OUTPUT_JSONL"
echo "log=$LOG"

"$PY" scripts/infer/vllm_rollout.py \
  --input-parquet "$INPUT_PARQUET" \
  --model-path "$MODEL_PATH" \
  --gpu-ids "$GPU_IDS" \
  --enable-thinking "$ENABLE_THINKING" \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3 \
  --num-prompts "$NUM_PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --output-jsonl "$OUTPUT_JSONL" \
  2>&1 | tee "$LOG"
