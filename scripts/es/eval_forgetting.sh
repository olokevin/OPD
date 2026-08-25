#!/bin/bash
# Prior-knowledge (forgetting) + new-task (MATH-500) evaluation of one model.
#
#   DEVICE=2 bash scripts/es/eval_forgetting.sh <tag> <model-path>
#
# Prior-knowledge suite = HellaSwag (the benchmark used by arXiv:2601.20861) plus the
# standard commonsense battery and MMLU, all 0-shot log-likelihood ranking.
# New task = greedy MATH-500 with the ES trainer's own prompt + grader.
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
TAG=$1
MODEL=$2
OUT=${OUT:-/data/yequan/es/forgetting}/$TAG
mkdir -p "$OUT"

export HF_HOME=/data/yequan/huggingface
export HF_DATASETS_TRUST_REMOTE_CODE=0
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${DEVICE:-2}
CONDA=/home/yequan/miniconda3/envs/verl/bin

TASKS=${TASKS:-hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa,boolq,mmlu}
LIMIT_ARG=""
[ "${LIMIT:-0}" != "0" ] && LIMIT_ARG="--limit ${LIMIT}"

if [ ! -f "$OUT/lm_eval.json" ]; then
  echo "=== [$TAG] lm_eval: $TASKS"
  $CONDA/lm_eval --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=False" \
    --tasks "$TASKS" --num_fewshot 0 --batch_size auto $LIMIT_ARG \
    --output_path "$OUT/lm_eval_raw" && \
  $CONDA/python - "$OUT" <<'PY'
import json, os, sys, glob
out = sys.argv[1]
f = sorted(glob.glob(os.path.join(out, "lm_eval_raw", "**", "results_*.json"), recursive=True))[-1]
res = json.load(open(f))["results"]
flat = {}
for task, m in res.items():
    for k, v in m.items():
        if k.endswith(",none") and isinstance(v, float):
            flat[f"{task}/{k[:-5]}"] = 100.0 * v
json.dump(flat, open(os.path.join(out, "lm_eval.json"), "w"), indent=1)
print(json.dumps({k: round(v, 2) for k, v in flat.items() if "/" in k and k.count("_") < 6}, indent=0))
PY
else
  echo "=== [$TAG] lm_eval: cached"
fi

if [ ! -f "$OUT/math500.json" ]; then
  echo "=== [$TAG] MATH-500 greedy"
  $CONDA/python scripts/es/eval_math500.py --model "$MODEL" --out "$OUT/math500.json" \
    ${LIMIT:+--limit $LIMIT} 2>&1 | tail -3
else
  echo "=== [$TAG] MATH-500: cached"
fi
echo "=== [$TAG] done"
