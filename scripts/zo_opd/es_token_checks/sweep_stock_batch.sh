#!/bin/bash
# sweep_stock_batch.sh -- stock vLLM continuous-batching decode reference.
# The packed es_token driver is capped at a few slots by the full-context
# scratch-KV reservation, while BP-OPD decodes the whole batch at once. This
# sweep gives the stock tok/s curve vs concurrency so the es_token numbers can
# be read against the right reference point (B = the OPD batch size, 64).
#   SB_GPU=5 bash scripts/zo_opd/es_token_checks/sweep_stock_batch.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${SB_GPU:-5}
PY=/home/yequan/miniconda3/envs/verl/bin/python
OUT=${OUT:-scripts/zo_opd/results/es_token_stock_batch.txt}
JDIR=logs/es_profile/throughput; mkdir -p "$JDIR"
WT=$(pwd)
TS=${T_SHORT:-64}; TL=${T_LONG:-320}
{
echo "stock vLLM decode throughput vs concurrency -- $(date +%Y-%m-%d\ %H:%M:%S)"
echo "GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU))"
echo "Qwen3-1.7B bf16, greedy, ignore_eos, enforce_eager (same engine settings"
echo "ms/token-step from the T=$TS -> T=$TL slope. Two engine modes:"
echo "  eager     = enforce_eager=True, the setting the es_token driver forces"
echo "              (it captures its own graphs) -- the in-harness comparison;"
echo "  cudagraph = enforce_eager=False, stock vLLM's real generation path,"
echo "              which is what BP-OPD actually runs."
echo "=============================================================="
for b in 4 8 16 32 64; do
  for mode in eager cudagraph; do
    FLAG=""; [ "$mode" = cudagraph ] && FLAG="--stock-cudagraph"
    echo "### stock batch=$b mode=$mode"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT/verl HF_HOME=/data/yequan/huggingface \
        $PY scripts/zo_opd/es_token_checks/bench_decode_throughput.py \
        --stock-only $FLAG --pack-width $b --t-short $TS --t-long $TL \
        --json-out "$JDIR/stock_b${b}_$mode.json" 2>&1 \
        | grep -E "^\[stock|^RESULT|Error|Traceback|assert"
    echo
  done
done
echo "done $(date +%H:%M:%S)"
} 2>&1 | tee "$OUT"
