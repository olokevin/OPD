#!/bin/bash
# sweep_decode_isolation.sh -- attribute the es_token per-token-step cost.
# Three deltas, same packed graphed driver at pack_width=4:
#   N=0 skip-noise  : bare graphed decode floor (buffers + replay + logits)
#   N=0             : + the fused per-(slot,token) noise draw over all layers
#   N=8 skip-noise  : + the 112-layer rank-1 rail compute, no noise draw
#   N=8             : the shipping configuration
#   ISO_GPU=6 bash scripts/zo_opd/es_token_checks/sweep_decode_isolation.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${ISO_GPU:-6}
PY=/home/yequan/miniconda3/envs/verl/bin/python
OUT=${OUT:-scripts/zo_opd/results/es_token_decode_isolation.txt}
WT=$(pwd); TS=${T_SHORT:-64}; TL=${T_LONG:-320}
run() {  # $1=n_sample  $2=skip_noise(0/1)
    echo "### n_sample=$1 skip_noise=$2"
    env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT/verl HF_HOME=/data/yequan/huggingface \
        ${2:+ES_BENCH_SKIP_NOISE=$2} \
        $PY scripts/zo_opd/es_token_checks/bench_decode_throughput.py \
        --n-sample "$1" --pack-width 4 --t-short $TS --t-long $TL 2>&1 \
        | grep -E "^\[packed|^RESULT|Error|Traceback|assert"
    echo
}
{
echo "es_token decode cost attribution -- $(date +%Y-%m-%d\ %H:%M:%S)"
echo "GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU))"
echo "Qwen3-1.7B bf16, 112 matched linears, pack_width=4, greedy, sigma=0.01."
echo "ES_BENCH_SKIP_NOISE=1 removes the per-token fused noise draw only."
echo "=============================================================="
run 0 1; run 0 ""; run 8 1; run 8 ""
echo "done $(date +%H:%M:%S)"
} 2>&1 | tee "$OUT"
