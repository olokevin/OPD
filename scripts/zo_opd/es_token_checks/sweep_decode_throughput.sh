#!/bin/bash
# sweep_decode_throughput.sh -- es_token decode-throughput profile.
# Sweeps the rail count N (0 = clean decode only through the same packed graph
# driver) at the production pack_width, plus stock vLLM continuous batching as
# the reference decode path, plus a small pack_width sweep at N=8 (the lever the
# wiki flags for the residual gap vs BP).
#   TP_GPU=6 bash scripts/zo_opd/es_token_checks/sweep_decode_throughput.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${TP_GPU:-6}
PY=/home/yequan/miniconda3/envs/verl/bin/python
OUT=${OUT:-scripts/zo_opd/results/es_token_decode_throughput.txt}
JDIR=logs/es_profile/throughput; mkdir -p "$JDIR"
WT=$(pwd)
TS=${T_SHORT:-64}; TL=${T_LONG:-320}

run() {  # $1=n_sample $2=pack_width $3=extra flags
    echo "### n_sample=$1 pack_width=$2 ${3:-}"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT/verl HF_HOME=/data/yequan/huggingface \
        $PY scripts/zo_opd/es_token_checks/bench_decode_throughput.py \
        --n-sample "$1" --pack-width "$2" --t-short $TS --t-long $TL \
        --json-out "$JDIR/n$1_pw$2.json" ${3:-} 2>&1 \
        | grep -E "^\[cfg\]|^\[packed|^\[stock|^RESULT|Error|assert|Traceback|does not fit"
    echo
}

{
echo "es_token decode throughput -- $(date +%Y-%m-%d\ %H:%M:%S)"
echo "GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU))"
echo "Qwen3-1.7B bf16, 112 matched decoder linears, greedy, sigma=0.01,"
echo "graphed packed decode, EOS disabled; ms/token-step from the slope of"
echo "wall-clock over T=$TS -> T=$TL (capture/prefill/teardown cancel out)."
echo "=============================================================="
run 0 4 --stock
for n in 1 2 4 8 16 32; do run $n 4; done
echo "---- pack_width sweep at N=8 ----"
for pw in 8 16; do run 8 $pw; done
echo "done $(date +%H:%M:%S)"
} 2>&1 | tee "$OUT"
