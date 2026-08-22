#!/bin/bash
# sweep_grad_cosine.sh -- es_token gradient-quality profile: cosine(es dW, autograd)
# swept over probe count K = n_sample * repeats, two layer shapes, to check
#   (i)  the sqrt(K/(K+d_out*d_in)) information bound is tracked,
#   (ii) rails vs repeats are interchangeable at equal K (rail orthogonality),
#   (iii) the d = d_out*d_in dependence.
#   COS_GPU=5 bash scripts/zo_opd/es_token_checks/sweep_grad_cosine.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
GPU=${COS_GPU:-5}
PY=/home/yequan/miniconda3/envs/verl/bin/python
OUT=${OUT:-scripts/zo_opd/results/es_token_grad_cosine_sweep.txt}
WT=$(pwd)

run() {  # $1=layer $2=N $3=repeats
    echo "### layer=$1 n_sample=$2 repeats=$3 K=$(( $2 * $3 ))"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT/verl HF_HOME=/data/yequan/huggingface \
        $PY scripts/zo_opd/es_token_checks/check_es_grad_cosine.py \
        --layer "$1" --n-sample "$2" --repeats "$3" --sigma 1e-3 2>&1 \
        | grep -E "^layer |^cosine|^theory|^training-scale|^PASS|^FAIL"
    echo
}

{
echo "es_token gradient cosine sweep -- $(date +%Y-%m-%d\ %H:%M:%S)"
echo "GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU))"
echo "Qwen3-1.7B fp32, sigma=1e-3, bernoulli (u,v), mean_baseline, seed 42"
echo "=============================================================="
for cfg in "model.layers.0.mlp.down_proj 8 50" \
           "model.layers.0.mlp.down_proj 8 300" \
           "model.layers.0.mlp.down_proj 16 150" \
           "model.layers.0.mlp.down_proj 32 150" \
           "model.layers.0.self_attn.o_proj 8 50" \
           "model.layers.0.self_attn.o_proj 16 150"; do
    run $cfg
done
echo "done $(date +%H:%M:%S)"
} 2>&1 | tee "$OUT"
