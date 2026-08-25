#!/bin/bash
# bench_es_token_vs_bp.sh — es_token GOAL PROOF: one-step wall-clock + peak-mem,
#   es_token (graphed packed ALL-LINEAR rank-1 rail ES) vs BP-OPD.
# Both: Qwen3-1.7B student, Keven16 4B teacher, batch=64, max_tokens=1024, greedy.
#
# ES: fully-CUDA-graphed packed decode (1 clean + N_SAMPLE=8 rail rows/token,
#     ALL 112 decoder linears perturbed rank-1 per token via fixed Hadamard sign
#     rails + one fused noise draw), sampled-token teacher loss (ONE prefill per
#     rollout, prompt_logprobs), chunked-GEMM assembly. ONE step = 64 prompts in
#     waves of PACK_WIDTH=4 disjoint-KV slots -> one all-layer update.
# BP: standard verl PPO token_reward_direct (opd_math_ref.sh), one step, with
#     stock vLLM CUDA-graph generation.
#
# Reference numbers (same harness): NP V3 one-step 2472 s (decode 1368 +
# assemble 835 + teacher ~250); BP 53.79 s cold / 27.54 s steady
# (scripts/zo_opd/results/np_vs_bp_alllayer_graphed.txt).
#
#   ES_GPU=2 BP_GPU=3 PACK_WIDTH=4 bash scripts/zo_opd/bench_es_token_vs_bp.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
mkdir -p logs/es_vs_bp scripts/zo_opd/results

ES_GPU=${ES_GPU:-2}
BP_GPU=${BP_GPU:-3}
PACK_WIDTH=${PACK_WIDTH:-4}
TS=$(date +%Y%m%d_%H%M%S)

ES_LOG=logs/es_vs_bp/es_${TS}.log
BP_LOG=logs/es_vs_bp/bp_${TS}.log
ES_MEM=logs/es_vs_bp/es_${TS}.peakmem
BP_MEM=logs/es_vs_bp/bp_${TS}.peakmem

poll_peak_mem() {  # $1=gpu  $2=outfile
    local gpu="$1" out="$2"; echo 0 > "$out"
    while true; do
        local u
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
        [ -n "$u" ] || { sleep 1; continue; }
        local m; m=$(cat "$out" 2>/dev/null); [ -n "$m" ] || m=0
        if [ "$u" -gt "$m" ] 2>/dev/null; then echo "$u" > "$out"; fi
        sleep 1
    done
}

# ================================================================ es_token side
poll_peak_mem "$ES_GPU" "$ES_MEM" &
ES_MEM_PID=$!
ES_DEBUG_DECODE=1 CUDA_VISIBLE_DEVICES=$ES_GPU \
  EXP=esvsbp PACK_WIDTH=$PACK_WIDTH \
  BATCH_SIZE=64 MAX_RESP_LENGTH=1024 N_SAMPLE=8 \
  NUM_ITERATIONS=1 EVAL_INTERVAL=999 VAL_MAX_SAMPLES=4 HELDOUT_PROBE_SIZE=0 \
  GPU_MEMORY_UTILIZATION=${ES_STU_GMU:-0.55} \
  TEACHER_GPU_MEMORY_UTILIZATION=${ES_TCH_GMU:-0.30} \
  LR=1e-3 LOG_DIR=logs/es_vs_bp ES_LOGGER='["console"]' \
  bash scripts/zo_opd/opd_es_token.sh > "$ES_LOG" 2>&1
kill $ES_MEM_PID 2>/dev/null; wait $ES_MEM_PID 2>/dev/null
ES_PEAK=$(cat "$ES_MEM" 2>/dev/null)
echo "ES done. step metrics:"
grep -E "step_time|decode_s|teacher_s|assemble_s" "$ES_LOG" | head -8
echo "ES peak mem (MiB): $ES_PEAK"

# ====================================================================== BP side
poll_peak_mem "$BP_GPU" "$BP_MEM" &
BP_MEM_PID=$!
BP_RAY_PORT=$(( 6700 + (RANDOM % 200) ))
BP_TRAIN=${BP_TRAIN:-datasets/train_data/math-lv3to5/train.parquet}
BP_TEST=${BP_TEST:-datasets/test_data/MATH-500/test.parquet}
CUDA_VISIBLE_DEVICES=$BP_GPU TEST_FREQ=9999 MAX_RESP_LENGTH=1024 MAX_VAL_RESP_LENGTH=1024 \
  TRAIN_DATASET="$BP_TRAIN" TEST_FILE="[\"$BP_TEST\"]" \
  RAY_ISOLATE=1 RAY_PORT=$BP_RAY_PORT RAY_TMPDIR=/tmp/ray_bp_${TS} \
  bash scripts/zo_opd/opd_math_ref.sh > "$BP_LOG" 2>&1 &
BP_PID=$!
BP_DEADLINE=$(( $(date +%s) + 900 ))
while true; do
    if grep -qE "perf/time_per_step|timing_s/step" "$BP_LOG" 2>/dev/null; then
        echo "BP first step logged."; break; fi
    if ! pgrep -f "main_ppo" >/dev/null 2>&1 && ! kill -0 $BP_PID 2>/dev/null \
       && grep -qE "Final validation|main_ppo.*Error|Killed|Error executing job|FileNotFoundError|Traceback" "$BP_LOG" 2>/dev/null; then
        echo "BP process ended before first-step metric -- see $BP_LOG"; break; fi
    if [ "$(date +%s)" -ge "$BP_DEADLINE" ]; then
        echo "BP wait timed out after 900s -- see $BP_LOG"; break; fi
    sleep 5
done
sleep 3
kill $BP_PID 2>/dev/null
pkill -f "main_ppo" 2>/dev/null
pkill -f "on_policy_distillation" 2>/dev/null
wait $BP_PID 2>/dev/null
kill $BP_MEM_PID 2>/dev/null; wait $BP_MEM_PID 2>/dev/null
BP_PEAK=$(cat "$BP_MEM" 2>/dev/null)
echo "BP done. first-step timing:"
grep -E "step:1 |perf/time_per_step|timing_s/step" "$BP_LOG" | head -5
echo "BP peak mem (MiB): $BP_PEAK"

echo "=== logs: $ES_LOG  $BP_LOG ==="
echo "=== peak mem: ES=$ES_PEAK MiB  BP=$BP_PEAK MiB ==="
