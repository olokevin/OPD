#!/bin/bash
# bench_np_vs_bp.sh — F3 GOAL PROOF: one-step wall-clock + peak-mem,
#   NP (packed_graphed ALL-LAYER) vs BP-OPD.
# Both: Qwen3-1.7B student, Keven16 4B teacher, batch=64, max_tokens=1024, greedy.
#
# NP: fully-CUDA-graphed packed ALL-LAYER decode (DECODE_MODE=packed_graphed,
#     EN_LAYERWISE=false -> all matched down_proj layers updated in ONE graphed
#     packed decode per wave). ONE step = 64 prompts decoded in waves of
#     PACK_WIDTH=4 disjoint-KV slots, scored against the teacher, one all-layer
#     delta_W applied. N_SAMPLE=8 perturbation rails per decode step.
# BP: standard verl PPO token_reward_direct (opd_math_ref.sh), one step.
#
# PACK_WIDTH=4 is the KV-safe value: _np_prefill_packed reserves
#   ceil(max_model_len/block_size)=2560 blocks PER prompt (the FULL context
#   window, independent of max_tokens), so the B_pack disjoint KV slices must fit
#   num_gpu_blocks. At the bench student GMU (0.55) pack_width=4 (4*2560=10240
#   blocks) fits; 8 may not -> 4 is the known-safe goal value. The per-step cost
#   is dominated by the forward COUNT (1 clean + N=8 perturbed rows per token),
#   not the wave count, so pack_width=4 vs 8 does not change the verdict.
#
# Worktree note: the NP launcher imports verl via the editable-installed MAIN
# checkout (stale). We prepend PYTHONPATH=$ROOT/verl (this worktree) to the NP
# invocation so the all-layer graphed code actually runs. The BP side uses stock
# verl PPO and does NOT need the worktree, so we leave its PYTHONPATH alone.
#
#   NP_GPU=1 BP_GPU=2 PACK_WIDTH=4 bash scripts/zo_opd/bench_np_vs_bp.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$PWD"                                  # = worktree root (.../np-alllayer-graphed)
mkdir -p logs/np_vs_bp scripts/zo_opd/results

NP_GPU=${NP_GPU:-1}
BP_GPU=${BP_GPU:-2}
PACK_WIDTH=${PACK_WIDTH:-4}
TS=$(date +%Y%m%d_%H%M%S)

NP_LOG=logs/np_vs_bp/np_${TS}.log
BP_LOG=logs/np_vs_bp/bp_${TS}.log
NP_MEM=logs/np_vs_bp/np_${TS}.peakmem
BP_MEM=logs/np_vs_bp/bp_${TS}.peakmem

# ---- peak-GPU-mem poller: writes the running MAX (MiB) of memory.used on $1 to $2.
# Trainer-agnostic: captures the TRUE device peak (student+teacher+graph+KV).
# Background PID is killed by the caller once the run finishes.
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

# ====================================================================== NP side
# packed_graphed ALL-LAYER, one step (NUM_ITERATIONS=1, eval off), debug timing on.
# PYTHONPATH=$ROOT/verl shadows the stale editable-installed verl with THIS worktree.
poll_peak_mem "$NP_GPU" "$NP_MEM" &
NP_MEM_PID=$!
PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" \
  NP_DEBUG_DECODE=1 CUDA_VISIBLE_DEVICES=$NP_GPU \
  EXP=npvsbp_np DECODE_MODE=packed_graphed EN_LAYERWISE=false USE_CUDA_GRAPH=true \
  PACK_WIDTH=$PACK_WIDTH \
  BATCH_SIZE=64 MAX_RESP_LENGTH=1024 N_SAMPLE=8 N_ROLLOUT=1 \
  NUM_ITERATIONS=1 EVAL_INTERVAL=999 \
  GPU_MEMORY_UTILIZATION=${NP_STU_GMU:-0.55} TEACHER_GPU_MEMORY_UTILIZATION=${NP_TCH_GMU:-0.30} \
  LR=3e-2 LOG_DIR=logs/np_vs_bp NP_LOGGER='["console"]' \
  bash scripts/zo_opd/opd_math_np.sh > "$NP_LOG" 2>&1
kill $NP_MEM_PID 2>/dev/null; wait $NP_MEM_PID 2>/dev/null
NP_PEAK=$(cat "$NP_MEM" 2>/dev/null)
echo "NP done. step_time:"
grep -E "step:0 .*step_time" "$NP_LOG" | head -1
echo "NP peak mem (MiB): $NP_PEAK"

# ====================================================================== BP side
# One OPD step (opd_math_ref.sh), capture first per-step timing then stop the run.
# verl logs at step:1 with perf/time_per_step. Watch only for that metric line; the
# launcher PID exiting early is the wrapper, not the python trainer, so do NOT break
# on it. Give the BP Ray head a UNIQUE port + tmp dir so it never collides with a
# stale isolated Ray cluster left on the default per-GPU port by an earlier run.
poll_peak_mem "$BP_GPU" "$BP_MEM" &
BP_MEM_PID=$!
BP_RAY_PORT=$(( 6700 + (RANDOM % 200) ))
# The MAIN checkout holds the packaged train_data/ that opd_math_ref.sh's
# TRAIN_DATASET_NAME=MATH resolves to (datasets/train_data/math-lv3to5/train.parquet);
# this WORKTREE's datasets/ only has the top-level parquets, NOT train_data/. Point
# TRAIN_DATASET + TEST_FILE at the main checkout's ABSOLUTE paths so the BP launcher
# finds them regardless of cwd. (BP uses stock verl PPO; no worktree code needed.)
MAIN_ROOT=/home/yequan/Project/compression/OPD
BP_TRAIN=${BP_TRAIN:-$MAIN_ROOT/datasets/train_data/math-lv3to5/train.parquet}
BP_TEST=${BP_TEST:-$MAIN_ROOT/datasets/test_data/MATH-500/test.parquet}
CUDA_VISIBLE_DEVICES=$BP_GPU TEST_FREQ=9999 MAX_RESP_LENGTH=1024 MAX_VAL_RESP_LENGTH=1024 \
  TRAIN_DATASET="$BP_TRAIN" TEST_FILE="[\"$BP_TEST\"]" \
  RAY_ISOLATE=1 RAY_PORT=$BP_RAY_PORT RAY_TMPDIR=/tmp/ray_bp_${TS} \
  bash scripts/zo_opd/opd_math_ref.sh > "$BP_LOG" 2>&1 &
BP_PID=$!
BP_DEADLINE=$(( $(date +%s) + 900 ))   # 15 min cap for vLLM+FSDP+teacher load + 1 step
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

echo "=== logs: $NP_LOG  $BP_LOG ==="
echo "=== peak mem: NP=$NP_PEAK MiB  BP=$BP_PEAK MiB ==="
