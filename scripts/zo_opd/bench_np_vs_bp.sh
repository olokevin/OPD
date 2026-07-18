#!/bin/bash
# bench_np_vs_bp.sh — one-step wall-clock + peak-mem: NP packed vs BP-OPD.
# Both: Qwen3-1.7B student, Keven16 4B teacher, batch=64, max_tokens=1024, greedy.
#
# NP: packed decode (one step = 64 prompts decoded in waves, scored, one delta_W).
# BP: standard verl PPO token_reward_direct (opd_math_ref.sh), one step.
#
#   NP_GPU=6 BP_GPU=7 PACK_WIDTH=8 bash scripts/zo_opd/bench_np_vs_bp.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
mkdir -p logs/np_vs_bp

NP_GPU=${NP_GPU:-6}
BP_GPU=${BP_GPU:-7}
PACK_WIDTH=${PACK_WIDTH:-8}
TS=$(date +%Y%m%d_%H%M%S)

# --- NP: one packed step (NUM_ITERATIONS=1, eval off), debug timing on ---
NP_DEBUG_DECODE=1 CUDA_VISIBLE_DEVICES=$NP_GPU \
  EXP=npvsbp_np DECODE_MODE=packed PACK_WIDTH=$PACK_WIDTH \
  BATCH_SIZE=64 MAX_RESP_LENGTH=1024 N_SAMPLE=8 N_ROLLOUT=1 \
  NUM_ITERATIONS=1 EVAL_INTERVAL=999 \
  GPU_MEMORY_UTILIZATION=${NP_STU_GMU:-0.55} TEACHER_GPU_MEMORY_UTILIZATION=${NP_TCH_GMU:-0.30} \
  LR=3e-2 LOG_DIR=logs/np_vs_bp NP_LOGGER='["console"]' \
  bash scripts/zo_opd/opd_math_np.sh > logs/np_vs_bp/np_${TS}.log 2>&1
echo "NP done. step_time:"
grep -E "step:0 .*step_time" logs/np_vs_bp/np_${TS}.log | head -1

# --- BP: one OPD step (run opd_math_ref.sh, capture first-step timing) ---
# Wait until the first per-step metric line is logged (verl logs at step:1 with
# perf/time_per_step), then stop the run. Watch only for that metric line; the
# launcher PID exiting early is the wrapper, not the python trainer, so do NOT
# break on it. (We do NOT abort on "Traceback" -- opd_math_ref.sh's Ray-start can
# print a benign retry trace -- we just bound the wait by a generous timeout.)
# Give the BP Ray head a UNIQUE port + tmp dir so it never collides with a stale
# isolated Ray cluster left on the default per-GPU port by an earlier run.
BP_LOG=logs/np_vs_bp/bp_${TS}.log
BP_RAY_PORT=$(( 6700 + (RANDOM % 200) ))
CUDA_VISIBLE_DEVICES=$BP_GPU TEST_FREQ=9999 MAX_RESP_LENGTH=1024 MAX_VAL_RESP_LENGTH=1024 \
  RAY_ISOLATE=1 RAY_PORT=$BP_RAY_PORT RAY_TMPDIR=/tmp/ray_bp_${TS} \
  bash scripts/zo_opd/opd_math_ref.sh > "$BP_LOG" 2>&1 &
BP_PID=$!
BP_DEADLINE=$(( $(date +%s) + 900 ))   # 15 min cap for vLLM+FSDP+teacher load + 1 step
while true; do
    if grep -qE "perf/time_per_step|timing_s/step" "$BP_LOG" 2>/dev/null; then
        echo "BP first step logged."; break; fi
    if ! pgrep -f "main_ppo" >/dev/null 2>&1 && ! kill -0 $BP_PID 2>/dev/null \
       && grep -qE "Final validation|main_ppo.*Error|Killed" "$BP_LOG" 2>/dev/null; then
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
echo "BP done. first-step timing:"
grep -E "step:1 |perf/time_per_step|timing_s/step" "$BP_LOG" | head -5

echo "=== logs: logs/np_vs_bp/np_${TS}.log  logs/np_vs_bp/bp_${TS}.log ==="
