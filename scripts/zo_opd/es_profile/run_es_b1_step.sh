#!/bin/bash
# run_es_b1_step.sh -- es_token ONE-PROMPT full training step (decode + teacher
# + assemble/apply) for the 0902 B=1 rail plot: step time vs N for a decode
# path. Same student/teacher/data as opd_es_token.sh, batch_size=1,
# pack_width=1, max_tokens=512, 3 steps (step 0 = graph capture; report 1-2), NO eval
# (the trainer hangs on the decode after an eval -- see es_profile_results.md 14).
#
#   ES_GPU=1 NS="1 8 32 128" ES_PATH=seq/stream/fused/graph \
#       bash scripts/zo_opd/es_profile/run_es_b1_step.sh
#   ES_PATH = attn/lm/rail/step   (rows/full/kernel/eager = shipping)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH   # verl env (ray, python3)
export PYTHONPATH=$REPO_ROOT/verl:${PYTHONPATH:-}          # THIS worktree's verl, not the editable main
mkdir -p logs/es_b1 scripts/zo_opd/es_profile/results
ES_GPU=${ES_GPU:-1}
NS=${NS:-"1 8 32 128"}
ES_PATH=${ES_PATH:-rows/full/kernel/eager}
IFS=/ read -r ATTN LM RAIL STEP <<< "$ES_PATH"
TAG=$(echo "$ES_PATH" | tr '/' '_')
OUT=scripts/zo_opd/es_profile/results/es_b1_step_${TAG}.tsv
echo -e "path\tN\tstep\tstep_time\tdecode_s\tteacher_s\tassemble_s\tn_records" > "$OUT"

for N in $NS; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG=logs/es_b1/es_b1_${TAG}_N${N}_${TS}.log
    echo "[es b1] path=$ES_PATH N=$N -> $LOG"
    CUDA_VISIBLE_DEVICES=$ES_GPU \
      EXP=b1_${TAG}_N${N} BATCH_SIZE=1 PACK_WIDTH=1 B_PACK_BUCKETS='[1]' \
      MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-512} N_SAMPLE=$N \
      ATTN_IMPL=$ATTN LM_HEAD_IMPL=$LM RAIL_IMPL=$RAIL STEP_IMPL=$STEP \
      NUM_ITERATIONS=${NUM_ITERATIONS:-3} EVAL_INTERVAL=0 VAL_MAX_SAMPLES=4 HELDOUT_PROBE_SIZE=0 \
      GPU_MEMORY_UTILIZATION=${ES_STU_GMU:-0.55} TEACHER_GPU_MEMORY_UTILIZATION=${ES_TCH_GMU:-0.30} \
      LR=1e-3 LOG_DIR=logs/es_b1 ES_LOGGER='["console"]' \
      bash scripts/zo_opd/opd_es_token.sh > "$LOG" 2>&1 &
    RUN_PID=$!
    LAST=$(( ${NUM_ITERATIONS:-3} - 1 ))
    T0=$(date +%s)
    # wait for the LAST step's metric line (the driver may hang in teardown afterwards)
    while kill -0 $RUN_PID 2>/dev/null; do
        grep -q "step:${LAST} - train/step_time" "$LOG" 2>/dev/null && { sleep 2; break; }
        [ $(( $(date +%s) - T0 )) -ge ${ES_TIMEOUT:-900} ] && { echo "[es b1] TIMEOUT N=$N"; break; }
        sleep 5
    done
    grep -o "step:[0-9]* - train/step_time:[0-9.]* - train/decode_s:[0-9.]* - train/teacher_s:[0-9.]* - train/assemble_s:[0-9.]* - train/n_token_records:[0-9]*" "$LOG" \
      | sed -E "s|step:([0-9]*) - train/step_time:([0-9.]*) - train/decode_s:([0-9.]*) - train/teacher_s:([0-9.]*) - train/assemble_s:([0-9.]*) - train/n_token_records:([0-9]*)|$ES_PATH\t$N\t\1\t\2\t\3\t\4\t\5\t\6|" >> "$OUT"
    # The driver hangs after the last step / after an eval (known, untriaged) and `timeout`
    # only reaps the wrapper bash: kill the trainer and its engines by their GPU PIDs.
    sleep 3
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$ES_GPU")
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}'); do
        kill -9 "$pid" 2>/dev/null
    done
    for pid in $(pgrep -f "experiment_name=es_token_.*b1_${TAG}_N${N}_"); do kill -9 "$pid" 2>/dev/null; done
    sleep 5
done
echo "=== $OUT"; cat "$OUT"
