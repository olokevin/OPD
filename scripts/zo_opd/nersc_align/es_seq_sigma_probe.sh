#!/bin/bash
# es_seq_sigma_probe.sh -- pick sigma (and hence alpha = sigma/2) by MEASUREMENT.
#
# One ES iteration per sigma, reading two numbers off iteration 1:
#   train/reward_mean = -KL(pi_n || q)  : how far the perturbation pushes the
#                                         population off the teacher. sigma=0 gives
#                                         the UNPERTURBED reference KL.
#   train/reward_std                    : the fitness spread = the ES signal. Too
#                                         small and the z-scores are noise; too
#                                         large and the population is off-policy.
# Good operating point: clear spread while perturbed KL stays close to the sigma=0
# reference (rule of thumb <= ~1.5-2x).
#
# MUST run at the production MAX_TOKENS: a fixed perturbation compounds along the
# trajectory, so a sigma inside the linear regime at 512 can be outside it at 1536
# (docs/results/zo_opd.md 13.4).
#   PROBE_GPU=3 bash scripts/zo_opd/nersc_align/es_seq_sigma_probe.sh
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$R"

# Reap only OUR vLLM processes ON THE TARGET GPU. A host-wide
# `pkill -f VLLM::EngineCore` also reaches engines belonging to other runs of
# ours on other cards, which is a real hazard while a multi-hour job is training.
reap_gpu () {  # $1 = gpu index
  local g="$1" p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
    if [ "$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -u)" ] \
       && ps -o args= -p "$p" 2>/dev/null | grep -qE "VLLM::EngineCore|vllm"; then
      kill -9 "$p" 2>/dev/null
    fi
  done
  for _ in $(seq 1 36); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 5
  done
  return 1
}
GPU=${PROBE_GPU:-3}
OUT=$R/logs/esseq/sigma_probe; mkdir -p "$OUT"
SIGMAS=${SIGMAS:-"0 1e-3 2e-3 3e-3 6e-3"}

for sg in $SIGMAS; do
  echo "=== sigma=$sg  $(date +%H:%M:%S) ==="
  # vLLM engines are subprocesses: reap ours on THIS card only.
  reap_gpu "$GPU" || echo "  WARNING: gpu $GPU did not free"
  TRAIN_GPUS=$GPU NUM_ENGINES=1 ENGINE_GPU_FRACTION=0.5 \
  SIGMA=$sg ALPHA=0.0 POPULATION_SIZE=8 NUM_ITERATIONS=1 \
  TRAIN_BATCH_SIZE=16 MAX_TOKENS=${PROBE_MAX_TOKENS:-1536} EVAL_INTERVAL=0 \
  VAL_MAX_SAMPLES=8 EVAL_BATCH_SIZE=8 \
  GPU_MEMORY_UTILIZATION=0.35 TEACHER_GPU_MEMORY_UTILIZATION=0.18 \
  LOGGER='[console]' EXPERIMENT_NAME=sigma_probe_${sg} LOG_DIR=$OUT \
    bash scripts/zo_opd/nersc_align/es_seq_opd_nersc.sh > "$OUT/probe_${sg}.log" 2>&1
  echo -n "  sigma=$sg  "
  grep -oE "train/reward_mean:[-0-9.e+]+|train/reward_std:[-0-9.e+]+|train/kl_mean:[-0-9.e+]+|train/kl_spread:[-0-9.e+]+|train/resp_len:[-0-9.e+]+" \
      "$OUT/probe_${sg}.log" | tr '\n' ' '
  echo
done

echo "=== summary ==="
printf "%-8s %14s %14s %12s\n" sigma KL spread resp_len
for sg in $SIGMAS; do
  kl=$(grep -oE "train/kl_mean:[-0-9.e+]+" "$OUT/probe_${sg}.log" | tail -1 | cut -d: -f2)
  sd=$(grep -oE "train/kl_spread:[-0-9.e+]+" "$OUT/probe_${sg}.log" | tail -1 | cut -d: -f2)
  rl=$(grep -oE "train/resp_len:[-0-9.e+]+" "$OUT/probe_${sg}.log" | tail -1 | cut -d: -f2)
  printf "%-8s %14s %14s %12s\n" "$sg" "${kl:-NA}" "${sd:-NA}" "${rl:-NA}"
done
