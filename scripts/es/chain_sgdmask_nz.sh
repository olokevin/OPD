#!/bin/bash
# Second sgdmask arm (section 18): the threshold-0 mask -- every entry whose bf16 value SGD
# changed at all (~0.13% at step 5), not just those moved by >1e-5 (~0.002%).  Waits for
# chain_sgdmask.sh (WAIT_PID) so GPU 0 stays serialised, reuses its kept step-10 HF dump.
#
#   DEVICE=0 WAIT_PID=<pid of chain_sgdmask.sh> bash scripts/es/chain_sgdmask_nz.sh
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
DEVICE=${DEVICE:-0}
RUN=${RUN:-/data/yequan/bp/BP-q2p5-7b/sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10}
OUT=${OUT:-${REPO}/datasets/es_math}
ITERS=${ITERS:-80}
SIGMAS=${SIGMAS:-"0.001 0.003 0.01 0.03 0.1 0.3"}

if [ -n "${WAIT_PID:-}" ]; then
  echo "=== [$(date)] waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
[ -f "$RUN/global_step_10/actor/huggingface/model.safetensors.index.json" ] || { echo "missing step-10 HF ckpt"; exit 2; }

echo "=== [$(date)] building threshold-0 mask"
python3 scripts/es/build_sgd_mask.py --ckpt "$RUN/global_step_10/actor/huggingface" \
  --out "$OUT/sgd_mask_qwen2p5_math_7b_st10_nz.pt" --threshold 0 --compare "$OUT/sgd_mask_qwen2p5_math_7b_st10.pt"
export MASK_PATH="$OUT/sgd_mask_qwen2p5_math_7b_st10_nz.pt"

echo "=== [$(date)] sigma probe (nz)"
DEVICE=$DEVICE MODE=sgdmask SIGMAS="$SIGMAS" bash scripts/es/probe_reward_std.sh 2>&1 | tee logs/es/probe_sgdmask_nz_gpu${DEVICE}.log
read -r SIG SIG_HI < <(python3 scripts/es/pick_sigma.py logs/es/probe_sgdmask_nz_gpu${DEVICE}.log --target 0.050)
SIG=$(python3 -c "print('%.3g'%$SIG)")
ALP=$(python3 -c "print('%.6g'%($SIG/2*(1/3)**0.5))")
echo "=== [$(date)] picked sigma=$SIG (2x bracket $SIG_HI) alpha=$ALP"

echo "=== [$(date)] sgdmask-nz N=10 sigma=$SIG alpha=$ALP iters=$ITERS on GPU $DEVICE"
DEVICE=$DEVICE PERTURB_MODE=sgdmask POPULATION_SIZE=10 SIGMA="$SIG" ALPHA="$ALP" NUM_ITERATIONS="$ITERS" FORGET_TASKS= \
  EXPERIMENT_NAME="sgdmask-st10nz_math-lv3to5-b64_sig${SIG}_a${ALP}_N10_it${ITERS}" \
  bash scripts/es/run_es_math.sh
echo "=== [$(date)] sgdmask-nz exited with $?"
echo "SGDMASK NZ CHAIN DONE"
