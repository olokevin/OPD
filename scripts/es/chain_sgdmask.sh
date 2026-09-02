#!/bin/bash
# SGD-mask ES chain (section 18): wait for the SGD-GRPO run, diff its bf16 checkpoints against
# the base into coordinate masks, pick sigma by the reward_std probe (section 17.1), then run
# the N=10 / 80-iteration ES arm that perturbs only the masked entries.
#
#   DEVICE=0 SGD_PID=<pid of run_sgd_mask.sh> bash scripts/es/chain_sgdmask.sh
set -u
REPO=${REPO:-/home/yequan/Project/compression/OPD}
cd "$REPO"
export PATH=/home/yequan/miniconda3/envs/verl/bin:$PATH
export HF_HOME=${HF_HOME:-/data/yequan/huggingface}
DEVICE=${DEVICE:-0}
RUN=${RUN:-/data/yequan/bp/BP-q2p5-7b/sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10}
OUT=${OUT:-${REPO}/datasets/es_math}
ITERS=${ITERS:-80}
SIGMAS=${SIGMAS:-"0.001 0.003 0.01 0.03 0.1"}

if [ -n "${SGD_PID:-}" ]; then
  echo "=== [$(date)] waiting for SGD run pid $SGD_PID"
  while kill -0 "$SGD_PID" 2>/dev/null; do sleep 60; done
fi
for s in 5 10; do
  [ -f "$RUN/global_step_$s/actor/huggingface/model.safetensors.index.json" ] || { echo "missing step-$s HF ckpt"; exit 2; }
done

echo "=== [$(date)] building masks"
python3 scripts/es/build_sgd_mask.py --ckpt "$RUN/global_step_10/actor/huggingface" \
  --out "$OUT/sgd_mask_qwen2p5_math_7b_st10.pt" --threshold 1e-5
python3 scripts/es/build_sgd_mask.py --ckpt "$RUN/global_step_5/actor/huggingface" \
  --out "$OUT/sgd_mask_qwen2p5_math_7b_st5.pt" --threshold 1e-5 --compare "$OUT/sgd_mask_qwen2p5_math_7b_st10.pt"
# /data is full: the step-5 dump (15 GB) is reproducible from the masks' stats; keep step 10.
rm -rf "$RUN/global_step_5/actor/huggingface"
export MASK_PATH="$OUT/sgd_mask_qwen2p5_math_7b_st10.pt"

echo "=== [$(date)] sigma probe"
DEVICE=$DEVICE MODE=sgdmask SIGMAS="$SIGMAS" bash scripts/es/probe_reward_std.sh 2>&1 | tee logs/es/probe_sgdmask_gpu${DEVICE}.log
read -r SIG SIG_HI < <(python3 scripts/es/pick_sigma.py logs/es/probe_sgdmask_gpu${DEVICE}.log --target 0.050)
SIG=$(python3 -c "print('%.3g'%$SIG)")
ALP=$(python3 -c "print('%.6g'%($SIG/2*(1/3)**0.5))")
echo "=== [$(date)] picked sigma=$SIG (2x bracket $SIG_HI) alpha=$ALP"

echo "=== [$(date)] sgdmask N=10 sigma=$SIG alpha=$ALP iters=$ITERS on GPU $DEVICE"
DEVICE=$DEVICE PERTURB_MODE=sgdmask POPULATION_SIZE=10 SIGMA="$SIG" ALPHA="$ALP" NUM_ITERATIONS="$ITERS" FORGET_TASKS= \
  EXPERIMENT_NAME="sgdmask-st10_math-lv3to5-b64_sig${SIG}_a${ALP}_N10_it${ITERS}" \
  bash scripts/es/run_es_math.sh
echo "=== [$(date)] sgdmask exited with $?"
echo "SGDMASK CHAIN DONE"
