#!/bin/bash
# opd_2node_inside.sh — runs under a live 2-node allocation (SLURM_JOB_ID set).
# Bootstraps a Ray cluster spanning both nodes, then runs the OPD driver on the
# head node attached to that cluster. Returns when the driver exits OR when the
# 4h allocation is revoked (Slurm kills the srun steps). The controller loops it.
set -x
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
cd "$OPD_REPO"

JID=$SLURM_JOB_ID
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=${#NODES[@]}
HEAD=${NODES[0]}
PORT=6379

# Ray ACTORS inherit env from the raylet (the `ray start` step), NOT from the
# driver — so every ray-start srun step must source the full runtime env. $ACT
# does that (conda + HF/wandb/vLLM/NCCL/caches); see opd_2node_rayenv.sh.
ACT="source $OPD_REPO/slurm/opd/opd_2node_rayenv.sh;"

# Head IP reachable by workers (management net is fine for Ray's control plane).
HEAD_IP=$(srun --jobid="$JID" -N1 -n1 -w "$HEAD" hostname --ip-address 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)
RAY_ADDR="${HEAD_IP}:${PORT}"
echo "HEAD=$HEAD HEAD_IP=$HEAD_IP NODES=${NODES[*]} RAY_ADDR=$RAY_ADDR"

# Clean any stale Ray on all allocated nodes.
srun --jobid="$JID" -N"$NNODES" -n"$NNODES" --overlap bash -c "$ACT ray stop --force" 2>/dev/null || true
sleep 3

# Start the Ray HEAD on node0 (kept alive by --block; backgrounded srun step).
srun --jobid="$JID" -N1 -n1 -w "$HEAD" --overlap bash -c \
  "$ACT exec ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=4 --dashboard-host=0.0.0.0 --block" &
sleep 20

# Start a Ray WORKER on every other node (kept alive by --block).
for ((i=1; i<NNODES; i++)); do
  srun --jobid="$JID" -N1 -n1 -w "${NODES[$i]}" --overlap bash -c \
    "$ACT exec ray start --address=$RAY_ADDR --num-gpus=4 --block" &
done
sleep 20

# Wait until the cluster reports the expected GPU count (8) before launching.
for t in $(seq 1 30); do
  GPUS=$(srun --jobid="$JID" -N1 -n1 -w "$HEAD" --overlap bash -c \
    "$ACT python -c \"import ray;ray.init(address='$RAY_ADDR');print(int(ray.cluster_resources().get('GPU',0)));ray.shutdown()\"" 2>/dev/null | tail -1)
  echo "cluster GPUs so far: ${GPUS:-?}"
  [ "${GPUS:-0}" = "$((4*NNODES))" ] && break
  sleep 5
done

# Run the OPD driver on the head node, attached to the external cluster.
# (env.sh self-activates conda; export the cluster vars via bash -c rather than
# the `env` binary, which PATH-resolves to a broken ~/.local/bin/env here.)
# ENV_SCRIPT selects the driver env (full vs fura, etc.); default keeps the
# original 2-node full config. srun forwards the controller's exported env, so
# CKPT_PATH/EXPERIMENT_NAME/WANDB_RUN_ID/LR/... overrides flow through.
ENV_SCRIPT="${ENV_SCRIPT:-opd/full/opd_2node_env.sh}"
srun --jobid="$JID" -N1 -n1 -w "$HEAD" --overlap bash -c \
  "export RAY_EXTERNAL=1 RAY_ADDRESS=$RAY_ADDR NNODES=$NNODES; exec bash $OPD_REPO/slurm/$ENV_SCRIPT"
RC=$?
echo "DRIVER exited rc=$RC"
srun --jobid="$JID" -N"$NNODES" -n"$NNODES" --overlap bash -c "$ACT ray stop --force" 2>/dev/null || true
exit $RC
