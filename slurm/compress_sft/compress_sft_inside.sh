#!/bin/bash
# compress_sft_inside.sh — runs under a live N-node alloc (SLURM_JOB_ID set).
# Launches LlamaFactory multi-node DDP across all allocated nodes: one srun task
# per node, each setting NODE_RANK=$SLURM_NODEID and exec'ing compress_sft_env.sh,
# which runs `llamafactory-cli train` (spawns torchrun --nproc_per_node=4). Returns
# when training finishes (rc=0) OR the 4h alloc is revoked (srun killed). The
# controller loops it.
#
# Inherits from the controller: ENV_SCRIPT, YAML, OUTPUT_DIR, RUN_NAME, WANDB_RUN_ID.
set -x
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
cd "$OPD_REPO"

JID=$SLURM_JOB_ID
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=${#NODES[@]}
HEAD=${NODES[0]}
MASTER_PORT=${MASTER_PORT:-29501}

# Master address = head node's high-speed (hsn) IP, reachable by all nodes.
MASTER_ADDR=$(srun --jobid="$JID" -N1 -n1 -w "$HEAD" hostname --ip-address 2>/dev/null \
  | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)
echo "inside.sh: NNODES=$NNODES HEAD=$HEAD MASTER=$MASTER_ADDR:$MASTER_PORT NODES=${NODES[*]}"

ENV_SCRIPT=${ENV_SCRIPT:-compress_sft/compress_sft_env.sh}

# One task per node. $SLURM_NODEID is the per-node rank 0..NNODES-1. Each task
# exec's the env script which spawns torchrun for that node's 4 local GPUs.
srun --jobid="$JID" -N"$NNODES" --ntasks-per-node=1 --gpus-per-node=4 --overlap bash -c \
  "export NNODES=$NNODES NODE_RANK=\$SLURM_NODEID NPROC_PER_NODE=4 \
          MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT FORCE_TORCHRUN=1 \
          OPD_REPO=$OPD_REPO; \
   exec bash $OPD_REPO/slurm/$ENV_SCRIPT"
RC=$?
echo "=== inside.sh done rc=$RC $(date) ==="
exit $RC
