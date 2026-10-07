#!/bin/bash
# rayenv.sh — runtime env for every Ray daemon of the ISO DAPO runs (actors inherit
# the raylet's env). = the OPD 2-node env + the verl repo on PYTHONPATH, because the
# DAPO trainer lives in verl/recipe/ (not part of the pip-installed verl package).
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
export WANDB_PROJECT=${WANDB_PROJECT:-iso_rlvr_qwen3_4b}
source "${OPD_REPO}/slurm/opd/opd_2node_rayenv.sh"
export PYTHONPATH="${OPD_REPO}/verl:${PYTHONPATH}"
