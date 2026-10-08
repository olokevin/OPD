#!/bin/bash
# env.sh — DRIVER for one ISO-paper DAPO run (Qwen3-4B-Base, DeepMath-103K) on 2 nodes
# x 4 A100-80GB. Runs on the head node under opd_2node_inside.sh with RAY_EXTERNAL=1;
# the arm (PEFT_MODE, LR, ...) and the fixed CKPT/EXP come from controller.sh.
# Recipe = paper App. H, see scripts/iso/dapo_math_4b.sh and docs/results/FURA/iso.md.
set -x
export OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
source "${OPD_REPO}/slurm/iso/rayenv.sh"

export REPO=$OPD_REPO
export PY_ENV_BIN=none                     # conda env already active (rayenv.sh)
export DEVICES=0,1,2,3
export NNODES=${NNODES:-2}
export RESUME_MODE=auto
export TOTAL_STEPS=${TOTAL_STEPS:-160}   # 2026-10-07: all four runs cut from 400 to 160
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-1}   # saves take 4-12 s; steps are 25-36 min, so save every step
export PPO_TOK=${PPO_TOK:-9216}
export PROJECT=${WANDB_PROJECT}
exec bash "${OPD_REPO}/scripts/iso/dapo_math_4b.sh"
