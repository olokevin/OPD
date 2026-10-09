#!/bin/bash
# pn_explore.sh -- BP-OPD with parameter-space-noise exploration (Plappert et al. 2018), ONE GPU.
#
# Same pipeline as bp_opd.sh (R1-Distill-1.5B <- JustRL-1.5B, DAPO-Math-17k, 64 prompts x n=4 =
# 256 seqs / step, ppo_mini 16 prompts -> 4 Adam updates of 64 seqs each, 1 epoch, k1 token reward),
# but the 4 rollouts per prompt are GREEDY: slot 0 = clean policy, slots 1-3 = the policy under
# three fresh seeded weight perturbations W + sigma*eps_k (algorithm.pn_explore, ppo/pn_explore.py).
# The teacher scores every token, the BP update is on the clean weights, and the perturbed slots
# get token-level truncated IS min(pi_clean / pi_pert, 2) (algorithm.rollout_correction).
#
#   ARM=full   TRAIN_GPU=2 bash scripts/zo_opd/ds15b/pn_explore.sh   # all params, Adam 1e-6
#   ARM=isobtt TRAIN_GPU=3 bash scripts/zo_opd/ds15b/pn_explore.sh   # isobtt small cores V_j only, 1e-5
#   SMOKE=1 ARM=full TRAIN_GPU=2 bash scripts/zo_opd/ds15b/pn_explore.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARM=${ARM:-full}
export N_RESPONSES=${N_RESPONSES:-4}            # 1 clean + 3 perturbed per prompt
export TEMPERATURE=0                            # greedy training rollout (standing ES setting)
export VAL_N=1 VAL_DO_SAMPLE=False VAL_TEMPERATURE=0 VAL_TOP_P=1.0   # greedy eval
export ACTOR_PARAM_OFFLOAD=False                # perturbations act on the resident fp32 master
export PN_SIGMA=${PN_SIGMA:-1e-3}
export PN_PERTURB_SET=${PN_PERTURB_SET:-layers}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPD_SYSTEMD=0                            # plain supervisor (shell nofile is already high)
export RAY_TMPDIR=${RAY_TMPDIR:-/data/yequan/temp/ray_opd_gpu${TRAIN_GPU:-4}}   # root disk is ~96% full
export RAY_ROTATION_MAX_BYTES=$((200 * 1024 * 1024)) RAY_ROTATION_BACKUP_COUNT=1  # Ray logs once filled /

case "$ARM" in
  full)
    export LR=${LR:-1e-6}
    ARM_ARGS="" ;;
  isobtt)
    # isobtt with only the small per-block cores V_j trained AND perturbed; the large cores U_j
    # and the singular values S_j are frozen, as are embeddings / norms / biases / lm_head.
    export LR=${LR:-1e-5}
    export PEFT_MODE=isobtt ISO_TRAIN_OTHERS=False ISO_TRAIN_FACTORS=v
    export PN_PERTURB_SET=all                   # = every trainable tensor = the V_j deltas
    ARM_ARGS="actor_rollout_ref.actor.optim.weight_decay=0" ;;   # ISO requires wd 0
  *) echo "unknown ARM=$ARM (full | isobtt)"; exit 1 ;;
esac

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_pn-explore_${ARM}_T0_b64n4_sig${PN_SIGMA}_lr${LR}}
export CKPT_PATH=${CKPT_PATH:-/data/yequan/opd/opd/DAPO-Math-17k/${EXPERIMENT_NAME}}
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EXTRA_HYDRA_ARGS="algorithm.pn_explore=True algorithm.pn_sigma=${PN_SIGMA} \
  algorithm.pn_perturb_set=${PN_PERTURB_SET} \
  ++algorithm.rollout_correction.rollout_is=token ++algorithm.rollout_correction.rollout_is_threshold=2.0 \
  ${ARM_ARGS} ${EXTRA_HYDRA_ARGS:-}"
export PROJECT_NAME=${PROJECT_NAME:-es_opd_JustRL_1p5b_0903}   # = the BP baseline's project (run s7x9sidi)
export LOG_DIR=${LOG_DIR:-logs/ds15b/pn_explore}

exec bash "$SCRIPT_DIR/bp_opd.sh"
