#!/bin/bash
# es_opd.sh -- forward-only ES-OPD in the thunlp reference setting, ONE GPU.
#
# Identical pipeline to bp_opd.sh (same student/teacher/data/rollout/teacher scoring/
# advantages/eval) with algorithm.es_update=True: the BP actor update is replaced by
# N teacher-forced "prefill rails" -- forwards of the actor over the fixed rollout under
# antithetic seeded Gaussian weight perturbations -- scored with BP's own objective, and an
# OpenAI-ES step (verl/trainer/ppo/es_update.py).  Everything else is shared, so any
# difference in the curves is the gradient estimator.
#
# Defaults: 16 prompts x n=4 = 64 seqs / step, N=32 rails (16 antithetic pairs), 300 steps.
# sigma/alpha: calibrate with scripts/zo_opd/ds15b/opd_curvature.py (KL-rise curve at the
# production length); the ES convention alpha = sigma/2 gives per-coordinate motion
# alpha/sqrt(n_pairs) per step.
#
#   TRAIN_GPU=1 ES_SIGMA=1e-3 ES_ALPHA=5e-4 bash scripts/zo_opd/ds15b/es_opd.sh
#   SMOKE=1 TRAIN_GPU=1 bash scripts/zo_opd/ds15b/es_opd.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ES_SIGMA=${ES_SIGMA:-1e-3}
export ES_ALPHA=${ES_ALPHA:-5e-4}
export ES_N_RAILS=${ES_N_RAILS:-32}
export ES_NORMALIZE=${ES_NORMALIZE:-zscore}
export ES_STEPS=${ES_STEPS:-300}
export ES_PERTURB_SET=${ES_PERTURB_SET:-all}          # all | layers (decoder blocks only, = es_token's set)

export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}        # 16 prompts x n=4 = 64 seqs / step
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ACTOR_PARAM_OFFLOAD=False                     # rails perturb the resident fp32 master
export ACTOR_OPTIM_OFFLOAD=True                      # optimizer is never materialised
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ds15b_es-prefill_k${LOG_PROB_TOP_K:-0}_N${ES_N_RAILS}_sig${ES_SIGMA}_a${ES_ALPHA}}
export EXTRA_HYDRA_ARGS="algorithm.es_update=True algorithm.es_sigma=${ES_SIGMA} \
  algorithm.es_antithetic=${ES_ANTITHETIC:-True} \
  algorithm.es_alpha=${ES_ALPHA} algorithm.es_n_rails=${ES_N_RAILS} \
  algorithm.es_normalize=${ES_NORMALIZE} algorithm.es_perturb_set=${ES_PERTURB_SET} trainer.total_training_steps=${ES_STEPS} \
  ${EXTRA_HYDRA_ARGS:-}"
export LOG_DIR=${LOG_DIR:-logs/ds15b/es}

exec bash "$SCRIPT_DIR/bp_opd.sh"
