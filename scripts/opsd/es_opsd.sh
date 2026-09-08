#!/bin/bash
# es_opsd.sh -- forward-only es-prefill OPSD, ONE GPU.
#
# Identical pipeline to bp_opsd.sh (same model, self-teacher with privileged context,
# same data, rollout, teacher scoring, clipped-forward-KL advantages and eval) with
# algorithm.es_update=True: the BP actor update is replaced by N teacher-forced
# "prefill rails" -- forwards of the actor over the FIXED rollout under antithetic
# seeded Gaussian perturbations of the TRAINABLE (= LoRA) weights -- scored with the
# very objective BP maximises, then an OpenAI-ES step
# (verl/trainer/ppo/es_update.py).  Everything else is shared, so any difference in
# the curves is the gradient estimator.
#
# Perturbing LoRA only (r=64 => ~70M coefficients vs 1.72B dense) is the point: the
# es-prefill displacement-budget law (docs/results/ZO_OPD/es_rails_formulation.md §3)
# says the random walk that ends ES learning scales with sqrt(D), so a 25x smaller D
# is a 5x larger budget at equal coherent motion.
#
#   TRAIN_GPU=6 ES_N_RAILS=8 ES_SIGMA=1e-3 ES_ALPHA=5e-4 bash scripts/opsd/es_opsd.sh
#   SMOKE=1 TRAIN_GPU=6 bash scripts/opsd/es_opsd.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# sigma/alpha are NOT the ds15b full-weight numbers: LoRA's B is zero-initialised, so
# RMS(W) over the trainable coefficients is a different scale entirely.  Calibrate with
# scripts/opsd/calibrate_es_opsd.sh (reads es/d_snr, es/probe_footprint, es/post_update_gain).
export ES_SIGMA=${ES_SIGMA:-1e-3}
export ES_ALPHA=${ES_ALPHA:-5e-4}
export ES_N_RAILS=${ES_N_RAILS:-8}
export ES_NORMALIZE=${ES_NORMALIZE:-zscore}
export ES_ANTITHETIC=${ES_ANTITHETIC:-True}
export ES_STEPS=${ES_STEPS:-300}

export ACTOR_PARAM_OFFLOAD=False                 # rails perturb the resident fp32 master
export ACTOR_OPTIM_OFFLOAD=True                  # optimizer is never materialised
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opsd_es-prefill_N${ES_N_RAILS}_sig${ES_SIGMA}_a${ES_ALPHA}_lora${LORA_RANK:-64}}
export EXTRA_HYDRA_ARGS="algorithm.es_update=True algorithm.es_sigma=${ES_SIGMA} \
  algorithm.es_antithetic=${ES_ANTITHETIC} \
  algorithm.es_alpha=${ES_ALPHA} algorithm.es_n_rails=${ES_N_RAILS} \
  algorithm.es_normalize=${ES_NORMALIZE} \
  ${EXTRA_HYDRA_ARGS:-}"
export OPSD_STEPS=${OPSD_STEPS:-$ES_STEPS}
export LOG_DIR=${LOG_DIR:-logs/opsd/es}
export TRAIN_GPU=${TRAIN_GPU:-6}

exec bash "$SCRIPT_DIR/bp_opsd.sh"
