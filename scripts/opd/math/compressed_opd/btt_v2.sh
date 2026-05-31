#!/bin/bash
# btt_v2.sh — compressed_opd math launcher: CALIB_MODE=v2 (forward-activation-
# aware BTT decomposition). Backward gradient covariance is NOT collected; the
# OPD-faithful loss (CALIB_LOSS=opd) is passed through but unused by this
# variant. Use btt_v2_combined.sh if you want the OPD loss to actually steer
# the decomposition via backward whitening.
#
# Defaults to GPU 4. Override per-run, e.g. `LR=5e-6 bash btt_v2.sh` or
# `CUDA_VISIBLE_DEVICES=4 bash btt_v2.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default pinning for this script (overridable by the caller).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

# Per-script knob.
export CALIB_MODE=v2

bash on_policy_distillation.sh
