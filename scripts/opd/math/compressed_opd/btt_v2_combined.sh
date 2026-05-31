#!/bin/bash
# btt_v2_combined.sh — compressed_opd math launcher: CALIB_MODE=v2_combined
# (forward activation + backward gradient covariances). With CALIB_LOSS=opd the
# backward whitening is driven by the actual teacher-student OPD policy
# gradient on the calibration corpus (see src/compress/calibration_opd_loss.py
# and compress.calibration.collect_both_covariances_from_loader_opd) — this is
# the "use the actual teacher-student OPD loss to compute gradient and collect
# gradient correlation matrix" path.
#
# Defaults to GPU 5. Override per-run, e.g. `LR=5e-6 bash btt_v2_combined.sh`
# or `CUDA_VISIBLE_DEVICES=5 bash btt_v2_combined.sh`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default pinning for this script (overridable by the caller).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

# Per-script knob.
export CALIB_MODE=v2_combined

bash on_policy_distillation.sh
