#!/bin/bash
# run_forward_only.sh — compress→OPD with FORWARD-ONLY calibration.
# Wiki cell D0 / repo default: SVD-V2 attn (input whitening) + Nystrom mlp
# (forward C_sigma only). No backward pass, no teacher loaded during calib.
#
# Equivalent invocation: OBJECTIVE=forward bash run_svd_nystrom_opd.sh
#
# Overridable env vars: GPU, RATIO, EXPERIMENT_TAG, STUDENT_DIR, STAGE, PY.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJECTIVE=forward bash "$SCRIPT_DIR/run_svd_nystrom_opd.sh" "$@"
