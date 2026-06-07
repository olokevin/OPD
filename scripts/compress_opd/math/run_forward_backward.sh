#!/bin/bash
# run_forward_backward.sh — compress→OPD with FORWARD+BACKWARD calibration.
# Wiki cell D2: bilateral (input + CE-backward) SVD-V2 for attn + joint
# Nystrom kernel K = Cf^{1/2} Cb Cf^{1/2} for mlp. Needs a full backward over
# the calib pass on the 4B base — heavier than forward-only (~30 GB extra),
# uses the new mask-aware sequence/full calibration default.
#
# Equivalent invocation: OBJECTIVE=combined bash run_svd_nystrom_opd.sh
#
# Overridable env vars: GPU, RATIO, EXPERIMENT_TAG, STUDENT_DIR, STAGE, PY.
#
# Memory hint: if the combined backward OOMs on your card, lower
# CALIB_NUM_SEQS (the bs=1 backward path is the ceiling, not throughput).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJECTIVE=combined bash "$SCRIPT_DIR/run_svd_nystrom_opd.sh" "$@"
