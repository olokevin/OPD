#!/bin/bash
# compress->SFT, FORWARD+BACKWARD combined calibration. 4 interactive nodes (16
# GPUs), 16-way DDP, retain 0.6, in-process svd_nystrom compress + SFT on
# OpenThought3-Qwen3-4B, auto-resume across the 4h interactive cap.
#
# Launch detached from a login node (UNIQUE name -> safe to pkill independently):
#   nohup bash slurm/compress_sft/compress_sft_combined_controller.sh \
#     > /pscratch/sd/$USER/opd/compress_sft/logs/combined_controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &
export OBJECTIVE=combined
export RATIO=${RATIO:-0.7}
export YAML=qwen3_4b_nersc_combined_r${RATIO}_sft.yaml
source "$(dirname "$0")/_compress_sft_controller_core.sh"
