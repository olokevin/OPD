#!/bin/bash
# build_sft_env.sh — create the `sft` conda env on NERSC scratch for the
# compress->SFT jobs: py3.11 + editable LlamaFactory + flash-attn (fa2) + metrics.
# Run on a LOGIN node (has internet). Logs to the file the caller tees.
set -uo pipefail
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
ENV_PREFIX=${ENV_PREFIX:-${DATA_ROOT}/envs/sft}

source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh

echo "=== [build_sft_env] $(date) -> $ENV_PREFIX ==="
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"
python -V; which pip

# Keep pip/HF caches off GPFS $HOME (flock + quota).
export PIP_CACHE_DIR=${DATA_ROOT}/.cache/pip
export HF_HOME=${DATA_ROOT}/huggingface
mkdir -p "$PIP_CACHE_DIR"

echo "=== [build_sft_env] editable LlamaFactory (pulls torch CUDA wheels) ==="
cd "$OPD_REPO/LlamaFactory"
pip install -e . 2>&1 | tail -5

echo "=== [build_sft_env] metrics deps ==="
pip install -r requirements/metrics.txt 2>&1 | tail -3

echo "=== [build_sft_env] compress-core runtime deps (loguru) ==="
pip install loguru 2>&1 | tail -2

echo "=== [build_sft_env] flash-attn (fa2) — may compile, ~20-40 min; non-fatal ==="
pip install flash-attn --no-build-isolation 2>&1 | tail -8 \
  && echo "[build_sft_env] flash-attn OK" \
  || echo "[build_sft_env] flash-attn FAILED -> use flash_attn: sdpa in the YAML"

echo "=== [build_sft_env] sanity import ==="
python - <<'PY'
import torch, transformers, llamafactory
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("llamafactory", llamafactory.__file__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn NOT available:", e)
import sys; sys.path.insert(0, "/global/u1/y/yequan/Project/OPD/src")
import compress; print("compress importable")
PY
echo "=== [build_sft_env] done $(date) ==="
