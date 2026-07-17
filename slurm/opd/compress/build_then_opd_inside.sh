#!/bin/bash
# build_then_opd_inside.sh — runs under a live N-node interactive alloc (salloc cmd,
# on the login node; srun's onto compute nodes). If the zero-padded compressed
# student ($ACTOR_MODEL_PATH) is missing, BUILD it on the head node (1 GPU, full
# node RAM), then hand off to opd_2node_inside.sh which Ray-bootstraps + runs OPD on
# all N*4 GPUs. On auto-resume the student already exists, so the build is skipped
# and it goes straight to OPD resume.
set -u
OPD_REPO=${OPD_REPO:-/global/u1/y/yequan/Project/OPD}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/y/yequan/opd}
cd "$OPD_REPO"

JID=$SLURM_JOB_ID
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD=${NODES[0]}
STU=${ACTOR_MODEL_PATH:?set ACTOR_MODEL_PATH}

if [ ! -f "$STU/config.json" ]; then
  echo "=== [build] student missing -> building zero-padded student on $HEAD (1 GPU) $(date) ==="
  srun --jobid="$JID" -N1 -n1 -w "$HEAD" --gpus-per-node=4 --overlap bash -c '
    set -uo pipefail
    OPD_REPO='"$OPD_REPO"'; DATA_ROOT='"$DATA_ROOT"'; STU='"$STU"'; cd "$OPD_REPO"
    source /pscratch/sd/y/yequan/miniconda3/etc/profile.d/conda.sh
    conda activate ${DATA_ROOT}/envs/verl
    export PYTHONPATH="${OPD_REPO}/src:${OPD_REPO}/verl${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME=${DATA_ROOT}/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    _C=/tmp/${USER}/zo_cache; export TRITON_CACHE_DIR=${_C}/triton XDG_CACHE_HOME=${_C}/xdg; mkdir -p "$_C"
    echo "[build] node=$(hostname) -> $STU"
    python scripts/compress_opd/math/build_svd_nystrom_student.py \
      --model Qwen/Qwen3-4B --ratio 0.7 --objective combined \
      --skip-last-layers 1 --save-zero-padded --skip-ppl --skip-math \
      --save-dir "$STU" --metrics-json "${STU}_pre_opd.json"'
  if [ ! -f "$STU/config.json" ]; then
    echo "=== [build] FAILED — no $STU/config.json; aborting (controller will retry) ==="; exit 1; fi
  echo "=== [build] done $(date) | student ready: $STU ==="
fi

echo "=== [opd] handing off to opd_2node_inside.sh on $SLURM_NNODES nodes $(date) ==="
exec bash "$OPD_REPO/slurm/opd/opd_2node_inside.sh"
