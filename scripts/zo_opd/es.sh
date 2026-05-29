#!/bin/bash
# ES training on GPU 2 (single GPU, 1 vLLM engine).
set -e

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=2 python3 -m verl.trainer.main_es \
    --config-name es_trainer \
    es.sigma=0.001 \
    es.alpha=0.0005 \
    es.population_size=30 \
    es.num_engines=1 \
    es.num_iterations=800 \
    data.train_files=data/countdown/train.parquet \
    data.val_files=data/countdown/test.parquet \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    'trainer.logger=[console, wandb]'
