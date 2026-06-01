"""Verify the BTT compression implementation by C4-calibrating the same
4B → 1.66B BTT decomposition the compressed_opd training uses, then
measuring C4 PPL on the result.

The compressed_opd training uses MATH-training-set calibration with the
OPD-faithful loss, which yields PPL > 1e6 on C4. That can be either an
implementation bug or the (expected) consequence of calibrating against
distributions far from C4. This script re-runs the SAME decomposition
(train_mode=btt_llm_v2, output_one_block, rank=0.36) but with C4
calibration data, so a working implementation should give PPL << 1e3.

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 scripts/opd/math/compressed_opd/eval_c4_calib_ppl.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compress.decomposition import DecompositionConfig, decompose_with_loader  # noqa: E402
from compress.loaders import build_c4_calib_loader  # noqa: E402
from compress.ppl_eval import evaluate_model_ppl  # noqa: E402

# Match _common.sh
TEACHER = "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mode", default="btt_llm_v2",
                    choices=["btt_llm_v2", "btt_llm_v2_combined"])
    ap.add_argument("--decomp-mode", default="output_one_block")
    ap.add_argument("--rank", type=float, default=0.36)
    ap.add_argument("--s-merged-to", default="split")
    ap.add_argument("--skip-layers", default="lm_head")
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=4)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--ppl-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading {TEACHER} in {args.dtype}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=dtype)
    model = model.to(args.device)
    orig_params = _count_params(model)
    print(f"  loaded: {orig_params / 1e9:.3f}B params", flush=True)

    print(f"\nMeasuring baseline C4 PPL of UNCOMPRESSED model...", flush=True)
    t0 = time.time()
    baseline = evaluate_model_ppl(
        model, tokenizer,
        seqlen=args.ppl_seqlen, seed=args.ppl_seed,
        datasets=("c4",), device=args.device,
    )
    print(f"  uncompressed C4 PPL = {baseline['c4']:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\nBuilding C4 calibration loader "
          f"(num_seqs={args.calib_num_seqs}, max_length={args.calib_max_length}, "
          f"batch_size={args.calib_batch_size})...", flush=True)
    calib_loader = build_c4_calib_loader(
        tokenizer,
        num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length,
        batch_size=args.calib_batch_size,
        seed=args.calib_seed,
    )

    cfg = DecompositionConfig(
        train_mode=args.train_mode,
        compression_ratio=args.rank,
        skip_layers=args.skip_layers,
        decomp_mode=args.decomp_mode,
        train_position="both",   # matches _common.sh; doesn't affect weights
        s_merged_to=args.s_merged_to,
        factorize_by_head=True,
        calib_num_seqs=args.calib_num_seqs,
        calib_max_length=args.calib_max_length,
        calib_seed=args.calib_seed,
    )
    print(f"\nDecomposing with {args.train_mode} "
          f"(decomp_mode={args.decomp_mode}, rank={args.rank}, "
          f"s_merged_to={args.s_merged_to})...", flush=True)
    t0 = time.time()
    model, stats = decompose_with_loader(
        model, cfg, calib_loader=calib_loader,
        device=args.device, return_trainability_stats=True,
    )
    print(f"  decomposition done in {time.time() - t0:.1f}s "
          f"(num_btt_layers={stats['num_btt_layers']})", flush=True)
    new_params = _count_params(model)
    print(f"  compressed: {new_params / 1e9:.3f}B params "
          f"(ratio {new_params / orig_params:.3f})", flush=True)

    print(f"\nMeasuring C4 PPL of compressed model...", flush=True)
    t0 = time.time()
    ppl = evaluate_model_ppl(
        model, tokenizer,
        seqlen=args.ppl_seqlen, seed=args.ppl_seed,
        datasets=("c4",), device=args.device,
    )
    print(f"  compressed C4 PPL = {ppl['c4']:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n========== SUMMARY ==========", flush=True)
    print(f"  train_mode           = {args.train_mode}", flush=True)
    print(f"  decomp_mode          = {args.decomp_mode}", flush=True)
    print(f"  compression_ratio    = {args.rank}", flush=True)
    print(f"  s_merged_to          = {args.s_merged_to}", flush=True)
    print(f"  calib                = C4 (num_seqs={args.calib_num_seqs}, "
          f"max_length={args.calib_max_length})", flush=True)
    print(f"  uncompressed C4 PPL  = {baseline['c4']:.4f}", flush=True)
    print(f"  compressed   C4 PPL  = {ppl['c4']:.4f}", flush=True)
    print(f"  param ratio          = {new_params / orig_params:.3f}", flush=True)


if __name__ == "__main__":
    main()
