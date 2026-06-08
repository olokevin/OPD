"""Subsystem Ablation — localize which subsystem causes the structured collapse.

After two negatives (steering-subspace thesis falsified; attention-tail M1 weak,
but MLP was held compressed throughout), the cheapest most-diagnostic test is:
compress ONE subsystem at a time at the 0.36 budget and eval MATH-500.

Conditions (all @ retain 0.36):
  - attn_only : SVD-LLM-V2 on attention, MLP left DENSE
  - mlp_only  : Nystrom on MLP, attention left DENSE
  - both      : the full collapse (reference)

Disambiguates (GPT-5.4-designed):
  - mlp_only ~= both ~= 0, attn_only healthy  -> MLP/Nystrom is the killer
  - attn_only & mlp_only both OK, both ~= 0    -> interaction / cross-layer
  - attn_only ~= 0 too                          -> attention low-rank structure harmful

Note these are NOT iso-param to the 1.7B target (each compresses only half the
compressible linears), so report params per cell. The question here is *mechanism
localization*, not a parameter-matched comparison.

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
    scripts/reasoning_aware_compress/subsystem_ablation.py \
      --math-limit 100 --out results/block0/subsystem_ablation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.calibration import collect_mixed_statistics  # noqa: E402
from compress.compress_model import MethodSpec  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402

from layer_sensitivity import (  # noqa: E402
    build_openthought3_loader, eval_math500,
)

MIXED_SPEC = MethodSpec(attn="svd_llm_v2", mlp="nystrom")


def _layer_idx(name):
    """Decoder layer index from a module name, or None."""
    try:
        return int(name.split(".layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def keep_compressible_layers(stats, n_layers, skip_last):
    """Drop stats for the top `skip_last` decoder layers so they stay DENSE.

    skip_layers in the compressors matches by leaf name only (lm_head, q_proj),
    so to spare a whole decoder layer we instead remove its covariance entries —
    the compressors only touch modules that have stats.
    """
    if skip_last <= 0:
        return dict(stats)
    protect = set(range(n_layers - skip_last, n_layers))
    return {n: s for n, s in stats.items() if _layer_idx(n) not in protect}


def _compress_attn(model, stats, ratio, device):
    attn_subset = {n: s.clone() for n, s in stats.items() if ".self_attn." in n}
    svd_llm_v2_compress_model(model, attn_subset, compression_ratio=ratio,
                              skip_layers=("lm_head",), device=device, objective="forward")


def _compress_mlp(model, stats, ratio, device):
    down_subset = {n: s.clone() for n, s in stats.items() if n.endswith(".down_proj")}
    nystrom_compress_model(model, down_subset, sparsity=1.0 - ratio,
                           skip_layers=("lm_head",), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="retain ratio (0.8 = keep 80%% of compressible params)")
    ap.add_argument("--skip-last-layers", type=int, default=1,
                    help="leave the top-N decoder layers DENSE (final-layer MLP is "
                         "critical per When-Reasoning-Meets-Compression)")
    ap.add_argument("--conditions", nargs="+",
                    default=["attn_only", "mlp_only", "both"],
                    choices=["attn_only", "mlp_only", "both"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=100)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--eval-baseline", action="store_true",
                    help="also eval the uncompressed dense model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    base_cpu.eval()

    # ---- mixed stats once ---- #
    model = copy.deepcopy(base_cpu).to(device)
    logger.info("Collecting OpenThought3 mixed statistics (one pass)...")
    calib_loader = build_openthought3_loader(
        tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
    )
    stats = collect_mixed_statistics(
        model, calib_loader, MIXED_SPEC, device=device, skip_layers=("lm_head",),
    )
    n_layers = base_cpu.config.num_hidden_layers
    n_before = len(stats)
    stats = keep_compressible_layers(stats, n_layers, args.skip_last_layers)
    logger.info(f"Skip-last-layers={args.skip_last_layers}: kept stats for "
                f"{len(stats)}/{n_before} modules (top {args.skip_last_layers} "
                f"decoder layer(s) of {n_layers} left dense)")
    del calib_loader, model
    torch.cuda.empty_cache()

    results = {"model": args.model, "ratio": args.ratio,
               "skip_last_layers": args.skip_last_layers,
               "config": vars(args), "cells": []}

    def _save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.eval_baseline:
        logger.info("=== baseline (dense) ===")
        m = copy.deepcopy(base_cpu).to(device)
        acc = eval_math500(m, tokenizer, device=device, limit=args.math_limit,
                           max_new_tokens=args.math_max_new_tokens,
                           batch_size=args.math_batch_size)
        nz = sum(p.numel() for p in m.parameters())
        results["cells"].append({"condition": "dense", "math500_acc": acc,
                                 "total_params": int(nz)})
        logger.info(f"  dense MATH-500={acc:.4f}  params={nz/1e9:.3f}B")
        _save(); del m; torch.cuda.empty_cache()

    for cond in args.conditions:
        t0 = time.time()
        logger.info(f"=== condition: {cond} @ ratio {args.ratio} ===")
        m = copy.deepcopy(base_cpu).to(device)
        if cond in ("attn_only", "both"):
            _compress_attn(m, stats, args.ratio, device)
        if cond in ("mlp_only", "both"):
            _compress_mlp(m, stats, args.ratio, device)
        m.to(device)
        nz = sum(p.numel() for p in m.parameters())
        acc = eval_math500(m, tokenizer, device=device, limit=args.math_limit,
                           max_new_tokens=args.math_max_new_tokens,
                           batch_size=args.math_batch_size)
        rec = {"condition": cond, "math500_acc": acc, "total_params": int(nz),
               "elapsed_s": time.time() - t0}
        results["cells"].append(rec)
        logger.info(f"  {cond}: MATH-500={acc:.4f}  params={nz/1e9:.3f}B  "
                    f"[{time.time()-t0:.0f}s]")
        _save(); del m; torch.cuda.empty_cache()

    logger.info("==== SUBSYSTEM ABLATION SUMMARY (MATH-500) ====")
    for c in results["cells"]:
        logger.info(f"  {c['condition']:10s}: MATH={c['math500_acc']:.4f}  "
                    f"params={c['total_params']/1e9:.3f}B")
    _save()
    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
