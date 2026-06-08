"""Block A — Low-Rank + Sparse Residual / "+Patch" (mechanism M1: the rank floor).

Causal M1 test: structured low-rank truncation drops the full-rank "escape edges"
SparseGPT keeps. Restore them as a SparseGPT-pruned residual S of R = W - UV at a
small leftover budget; the total (UV + S) stays at the block retain ratio. If A2
holds dense-level accuracy to a LOWER ratio than A0 (pure SVD), the full-rank
escape edges are the missing ingredient → M1 confirmed causally (succeeding where
attention tail-rescue's 0→4% failed, because that only re-added LOW-rank tail, not
FULL-rank sparse edges).

Cells (attention; MLP = standing Nystrom forward throughout):
  A0  pure SVD-V2 at the block ratio (= D0)                       — no-patch baseline
  A1  LR + sparse-residual, R fit against DENSE activations       — cheap baseline
  A2  LR + sparse-residual, R fit against COMPRESSED-upstream acts — the real claim
  A3  budget-split sweep (S share of retained budget ∈ sweep)     — where the tail earns params

Operating point: retain 0.8 first, last decoder layer left dense. Eval MATH/100 + C4 PPL.
Keep S small and ABLATE it (A0 vs A2) so the claim stays "we fixed *structured*
compression", not "a sparse tail did the work".

Usage (GPU 5):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/reasoning_aware_compress/lr_sparse_residual.py \
        --cells A0 A1 A2 --ratio 0.8 --sparse-frac 0.075 --math-limit 100 \
        --out scripts/reasoning_aware_compress/results/blockA/lr_sparse_r0.8.json
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn
from loguru import logger

import compress_common as cc

from compress.calibration import (  # noqa: E402
    collect_mixed_statistics, collect_covariances_from_loader,
)
from compress.compress_model import MethodSpec  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.hybrid.lr_sparse import lr_sparse_compress_attn  # noqa: E402

ALL_CELLS = ("A0", "A1", "A2", "A3")


def _capture_dense_attn(model, attn_names):
    """Snapshot dense W and bias for each attn linear before compression."""
    W, B = {}, {}
    mods = dict(model.named_modules())
    for n in attn_names:
        m = mods[n]
        W[n] = m.weight.data.detach().clone()
        B[n] = m.bias.data.detach().clone() if m.bias is not None else None
    return W, B


def compress_mlp_nystrom(model, mlp_stats, *, ratio, device, protect):
    stats = cc.drop_protected_stats(mlp_stats, protect)
    logger.disable("compress")
    try:
        nystrom_compress_model(model, stats, sparsity=1.0 - ratio,
                               skip_layers=("lm_head",), device=device)
    finally:
        logger.enable("compress")
    return model


def compress_attn_svd(model, fwd_cov_attn, *, ratio, device):
    logger.disable("compress")
    try:
        svd_llm_v2_compress_model(model, fwd_cov_attn, compression_ratio=ratio,
                                  skip_layers=("lm_head",), device=device,
                                  objective="forward")
    finally:
        logger.enable("compress")
    return model


def main():
    ap = argparse.ArgumentParser()
    cc.add_cli_common(ap)
    ap.add_argument("--cells", nargs="+", default=["A0", "A1", "A2"],
                    choices=list(ALL_CELLS))
    ap.add_argument("--sparse-frac", type=float, default=0.075,
                    help="S's share of the retained budget (A1/A2). 0.075 of 0.8 ≈ LR 0.74 + S 0.06")
    ap.add_argument("--a3-sweep", nargs="+", type=float,
                    default=[0.125, 0.075, 0.0375],
                    help="A3: S-share-of-retained values (≈ S taking 1/3,1/4,1/8 of the compressed-away mass)")
    ap.add_argument("--refine-passes", type=int, default=1,
                    help="A2: extra Hessian passes re-fitting R vs the deployed UV+S prefix "
                         "(1 = fit against true deployed activations; 0 = low-rank-upstream approx)")
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    device = args.device
    tokenizer = cc.load_tokenizer(args.model)

    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = cc.load_model(args.model, dtype, "cpu")
    base_cpu.eval()
    protect = cc.protected_layer_set(base_cpu, args.skip_last_layers)
    n_layers = cc.num_hidden_layers(base_cpu)
    attn_names = cc.attn_names_excluding_protected(base_cpu, protect)
    logger.info(f"skip_last_layers={args.skip_last_layers}: protecting layers "
                f"{sorted(protect)} of {n_layers}; {len(attn_names)} attn linears compressible")

    results = {"block": "A", "model": args.model, "ratio": args.ratio,
               "skip_last_layers": args.skip_last_layers,
               "protected_layers": sorted(protect),
               "sparse_frac": args.sparse_frac, "config": vars(args), "cells": []}

    # ---- shared statistics: MLP Nystrom C_sigma + attn input cov ---------- #
    logger.info("Collecting statistics once (MLP C_sigma + attn input cov)...")
    m0 = copy.deepcopy(base_cpu).to(device)
    calib = cc.build_calib_loader(
        args.calib, tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
        seed=args.calib_seed)
    mlp_stats_full = collect_mixed_statistics(
        m0, calib, MethodSpec(attn=None, mlp="nystrom"),
        device=device, skip_layers=("lm_head",))
    del m0, calib
    torch.cuda.empty_cache()

    m1 = copy.deepcopy(base_cpu).to(device)
    calib = cc.build_calib_loader(
        args.calib, tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
        seed=args.calib_seed)
    attn_fwd_cov_full = collect_covariances_from_loader(
        m1, calib, device=device, skip_layers=("lm_head",))
    del m1, calib
    torch.cuda.empty_cache()
    # attn-only, protected layers dropped
    attn_fwd_cov = cc.drop_protected_stats(
        {k: v for k, v in attn_fwd_cov_full.items() if ".self_attn." in k}, protect)

    def _fresh_with_mlp():
        model = copy.deepcopy(base_cpu).to(device)
        compress_mlp_nystrom(
            model, {k: v.clone() for k, v in mlp_stats_full.items()},
            ratio=args.ratio, device=device, protect=protect)
        return model

    def _build_A0():
        model = _fresh_with_mlp()
        compress_attn_svd(model, {k: v.clone() for k, v in attn_fwd_cov.items()},
                          ratio=args.ratio, device=device)
        return model

    def _build_patch(activation_source, sparse_frac):
        model = _fresh_with_mlp()
        dense_W, dense_B = _capture_dense_attn(model, attn_names)

        def _loader_factory():
            return cc.build_calib_loader(
                args.calib, tokenizer, num_seqs=args.calib_num_seqs,
                max_length=args.calib_max_length, batch_size=args.calib_batch_size,
                seed=args.calib_seed)

        lr_sparse_compress_attn(
            model, attn_names, dense_W, dense_B,
            {k: v.clone() for k, v in attn_fwd_cov.items()}, _loader_factory,
            total_ratio=args.ratio, sparse_frac=sparse_frac,
            activation_source=activation_source, device=device,
            refine_passes=args.refine_passes)
        del dense_W, dense_B
        torch.cuda.empty_cache()
        return model

    def _run(cell, model, extra=None):
        metrics = cc.eval_cell(
            model, tokenizer, device=device, math_limit=args.math_limit,
            math_max_new_tokens=args.math_max_new_tokens,
            math_batch_size=args.math_batch_size, ppl_seqlen=args.ppl_seqlen,
            skip_math=args.skip_math)
        rec = {"cell": cell, "ratio": args.ratio, **(extra or {}), **metrics}
        results["cells"].append(rec)
        cc.save_json(results, args.out)
        macc = "-" if rec["math500_acc"] is None else f"{rec['math500_acc']*100:.2f}%"
        logger.info(f"[A {cell}] nz={rec['params_nonzero_B']:.3f}B "
                    f"c4_ppl={rec['c4_ppl']:.4f} math={macc}")

    for cell in args.cells:
        logger.info(f"=== Block A cell {cell} @ retain {args.ratio} ===")
        t0 = time.time()
        if cell == "A0":
            _run("A0", _build_A0(), {"elapsed_s_start": t0})
        elif cell == "A1":
            _run("A1", _build_patch("dense", args.sparse_frac),
                 {"sparse_frac": args.sparse_frac})
        elif cell == "A2":
            _run("A2", _build_patch("compressed", args.sparse_frac),
                 {"sparse_frac": args.sparse_frac})
        elif cell == "A3":
            for sf in args.a3_sweep:
                logger.info(f"--- A3 sparse_frac={sf} (compressed acts) ---")
                _run(f"A3_sf{sf}", _build_patch("compressed", sf),
                     {"sparse_frac": sf})
        torch.cuda.empty_cache()
        logger.info(f"  ({time.time() - t0:.0f}s)")

    print("\n========== Block A SUMMARY ==========")
    print(f"{'cell':12s} {'nz params':>10s} {'C4 PPL':>10s} {'MATH-500':>10s}")
    for r in results["cells"]:
        macc = "-" if r["math500_acc"] is None else f"{r['math500_acc']*100:.2f}%"
        print(f"{r['cell']:12s} {r['params_nonzero_B']:>9.3f}B "
              f"{r['c4_ppl']:>10.4f} {macc:>10s}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
