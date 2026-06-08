"""Block D — OPD-Weighted Bi-Whitened SVD (mechanism M2: the objective lever).

Tests whether the reasoning-collapse lever is a *better reconstruction objective*
(loss/OPD-weighted) rather than *more rank* (Block A). Replaces input-variance
whitening on the ATTENTION SVD path with a bilateral objective
    min ‖C_dy^{1/2} (W - Ŵ) (XᵀX)^{1/2}‖_F²
(truncated SVD of C_dy^{1/2} W (XᵀX)^{1/2}), where the output-grad covariance
C_dy is driven by the OPD/teacher loss rather than next-token CE.

The MLP path is held at the standing Nystrom (forward) compression for every cell
so the attention objective is measured on an otherwise-working model.

Cells (attention objective varies; MLP = Nystrom forward-only throughout):
  D0  forward-only input whitening (objective="forward")        — within-block baseline (= A0/B0)
  D1  backward-only whitening, C_dy from next-token CE           — isolates grad- vs input-weighting
  D2  bilateral, C_dy from next-token CE (objective="combined")  — OBD-LLM-style prior-art baseline
  D3  bilateral, C_dy from OPD/teacher loss                      — the Idea-D claim
  (D4 transition-token C_dy — not implemented here; bridge to TRACER C1)

Operating point: retain 0.8 first (sweep down later), last decoder layer left
dense (--skip-last-layers 1). Eval: MATH-500/100 + C4 PPL per cell.

Success: along the sweep D3 > D2 ≥ D1 > D0. Falsifier: D3 ≈ D0 at every ratio →
M2 is not a separable lever, the rank floor (M1, Block A) dominates.

Note on D3's teacher (GPT-5.4 review, CRITICAL): D3 needs a DISTINCT teacher.
If teacher == student the OPD KL is identically 0, the backward covariance is zero,
and D3 silently collapses to the forward-only baseline. The driver now REJECTS
teacher==student unless --allow-degenerate-opd. Pass --teacher <distinct ckpt>
(e.g. a stronger model, or the dense 4B distilling a separately-compressed student).

Known limitation (GPT-5.4 review, MAJOR — shared with the standing OPD pipeline):
the calibration windows are packed chat text with all-ones attention masks, so the
OPD/CE backward includes prompt/system tokens, not response-only. This is a
faithfulness caveat on D1/D2/D3 (and B2), not a crash; a response-span-aware
calibration loader is the documented refinement.

Usage (one GPU):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/reasoning_aware_compress/bi_whitened_svd.py \
        --cells D0 D1 D2 D3 --ratio 0.8 --math-limit 100 \
        --out scripts/reasoning_aware_compress/results/blockD/bi_whitened_r0.8.json
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
from loguru import logger

import compress_common as cc
from compress_common import REPO_ROOT  # noqa: F401  (re-export for clarity)

from compress.calibration import (  # noqa: E402
    collect_mixed_statistics,
    collect_both_covariances_from_loader,
    collect_both_covariances_from_loader_opd,
)
from compress.compress_model import MethodSpec  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402

MIXED_SPEC = MethodSpec(attn="svd_llm_v2", mlp="nystrom")
ALL_CELLS = ("D0", "D1", "D2", "D3")


# --------------------------------------------------------------------------- #
# attention-only key filter (the D objective varies on attention only)
# --------------------------------------------------------------------------- #
def _is_attn(name: str) -> bool:
    return ".self_attn." in name


def _attn_only(d):
    return {k: v for k, v in d.items() if _is_attn(k)}


def compress_attn(model, fwd_cov, bwd_cov, *, ratio, objective, device, protect):
    """Compress attention linears with the given objective, last layers dropped.

    fwd_cov / bwd_cov are full (attn+mlp) covariance dicts; we slice to attn,
    drop protected layers, then call svd_llm_v2_compress_model. A layer absent
    from the dict is left dense by the core (svd_llm_v2.py:256-258)."""
    fwd = cc.drop_protected_stats(_attn_only(fwd_cov), protect)
    bwd = cc.drop_protected_stats(_attn_only(bwd_cov), protect) if bwd_cov else None
    logger.disable("compress")
    try:
        svd_llm_v2_compress_model(
            model, fwd, compression_ratio=ratio,
            skip_layers=("lm_head",), device=device,
            objective=objective, backward_covariances=bwd,
        )
    finally:
        logger.enable("compress")
    return model


def compress_mlp_nystrom(model, mlp_stats, *, ratio, device, protect):
    """Standing Nystrom (forward) MLP compression, last layers dropped."""
    stats = cc.drop_protected_stats(mlp_stats, protect)
    logger.disable("compress")
    try:
        nystrom_compress_model(
            model, stats, sparsity=1.0 - ratio,
            skip_layers=("lm_head",), device=device,
        )
    finally:
        logger.enable("compress")
    return model


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    cc.add_cli_common(ap)
    ap.add_argument("--cells", nargs="+", default=list(ALL_CELLS),
                    choices=list(ALL_CELLS))
    ap.add_argument("--teacher", default=None,
                    help="DISTINCT teacher checkpoint for D3 OPD covariance "
                         "(required for a non-degenerate D3; default same as --model is rejected)")
    ap.add_argument("--allow-degenerate-opd", action="store_true",
                    help="run D3 even when teacher==student (KL≡0 null cell; documented sanity only)")
    ap.add_argument("--opd-top-k", type=int, default=16)
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    device = args.device
    tokenizer = cc.load_tokenizer(args.model)

    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = cc.load_model(args.model, dtype, "cpu")
    base_cpu.eval()
    protect = cc.protected_layer_set(base_cpu, args.skip_last_layers)
    n_layers = cc.num_hidden_layers(base_cpu)
    logger.info(f"skip_last_layers={args.skip_last_layers}: protecting decoder "
                f"layers {sorted(protect)} of {n_layers} (left dense)")

    results = {"block": "D", "model": args.model, "ratio": args.ratio,
               "skip_last_layers": args.skip_last_layers,
               "protected_layers": sorted(protect),
               "config": vars(args), "cells": []}

    # ---- shared MLP Nystrom statistics (forward C_sigma), collected once ---- #
    logger.info("Collecting MLP Nystrom statistics (forward C_sigma) once...")
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

    # ---- attention covariances, collected per loss-type as needed ---- #
    # forward (input) cov is the same for every cell; backward cov differs.
    fwd_ce = bwd_ce = None       # next-token CE both-pass (D0/D1/D2 use fwd_ce; D1/D2 use bwd_ce)
    fwd_opd = bwd_opd = None      # OPD/teacher both-pass (D3)
    teacher = None

    def _ensure_ce():
        nonlocal fwd_ce, bwd_ce
        if fwd_ce is not None:
            return
        logger.info("Collecting fwd+bwd covariances (next-token CE)...")
        m = copy.deepcopy(base_cpu).to(device)
        cl = cc.build_calib_loader(
            args.calib, tokenizer, num_seqs=args.calib_num_seqs,
            max_length=args.calib_max_length, batch_size=args.calib_batch_size,
            seed=args.calib_seed)
        fwd_ce, bwd_ce = collect_both_covariances_from_loader(
            m, cl, device=device, skip_layers=("lm_head",), loss_fn=None)
        del m, cl
        torch.cuda.empty_cache()

    def _ensure_opd():
        nonlocal fwd_opd, bwd_opd, teacher
        if fwd_opd is not None:
            return
        t_name = args.teacher or args.model
        # CRITICAL guard (GPT-5.4 review): if teacher == student, the OPD KL is
        # identically 0, the backward covariance is zero, and D3 silently
        # collapses to the forward-only baseline (or NaN whitening). A genuine
        # OPD objective needs a DISTINCT teacher checkpoint.
        if not args.allow_degenerate_opd and t_name == args.model:
            raise SystemExit(
                "[D3] teacher == student (--teacher not set) → OPD loss is "
                "degenerate (KL≡0, zero backward cov). Pass a DISTINCT --teacher "
                "checkpoint, or --allow-degenerate-opd to run it anyway as a "
                "documented null/sanity cell.")
        logger.info(f"Collecting fwd+bwd covariances (OPD/teacher={t_name})...")
        student = copy.deepcopy(base_cpu).to(device)
        teacher = cc.load_model(t_name, dtype, device)
        cl = cc.build_calib_loader(
            args.calib, tokenizer, num_seqs=args.calib_num_seqs,
            max_length=args.calib_max_length, batch_size=args.calib_batch_size,
            seed=args.calib_seed)
        fwd_opd, bwd_opd = collect_both_covariances_from_loader_opd(
            student, teacher, cl, device=device, skip_layers=("lm_head",),
            top_k=args.opd_top_k)
        del student, cl, teacher
        teacher = None
        torch.cuda.empty_cache()

    def _build_cell(cell):
        """Return a fresh compressed model for the given D cell."""
        model = copy.deepcopy(base_cpu).to(device)
        # MLP Nystrom (forward) on every cell — clone stats (compress pops them).
        compress_mlp_nystrom(
            model, {k: v.clone() for k, v in mlp_stats_full.items()},
            ratio=args.ratio, device=device, protect=protect)
        if cell == "D0":
            _ensure_ce()
            compress_attn(model, fwd_ce, None, ratio=args.ratio,
                          objective="forward", device=device, protect=protect)
        elif cell == "D1":
            _ensure_ce()
            # backward-only: svd_compress_layer_backward whitens by C_dy; pass
            # the bwd cov as the "covariances" arg with objective="backward".
            compress_attn(model, bwd_ce, None, ratio=args.ratio,
                          objective="backward", device=device, protect=protect)
        elif cell == "D2":
            _ensure_ce()
            compress_attn(model, fwd_ce, bwd_ce, ratio=args.ratio,
                          objective="combined", device=device, protect=protect)
        elif cell == "D3":
            _ensure_opd()
            compress_attn(model, fwd_opd, bwd_opd, ratio=args.ratio,
                          objective="combined", device=device, protect=protect)
        else:
            raise ValueError(cell)
        return model.to(device)

    for cell in args.cells:
        logger.info(f"=== Block D cell {cell} @ retain {args.ratio} ===")
        t0 = time.time()
        model = _build_cell(cell)
        metrics = cc.eval_cell(
            model, tokenizer, device=device, math_limit=args.math_limit,
            math_max_new_tokens=args.math_max_new_tokens,
            math_batch_size=args.math_batch_size, ppl_seqlen=args.ppl_seqlen,
            skip_math=args.skip_math)
        rec = {"cell": cell, "ratio": args.ratio,
               "elapsed_s": time.time() - t0, **metrics}
        results["cells"].append(rec)
        cc.save_json(results, args.out)
        macc = "-" if rec["math500_acc"] is None else f"{rec['math500_acc']*100:.2f}%"
        logger.info(f"[D {cell}] nz={rec['params_nonzero_B']:.3f}B "
                    f"c4_ppl={rec['c4_ppl']:.4f} math={macc} "
                    f"({rec['elapsed_s']:.0f}s)")
        del model
        torch.cuda.empty_cache()

    print("\n========== Block D SUMMARY ==========")
    print(f"{'cell':6s} {'nz params':>10s} {'C4 PPL':>10s} {'MATH-500':>10s}")
    for r in results["cells"]:
        macc = "-" if r["math500_acc"] is None else f"{r['math500_acc']*100:.2f}%"
        print(f"{r['cell']:6s} {r['params_nonzero_B']:>9.3f}B "
              f"{r['c4_ppl']:>10.4f} {macc:>10s}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
