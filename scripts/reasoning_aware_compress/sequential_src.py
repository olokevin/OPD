"""Block B — Sequential Re-Linearized Structured Compression / SRC (mechanism M3).

The standing pipeline collects ALL covariances in one DENSE pass, so no layer
sees its compressed upstream and errors compound across depth (M3). SRC compresses
layers in depth order, re-collecting each layer's input covariance through the
ALREADY-COMPRESSED prefix so it reconstructs against the distribution it will
actually receive. One-shot, no SGD.

Cells (attn SVD + MLP Nystrom; both re-linearized for B1+):
  B0  dense-pass layer-independent (= D0)                   — current pipeline at this ratio
  B1  SRC, forward cov re-collected on compressed prefix    — SAES-SVD-style accumulation fix
  B2  SRC, forward + OPD-backward cov on compressed prefix  — Idea-B delta (B's fix + D's objective)
  (B3 = SRC + per-layer sparse residual from A — combined synergy; --cells B3, needs Block A core)

Operating point: retain 0.8 first, last decoder layer left dense. Eval MATH/100 + C4 PPL.
Success: B1 > B0 (accumulation matters) and B2 ≥ B1 (OPD objective adds on top).

Usage (GPU 5):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/reasoning_aware_compress/sequential_src.py \
        --cells B0 B1 B2 --ratio 0.8 --math-limit 100 \
        --out scripts/reasoning_aware_compress/results/blockB/src_r0.8.json
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
from loguru import logger

import compress_common as cc

from compress.calibration import collect_mixed_statistics  # noqa: E402
from compress.calibration_opd_loss import opd_calibration_loss  # noqa: E402
from compress.compress_model import MethodSpec  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.sequential.relinearized import sequential_relinearized_compress  # noqa: E402

ALL_CELLS = ("B0", "B1", "B2", "B3")


def _build_B0(base_cpu, tokenizer, args, protect, device):
    """Dense-pass layer-independent baseline (= D0): one dense covariance pass,
    standing svd_v2 attn + nystrom mlp, last layer dropped."""
    model = copy.deepcopy(base_cpu).to(device)
    calib = cc.build_calib_loader(
        args.calib, tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
        seed=args.calib_seed)
    stats = collect_mixed_statistics(
        model, calib, MethodSpec(attn="svd_llm_v2", mlp="nystrom"),
        device=device, skip_layers=("lm_head",))
    stats = cc.drop_protected_stats(stats, protect)
    attn = {k: v for k, v in stats.items() if ".self_attn." in k}
    down = {k: v for k, v in stats.items() if k.endswith(".mlp.down_proj")}
    logger.disable("compress")
    try:
        nystrom_compress_model(model, {k: v.clone() for k, v in down.items()},
                               sparsity=1.0 - args.ratio, skip_layers=("lm_head",),
                               device=device)
        svd_llm_v2_compress_model(model, {k: v.clone() for k, v in attn.items()},
                                  compression_ratio=args.ratio,
                                  skip_layers=("lm_head",), device=device,
                                  objective="forward")
    finally:
        logger.enable("compress")
    del calib
    torch.cuda.empty_cache()
    return model


def _build_SRC(base_cpu, tokenizer, args, protect, device, *, objective, teacher=None):
    model = copy.deepcopy(base_cpu).to(device)

    def _loader_factory():
        return cc.build_calib_loader(
            args.calib, tokenizer, num_seqs=args.calib_num_seqs,
            max_length=args.calib_max_length, batch_size=args.calib_batch_size,
            seed=args.calib_seed)

    opd_loss_fn = None
    if objective == "combined" and teacher is not None:
        def opd_loss_fn(logits, batch):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False)
            response_mask = attention_mask.bool() if attention_mask is not None else None
            return opd_calibration_loss(
                logits, t_out.logits.detach(), top_k=args.opd_top_k,
                response_mask=response_mask)

    sequential_relinearized_compress(
        model, _loader_factory, ratio=args.ratio, protect=protect,
        device=device, objective=objective, opd_loss_fn=opd_loss_fn)
    return model


def main():
    ap = argparse.ArgumentParser()
    cc.add_cli_common(ap)
    ap.add_argument("--cells", nargs="+", default=["B0", "B1", "B2"],
                    choices=list(ALL_CELLS))
    ap.add_argument("--teacher", default=None,
                    help="DISTINCT teacher checkpoint for B2 OPD backward cov "
                         "(default same as --model is rejected as degenerate)")
    ap.add_argument("--allow-degenerate-opd", action="store_true",
                    help="run B2 even when teacher==student (KL≡0 null cell)")
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
    logger.info(f"skip_last_layers={args.skip_last_layers}: protecting layers "
                f"{sorted(protect)} of {n_layers}")

    results = {"block": "B", "model": args.model, "ratio": args.ratio,
               "skip_last_layers": args.skip_last_layers,
               "protected_layers": sorted(protect),
               "config": vars(args), "cells": []}

    teacher = None
    if "B2" in args.cells:
        t_name = args.teacher or args.model
        # CRITICAL guard (GPT-5.4 review): teacher == student → OPD KL≡0, zero
        # backward cov, B2 silently collapses to B1. Require a distinct teacher.
        if not args.allow_degenerate_opd and t_name == args.model:
            raise SystemExit(
                "[B2] teacher == student (--teacher not set) → OPD loss degenerate "
                "(KL≡0). Pass a DISTINCT --teacher, or --allow-degenerate-opd to "
                "run it as a documented null cell.")
        logger.info(f"Loading teacher {t_name} for B2 OPD backward cov...")
        teacher = cc.load_model(t_name, dtype, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    def _run(cell, model):
        metrics = cc.eval_cell(
            model, tokenizer, device=device, math_limit=args.math_limit,
            math_max_new_tokens=args.math_max_new_tokens,
            math_batch_size=args.math_batch_size, ppl_seqlen=args.ppl_seqlen,
            skip_math=args.skip_math)
        rec = {"cell": cell, "ratio": args.ratio, **metrics}
        results["cells"].append(rec)
        cc.save_json(results, args.out)
        macc = "-" if rec["math500_acc"] is None else f"{rec['math500_acc']*100:.2f}%"
        logger.info(f"[B {cell}] nz={rec['params_nonzero_B']:.3f}B "
                    f"c4_ppl={rec['c4_ppl']:.4f} math={macc}")

    for cell in args.cells:
        logger.info(f"=== Block B cell {cell} @ retain {args.ratio} ===")
        t0 = time.time()
        if cell == "B0":
            model = _build_B0(base_cpu, tokenizer, args, protect, device)
        elif cell == "B1":
            model = _build_SRC(base_cpu, tokenizer, args, protect, device,
                               objective="forward")
        elif cell == "B2":
            model = _build_SRC(base_cpu, tokenizer, args, protect, device,
                               objective="combined", teacher=teacher)
        elif cell == "B3":
            raise NotImplementedError(
                "B3 (SRC + per-layer sparse residual) is a synergy cell; run after "
                "A and B independently show signal (plan run-order).")
        else:
            raise ValueError(cell)
        _run(cell, model)
        del model
        torch.cuda.empty_cache()
        logger.info(f"  ({time.time() - t0:.0f}s)")

    print("\n========== Block B SUMMARY ==========")
    print(f"{'cell':6s} {'nz params':>10s} {'C4 PPL':>10s} {'MATH-500':>10s}")
    for r in results["cells"]:
        macc = "-" if r["math500_acc"] is None else f"{r['math500_acc']*100:.2f}%"
        print(f"{r['cell']:6s} {r['params_nonzero_B']:>9.3f}B "
              f"{r['c4_ppl']:>10.4f} {macc:>10s}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
