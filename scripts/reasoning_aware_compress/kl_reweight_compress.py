"""KL-importance reweighted calibration: compress, measure damage, recompress.

docs/wiki/reweighted_compress.md. Forward-only SVD-V2 (attn) + Nystrom (MLP),
last layer dense, OpenThought3 full-seq calibration — the SAME method/operating
point as fullseq_calib_sweep.py — but the within-sequence token weights come from
the forward KL between the uncompressed teacher and a uniformly-compressed
student (the "probe").

Two passes per cell:
  Pass 0 (probe)   : uniform sequence-reweight compress at `ratio` -> student S.
  Damage           : teacher(uncompressed) vs S, per-token KL -> token weights.
  Pass 1 (final)   : recompress ORIGINAL weights with the KL-weighted covariance.
Eval: eval_math_capture (strict/relaxed/gen_len/tok2correct) + C4 PPL.

Cells:
  beta=0            : ablation anchor — must reproduce fullseq sequence:full @ratio.
  beta>0            : KL tilt (the method).

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
    scripts/reasoning_aware_compress/kl_reweight_compress.py \
    --ratio 0.7 --cells B:0 K-mid:1:5 K-sharp:2:8 --math-limit 100 \
    --out scripts/reasoning_aware_compress/results/reweight/kl_r0.7.json
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from loguru import logger

import compress_common as cc

from compress.calibration import (
    collect_covariances_reweighted,
    collect_covariances_weighted,
)
from compress.kl_reweight import compute_kl_token_weights
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model
from compress.structured.nystrom import nystrom_compress_model

from layer_sensitivity import _openthought3_texts


def build_fullseq_loader(tokenizer, *, length_filter, num_seqs):
    """Same loader fullseq_calib_sweep uses — full sequences, bs=1 (the weights
    list is indexed by this exact iteration order, shuffle=False)."""
    from compress.loaders import build_fullseq_calib_loader
    path = cc.REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
    texts = _openthought3_texts(tokenizer, path, n=num_seqs * 20)
    # bs=1 so each batch is one full sequence (weights align trivially; per-row
    # masking keeps every reweight mode correct regardless).
    return build_fullseq_calib_loader(
        tokenizer, texts, num_seqs=num_seqs, length_filter=length_filter, batch_size=1)


def compress_fwd_only(model, full_cov, *, ratio, device, protect):
    """Forward-only SVD-V2 attn + Nystrom MLP from a forward covariance dict."""
    attn = cc.drop_protected_stats(
        {k: v for k, v in full_cov.items() if ".self_attn." in k}, protect)
    down = cc.drop_protected_stats(
        {k: v for k, v in full_cov.items() if k.endswith(".mlp.down_proj")}, protect)
    logger.disable("compress")
    try:
        nystrom_compress_model(model, {k: v.clone() for k, v in down.items()},
                               sparsity=1.0 - ratio, skip_layers=("lm_head",), device=device)
        svd_llm_v2_compress_model(model, {k: v.clone() for k, v in attn.items()},
                                  compression_ratio=ratio, skip_layers=("lm_head",),
                                  device=device, objective="forward")
    finally:
        logger.enable("compress")
    return model


def run_cell(base_cpu, tokenizer, *, beta, w_max, ratio, length_filter, protect,
             device, args, out_dir, tag):
    logger.info(f"=== cell {tag}: beta={beta} w_max={w_max} ratio={ratio} ===")
    t0 = time.time()

    # ---- uniform covariance (used for the probe AND as the beta=0 final) ----
    uni_loader = build_fullseq_loader(tokenizer, length_filter=length_filter,
                                      num_seqs=args.calib_num_seqs)
    m_probe = copy.deepcopy(base_cpu).to(device)
    uni_cov = collect_covariances_reweighted(
        m_probe, uni_loader, device=device, skip_layers=("lm_head",), reweight="sequence")
    del m_probe
    torch.cuda.empty_cache()

    if beta == 0.0:
        # ablation anchor: identical to fullseq sequence:full — no probe/KL needed.
        final_cov = uni_cov
    else:
        # ---- Pass 0: probe student from the uniform covariance ----
        student = compress_fwd_only(copy.deepcopy(base_cpu).to(device),
                                    {k: v.clone() for k, v in uni_cov.items()},
                                    ratio=ratio, device=device, protect=protect)
        # ---- damage: teacher (uncompressed) vs student (compressed) KL ----
        teacher = copy.deepcopy(base_cpu).to(device)
        kl_loader = build_fullseq_loader(tokenizer, length_filter=length_filter,
                                         num_seqs=args.calib_num_seqs)
        weights = compute_kl_token_weights(
            teacher, student, kl_loader, device=device, beta=beta, w_max=w_max)
        del teacher, student
        torch.cuda.empty_cache()
        # ---- Pass 1: reweighted covariance on the ORIGINAL weights ----
        w_loader = build_fullseq_loader(tokenizer, length_filter=length_filter,
                                        num_seqs=args.calib_num_seqs)
        m_w = copy.deepcopy(base_cpu).to(device)
        final_cov = collect_covariances_weighted(
            m_w, w_loader, weights, device=device, skip_layers=("lm_head",),
            reweight="sequence")
        del m_w, uni_cov
        torch.cuda.empty_cache()

    # ---- final compressed model + eval ----
    model = compress_fwd_only(copy.deepcopy(base_cpu).to(device), final_cov,
                              ratio=ratio, device=device, protect=protect)
    del final_cov
    torch.cuda.empty_cache()
    ppl = cc.eval_c4_ppl(model, tokenizer, seqlen=args.ppl_seqlen, device=device)
    m = cc.eval_math_capture(
        model, tokenizer, device=device, limit=args.math_limit,
        max_new_tokens=args.math_max_new_tokens, batch_size=args.math_batch_size,
        save_path=str(Path(out_dir) / f"responses_{tag}.json"))
    rec = {"tag": tag, "beta": beta, "w_max": w_max, "ratio": ratio,
           "length_filter": length_filter, "c4_ppl": ppl,
           "elapsed_s": time.time() - t0, **m}
    del model
    torch.cuda.empty_cache()
    logger.info(f"[{tag}] strict={rec['strict_acc']*100:.1f}% relaxed={rec['relaxed_acc']*100:.1f}% "
                f"gen_len={rec['mean_gen_tokens']:.0f} ppl={rec['c4_ppl']:.1f}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cc.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(cc.DTYPE_MAP))
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--length-filter", default="full", choices=["full", "lt2048"])
    ap.add_argument("--cells", nargs="+", default=["B:0", "K-mid:1:5", "K-sharp:2:8"],
                    help="name:beta[:w_max] (w_max default 5)")
    ap.add_argument("--skip-last-layers", type=int, default=1)
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=100)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    device = args.device
    tokenizer = cc.load_tokenizer(args.model)
    base_cpu = cc.load_model(args.model, dtype, "cpu")
    base_cpu.eval()
    protect = cc.protected_layer_set(base_cpu, args.skip_last_layers)
    logger.info(f"Protecting decoder layers {sorted(protect)} (dense)")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "config": vars(args),
               "protected_layers": sorted(protect), "cells": []}

    for spec in args.cells:
        parts = spec.split(":")
        name = parts[0]
        beta = float(parts[1])
        w_max = float(parts[2]) if len(parts) > 2 else 5.0
        rec = run_cell(base_cpu, tokenizer, beta=beta, w_max=w_max, ratio=args.ratio,
                       length_filter=args.length_filter, protect=protect, device=device,
                       args=args, out_dir=out_dir, tag=f"{name}_b{beta}_r{args.ratio}")
        results["cells"].append(rec)
        cc.save_json(results, args.out)

    print("\n========== KL-REWEIGHTED CALIB ==========")
    print(f"{'tag':28s} {'beta':>5s} {'strict':>7s} {'relaxed':>8s} {'gen_len':>8s} {'C4 PPL':>9s}")
    for r in results["cells"]:
        print(f"{r['tag']:28s} {r['beta']:>5.1f} {r['strict_acc']*100:>6.1f}% "
              f"{r['relaxed_acc']*100:>7.1f}% {r['mean_gen_tokens']:>8.0f} {r['c4_ppl']:>9.1f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
