"""Full-sequence calibration: reweighting × length-filter, then ratio sweep.

Forward-only compression (SVD-V2 input-whitening attn + Nystrom MLP, last layer
dense) — same method as the 2048-window sweep — but the calibration covariances
come from FULL (un-windowed) OpenThought3 conversations, with two knobs:

  reweight  ∈ {token, sequence}   how cross-sequence averaging is done:
                token    = every token weighted equally (long traces dominate)
                sequence = every conversation weighted equally
  length    ∈ {full, lt2048}      which conversations enter calibration:
                full   = whole conversation, no length cap (true full length)
                lt2048 = only conversations shorter than 2048 tokens

Both are mask-aware (pad positions excluded). Calibration set = 128 sequences.

Stage 1 (--stage tune): the 4 settings at retain 0.7 → pick best by STRICT MATH.
Stage 2 (--stage sweep): the chosen setting at retain 0.6/0.5/0.4.

Per cell, one generation pass reports (compress_common.eval_math_capture):
  strict_acc · relaxed_acc (contains-correct) · mean_gen_tokens · mean_tok_to_correct
plus C4 PPL. Captured per-response records saved for audit.

Usage (Stage 1, split across two GPUs):
  CUDA_VISIBLE_DEVICES=2 ... fullseq_calib_sweep.py --stage tune \
    --settings token:full sequence:full --ratio 0.7 --out results/fullseq/tune_g2.json
  CUDA_VISIBLE_DEVICES=3 ... fullseq_calib_sweep.py --stage tune \
    --settings token:lt2048 sequence:lt2048 --ratio 0.7 --out results/fullseq/tune_g3.json
Stage 2:
  CUDA_VISIBLE_DEVICES=2 ... fullseq_calib_sweep.py --stage sweep \
    --setting <best> --ratios 0.6 0.5 0.4 --out results/fullseq/sweep_best.json
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from loguru import logger

import compress_common as cc

from compress.loaders import build_fullseq_calib_loader
from compress.calibration import collect_covariances_reweighted
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model
from compress.structured.nystrom import nystrom_compress_model

from layer_sensitivity import _openthought3_texts  # raw OpenThought3 chat→text


def build_fullseq_loader(tokenizer, *, length_filter, num_seqs):
    path = cc.REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
    # over-sample raw convs; the loader filters/keeps the first num_seqs that pass.
    texts = _openthought3_texts(tokenizer, path, n=num_seqs * 20)
    # batch size: lt2048 is short → batch 4; full is up to ~10k tokens → batch 2
    # (per-row masking makes both reweight modes correct at any batch size).
    bs = 4 if length_filter == "lt2048" else 2
    return build_fullseq_calib_loader(
        tokenizer, texts, num_seqs=num_seqs, length_filter=length_filter,
        batch_size=bs)


def collect_setting_cov(base_cpu, tokenizer, *, reweight, length_filter, num_seqs,
                        device):
    """Forward covariances for ALL non-skip linears (attn input cov + down_proj
    C_sigma in one pass) under the chosen reweighting + length filter."""
    m = copy.deepcopy(base_cpu).to(device)
    loader = build_fullseq_loader(tokenizer, length_filter=length_filter, num_seqs=num_seqs)
    cov = collect_covariances_reweighted(
        m, loader, device=device, skip_layers=("lm_head",), reweight=reweight)
    del m, loader
    torch.cuda.empty_cache()
    return cov


def compress_fwd_only(model, full_cov, *, ratio, device, protect):
    attn = cc.drop_protected_stats({k: v for k, v in full_cov.items() if ".self_attn." in k}, protect)
    down = cc.drop_protected_stats({k: v for k, v in full_cov.items() if k.endswith(".mlp.down_proj")}, protect)
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


def run_cell(base_cpu, tokenizer, *, reweight, length_filter, ratio, protect,
             device, args, out_dir, tag):
    logger.info(f"=== cell {tag}: reweight={reweight} length={length_filter} ratio={ratio} ===")
    t0 = time.time()
    cov = collect_setting_cov(base_cpu, tokenizer, reweight=reweight,
                              length_filter=length_filter, num_seqs=args.calib_num_seqs,
                              device=device)
    model = copy.deepcopy(base_cpu).to(device)
    compress_fwd_only(model, cov, ratio=ratio, device=device, protect=protect)
    del cov
    torch.cuda.empty_cache()
    ppl = cc.eval_c4_ppl(model, tokenizer, seqlen=args.ppl_seqlen, device=device)
    m = cc.eval_math_capture(
        model, tokenizer, device=device, limit=args.math_limit,
        max_new_tokens=args.math_max_new_tokens, batch_size=args.math_batch_size,
        save_path=str(Path(out_dir) / f"responses_{tag}.json"))
    rec = {"tag": tag, "reweight": reweight, "length_filter": length_filter,
           "ratio": ratio, "c4_ppl": ppl, "elapsed_s": time.time() - t0, **m}
    del model
    torch.cuda.empty_cache()
    logger.info(f"[{tag}] strict={rec['strict_acc']*100:.1f}% relaxed={rec['relaxed_acc']*100:.1f}% "
                f"gen_len={rec['mean_gen_tokens']:.0f} tok2correct="
                f"{rec['mean_tok_to_correct'] if rec['mean_tok_to_correct'] is None else round(rec['mean_tok_to_correct'])} "
                f"ppl={rec['c4_ppl']:.1f}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cc.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(cc.DTYPE_MAP))
    ap.add_argument("--stage", required=True, choices=["tune", "sweep"])
    ap.add_argument("--settings", nargs="+", default=["token:full", "sequence:full",
                                                      "token:lt2048", "sequence:lt2048"],
                    help="tune stage: list of reweight:length_filter")
    ap.add_argument("--setting", default=None, help="sweep stage: single reweight:length_filter")
    ap.add_argument("--ratio", type=float, default=0.7, help="tune stage ratio")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.6, 0.5, 0.4],
                    help="sweep stage ratios")
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
    results = {"stage": args.stage, "model": args.model, "config": vars(args),
               "protected_layers": sorted(protect), "cells": []}

    if args.stage == "tune":
        for s in args.settings:
            rw, lf = s.split(":")
            rec = run_cell(base_cpu, tokenizer, reweight=rw, length_filter=lf,
                           ratio=args.ratio, protect=protect, device=device,
                           args=args, out_dir=out_dir, tag=f"{rw}_{lf}_r{args.ratio}")
            results["cells"].append(rec)
            cc.save_json(results, args.out)
    else:  # sweep
        assert args.setting, "--setting required for sweep stage"
        rw, lf = args.setting.split(":")
        for ratio in args.ratios:
            rec = run_cell(base_cpu, tokenizer, reweight=rw, length_filter=lf,
                           ratio=ratio, protect=protect, device=device,
                           args=args, out_dir=out_dir, tag=f"{rw}_{lf}_r{ratio}")
            results["cells"].append(rec)
            cc.save_json(results, args.out)

    print("\n========== FULL-SEQ CALIB ==========")
    print(f"{'tag':28s} {'strict':>7s} {'relaxed':>8s} {'gen_len':>8s} {'tok2corr':>9s} {'C4 PPL':>9s}")
    for r in results["cells"]:
        ttc = "-" if r["mean_tok_to_correct"] is None else f"{r['mean_tok_to_correct']:.0f}"
        print(f"{r['tag']:28s} {r['strict_acc']*100:>6.1f}% {r['relaxed_acc']*100:>7.1f}% "
              f"{r['mean_gen_tokens']:>8.0f} {ttc:>9s} {r['c4_ppl']:>9.1f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
