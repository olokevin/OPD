"""Ratio sweep (forward-only input whitening) + reasoning-trace diff per ratio.

For each retain ratio in the sweep, compress Qwen3-4B with the standing
forward-only path (SVD-LLM-V2 input-whitening on attention + Nystrom MLP, last
decoder layer left dense), then:
  1. eval MATH-500/100 + C4 PPL (the sweep accuracy/PPL curve), and
  2. regenerate the 5 frozen dense-correct probe traces on the compressed model
     and diff them against the dense reference (Block T) — so we can read WHERE
     the reasoning trace breaks as the ratio drops.

This is the forward-only ("D0/A0-style", no backward, no sparse residual) sweep
the user asked for: the plain structured-compression cliff. Trace diffs at every
ratio localize the first-divergence point and the failure mode as accuracy falls.

Output:
  - results/sweep/fwd_only_sweep.json : per-ratio {ratio, nz, c4_ppl, math, trace_summary}
  - results/sweep/traces_r{ratio}.json : per-ratio full 5-probe dense-vs-compressed traces

Usage (GPU 5):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/opd/math/compressed_opd/ratio_sweep_trace.py \
        --ratios 0.8 0.7 0.6 0.5 0.4 0.36 --math-limit 100 \
        --probe-set scripts/opd/math/compressed_opd/results/blockT/trace_probe_set.json \
        --out-dir scripts/opd/math/compressed_opd/results/sweep
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from loguru import logger

import compress_common as cc

from compress.calibration import collect_mixed_statistics, collect_covariances_from_loader
from compress.compress_model import MethodSpec
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model
from compress.structured.nystrom import nystrom_compress_model

import trace_diff as td


def compress_fwd_only(model, attn_cov, mlp_stats, *, ratio, device, protect):
    """SVD-V2 (forward input whitening) attn + Nystrom MLP, protected layers dense."""
    attn = cc.drop_protected_stats({k: v for k, v in attn_cov.items() if ".self_attn." in k}, protect)
    mlp = cc.drop_protected_stats(mlp_stats, protect)
    logger.disable("compress")
    try:
        nystrom_compress_model(model, {k: v.clone() for k, v in mlp.items()},
                               sparsity=1.0 - ratio, skip_layers=("lm_head",), device=device)
        svd_llm_v2_compress_model(model, {k: v.clone() for k, v in attn.items()},
                                  compression_ratio=ratio, skip_layers=("lm_head",),
                                  device=device, objective="forward")
    finally:
        logger.enable("compress")
    return model


def summarize_traces(traces):
    """Compact per-ratio read: how many of the 5 probes still solve correct, and
    the first-divergence character index vs the dense trace (proxy for 'where the
    trace breaks')."""
    out = []
    n_correct = 0
    for t in traces:
        dense, comp = t["dense_text"], t["comp_text"]
        # first char index where compressed departs from dense
        div = next((i for i, (a, b) in enumerate(zip(dense, comp)) if a != b), min(len(dense), len(comp)))
        n_correct += int(t["comp_correct"])
        out.append({
            "problem_id": t["problem_id"],
            "comp_correct": t["comp_correct"],
            "first_divergence_char": div,
            "dense_len": len(dense),
            "comp_len": len(comp),
            "div_frac": round(div / max(1, len(dense)), 3),
        })
    return {"n_comp_correct": n_correct, "n_probes": len(traces), "per_probe": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cc.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(cc.DTYPE_MAP))
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.8, 0.7, 0.6, 0.5, 0.4, 0.36])
    ap.add_argument("--skip-last-layers", type=int, default=1)
    ap.add_argument("--calib", default="openthought3", choices=["openthought3", "c4"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=100)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    device = args.device
    tokenizer = cc.load_tokenizer(args.model)
    probe_set = td.load_probe_set(args.probe_set)
    logger.info(f"Loaded {len(probe_set)} trace probes from {args.probe_set}")

    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = cc.load_model(args.model, dtype, "cpu")
    base_cpu.eval()
    protect = cc.protected_layer_set(base_cpu, args.skip_last_layers)
    logger.info(f"Protecting decoder layers {sorted(protect)} (dense)")

    # ---- statistics collected ONCE on the dense model (forward-only) -------- #
    logger.info("Collecting forward statistics once (attn input cov + MLP C_sigma)...")
    m0 = copy.deepcopy(base_cpu).to(device)
    calib = cc.build_calib_loader(args.calib, tokenizer, num_seqs=args.calib_num_seqs,
                                  max_length=args.calib_max_length,
                                  batch_size=args.calib_batch_size, seed=args.calib_seed)
    mlp_stats = collect_mixed_statistics(m0, calib, MethodSpec(attn=None, mlp="nystrom"),
                                         device=device, skip_layers=("lm_head",))
    del calib
    m1 = m0  # reuse
    calib = cc.build_calib_loader(args.calib, tokenizer, num_seqs=args.calib_num_seqs,
                                  max_length=args.calib_max_length,
                                  batch_size=args.calib_batch_size, seed=args.calib_seed)
    attn_cov = collect_covariances_from_loader(m1, calib, device=device, skip_layers=("lm_head",))
    del calib, m0
    torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    results = {"model": args.model, "ratios": args.ratios,
               "skip_last_layers": args.skip_last_layers,
               "protected_layers": sorted(protect), "config": vars(args), "sweep": []}

    for ratio in args.ratios:
        logger.info(f"=== ratio {ratio} (forward-only) ===")
        t0 = time.time()
        model = copy.deepcopy(base_cpu).to(device)
        compress_fwd_only(model, attn_cov, mlp_stats, ratio=ratio, device=device, protect=protect)
        metrics = cc.eval_cell(model, tokenizer, device=device, math_limit=args.math_limit,
                               math_max_new_tokens=args.math_max_new_tokens,
                               math_batch_size=args.math_batch_size, ppl_seqlen=args.ppl_seqlen)
        traces = td.generate_traces(model, tokenizer, probe_set, device=device,
                                    method="fwd_only", ratio=ratio,
                                    max_new_tokens=args.math_max_new_tokens)
        cc.save_json({"ratio": ratio, "traces": traces}, str(out_dir / f"traces_r{ratio}.json"))
        rec = {"ratio": ratio, "elapsed_s": time.time() - t0,
               "trace_summary": summarize_traces(traces), **metrics}
        results["sweep"].append(rec)
        cc.save_json(results, str(out_dir / "fwd_only_sweep.json"))
        macc = "-" if rec["math500_acc"] is None else f"{rec['math500_acc']*100:.2f}%"
        ts = rec["trace_summary"]
        logger.info(f"[sweep r={ratio}] nz={rec['params_nonzero_B']:.3f}B "
                    f"c4_ppl={rec['c4_ppl']:.2f} math={macc} "
                    f"probes_correct={ts['n_comp_correct']}/{ts['n_probes']}")
        del model
        torch.cuda.empty_cache()

    print("\n========== FORWARD-ONLY RATIO SWEEP ==========")
    print(f"{'ratio':>6s} {'nz':>9s} {'C4 PPL':>10s} {'MATH-500':>9s} {'probes✓':>8s} {'med div_frac':>12s}")
    for r in results["sweep"]:
        macc = "-" if r["math500_acc"] is None else f"{r['math500_acc']*100:.1f}%"
        ts = r["trace_summary"]
        divs = sorted(p["div_frac"] for p in ts["per_probe"])
        med = divs[len(divs) // 2] if divs else 0.0
        print(f"{r['ratio']:>6.2f} {r['params_nonzero_B']:>8.3f}B {r['c4_ppl']:>10.2f} "
              f"{macc:>9s} {ts['n_comp_correct']:>4d}/{ts['n_probes']:<3d} {med:>12.3f}")
    print(f"\nWrote {out_dir/'fwd_only_sweep.json'} + per-ratio traces")


if __name__ == "__main__":
    main()
