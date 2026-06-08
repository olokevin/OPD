"""Tail-Rescue — causal test of the M1 (rank-deficiency) mechanism.

Block-0 falsified the steering-subspace thesis (steering directions are the
BEST-preserved part of the collapsed model). The leading live mechanism is now
M1: low-rank truncation discards off-subspace / tail singular mass the residual
stream needs. This experiment tests M1 *causally*:

  Compress the model at the 0.36 budget (attn via SVD-LLM-V2, mlp via Nystrom),
  then on the ATTENTION SVD modules add back the first `k` DISCARDED singular
  components (ranks [r : r+k] of the same whitened SVD). Sweep small k and eval
  MATH-500.

Prediction (M1):
  MATH recovers monotonically as a tiny tail `k` is reinjected -> the lost tail
  mass is the culprit. If MATH stays at 0 -> "missing tail" is not the story
  (look at cross-layer accumulation / a few under-ranked layers / MLP side).

Why attention-only tail: for SVD the "discarded singular components" are exactly
defined (ranks beyond r). Nystrom/MLP discards neurons (a different object), so
we hold MLP at 0.36 and vary only the attn tail — a clean one-shot M1 probe.
k is expressed PER MODULE (so total added params scale with #attn linears).

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    python scripts/reasoning_aware_compress/tail_rescue.py \
      --k-sweep 0 4 8 16 32 --math-limit 100 \
      --out results/block0/tail_rescue.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.calibration import collect_mixed_statistics  # noqa: E402
from compress.compress_model import MethodSpec  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.svd.svd_linear import SVDCompressedLinear  # noqa: E402
from compress.utils.whitening import compute_whitening  # noqa: E402

from layer_sensitivity import (  # noqa: E402
    build_openthought3_loader, attn_linear_names, mlp_down_names, eval_math500,
)

MIXED_SPEC = MethodSpec(attn="svd_llm_v2", mlp="nystrom")


def _budget_rank(d_out, d_in, ratio):
    """The rank svd_llm_v2_compress_model picks for this (d_out,d_in) at `ratio`."""
    r = int(ratio * d_out * d_in / (d_out + d_in))
    return max(1, min(r, min(d_out, d_in)))


def _svd_layer_with_tail(weight, ratio, k, cov, bias, device):
    """Whitened SVD keeping rank r=budget(ratio); +k = keep ranks [0 : r+k].

    k=0 reproduces svd_llm_v2 exactly (the 0.36 collapse). k>0 adds the first k
    discarded ('tail') singular components -> tests their causal contribution.
    """
    d_out, d_in = weight.shape
    r = _budget_rank(d_out, d_in, ratio)
    keep = min(r + k, min(d_out, d_in))
    W = weight.to(device=device, dtype=torch.float32)
    Phi, Phi_inv = compute_whitening(cov.to(device), device=device)
    U, S, Vh = torch.linalg.svd(W @ Phi, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :keep], S[:keep], Vh[:keep, :]
    sqrt_S = torch.sqrt(S_r)
    V_r = (Phi_inv.t() @ Vh_r.t() * sqrt_S.unsqueeze(0))   # (d_in, keep)
    U_r = (U_r * sqrt_S.unsqueeze(0)).t()                   # (keep, d_out)
    return SVDCompressedLinear(U_r.cpu().to(weight.dtype), V_r.cpu().to(weight.dtype),
                               bias=bias), keep, r


def _set_module(model, name, new_mod):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod.to(next(model.parameters()).device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--skip-last-layers", type=int, default=1,
                    help="leave the top-N decoder layers DENSE")
    ap.add_argument("--k-sweep", nargs="+", type=int, default=[0, 4, 8, 16, 32])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=100)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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

    n_layers = base_cpu.config.num_hidden_layers
    protect = set(range(n_layers - args.skip_last_layers, n_layers)) \
        if args.skip_last_layers > 0 else set()

    def _layer_idx(name):
        try:
            return int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return None

    attn_by_layer = attn_linear_names(base_cpu)
    attn_names = [n for li, names in attn_by_layer.items() if li not in protect
                  for n in names]
    down_by_layer = mlp_down_names(base_cpu)

    # ---- mixed stats once (attn input cov + down_proj C_sigma) ---- #
    model = copy.deepcopy(base_cpu).to(device)
    logger.info("Collecting OpenThought3 mixed statistics (one pass)...")
    calib_loader = build_openthought3_loader(
        tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
    )
    stats = collect_mixed_statistics(
        model, calib_loader, MIXED_SPEC, device=device, skip_layers=("lm_head",),
    )
    # leave the top-N decoder layers dense (drop their stats)
    stats = {n: s for n, s in stats.items() if _layer_idx(n) not in protect}
    logger.info(f"Skip-last-layers={args.skip_last_layers}: top {args.skip_last_layers} "
                f"of {n_layers} decoder layer(s) left dense")
    del calib_loader, model
    torch.cuda.empty_cache()

    results = {"model": args.model, "ratio": args.ratio, "k_sweep": args.k_sweep,
               "skip_last_layers": args.skip_last_layers,
               "config": vars(args), "cells": []}

    def _save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    for k in args.k_sweep:
        t0 = time.time()
        logger.info(f"=== tail k={k} (attn rank r -> r+{k}; mlp Nystrom @ {args.ratio}) ===")
        m = copy.deepcopy(base_cpu).to(device)

        # attn: SVD with +k tail per module
        added = 0
        ranks_seen = []
        for nm in attn_names:
            if nm not in stats:
                continue
            mod = m
            for p in nm.split("."):
                mod = getattr(mod, p)
            new_mod, keep, r = _svd_layer_with_tail(
                mod.weight.data, args.ratio, k, stats[nm],
                mod.bias.data if mod.bias is not None else None, device,
            )
            _set_module(m, nm, new_mod)
            added += (keep - r) * (new_mod.in_features + new_mod.out_features)
            ranks_seen.append((r, keep))

        # mlp: Nystrom at the same ratio (held fixed)
        down_subset = {n: stats[n].clone() for n in stats if n.endswith(".down_proj")}
        nystrom_compress_model(m, down_subset, sparsity=1.0 - args.ratio,
                               skip_layers=("lm_head",), device=device)

        m.to(device)
        nz = sum(p.numel() for p in m.parameters())
        acc = eval_math500(
            m, tokenizer, device=device, limit=args.math_limit,
            max_new_tokens=args.math_max_new_tokens, batch_size=args.math_batch_size,
        )
        rec = {"k": k, "math500_acc": acc, "total_params": int(nz),
               "attn_extra_params": int(added),
               "example_rank": ranks_seen[0] if ranks_seen else None,
               "elapsed_s": time.time() - t0}
        results["cells"].append(rec)
        logger.info(f"  k={k}: MATH-500={acc:.4f}  total_params={nz/1e9:.3f}B  "
                    f"(+{added/1e6:.1f}M attn tail)  [{time.time()-t0:.0f}s]")
        _save()
        del m
        torch.cuda.empty_cache()

    logger.info("==== TAIL-RESCUE SUMMARY (MATH-500 vs k) ====")
    for c in results["cells"]:
        logger.info(f"  k={c['k']:3d}: MATH={c['math500_acc']:.4f}  "
                    f"params={c['total_params']/1e9:.3f}B")
    accs = [c["math500_acc"] for c in results["cells"]]
    if len(accs) >= 2:
        monotone = all(accs[i] <= accs[i+1] + 1e-9 for i in range(len(accs)-1))
        logger.info(f"  monotonic recovery with k? {monotone}; "
                    f"delta(first->last) = {accs[-1]-accs[0]:+.4f} "
                    f"(M1 predicts monotone increase from ~0)")
    _save()
    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
