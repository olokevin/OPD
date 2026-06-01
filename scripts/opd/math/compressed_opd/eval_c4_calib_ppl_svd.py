"""Same as eval_c4_calib_ppl.py but with train_mode=svd_llm_v2 to compare
against BTT. If SVD-LLM gives sane PPL at the same ratio and BTT doesn't,
the BTT implementation is the problem."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compress.decomposition import DecompositionConfig, decompose_with_loader  # noqa
from compress.loaders import build_c4_calib_loader  # noqa
from compress.ppl_eval import evaluate_model_ppl  # noqa

TEACHER = "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mode", default="svd_llm_v2",
                    choices=["svd_llm_v2", "svd_llm_v2_combined", "btt_llm_v2"])
    ap.add_argument("--ratio", type=float, default=0.36)
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=4)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--ppl-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = torch.bfloat16

    print(f"Loading {TEACHER}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=dtype).to(args.device)
    orig_params = _count_params(model)
    print(f"  loaded: {orig_params / 1e9:.3f}B params", flush=True)

    print(f"Baseline C4 PPL...", flush=True)
    t0 = time.time()
    baseline = evaluate_model_ppl(model, tokenizer, seqlen=args.ppl_seqlen,
                                   seed=args.ppl_seed, datasets=("c4",), device=args.device)
    print(f"  uncompressed C4 PPL = {baseline['c4']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"Building C4 calib loader...", flush=True)
    calib_loader = build_c4_calib_loader(
        tokenizer,
        num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length,
        batch_size=args.calib_batch_size,
        seed=args.calib_seed,
    )

    cfg = DecompositionConfig(
        train_mode=args.train_mode,
        compression_ratio=args.ratio,
        skip_layers="lm_head",
        decomp_mode="output_one_block",  # only used by btt modes
        train_position="both",
        s_merged_to="split",
        factorize_by_head=True,
        calib_num_seqs=args.calib_num_seqs,
        calib_max_length=args.calib_max_length,
        calib_seed=args.calib_seed,
    )

    print(f"\nDecomposing with {args.train_mode} at ratio {args.ratio}...", flush=True)
    t0 = time.time()
    out = decompose_with_loader(model, cfg, calib_loader=calib_loader,
                                 device=args.device, return_trainability_stats=True)
    model = out[0] if isinstance(out, tuple) else out
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    new_params = _count_params(model)
    print(f"  compressed: {new_params / 1e9:.3f}B params (param ratio {new_params/orig_params:.3f})", flush=True)

    print(f"\nCompressed C4 PPL...", flush=True)
    t0 = time.time()
    ppl = evaluate_model_ppl(model, tokenizer, seqlen=args.ppl_seqlen,
                              seed=args.ppl_seed, datasets=("c4",), device=args.device)
    print(f"  compressed C4 PPL = {ppl['c4']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n========== SUMMARY ==========", flush=True)
    print(f"  train_mode           = {args.train_mode}", flush=True)
    print(f"  compression_ratio    = {args.ratio}", flush=True)
    print(f"  uncompressed C4 PPL  = {baseline['c4']:.4f}", flush=True)
    print(f"  compressed   C4 PPL  = {ppl['c4']:.4f}", flush=True)
    print(f"  param ratio          = {new_params/orig_params:.3f}", flush=True)


if __name__ == "__main__":
    main()
