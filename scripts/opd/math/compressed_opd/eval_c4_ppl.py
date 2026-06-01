"""Report C4 perplexity for the original Qwen3-4B teacher vs the BTT-compressed
v2 / v2_combined variants, using the calibration caches saved by the
compressed_opd runs.

The compressed checkpoints live as a (topology JSON + safetensors) pair under
``$HF_HOME/opd_calib_cache/<sha>/`` — exactly what
``BlockTTAdapter._try_load_calibrated_cache`` consumes at training start.

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 scripts/opd/math/compressed_opd/eval_c4_ppl.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402
from compress.topology import rebuild_btt_from_topology  # noqa: E402


TEACHER = "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"
CACHE_ROOT = Path(os.environ.get("HF_HOME", "/data/yequan/huggingface")) / "opd_calib_cache"
# Match the sha256 dirs saved by btt_v2 / btt_v2_combined runs (see the
# signature.json in each dir).
CACHES = {
    "btt_v2": CACHE_ROOT / "6fe781d2b58145a5",
    "btt_v2_combined": CACHE_ROOT / "0a443da3981d6f24",
}


def _load_compressed_model(cache_dir: Path, base_model: str, dtype: torch.dtype, device: str):
    """Rebuild BTT topology on a fresh from_pretrained model, then load the
    cached btt_l / btt_r / btt_s + embeddings + lm_head."""
    print(f"  loading base model {base_model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    print("  rebuilding BTT topology ...", flush=True)
    with open(cache_dir / "btt_topology.json") as f:
        topology = json.load(f)
    rebuild_btt_from_topology(model, topology)
    print(f"  loading cached BTT weights from {cache_dir} ...", flush=True)
    state = load_file(str(cache_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]} ...", flush=True)
    if missing:
        # Some norm/embedding keys may be missing in older caches; that's ok
        # as long as none of the BTT factors are missing.
        btt_missing = [k for k in missing if "btt_" in k]
        if btt_missing:
            raise RuntimeError(f"missing BTT factor keys: {btt_missing[:5]}")
        print(f"  note: {len(missing)} non-BTT keys missing (using from_pretrained init for those)", flush=True)
    # rebuild_btt_from_topology may create BTT cores in default dtype (fp32)
    # for any factor not present in the safetensors; cast the whole model to
    # the requested dtype so BTT's einsum doesn't choke on mixed Float/BFloat16.
    return model.to(device=device, dtype=dtype)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=str, default="", help="comma-separated list to skip: original,btt_v2,btt_v2_combined")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)

    results: dict[str, dict[str, float]] = {}
    timings: dict[str, float] = {}
    param_counts: dict[str, int] = {}

    targets = []
    if "original" not in skip:
        targets.append(("original_Qwen3-4B", None))
    for name, cache in CACHES.items():
        if name not in skip:
            if cache.is_dir():
                targets.append((name, cache))
            else:
                print(f"[skip] {name}: cache dir not found at {cache}", flush=True)

    for label, cache_dir in targets:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.time()
        if cache_dir is None:
            model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=dtype).to(args.device)
        else:
            model = _load_compressed_model(cache_dir, TEACHER, dtype, args.device)
        param_counts[label] = _count_params(model)
        print(f"  param count: {param_counts[label] / 1e9:.3f}B", flush=True)
        print(f"  running C4 PPL (seqlen={args.seqlen}, seed={args.seed}) ...", flush=True)
        ppl = evaluate_model_ppl(
            model, tokenizer,
            seqlen=args.seqlen, seed=args.seed,
            datasets=("c4",), device=args.device,
        )
        results[label] = ppl
        timings[label] = time.time() - t0
        print(f"  C4 PPL: {ppl['c4']:.4f}  (elapsed {timings[label]:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'model':30s}  {'params':>10s}  {'C4 PPL':>10s}  {'elapsed':>10s}", flush=True)
    for label in results:
        print(
            f"{label:30s}  "
            f"{param_counts[label] / 1e9:>9.3f}B  "
            f"{results[label]['c4']:>10.4f}  "
            f"{timings[label]:>9.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
