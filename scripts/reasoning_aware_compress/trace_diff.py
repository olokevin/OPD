"""Block T — Reasoning-trace diff: uncompressed vs compressed.

Diagnostic / inspiration (NOT a headline metric): for each compression method,
look at *how the reasoning itself changes*, not just the accuracy number. Pick 5
fixed MATH-500 problems the uncompressed Qwen3-4B solves correctly (greedy), then
for each method/cell regenerate the trace on the compressed model under the SAME
prompt + greedy decoding and diff it against the dense trace.

Two entry points:
  1. `--mode build` — run dense model greedy on MATH-500, keep the first
     --n-probes graded CORRECT, freeze {problem_id, prompt, dense_text, gold} to
     trace_probe_set.json so every method diffs against the same reference.
  2. `generate_traces(model, tokenizer, probe, ...)` — importable helper the
     method drivers (D/A/B) call to dump per-item compressed traces. Also exposed
     via `--mode dense-only` to regenerate the dense reference traces.

Records saved per item: {problem_id, dense_text, comp_text, dense_correct=True,
comp_correct, method, ratio}. Failure-mode tagging + side-by-side excerpts are a
light manual/LLM pass on the saved JSON (see TRACE_DIFF.md), kept out of the
generation path.

Settings match eval_math500 exactly: enable_thinking=False, do_sample=False,
max_new_tokens=2048, left-padding, ttrl_math grader on dataset gold.

Usage — build the probe set once (dense, GPU 5):
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/reasoning_aware_compress/trace_diff.py --mode build \
        --n-probes 5 --scan-limit 60 \
        --probe-set scripts/reasoning_aware_compress/results/blockT/trace_probe_set.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import compress_common as cc


# --------------------------------------------------------------------------- #
# probe-set construction (dense reference)
# --------------------------------------------------------------------------- #
def _load_math500(limit: int):
    import pandas as pd
    df = pd.read_parquet(
        cc.REPO_ROOT / "datasets" / "test_data" / "MATH-500" / "test.parquet")
    if limit > 0:
        df = df.iloc[:limit]
    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "problem_id": int(idx),
            "messages": list(row["prompt"]),
            "gold": str(row["reward_model"]["ground_truth"]),
        })
    return rows


def _render_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _generate_one(model, tokenizer, prompt, *, device, max_new_tokens):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    enc = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


def _grade(text, gold):
    from verl.utils.reward_score.ttrl_math import compute_score
    res = compute_score(text, str(gold))
    return bool(res.get("acc", False))


def build_probe_set(model, tokenizer, *, device, n_probes, scan_limit,
                    max_new_tokens):
    """Run dense model greedy over MATH-500[:scan_limit], keep first n_probes
    graded CORRECT. Returns the frozen probe list."""
    model.eval()
    rows = _load_math500(scan_limit)
    probes = []
    for r in rows:
        prompt = _render_prompt(tokenizer, r["messages"])
        text = _generate_one(model, tokenizer, prompt, device=device,
                             max_new_tokens=max_new_tokens)
        correct = _grade(text, r["gold"])
        logger.info(f"  probe scan pid={r['problem_id']} correct={correct} "
                    f"({len(probes)}/{n_probes} kept)")
        if correct:
            probes.append({
                "problem_id": r["problem_id"],
                "prompt": prompt,
                "gold": r["gold"],
                "dense_text": text,
                "dense_correct": True,
            })
        if len(probes) >= n_probes:
            break
    if len(probes) < n_probes:
        logger.warning(f"Only {len(probes)}/{n_probes} dense-correct probes found "
                       f"in first {scan_limit} problems; increase --scan-limit.")
    return probes


# --------------------------------------------------------------------------- #
# importable helper for method drivers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_traces(model, tokenizer, probe_set, *, device, method, ratio,
                    max_new_tokens=2048):
    """Regenerate traces on `model` (compressed) for every probe, diff vs dense.

    Returns a list of records: {problem_id, dense_text, comp_text,
    dense_correct, comp_correct, method, ratio}. The dense reference text/gold
    come from the frozen probe_set (dataset ground truth, not a model output)."""
    model.eval()
    out = []
    for p in probe_set:
        comp_text = _generate_one(model, tokenizer, p["prompt"], device=device,
                                  max_new_tokens=max_new_tokens)
        comp_correct = _grade(comp_text, p["gold"])
        out.append({
            "problem_id": p["problem_id"],
            "method": method,
            "ratio": ratio,
            "dense_correct": p.get("dense_correct", True),
            "comp_correct": comp_correct,
            "dense_text": p["dense_text"],
            "comp_text": comp_text,
        })
        logger.info(f"  trace pid={p['problem_id']} method={method} "
                    f"comp_correct={comp_correct}")
    return out


def load_probe_set(path) -> list:
    with open(path) as f:
        obj = json.load(f)
    return obj["probes"] if isinstance(obj, dict) and "probes" in obj else obj


# --------------------------------------------------------------------------- #
# main (build mode + dense-only regeneration)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="build", choices=["build", "dense-only"])
    ap.add_argument("--model", default=cc.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(cc.DTYPE_MAP))
    ap.add_argument("--n-probes", type=int, default=5)
    ap.add_argument("--scan-limit", type=int, default=60,
                    help="how many MATH-500 problems to scan for dense-correct probes")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--probe-set", required=True,
                    help="output path for trace_probe_set.json")
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    tokenizer = cc.load_tokenizer(args.model)
    logger.info(f"Loading dense model {args.model}...")
    model = cc.load_model(args.model, dtype, args.device)

    t0 = time.time()
    probes = build_probe_set(
        model, tokenizer, device=args.device, n_probes=args.n_probes,
        scan_limit=args.scan_limit, max_new_tokens=args.max_new_tokens)
    obj = {"model": args.model, "n_probes": len(probes),
           "scan_limit": args.scan_limit, "config": vars(args),
           "elapsed_s": time.time() - t0, "probes": probes}
    cc.save_json(obj, args.probe_set)
    logger.info(f"Wrote {len(probes)} probes -> {args.probe_set} "
                f"({time.time() - t0:.0f}s)")
    print(f"\nFroze {len(probes)} dense-correct MATH-500 probes: "
          f"pids={[p['problem_id'] for p in probes]}")


if __name__ == "__main__":
    main()
