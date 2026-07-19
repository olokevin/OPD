#!/usr/bin/env python3
"""Offline MATH-500 eval faithful to SimpleRL-Zoo, for GRPO-via-full.sh checkpoints.

verl's in-training validation is NOT comparable to SimpleRL-Zoo's reported numbers:
it samples at temp=1.0/top_p=0.95/n=8 and applies the model chat template (which
injects a "You are a helpful assistant." system prompt). SimpleRL reports MATH-500
as greedy pass@1 (their default eval_math_nodes.sh: temperature=0.0, top_p=1,
n_sampling=1, max_tokens=16000) with the ``qwen25-math-cot`` template.

This script reproduces that:
  - prompt = qwen25-math-cot (system=instruction, user=raw question), built from the
    restructured MATH-500_simplerl parquet + the stock Qwen2.5 chat template — the
    same prompt full.sh trains on;
  - decoding = greedy, top_p=1.0, max_tokens=16000 (SimpleRL defaults; override via
    flags to get their temp=1.0/top_p=0.95 zero-RL setting);
  - grading = verl's ttrl_math compute_score (the same grader used in training).

Run with the verl conda env (needs vLLM + ray):
  /home/yequan/miniconda3/envs/verl/bin/python scripts/grpo/eval_math500_simplerl.py \
      --model <hf_checkpoint_dir> --tp 1

Reports pass@1 (n_sampling=1) or avg@n accuracy. Does NOT touch OPD assets.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Make verl's ttrl_math grader importable.
sys.path.insert(0, str(REPO_ROOT / "verl"))
from verl.utils.reward_score.ttrl_math import compute_score  # noqa: E402

DEFAULT_TEST = REPO_ROOT / "datasets/test_data/MATH-500_simplerl/test.parquet"


def build_prompts(df, tokenizer):
    """Render each restructured [system,user] prompt via the stock chat template.

    With the MATH-500_simplerl parquet (system=instruction, user=raw question) this
    reproduces SimpleRL's ``qwen25-math-cot`` byte-for-byte.
    """
    prompts = []
    for msgs in df["prompt"]:
        text = tokenizer.apply_chat_template(
            list(msgs), tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir to evaluate")
    ap.add_argument("--test-parquet", default=str(DEFAULT_TEST))
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    # SimpleRL eval_math_nodes.sh defaults: greedy pass@1, max_tokens 16000.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--n-sampling", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="eval only first N (0=all)")
    ap.add_argument("--out", default="", help="optional JSONL dump of generations")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.test_parquet)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    gts = [r["ground_truth"] for r in df["reward_model"]]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(df, tokenizer)
    print(f"[eval] {len(prompts)} problems | model={args.model}")
    print(f"[eval] sample prompt:\n{prompts[0]!r}\n")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens + 2048,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=args.n_sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=0,
    )
    outputs = llm.generate(prompts, sp)

    n_correct = 0
    n_total = 0
    rows = []
    for out, gt in zip(outputs, gts):
        # avg@n: fraction of the n samples that are correct.
        sample_correct = 0
        preds = []
        for comp in out.outputs:
            res = compute_score(comp.text, str(gt))
            sample_correct += int(bool(res["acc"]))
            preds.append({"pred": res["pred"], "acc": bool(res["acc"])})
        n_correct += sample_correct
        n_total += len(out.outputs)
        rows.append({"gt": gt, "n": len(out.outputs), "correct": sample_correct, "preds": preds})

    acc = n_correct / n_total if n_total else 0.0
    tag = "pass@1" if args.n_sampling == 1 else f"avg@{args.n_sampling}"
    print(f"\n[eval] MATH-500 {tag} accuracy: {acc:.4f}  ({n_correct}/{n_total})")

    if args.out:
        import json

        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] wrote per-problem results to {args.out}")


if __name__ == "__main__":
    main()
