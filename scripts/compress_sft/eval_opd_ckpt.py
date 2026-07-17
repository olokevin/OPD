"""Apples-to-apples post-OPD eval: same recipe as build_sparsegpt_student.py
(greedy MATH-500 first 200 problems, C4 sliding-window PPL, sparsity check).

Use on an HF-format ckpt dir (after merging FSDP shards via
verl/scripts/legacy_model_merger.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402


@torch.no_grad()
def count_params(model):
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero


@torch.no_grad()
def linear_sparsity(model, skip=("lm_head",)):
    tot = zer = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.split(".")[-1] not in skip:
            w = m.weight.data
            tot += w.numel()
            zer += (w == 0).sum().item()
    return zer / max(tot, 1), tot, zer


@torch.no_grad()
def eval_math500(model, tokenizer, device, limit, max_new_tokens, batch_size):
    import pandas as pd
    from verl.utils.reward_score.ttrl_math import compute_score
    df = pd.read_parquet(REPO_ROOT / "datasets" / "test_data" / "MATH-500" / "test.parquet")
    if limit > 0:
        df = df.iloc[:limit]
    prompts, gts = [], []
    for _, row in df.iterrows():
        messages = list(row["prompt"])
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)
        gts.append(row["reward_model"]["ground_truth"])
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    n_correct = 0
    n_total = len(prompts)
    for i in range(0, n_total, batch_size):
        bp = prompts[i:i + batch_size]
        bg = gts[i:i + batch_size]
        enc = tokenizer(bp, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for resp, gt in zip(decoded, bg):
            res = compute_score(resp, str(gt))
            n_correct += int(res.get("acc", False))
        done = min(i + batch_size, n_total)
        print(f"  MATH-500 {done}/{n_total}  acc={n_correct / done:.4f}", flush=True)
    return n_correct / n_total if n_total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--label", required=True, help="short label for the run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=200)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"=== Eval {args.label} ===")
    print(f"Loading {args.model_dir} in {args.dtype} on {args.device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(args.device)
    total, nonzero = count_params(model)
    lin_z_frac, lin_tot, lin_z = linear_sparsity(model)
    print(f"Params: total={total / 1e9:.3f}B  nonzero={nonzero / 1e9:.3f}B "
          f"(overall zero rate {(1 - nonzero / total) * 100:.2f}%)")
    print(f"Linear (skip lm_head): zero rate {lin_z_frac * 100:.2f}% "
          f"({lin_z / 1e9:.3f}B / {lin_tot / 1e9:.3f}B)")

    ppl = None
    if not args.skip_ppl:
        ppl = evaluate_model_ppl(model, tokenizer, seqlen=args.ppl_seqlen, seed=0,
                                 datasets=("c4",), device=args.device)["c4"]
        print(f"C4 PPL = {ppl:.4f}")

    math_acc = None
    if not args.skip_math:
        math_acc = eval_math500(model, tokenizer, device=args.device,
                                limit=args.math_limit,
                                max_new_tokens=args.math_max_new_tokens,
                                batch_size=args.math_batch_size)
        print(f"MATH-500 ({args.math_limit} greedy) = {math_acc * 100:.2f}%")

    metrics = {
        "label": args.label,
        "model_dir": args.model_dir,
        "params_total_B": total / 1e9,
        "params_nonzero_B": nonzero / 1e9,
        "linear_zero_frac": lin_z_frac,
        "linear_zero_B": lin_z / 1e9,
        "linear_total_B": lin_tot / 1e9,
        "c4_ppl": ppl,
        "math500_acc": math_acc,
        "elapsed_s": time.time() - t0,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
