"""MMLU-Pro eval for a saved HF checkpoint, via the same greedy-generate + ttrl_math
grader used for MATH-500. MMLU-Pro prompts already instruct the model to put its
final answer in \\boxed{}, so the answer letter (A-J) is extracted and graded exactly
like a math answer.

Run in the `verl` conda env (the grader imports verl/ray):
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface \\
    /home/yequan/miniconda3/envs/verl/bin/python \\
    scripts/compress_sft/eval_mmlu_pro.py \\
      --model-dir <ckpt> --label <name> --metrics-json <out.json> \\
      --limit 1000 --mmlu-max-new-tokens 512

The OLMoE chat-template fallback (no chat template on base models) mirrors
build_sparsegpt_student.py / compress_setup._render_calib_texts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))


def _render_prompt(tokenizer, messages):
    """Apply the chat template; fall back gracefully for models that don't accept
    enable_thinking (TypeError) or have no chat template at all (ValueError)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            pass
    except ValueError:
        pass
    # base model with no chat template: plain role-tagged concat + answer cue.
    body = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    return body + "\nassistant:"


def _stratified_indices(df, limit):
    """Take ``limit`` rows balanced across the `category` column (round-robin),
    so a small subset still covers every MMLU-Pro domain."""
    if limit <= 0 or limit >= len(df):
        return list(range(len(df)))
    by_cat: dict = {}
    for i, cat in enumerate(df["category"].tolist()):
        by_cat.setdefault(cat, []).append(i)
    order: list = []
    cats = list(by_cat.values())
    pos = 0
    while len(order) < limit and any(pos < len(c) for c in cats):
        for c in cats:
            if pos < len(c):
                order.append(c[pos])
                if len(order) >= limit:
                    break
        pos += 1
    return sorted(order[:limit])


@torch.no_grad()
def eval_mmlu_pro(model, tokenizer, device, limit, max_new_tokens, batch_size):
    import pandas as pd
    from verl.utils.reward_score.ttrl_math import compute_score

    df = pd.read_parquet(REPO_ROOT / "datasets" / "test_data" / "MMLU-Pro" / "test.parquet")
    idxs = _stratified_indices(df, limit)
    sub = df.iloc[idxs]

    prompts, gts, cats = [], [], []
    for _, row in sub.iterrows():
        prompts.append(_render_prompt(tokenizer, list(row["prompt"])))
        gts.append(row["reward_model"]["ground_truth"])
        cats.append(row["category"])

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_total = len(prompts)
    n_correct = 0
    per_cat: dict = {}
    for i in range(0, n_total, batch_size):
        bp = prompts[i:i + batch_size]
        bg = gts[i:i + batch_size]
        bc = cats[i:i + batch_size]
        enc = tokenizer(bp, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for resp, gt, cat in zip(decoded, bg, bc):
            ok = int(compute_score(resp, str(gt)).get("acc", False))
            n_correct += ok
            agg = per_cat.setdefault(cat, [0, 0])
            agg[0] += ok
            agg[1] += 1
        done = min(i + batch_size, n_total)
        print(f"  MMLU-Pro {done}/{n_total}  acc={n_correct / done:.4f}", flush=True)

    cat_acc = {c: (h / n if n else 0.0) for c, (h, n) in sorted(per_cat.items())}
    return (n_correct / n_total if n_total else 0.0), n_total, cat_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer source; defaults to --model-dir (use base model "
                         "id when the eval env's transformers can't read the ckpt one).")
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--label", required=True, help="short label for the run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=1000,
                    help="number of problems (category-stratified); <=0 means all 12032")
    ap.add_argument("--mmlu-max-new-tokens", type=int, default=512)
    ap.add_argument("--mmlu-batch-size", type=int, default=16)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"=== Eval MMLU-Pro {args.label} ===")
    print(f"Loading {args.model_dir} in {args.dtype} on {args.device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_dir)
    # heterogeneous svd_nystrom merged ckpt -> custom loader (see hetero_load).
    from hetero_load import load_compressed_merged
    model = load_compressed_merged(args.model_dir, dtype=dtype, device=args.device)

    acc, n, cat_acc = eval_mmlu_pro(
        model, tokenizer, device=args.device, limit=args.limit,
        max_new_tokens=args.mmlu_max_new_tokens, batch_size=args.mmlu_batch_size,
    )
    print(f"MMLU-Pro ({n} greedy) = {acc * 100:.2f}%")

    metrics = {
        "label": args.label,
        "model_dir": args.model_dir,
        "mmlu_pro_acc": acc,
        "n_problems": n,
        "mmlu_pro_acc_by_category": cat_acc,
        "elapsed_s": time.time() - t0,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
