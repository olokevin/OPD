"""Compress Qwen3-4B (non-thinking) with SparseGPT @ 64% unstructured sparsity
and save as an HF model directory (so the verl OPD trainer can load it as the
actor model via AutoModelForCausalLM.from_pretrained).

Matches the calibration recipe used by `compare_compression.py` (128 seqs ×
2048 tokens, batch 2, seed 3, OBS prune w/ 8 GB Hessian-group cap), but adds
optional MATH-500 eval and `save_pretrained` of the compressed weights.

Calibration source modes:
  - `openthought3` : datasets/OpenThought3-Qwen3-4B/data/train.jsonl
                     (the dataset's stored prompt+teacher conversations)
  - `jsonl <path>` : a custom jsonl with the same {"messages": [...]} shape
                     (e.g. our V2 fresh teacher-generated calibration data)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402
from compress.loaders import build_text_calib_loader  # noqa: E402
from compress.unstructured import sparsegpt_prune, sparsity_for_param_ratio  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B"
RETAIN_RATIO = 0.36
SPARSITY = sparsity_for_param_ratio(RETAIN_RATIO)  # 0.64


def _render_msgs(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )


def load_calib_texts(tokenizer, calib_jsonl: Path, n: int):
    if not calib_jsonl.is_file():
        raise FileNotFoundError(calib_jsonl)
    # Over-sample raw rows since the window packer drops short ones.
    texts = []
    with open(calib_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not msgs:
                continue
            texts.append(_render_msgs(tokenizer, msgs))
            if len(texts) >= n * 12:
                break
    if not texts:
        raise ValueError(f"No texts read from {calib_jsonl}")
    return texts


@torch.no_grad()
def count_params(model):
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero


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
        logger.info(f"  MATH-500 {min(i + batch_size, n_total)}/{n_total} "
                    f"acc={n_correct / min(i + batch_size, n_total):.4f}")
    return n_correct / n_total if n_total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calib-jsonl", required=True,
                    help="JSONL of {messages: [user, assistant]} for calibration.")
    ap.add_argument("--save-dir", required=True,
                    help="HF model dir to save the SparseGPT-pruned weights into.")
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=200)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--sparsegpt-mem-gb", type=float, default=8.0)
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--skip-save", action="store_true")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    t0 = time.time()
    logger.info(f"Loading {args.model} in {args.dtype} on {args.device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device)

    # Build calib loader from jsonl
    texts = load_calib_texts(tokenizer, Path(args.calib_jsonl), args.calib_num_seqs)
    logger.info(f"Loaded {len(texts)} candidate calib texts from {args.calib_jsonl}")
    calib_loader = build_text_calib_loader(
        tokenizer, texts,
        num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length,
        batch_size=args.calib_batch_size,
    )

    # SparseGPT prune
    logger.info(f"SparseGPT @ {SPARSITY * 100:.0f}% unstructured "
                f"(retain {RETAIN_RATIO:.2f}); Hessian-group cap {args.sparsegpt_mem_gb} GB")
    model = sparsegpt_prune(
        model, calib_loader,
        sparsity=SPARSITY,
        device=args.device,
        scope="all",
        skip_layers=("lm_head",),
        memory_limit_gb=args.sparsegpt_mem_gb,
    )
    model = model.to(args.device)

    total, nonzero = count_params(model)
    logger.info(f"Params: total={total / 1e9:.3f}B  nonzero={nonzero / 1e9:.3f}B "
                f"(zero rate {(1 - nonzero / total) * 100:.1f}%)")

    # Eval
    ppl = evaluate_model_ppl(
        model, tokenizer, seqlen=args.ppl_seqlen, seed=0,
        datasets=("c4",), device=args.device,
    )["c4"]
    logger.info(f"C4 PPL = {ppl:.4f}")
    math_acc = None
    if not args.skip_math:
        math_acc = eval_math500(
            model, tokenizer, device=args.device,
            limit=args.math_limit,
            max_new_tokens=args.math_max_new_tokens,
            batch_size=args.math_batch_size,
        )
        logger.info(f"MATH-500 acc = {math_acc * 100:.2f}% ({int(math_acc * args.math_limit)}/{args.math_limit})")

    # Save
    if not args.skip_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save as plain HF dense weights (zeros included). The tokenizer too,
        # so the OPD trainer's AutoTokenizer.from_pretrained(actor_path) works.
        logger.info(f"Saving HF weights -> {save_dir}")
        model.save_pretrained(str(save_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "config": vars(args),
        "retain_ratio": RETAIN_RATIO,
        "sparsity": SPARSITY,
        "params_total_B": total / 1e9,
        "params_nonzero_B": nonzero / 1e9,
        "c4_ppl": ppl,
        "math500_acc": math_acc,
        "elapsed_s": time.time() - t0,
    }
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote metrics -> {args.metrics_json}")
    print("==== METRICS ====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
