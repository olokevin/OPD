"""Compare compression strategies for Qwen3-4B (non-thinking) at a ~1.7B target.

Strategies (all targeting per-linear retain ratio = 0.36 → ~1.7B effective params):
  - uncompressed           : baseline Qwen/Qwen3-4B
  - sparsegpt              : unstructured SparseGPT pruning at ~64% sparsity
                             (iso-nonzero with the 1.7B structured footprint)
  - svd_v2_all             : SVD-LLM-V2 on all compressible linears
  - svd_v2_attn_nystrom_mlp: SVD-LLM-V2 on attention linears + Nystrom on MLP

Calibration datasets:
  - c4            : streamed allenai/c4 en
  - openthought3  : datasets/OpenThought3-Qwen3-4B (chat messages rendered via
                    the non-thinking chat template, then sampled)

Metrics per (calib, strategy):
  - C4 validation perplexity (compress.ppl_eval.evaluate_model_ppl)
  - MATH-500 accuracy (HF model.generate, non-thinking template, ttrl_math grader)

The uncompressed baseline is calibration-independent and evaluated once.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python3 scripts/opd/math/compressed_opd/compare_compression.py \
      --calib c4 openthought3 \
      --strategies sparsegpt svd_v2_all svd_v2_attn_nystrom_mlp \
      --math-limit 200 --out results_compare.json
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
# verl reward grader lives under verl/
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402
from compress.loaders import build_c4_calib_loader, build_text_calib_loader  # noqa: E402
from compress.compress_model import compress_model_with_loader  # noqa: E402
from compress.unstructured import sparsegpt_prune, sparsity_for_param_ratio  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B"
RETAIN_RATIO = 0.36          # → ~1.7B effective params
SPARSITY = sparsity_for_param_ratio(RETAIN_RATIO)  # 0.64
SPARSEGPT_MEM_GB = 8.0       # Hessian-group budget (overridable via --sparsegpt-mem-gb)

STRATEGIES = ("sparsegpt", "svd_v2_all", "svd_v2_attn_nystrom_mlp")


# --------------------------------------------------------------------------- #
# model / calibration helpers
# --------------------------------------------------------------------------- #
def load_model(model_name: str, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device)


def count_params(model) -> tuple[int, int]:
    """Return (total_params, nonzero_params)."""
    total = nonzero = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
    return total, nonzero


def _openthought3_texts(tokenizer, path: Path, n: int):
    """Render OpenThought3 chat conversations into non-thinking-template texts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not messages:
                continue
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            out.append(text)
            if len(out) >= n:
                break
    if not out:
        raise ValueError(f"No OpenThought3 conversations read from {path}")
    return out


def build_calib_loader(calib: str, tokenizer, *, num_seqs: int, max_length: int,
                       batch_size: int, seed: int):
    if calib == "c4":
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size, seed=seed,
        )
    if calib == "openthought3":
        path = REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
        # Over-sample raw conversations (many are shorter than max_length and get
        # dropped by the window packer), then pack into max_length windows.
        texts = _openthought3_texts(tokenizer, path, n=num_seqs * 12)
        return build_text_calib_loader(
            tokenizer, texts, num_seqs=num_seqs,
            max_length=max_length, batch_size=batch_size,
        )
    raise ValueError(f"Unknown calib dataset {calib!r}")


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def apply_strategy(model, strategy: str, calib_loader, device: str,
                   sparsegpt_mem_gb: float = SPARSEGPT_MEM_GB):
    if strategy == "sparsegpt":
        return sparsegpt_prune(
            model, calib_loader, sparsity=SPARSITY,
            device=device, scope="all",
            skip_layers=("lm_head",),
            # Keep peak Hessian memory modest so the run survives a shared GPU.
            # ~8 GB groups → several calibration forward passes, far lower peak.
            memory_limit_gb=sparsegpt_mem_gb,
        )
    if strategy == "svd_v2_all":
        return compress_model_with_loader(
            model, calib_loader,
            compression_ratio=RETAIN_RATIO,
            method="svd_llm_v2",
            device=device,
            skip_layers=("lm_head",),
        )
    if strategy == "svd_v2_attn_nystrom_mlp":
        return compress_model_with_loader(
            model, calib_loader,
            compression_ratio=RETAIN_RATIO,
            method={"attn": "svd_llm_v2", "mlp": "nystrom"},
            device=device,
            skip_layers=("lm_head",),
        )
    raise ValueError(f"Unknown strategy {strategy!r}")


# --------------------------------------------------------------------------- #
# MATH-500 eval (HF generate + ttrl grader)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_math500(model, tokenizer, *, device: str, limit: int,
                 max_new_tokens: int, batch_size: int):
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
        batch_prompts = prompts[i:i + batch_size]
        batch_gts = gts[i:i + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for resp, gt in zip(decoded, batch_gts):
            res = compute_score(resp, str(gt))
            n_correct += int(res.get("acc", False))
        logger.info(f"  MATH-500 {min(i + batch_size, n_total)}/{n_total} "
                    f"running acc={n_correct / min(i + batch_size, n_total):.4f}")
    return n_correct / n_total if n_total else 0.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calib", nargs="+", default=["c4", "openthought3"],
                    choices=["c4", "openthought3"])
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                    choices=list(STRATEGIES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=4)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=200,
                    help="number of MATH-500 problems to grade (0 = all 500)")
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--sparsegpt-mem-gb", type=float, default=SPARSEGPT_MEM_GB,
                    help="SparseGPT Hessian-group memory budget (GB); lower = less peak memory")
    ap.add_argument("--out", default="results_compare.json")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results: dict = {"config": vars(args), "retain_ratio": RETAIN_RATIO,
                     "sparsity": SPARSITY, "runs": []}

    def _record(label, calib, model, t0):
        total, nonzero = count_params(model)
        ppl = evaluate_model_ppl(
            model, tokenizer, seqlen=args.ppl_seqlen, seed=0,
            datasets=("c4",), device=args.device,
        )["c4"]
        math_acc = None
        if not args.skip_math:
            math_acc = eval_math500(
                model, tokenizer, device=args.device,
                limit=args.math_limit,
                max_new_tokens=args.math_max_new_tokens,
                batch_size=args.math_batch_size,
            )
        rec = {
            "label": label, "calib": calib,
            "params_total_B": total / 1e9,
            "params_nonzero_B": nonzero / 1e9,
            "c4_ppl": ppl, "math500_acc": math_acc,
            "elapsed_s": time.time() - t0,
        }
        results["runs"].append(rec)
        logger.info(f"[{label} | calib={calib}] params(nz)={nonzero/1e9:.3f}B "
                    f"c4_ppl={ppl:.4f} math_acc={math_acc}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # --- uncompressed baseline (calib-independent) ---
    if not args.skip_baseline:
        logger.info("=== uncompressed baseline ===")
        t0 = time.time()
        model = load_model(args.model, dtype, args.device)
        _record("uncompressed", None, model, t0)
        del model
        torch.cuda.empty_cache()

    # --- compression matrix ---
    for calib in args.calib:
        for strategy in args.strategies:
            logger.info(f"=== {strategy} | calib={calib} ===")
            t0 = time.time()
            model = load_model(args.model, dtype, args.device)
            calib_loader = build_calib_loader(
                calib, tokenizer,
                num_seqs=args.calib_num_seqs,
                max_length=args.calib_max_length,
                batch_size=args.calib_batch_size,
                seed=args.calib_seed,
            )
            model = apply_strategy(model, strategy, calib_loader, args.device,
                                   sparsegpt_mem_gb=args.sparsegpt_mem_gb)
            model = model.to(args.device)
            _record(strategy, calib, model, t0)
            del model, calib_loader
            torch.cuda.empty_cache()

    # --- print summary table ---
    print("\n========== SUMMARY ==========")
    print(f"{'strategy':28s} {'calib':14s} {'nz params':>10s} {'C4 PPL':>10s} {'MATH-500':>10s}")
    for r in results["runs"]:
        macc = "-" if r["math500_acc"] is None else f"{r['math500_acc']*100:.2f}%"
        print(f"{r['label']:28s} {str(r['calib']):14s} "
              f"{r['params_nonzero_B']:>9.3f}B {r['c4_ppl']:>10.4f} {macc:>10s}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
