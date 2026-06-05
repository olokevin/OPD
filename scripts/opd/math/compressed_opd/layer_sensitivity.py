"""Per-layer compression-sensitivity sweep for Qwen3-4B (non-thinking).

For each decoder layer we compress exactly ONE module at a time — either the
whole ``self_attn`` (its 4 linears q/k/v/o_proj via SVD-LLM-V2) or the whole
``mlp`` triplet (gate/up/down_proj, jointly, via Nystrom) — leaving every other
module of the model untouched, then measure MATH-500 accuracy. Sweeping the
retain ratio (0.9 / 0.8 / 0.6 / 0.5) gives one accuracy curve per ratio over the
layer index, separately for self_attn and mlp, separately per model.

Method matches docs/results/compressed_opd.md:
  - self_attn : SVD-LLM-V2 (``svd_llm_v2``) on q/k/v/o_proj, forward whitening.
  - mlp       : Nystrom joint compression of the gate/up/down triplet, using the
                down_proj intermediate-activation covariance C_sigma.
  - calib     : OpenThought3-Qwen3-4B math traces, non-thinking chat template,
                128 x 2048-token windows, seed 3 (same as compare_compression).
  - eval      : MATH-500 first 200, greedy HF generate, ttrl_math grader.

The expensive calibration statistics are collected ONCE per model (a single
forward pass hooks every attn linear input + every down_proj input) and reused
across every (layer, module, ratio) cell. Each cell starts from a pristine CPU
copy of the base weights reloaded into a single reused GPU model, so cells never
contaminate each other.

Usage (single GPU, one shard of layers):
  CUDA_VISIBLE_DEVICES=4 python3 layer_sensitivity.py \
      --model Qwen/Qwen3-4B --model-tag qwen3-4b-base \
      --layers 0-8 --ratios 0.9 0.8 0.6 0.5 \
      --out results/layer_sens_qwen3-4b-base_l0-8.json
"""
from __future__ import annotations

import argparse
import copy
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

from compress.compress_model import MethodSpec  # noqa: E402
from compress.calibration import collect_mixed_statistics  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.loaders import build_text_calib_loader  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B"
ATTN_METHOD = "svd_llm_v2"
MLP_METHOD = "nystrom"
# Shared mixed spec used for the (one-shot) statistics collection: hooks every
# attn linear input AND every down_proj input (the Nystrom C_sigma).
MIXED_SPEC = MethodSpec(attn=ATTN_METHOD, mlp=MLP_METHOD)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
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
                    messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            out.append(text)
            if len(out) >= n:
                break
    if not out:
        raise ValueError(f"No OpenThought3 conversations read from {path}")
    return out


def build_openthought3_loader(tokenizer, *, num_seqs, max_length, batch_size,
                              length="full"):
    """OpenThought3 calibration loader.

    length="full" (DEFAULT, 2026-06-04) — full un-windowed sequences (pairs with
    sequence-reweighted covariance collection; beats windowing on reasoning, see
    FULLSEQ_CALIB_RESULTS.md). "lt2048" keeps only <2048-token conversations.
    "window2048" = legacy 2048-token windows; pass it (+ reweight="token" in the
    collector) to reproduce pre-2026-06-04 baselines.
    """
    path = REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
    if length in ("full", "lt2048"):
        from compress.loaders import build_fullseq_calib_loader
        texts = _openthought3_texts(tokenizer, path, n=num_seqs * 20)
        bs = 4 if length == "lt2048" else 1  # full truncated to 4096; bs=1 for bwd safety
        return build_fullseq_calib_loader(
            tokenizer, texts, num_seqs=num_seqs, length_filter=length, batch_size=bs)
    texts = _openthought3_texts(tokenizer, path, n=num_seqs * 12)
    return build_text_calib_loader(
        tokenizer, texts, num_seqs=num_seqs,
        max_length=max_length, batch_size=batch_size,
    )


# --------------------------------------------------------------------------- #
# single-module targeting
# --------------------------------------------------------------------------- #
def attn_linear_names(model):
    """All self_attn projection linears, grouped by decoder layer index."""
    by_layer: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if ".self_attn." not in name:
            continue
        # name like model.layers.{i}.self_attn.q_proj
        try:
            li = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        by_layer.setdefault(li, []).append(name)
    return by_layer


def mlp_down_names(model):
    """down_proj linear name per decoder layer (the Nystrom C_sigma key)."""
    by_layer: dict[int, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(".mlp.down_proj"):
            continue
        try:
            li = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        by_layer[li] = name
    return by_layer


def compress_one_module(model, *, module_type, layer_idx, ratio, stats,
                        attn_by_layer, down_by_layer, device):
    """Compress exactly one (layer, module) in-place. Returns count of linears hit.

    ``stats`` is the full mixed-statistics dict; we pass a fresh single-module
    *subset copy* to the compressor (it pops the dict, so never share the real one).
    """
    # Passing only the target module's keys means the compressor logs a
    # "No covariance for ..." WARNING for every other linear in the model
    # (~178/cell). Mute compress.* logging during the call so the per-shard log
    # stays readable and real errors aren't buried; our own __main__ logs stay.
    logger.disable("compress")
    try:
        if module_type == "self_attn":
            names = attn_by_layer[layer_idx]
            subset = {n: stats[n].clone() for n in names if n in stats}
            n_targeted = len(subset)  # capture before compress: it pops the dict
            if not subset:
                raise RuntimeError(f"no attn stats for layer {layer_idx}: {names}")
            svd_llm_v2_compress_model(
                model, subset, compression_ratio=ratio,
                skip_layers=("lm_head",), device=device, objective="forward",
            )
            return n_targeted
        elif module_type == "mlp":
            down = down_by_layer[layer_idx]
            if down not in stats:
                raise RuntimeError(f"no down_proj C_sigma for layer {layer_idx}: {down}")
            subset = {down: stats[down].clone()}
            nystrom_compress_model(
                model, subset, sparsity=1.0 - ratio,
                skip_layers=("lm_head",), device=device,
            )
            return 1  # one triplet (3 linears) rewritten
        raise ValueError(module_type)
    finally:
        logger.enable("compress")


# --------------------------------------------------------------------------- #
# MATH-500 eval (HF generate + ttrl grader) — identical to compare_compression
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_math500(model, tokenizer, *, device, limit, max_new_tokens, batch_size):
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
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for resp, gt in zip(decoded, bg):
            res = compute_score(resp, str(gt))
            n_correct += int(res.get("acc", False))
    return n_correct / n_total if n_total else 0.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_layers(spec: str, n_layers: int):
    """Parse a layer spec, PRESERVING the caller's order (dedup keeps first seen).

    Order matters: callers may want late layers swept first. 'all' -> 0..n-1,
    'a-b' expands ascending, comma lists keep their given order.
    """
    if spec.strip().lower() in ("all", ""):
        return list(range(n_layers))
    out: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        items = (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                 if "-" in part else [int(part)])
        for x in items:
            if 0 <= x < n_layers and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-tag", required=True,
                    help="short id for output records (e.g. qwen3-4b-base)")
    ap.add_argument("--layers", default="all",
                    help="layer indices: 'all', '0-8', or '0,3,5'")
    ap.add_argument("--modules", nargs="+", default=["self_attn", "mlp"],
                    choices=["self_attn", "mlp"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.9, 0.8, 0.6, 0.5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--math-limit", type=int, default=200)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--eval-baseline", action="store_true",
                    help="also eval the uncompressed model once")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- pristine CPU model kept as the per-cell source of truth ---- #
    # Compression REPLACES nn.Linear modules with new module types / shapes, so
    # the module tree changes structurally and state_dict restore is impossible.
    # Each cell therefore gets a fresh deepcopy of this pristine CPU model moved
    # onto the GPU; the compressed copy is discarded after eval. (deepcopy of an
    # ~8GB bf16 model on CPU is far cheaper than reloading from disk 576x.)
    logger.info(f"Loading base model {args.model} (kept on CPU) ...")
    base_cpu = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    base_cpu.eval()
    n_layers = base_cpu.config.num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    logger.info(f"{args.model_tag}: {n_layers} layers; sweeping {layers}")

    attn_by_layer = attn_linear_names(base_cpu)
    down_by_layer = mlp_down_names(base_cpu)

    # Live GPU working model (rebuilt per cell from base_cpu).
    model = copy.deepcopy(base_cpu).to(device)

    # ---- collect mixed statistics ONCE (attn inputs + down_proj C_sigma) --- #
    logger.info("Collecting OpenThought3 mixed statistics (one pass)...")
    calib_loader = build_openthought3_loader(
        tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
    )
    stats = collect_mixed_statistics(
        model, calib_loader, MIXED_SPEC, device=device, skip_layers=("lm_head",),
    )
    del calib_loader
    torch.cuda.empty_cache()
    logger.info(f"Collected {len(stats)} statistic tensors.")

    results = {
        "model": args.model, "model_tag": args.model_tag,
        "n_layers": n_layers, "layers": layers,
        "modules": args.modules, "ratios": args.ratios,
        "attn_method": ATTN_METHOD, "mlp_method": MLP_METHOD,
        "calib": "openthought3", "math_limit": args.math_limit,
        "config": vars(args), "baseline_acc": None, "cells": [],
    }

    def _restore():
        """Rebuild the live GPU model from the pristine CPU base.

        Returns the fresh model (compression mutates the module tree in place).
        """
        nonlocal model
        del model
        torch.cuda.empty_cache()
        model = copy.deepcopy(base_cpu).to(device)
        return model

    def _save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- optional uncompressed baseline ---- #
    if args.eval_baseline:
        logger.info("=== baseline (uncompressed) ===")
        _restore()
        acc = eval_math500(
            model, tokenizer, device=device, limit=args.math_limit,
            max_new_tokens=args.math_max_new_tokens, batch_size=args.math_batch_size,
        )
        results["baseline_acc"] = acc
        logger.info(f"baseline MATH-500 acc = {acc:.4f}")
        _save()

    # ---- sweep ---- #
    for module_type in args.modules:
        for layer_idx in layers:
            for ratio in args.ratios:
                t0 = time.time()
                logger.info(f"=== {args.model_tag} L{layer_idx} {module_type} "
                            f"ratio={ratio} ===")
                _restore()
                try:
                    n_hit = compress_one_module(
                        model, module_type=module_type, layer_idx=layer_idx,
                        ratio=ratio, stats=stats, attn_by_layer=attn_by_layer,
                        down_by_layer=down_by_layer, device=device,
                    )
                    model.to(device)
                    acc = eval_math500(
                        model, tokenizer, device=device, limit=args.math_limit,
                        max_new_tokens=args.math_max_new_tokens,
                        batch_size=args.math_batch_size,
                    )
                    err = None
                except Exception as e:  # keep the sweep alive; log the cell
                    logger.exception(f"cell failed: {e}")
                    acc, n_hit, err = None, 0, repr(e)
                rec = {
                    "model_tag": args.model_tag, "layer": layer_idx,
                    "module": module_type, "ratio": ratio,
                    "math500_acc": acc, "n_linears_compressed": n_hit,
                    "elapsed_s": time.time() - t0, "error": err,
                }
                results["cells"].append(rec)
                logger.info(f"  -> acc={acc} ({time.time()-t0:.0f}s)")
                _save()
                torch.cuda.empty_cache()

    logger.info(f"Done. Wrote {args.out} ({len(results['cells'])} cells).")


if __name__ == "__main__":
    main()
