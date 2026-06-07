"""Standalone svd_nystrom compression (NO training): compress Qwen3-4B with SVD-LLM-V2
on self_attn + Nystrom on MLP (retain 0.7, sequence-reweighted full-seq OpenThought3
calib), materialize to a DENSE smaller HF checkpoint that reloads for future SFT
(finetuning_type: full | svd_nystrom | vLLM), and report MATH-500 + MMLU-Pro accuracy
on the compressed (pre-SFT) model.

This uses the SAME recipe as LlamaFactory's _init_compress_svd_nystrom, but driven
directly from src/compress so it runs entirely in the `verl` conda env (which has the
ttrl_math grader). Mirrors build_sparsegpt_student.py.

Run (verl env):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface \\
    /home/yequan/miniconda3/envs/verl/bin/python \\
    scripts/opd/math/compressed_opd/build_svd_nystrom_student.py \\
      --model Qwen/Qwen3-4B-Base --objective forward --ratio 0.7 \\
      --save-dir /data/yequan/compress_sft/ckpts/qwen3_4b_base/forward_r0.7_precompress \\
      --metrics-json /data/yequan/compress_sft/metrics/qwen3_4b_base/forward_r0.7_precompress.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "verl"))

from compress.loaders import build_fullseq_calib_loader  # noqa: E402
from compress.structured.nystrom import (  # noqa: E402
    find_mlp_triplets, nystrom_compress_model, nystrom_combined_compress_model,
)
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.svd.svd_linear import SVDCompressedLinear  # noqa: E402
from compress.integration import materialize_svd_to_linear  # noqa: E402
from compress.ppl_eval import evaluate_model_ppl  # noqa: E402

CALIB = REPO / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"


def _render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except (TypeError, ValueError):
            pass
    except ValueError:
        pass
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)


def _calib_texts(tokenizer, n):
    texts = []
    with open(CALIB) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages")
            if msgs:
                texts.append(_render(tokenizer, msgs))
            if len(texts) >= n:
                break
    return texts


def _layer_idx(name):
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _protect(model, skip_last):
    n = model.config.num_hidden_layers
    return set(range(n - skip_last, n)) if skip_last > 0 else set()


def _drop_protected(stats, protect):
    return {k: v for k, v in stats.items() if _layer_idx(k) not in protect}


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
    df = pd.read_parquet(REPO / "datasets" / "test_data" / "MATH-500" / "test.parquet")
    if limit > 0:
        df = df.iloc[:limit]
    prompts, gts = [], []
    for _, row in df.iterrows():
        prompts.append(_render_gen(tokenizer, list(row["prompt"])))
        gts.append(row["reward_model"]["ground_truth"])
    return _gen_grade(model, tokenizer, device, prompts, gts, max_new_tokens, batch_size, compute_score, "MATH-500")


@torch.no_grad()
def eval_mmlu_pro(model, tokenizer, device, limit, max_new_tokens, batch_size):
    import pandas as pd
    from verl.utils.reward_score.ttrl_math import compute_score
    df = pd.read_parquet(REPO / "datasets" / "test_data" / "MMLU-Pro" / "test.parquet")
    # category-stratified subset
    if 0 < limit < len(df):
        by = {}
        for i, c in enumerate(df["category"].tolist()):
            by.setdefault(c, []).append(i)
        order, cats, pos = [], list(by.values()), 0
        while len(order) < limit and any(pos < len(c) for c in cats):
            for c in cats:
                if pos < len(c):
                    order.append(c[pos])
                    if len(order) >= limit:
                        break
            pos += 1
        df = df.iloc[sorted(order[:limit])]
    prompts, gts = [], []
    for _, row in df.iterrows():
        prompts.append(_render_gen(tokenizer, list(row["prompt"])))
        gts.append(row["reward_model"]["ground_truth"])
    return _gen_grade(model, tokenizer, device, prompts, gts, max_new_tokens, batch_size, compute_score, "MMLU-Pro")


def _render_gen(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            pass
    except ValueError:
        pass
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages) + "\nassistant:"


@torch.no_grad()
def _gen_grade(model, tokenizer, device, prompts, gts, max_new_tokens, batch_size, compute_score, tag):
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    n_correct, n_total = 0, len(prompts)
    for i in range(0, n_total, batch_size):
        bp, bg = prompts[i:i + batch_size], gts[i:i + batch_size]
        enc = tokenizer(bp, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        dec = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for resp, gt in zip(dec, bg):
            n_correct += int(compute_score(resp, str(gt)).get("acc", False))
        done = min(i + batch_size, n_total)
        logger.info(f"  {tag} {done}/{n_total} acc={n_correct/done:.4f}")
    return n_correct / n_total if n_total else 0.0


def compress(model, tokenizer, *, ratio, objective, skip_last, device, calib_num_seqs):
    skip = ("lm_head",)
    attn_protect = _protect(model, skip_last)
    triplets = find_mlp_triplets(model, skip)
    mlp_down_keys = {d for (_p, _g, _u, d) in triplets}
    logger.info(f"svd_nystrom: ratio={ratio} objective={objective} | {len(triplets)} triplets, "
                f"attn_protect={sorted(attn_protect)}")

    texts = _calib_texts(tokenizer, calib_num_seqs * 20)
    bs = 1 if objective == "combined" else 2
    loader = build_fullseq_calib_loader(tokenizer, texts, num_seqs=calib_num_seqs,
                                        length_filter="full", batch_size=bs)
    model.eval()
    if objective == "forward":
        from compress.calibration import collect_covariances_reweighted
        cov = collect_covariances_reweighted(model, loader, device=device, skip_layers=skip, reweight="sequence")
        attn = _drop_protected({k: v for k, v in cov.items() if ".self_attn." in k}, attn_protect)
        down = {k: v for k, v in cov.items() if k in mlp_down_keys}
        n_missing = len(mlp_down_keys) - len(down)
        nystrom_compress_model(model, {k: v.clone() for k, v in down.items()},
                               sparsity=1.0 - ratio, skip_layers=skip, device=device)
        svd_llm_v2_compress_model(model, {k: v.clone() for k, v in attn.items()},
                                  compression_ratio=ratio, skip_layers=skip, device=device, objective="forward")
    else:
        from compress.calibration import (collect_both_covariances_from_loader,
                                          collect_nystrom_combined_statistics)
        fwd, bwd = collect_both_covariances_from_loader(model, loader, device=device,
                                                        skip_layers=skip, reweight="sequence")
        attn_fwd = _drop_protected({k: v for k, v in fwd.items() if ".self_attn." in k}, attn_protect)
        attn_bwd = _drop_protected({k: v for k, v in bwd.items() if ".self_attn." in k}, attn_protect)
        stats = collect_nystrom_combined_statistics(model, loader, device=device, skip_layers=skip, reweight="sequence")
        n_missing = len(mlp_down_keys) - len(stats)
        nystrom_combined_compress_model(model, stats, sparsity=1.0 - ratio, skip_layers=skip, device=device)
        svd_llm_v2_compress_model(model, {k: v.clone() for k, v in attn_fwd.items()},
                                  compression_ratio=ratio, skip_layers=skip, device=device,
                                  objective="combined",
                                  backward_covariances={k: v.clone() for k, v in attn_bwd.items()})
    del loader
    torch.cuda.empty_cache()
    if n_missing:
        raise RuntimeError(f"{n_missing} MLP triplets had no calib cov; increase --calib-num-seqs")

    # materialize SVD factors -> dense nn.Linear so the saved ckpt is a standard
    # (smaller) Qwen3 that reloads for any future SFT / vLLM.
    materialize_svd_to_linear(model)
    # record the Nystrom-shrunk intermediate_size from a compressed triplet so the
    # saved config matches the (smaller) MLP weights and reloads correctly.
    mods = dict(model.named_modules())
    k = int(mods[triplets[0][0]].gate_proj.weight.shape[0])
    model.config.intermediate_size = k
    logger.info(f"materialized SVD->dense; config.intermediate_size -> {k}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--objective", choices=["forward", "combined"], default="forward")
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--skip-last-layers", type=int, default=1)
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--math-limit", type=int, default=500)
    ap.add_argument("--math-max-new-tokens", type=int, default=4096)
    ap.add_argument("--mmlu-limit", type=int, default=500)
    ap.add_argument("--mmlu-max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-save", action="store_true")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    t0 = time.time()
    logger.info(f"Loading {args.model} ({args.dtype}) on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    model = compress(model, tokenizer, ratio=args.ratio, objective=args.objective,
                     skip_last=args.skip_last_layers, device=args.device,
                     calib_num_seqs=args.calib_num_seqs)

    total, nonzero = count_params(model)
    logger.info(f"Params: total={total/1e9:.3f}B nonzero={nonzero/1e9:.3f}B "
                f"(retain {nonzero/total:.3f})")

    if not args.skip_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(save_dir))
        logger.info(f"Saved reloadable dense HF checkpoint -> {save_dir}")

    ppl = math_acc = mmlu_acc = None
    if not args.skip_eval:
        ppl = evaluate_model_ppl(model, tokenizer, seqlen=args.ppl_seqlen, seed=0,
                                 datasets=("c4",), device=args.device)["c4"]
        logger.info(f"C4 PPL = {ppl:.4f}")
        math_acc = eval_math500(model, tokenizer, args.device, args.math_limit,
                                args.math_max_new_tokens, args.batch_size)
        logger.info(f"MATH-500 ({args.math_limit}) = {math_acc*100:.2f}%")
        mmlu_acc = eval_mmlu_pro(model, tokenizer, args.device, args.mmlu_limit,
                                 args.mmlu_max_new_tokens, args.batch_size)
        logger.info(f"MMLU-Pro ({args.mmlu_limit}) = {mmlu_acc*100:.2f}%")

    metrics = {
        "model": args.model, "objective": args.objective, "ratio": args.ratio,
        "skip_last_layers": args.skip_last_layers, "calib_num_seqs": args.calib_num_seqs,
        "save_dir": None if args.skip_save else args.save_dir,
        "params_total_B": total / 1e9, "params_nonzero_B": nonzero / 1e9,
        "retain_realized": nonzero / total,
        "c4_ppl": ppl, "math500_acc": math_acc, "mmlu_pro_acc": mmlu_acc,
        "elapsed_s": time.time() - t0,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))
    logger.info(f"Wrote metrics -> {args.metrics_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
