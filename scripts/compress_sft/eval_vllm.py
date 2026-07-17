"""Fast vLLM eval for a compress_sft (heterogeneous-MLP) merged checkpoint.

The svd_nystrom merged checkpoint has per-layer MLP widths (skip_last=1), which vLLM
cannot load. So: load it via hetero_load, ZERO-PAD the shrunk MLPs to uniform
intermediate_size (-> stock Qwen3, the padding is inert in the forward), save a temp
dir, then vLLM-generate MATH-500 + MMLU-Pro and grade with ttrl_math. ~15x faster than
HF-generate, so evals actually finish. AIME is intentionally NOT included.

Run in the verl env (vLLM + ray grader). Writes math500.json + mmlu_pro.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "verl"))
sys.path.insert(0, str(REPO / "scripts" / "compress_sft"))  # hetero_load

from hetero_load import load_compressed_merged  # noqa: E402


@torch.no_grad()
def zero_pad_mlps(model) -> tuple[int, int]:
    """Zero-pad shrunk MLPs up to config.intermediate_size -> stock (vLLM-loadable).
    The padding (exact zeros) is inert in the forward, so generation == the compressed
    model. Returns (n_padded, I)."""
    I = int(model.config.intermediate_size)
    n = 0
    for m in model.modules():
        if not (hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj")):
            continue
        w = m.gate_proj.weight.shape[0]
        if w >= I:
            continue
        H_in = m.gate_proj.weight.shape[1]
        H_out = m.down_proj.weight.shape[0]
        dev, dt = m.gate_proj.weight.device, m.gate_proj.weight.dtype
        for name in ("gate_proj", "up_proj"):
            old = getattr(m, name)
            new = nn.Linear(H_in, I, bias=old.bias is not None, device=dev, dtype=dt)
            new.weight.zero_(); new.weight[:w].copy_(old.weight)
            if old.bias is not None:
                new.bias.zero_(); new.bias[:w].copy_(old.bias)
            setattr(m, name, new)
        old = m.down_proj
        new = nn.Linear(I, H_out, bias=old.bias is not None, device=dev, dtype=dt)
        new.weight.zero_(); new.weight[:, :w].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
        m.down_proj = new
        n += 1
    return n, I


def _render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _stratified_indices(df, limit):
    if limit <= 0 or limit >= len(df):
        return list(range(len(df)))
    by_cat: dict = {}
    for i, cat in enumerate(df["category"].tolist()):
        by_cat.setdefault(cat, []).append(i)
    order, cats, pos = [], list(by_cat.values()), 0
    while len(order) < limit and any(pos < len(c) for c in cats):
        for c in cats:
            if pos < len(c):
                order.append(c[pos])
                if len(order) >= limit:
                    break
        pos += 1
    return sorted(order[:limit])


def _grade(responses, gts):
    from verl.utils.reward_score.ttrl_math import compute_score
    return sum(int(compute_score(r, str(g)).get("acc", False)) for r, g in zip(responses, gts))


def main():
    import pandas as pd
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--math-json", required=True)
    ap.add_argument("--mmlu-json", required=True)
    ap.add_argument("--label", default="eval")
    ap.add_argument("--math-limit", type=int, default=500)
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--mmlu-limit", type=int, default=1000)
    ap.add_argument("--mmlu-max-new-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--tmp-dir", default=None, help="where to write the padded stock model")
    args = ap.parse_args()
    t0 = time.time()

    # 1. heterogeneous merged -> zero-padded stock -> temp dir
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[eval-vllm] loading heterogeneous ckpt {args.model_dir}", flush=True)
    model = load_compressed_merged(args.model_dir, dtype=torch.bfloat16, device="cpu")
    n_pad, I = zero_pad_mlps(model)
    tmp = Path(args.tmp_dir or (args.model_dir.rstrip("/") + "_padded_vllm"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"[eval-vllm] zero-padded {n_pad} MLPs -> uniform {I}; saving stock -> {tmp}", flush=True)
    model.save_pretrained(str(tmp), safe_serialization=True)
    tok.save_pretrained(str(tmp))
    del model

    # 2. vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(tmp), dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=max(args.math_max_new_tokens, args.mmlu_max_new_tokens) + 2048,
              enforce_eager=False, trust_remote_code=True)

    # 3. MATH-500
    mdf = pd.read_parquet(REPO / "datasets" / "test_data" / "MATH-500" / "test.parquet")
    if args.math_limit > 0:
        mdf = mdf.iloc[:args.math_limit]
    mprompts = [_render(tok, list(r["prompt"])) for _, r in mdf.iterrows()]
    mgts = [r["reward_model"]["ground_truth"] for _, r in mdf.iterrows()]
    msp = SamplingParams(temperature=0.0, max_tokens=args.math_max_new_tokens)
    mout = llm.generate(mprompts, msp)
    mresp = [o.outputs[0].text for o in mout]
    mcorr = _grade(mresp, mgts)
    math_acc = mcorr / max(len(mgts), 1)
    print(f"[eval-vllm] MATH-500 ({len(mgts)}) = {math_acc*100:.2f}%", flush=True)

    # 4. MMLU-Pro
    udf = pd.read_parquet(REPO / "datasets" / "test_data" / "MMLU-Pro" / "test.parquet")
    uidx = _stratified_indices(udf, args.mmlu_limit)
    usub = udf.iloc[uidx]
    uprompts = [_render(tok, list(r["prompt"])) for _, r in usub.iterrows()]
    ugts = [r["reward_model"]["ground_truth"] for _, r in usub.iterrows()]
    usp = SamplingParams(temperature=0.0, max_tokens=args.mmlu_max_new_tokens)
    uout = llm.generate(uprompts, usp)
    uresp = [o.outputs[0].text for o in uout]
    ucorr = _grade(uresp, ugts)
    mmlu_acc = ucorr / max(len(ugts), 1)
    print(f"[eval-vllm] MMLU-Pro ({len(ugts)}) = {mmlu_acc*100:.2f}%", flush=True)

    Path(args.math_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"label": args.label, "model_dir": args.model_dir, "math500_acc": math_acc,
               "aime24_acc": None, "n_problems": len(mgts), "elapsed_s": time.time() - t0},
              open(args.math_json, "w"), indent=2)
    json.dump({"label": args.label, "model_dir": args.model_dir, "mmlu_pro_acc": mmlu_acc,
               "n_problems": len(ugts), "elapsed_s": time.time() - t0},
              open(args.mmlu_json, "w"), indent=2)
    # drop the temp padded model
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[eval-vllm] done in {time.time()-t0:.0f}s | MATH {math_acc*100:.1f}% MMLU {mmlu_acc*100:.1f}%",
          flush=True)


if __name__ == "__main__":
    main()
