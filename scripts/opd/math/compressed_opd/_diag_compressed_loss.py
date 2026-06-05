"""Diagnose the loss=0 / NaN seen in svd_nystrom SFT: compress Qwen3-4B (forward,
ratio 0.7) in-process, then run ONE forward+backward on a real OpenThought3 batch
and report loss, logits stats, and whether anything is NaN/Inf.

Run in the sft env:
  CUDA_VISIBLE_DEVICES=6 HF_HOME=/data/yequan/huggingface \
    /home/yequan/miniconda3/envs/sft/bin/python \
    scripts/opd/math/compressed_opd/_diag_compressed_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "LlamaFactory" / "src"))
sys.path.insert(0, str(REPO / "src"))

MODEL = "Qwen/Qwen3-4B-Base"
CALIB = "/home/yequan/Project/compression/OPD/datasets/OpenThought3-Qwen3-4B/data/train.jsonl"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.model.compress_setup import init_compress_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    fa = FinetuningArguments(
        finetuning_type="svd_nystrom", calib_mode="svd_v2",
        compression_ratio=0.7, skip_last_layers=1,
        calib_source="traces", calib_traces_path=CALIB, calib_num_seqs=16,
    )

    class _MA:
        model_name_or_path = MODEL
        trust_remote_code = True
    model = init_compress_model(cfg, model, _MA(), fa, is_trainable=True)
    model = model.to("cuda")

    # --- check weights for NaN/Inf right after compression ---
    bad_w = [n for n, p in model.named_parameters()
             if not torch.isfinite(p).all()]
    print(f"[diag] params with non-finite values after compression: {len(bad_w)}",
          bad_w[:5])

    # --- build one real OpenThought3 training example (prompt masked, response supervised) ---
    import json
    with open(CALIB) as f:
        row = json.loads(f.readline())
    msgs = row["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    labels = enc["input_ids"].clone()

    model.train()
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                labels=labels)
    loss = out.loss
    logits = out.logits
    print(f"[diag] loss = {loss.item()}")
    print(f"[diag] logits finite={torch.isfinite(logits).all().item()} "
          f"min={logits.min().item():.2f} max={logits.max().item():.2f} "
          f"absmax={logits.abs().max().item():.2f}")

    loss.backward()
    gnan = [n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()]
    gmax = max((p.grad.abs().max().item() for _, p in model.named_parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()), default=0.0)
    print(f"[diag] grads non-finite in {len(gnan)} params {gnan[:5]}; "
          f"max finite grad={gmax:.3g}")

    print("[diag] DONE")


if __name__ == "__main__":
    main()
