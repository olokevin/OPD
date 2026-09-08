#!/usr/bin/env python3
"""Merge a verl PEFT-LoRA checkpoint into a standalone HF model vLLM can load.

verl writes the adapter (adapter_config.json + adapter_model.safetensors) to
`<ckpt>/actor/merged_hf/` -- the directory name is misleading, it is NOT merged.

    python scripts/opsd/merge_lora.py --adapter <ckpt>/actor/merged_hf --out /tmp/step50
"""
import argparse, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ.setdefault("HF_HOME", "/data/yequan/huggingface")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, args.adapter, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
# tokenizer: prefer the checkpoint's own copy, fall back to the base
try:
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
except Exception:
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print("merged ->", args.out)
