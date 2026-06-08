"""Regenerate the OpenThought3-Qwen3-4B calibration traces that Stage-1
reasoning-aware compression calibrates on.

`compress_common.build_calib_loader(calib="openthought3")` reads
`datasets/OpenThought3-Qwen3-4B/data/train.jsonl`, one row per conversation:

    {"messages": [{"role": "user", "content": <math problem>},
                  {"role": "assistant", "content": <Qwen3-4B reasoning trace>}]}

`_openthought3_texts` renders each row with the Qwen3 chat template
(add_generation_prompt=False, enable_thinking=False) so the assistant trace is
*included* in the calibration sequence. This script rolls the assistant traces
out with vLLM from the **uncompressed Qwen3-4B (non-thinking)** on the OpenThoughts3
math prompts we already ship in `datasets/OpenThoughts3_opd.parquet`, reproducing
the dataset that was built on the original H100 box but is absent on NERSC.

Offline: the base model must already be in the HF cache (HF_HUB_OFFLINE ok).

Usage (single GPU is enough; vLLM continuous-batches):
  CUDA_VISIBLE_DEVICES=0 python build_ot3_calib_jsonl.py \
    --model Qwen/Qwen3-4B --num-prompts 512 --max-new-tokens 4096 \
    --out datasets/OpenThought3-Qwen3-4B/data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--prompts-parquet",
                    default=str(REPO_ROOT / "datasets" / "OpenThoughts3_opd.parquet"))
    ap.add_argument("--num-prompts", type=int, default=512,
                    help="how many traces to roll out (the calibrator picks 128 "
                         "from the pool; 512 leaves room after length filtering)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B"
                                / "data" / "train.jsonl"))
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.prompts_parquet)
    n = min(args.num_prompts, len(df))
    df = df.iloc[:n]

    tok = AutoTokenizer.from_pretrained(args.model)
    user_contents, rendered = [], []
    for _, row in df.iterrows():
        msgs = list(row["prompt"])  # [{"role": "user", "content": ...}]
        user_contents.append(msgs[-1]["content"])
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        rendered.append(text)

    print(f"[calib] rolling out {n} traces from {args.model} "
          f"(max_new_tokens={args.max_new_tokens}, T={args.temperature})")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              dtype="bfloat16", seed=args.seed, enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, seed=args.seed)
    outs = llm.generate(rendered, sp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for user, o in zip(user_contents, outs):
            resp = o.outputs[0].text.strip()
            if not resp:
                continue
            row = {"messages": [{"role": "user", "content": user},
                                {"role": "assistant", "content": resp}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[calib] wrote {kept} conversations -> {out_path}")


if __name__ == "__main__":
    main()
