"""Generate V2 calibration data for SparseGPT.

Sample 128 prompts from OpenThought3-Qwen3-4B's user turns (math problems with
the \\boxed{} instruction suffix), then have the *original uncompressed*
Qwen/Qwen3-4B (non-thinking) generate fresh assistant responses with the OPD
train-rollout generation settings (T=0.6, top_p=1.0, max_new_tokens=3072, n=1).

Output: jsonl of {"messages": [{"role":"user", ...}, {"role":"assistant", ...}]}
matching the format consumed by compare_compression.py's _openthought3_texts().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
    ap.add_argument("--prompts-src",
                    default="datasets/OpenThought3-Qwen3-4B/data/train.jsonl",
                    help="Source jsonl; we take the user turn from the first N rows.")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    # ----- collect prompts -----
    prompts_user = []
    with open(args.prompts_src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages") or []
            if not msgs or msgs[0].get("role") != "user":
                continue
            user_content = msgs[0]["content"]
            if not user_content:
                continue
            prompts_user.append(user_content)
            if len(prompts_user) >= args.n:
                break
    if len(prompts_user) < args.n:
        raise SystemExit(
            f"Only got {len(prompts_user)} prompts from {args.prompts_src}; needed {args.n}"
        )
    print(f"Sampled {len(prompts_user)} user prompts from {args.prompts_src}")

    # ----- render via non-thinking chat template -----
    tok = AutoTokenizer.from_pretrained(args.teacher)
    rendered = []
    for u in prompts_user:
        msgs = [{"role": "user", "content": u}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        rendered.append(text)

    # ----- vLLM generate -----
    llm = LLM(
        model=args.teacher,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        seed=args.seed,
    )
    outs = llm.generate(rendered, sp)

    # ----- write jsonl -----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as f:
        for u_text, o in zip(prompts_user, outs):
            resp = o.outputs[0].text
            row = {"messages": [
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": resp},
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    main()
