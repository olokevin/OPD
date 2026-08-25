"""Standalone greedy MATH-500 eval, identical to the ES trainer's held-out loop.

Same prompt processor and grader as `verl.trainer.es` (`task_type=qwen_math`,
`ttrl_math` with `fast=True`), same greedy decoding and 3000-token budget, so the
numbers are directly comparable to the in-trainer `eval/accuracy` curve in
docs/results/ES/es_results.md.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "verl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "datasets/es_math/math500_qwenmath_test.parquet"))
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.trainer.es.task_utils import get_task_components

    rows = pd.read_parquet(args.data).to_dict("records")
    if args.limit > 0:
        rows = rows[: args.limit]
    prompt_proc, reward_fn = get_task_components("qwen_math")
    tok = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=False, seed=args.seed)
    sp = SamplingParams(temperature=0.0, seed=args.seed, max_tokens=args.max_tokens)

    prompts = [prompt_proc(r, tok) for r in rows]
    outs = llm.generate(prompts, sp)
    rewards = [reward_fn(o.outputs[0].text, r)["reward"] for o, r in zip(outs, rows)]
    acc = 100.0 * sum(rewards) / len(rewards)
    res = {"model": args.model, "n": len(rewards), "math500_acc": acc,
           "max_tokens": args.max_tokens}
    print(json.dumps(res))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
