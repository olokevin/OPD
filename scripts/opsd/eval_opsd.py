#!/usr/bin/env python3
"""Paper-protocol eval for OPSD checkpoints (arXiv:2601.18734, Table 2 / Table 8).

Avg@12, temperature 1.0, top-p 0.95, top-k -1, min-p 0.0, max 38912 new tokens,
Qwen3 THINKING MODE ENABLED -- note the student is *trained* thinking-off at 1024
tokens; the paper still evaluates thinking-on, so this script defaults that way.

    python scripts/opsd/eval_opsd.py --model Qwen/Qwen3-1.7B --gpu 5 --tag base
    python scripts/opsd/eval_opsd.py --model <ckpt>/merged_hf --gpu 5 --tag step50

`--fast` swaps in the cheap monitoring protocol (n=4, 8192 tokens) for triage.
"""
import argparse, collections, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO, "verl"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", default=None)
ap.add_argument("--gpu", default="5")
ap.add_argument("--benches", default="AIME24,AIME25,HMMT25")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=-1)
ap.add_argument("--min-p", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=38912)
ap.add_argument("--enable-thinking", default="true")
ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--fast", action="store_true", help="n=4 @ 8192 tokens -- triage only")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
args = ap.parse_args()
if args.fast:
    args.n, args.max_tokens = 4, 8192

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ.setdefault("HF_HOME", "/data/yequan/huggingface")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score.ttrl_math import reward_func

tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model,
          gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
          max_model_len=1024 + args.max_tokens, seed=0)
sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, min_p=args.min_p, max_tokens=args.max_tokens, seed=0)

result = {"model": args.model, "tag": args.tag, "n": args.n,
          "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
          "max_tokens": args.max_tokens, "enable_thinking": args.enable_thinking,
          "benches": {}}
for bench in args.benches.split(","):
    df = pd.read_parquet(os.path.join(REPO, f"datasets/test_data/{bench}/test.parquet"))
    tkw = {} if args.enable_thinking.lower() in ("", "none") else {
        "enable_thinking": args.enable_thinking.lower() == "true"}
    prompts = [tok.apply_chat_template(list(r), tokenize=False,
                                       add_generation_prompt=True, **tkw)
               for r in df["prompt"]]
    gts = [r["ground_truth"] for r in df["reward_model"]]
    srcs = list(df["data_source"])
    outs = llm.generate(prompts, sp)

    per_prompt, lens, fin = [], [], collections.Counter()
    for o, gt, src in zip(outs, gts, srcs):
        hits = []
        for c in o.outputs:
            fin[c.finish_reason] += 1
            lens.append(len(c.token_ids))
            try:
                s = reward_func(src, c.text, gt)
                s = s["score"] if isinstance(s, dict) else float(s)
            except Exception:
                s = 0.0
            hits.append(float(s) > 0)
        per_prompt.append(np.mean(hits))
    lens = np.array(lens)
    acc = float(np.mean(per_prompt))
    sem = float(np.std(per_prompt) / np.sqrt(len(per_prompt)))
    result["benches"][bench] = {
        "n_prompts": len(per_prompt), f"avg@{args.n}": acc, "sem": sem,
        "resp_len_mean": float(lens.mean()),
        "hit_cap_rate": float((lens >= args.max_tokens).mean()),
        "stop_rate": fin["stop"] / len(lens),
    }
    print(f"[{args.tag or args.model}] {bench}: avg@{args.n}={100*acc:.1f} "
          f"(+-{100*sem:.1f}, {len(per_prompt)} prompts) len={lens.mean():.0f} "
          f"cap={(lens >= args.max_tokens).mean():.3f}", flush=True)

vals = [v[f"avg@{args.n}"] for v in result["benches"].values()]
result["average"] = float(np.mean(vals))
print(f"[{args.tag or args.model}] AVERAGE avg@{args.n} = {100*result['average']:.1f}")
print("\n" + json.dumps(result, indent=2))
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out)
