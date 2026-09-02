"""Offline math eval on one model, identical protocol for every checkpoint.

The BP run validates in-loop at n=4/T=1.0 and the ES run at greedy n=1, so their
in-run curves are not comparable to each other. This script is the common ruler:
same benchmarks, same sampling, same token budget, run on whatever checkpoints
came out.

  python eval_math.py --model <hf_path_or_id> --gpu 7 --benches MATH-500,AMC23,AIME24
"""
import argparse, json, os, sys, collections
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "verl"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", default=None, help="defaults to --model")
ap.add_argument("--gpu", default="7")
ap.add_argument("--benches", default="MATH-500,AMC23,AIME24")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--max-tokens", type=int, default=3072)
ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--enable-thinking", default="false",
                help="forwarded to apply_chat_template; MUST be false for the "
                     "non-thinking Qwen3 pair or the eval measures a different "
                     "model than was trained")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
args = ap.parse_args()

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
                    max_tokens=args.max_tokens, seed=0)

result = {"model": args.model, "n": args.n, "temperature": args.temperature,
          "top_p": args.top_p, "max_tokens": args.max_tokens, "benches": {}}
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
        "n_prompts": len(per_prompt), f"acc_mean@{args.n}": acc, "sem": sem,
        "resp_len_mean": float(lens.mean()),
        "hit_cap_rate": float((lens >= args.max_tokens).mean()),
        "stop_rate": fin["stop"] / len(lens),
    }
    print(f"[{args.tag or args.model}] {bench}: acc@{args.n}={acc:.4f} "
          f"(+-{sem:.4f}, {len(per_prompt)} prompts) len={lens.mean():.0f} "
          f"cap={(lens >= args.max_tokens).mean():.3f}")

print("\n" + json.dumps(result, indent=2))
if args.out:
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out)
