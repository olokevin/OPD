"""Probe a model's rollout behaviour on DAPO-Math prompts.

Answers the two questions that blocked the previous OPD runs:
  1) does generation actually STOP (emit EOS) inside the token budget, or does
     every rollout hit the cap?
  2) what accuracy does it get, i.e. is the eval metric off its floor?

  python probe_rollout.py --model Qwen/Qwen3-1.7B-Base --gpu 6
"""
import argparse, json, os, sys, collections
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "verl"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gpu", default="6")
ap.add_argument("--data", default="datasets/dapo-math-17k-processed.parquet")
ap.add_argument("--n-prompts", type=int, default=64)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--max-tokens", type=int, default=3072)
ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--tag", default="")
ap.add_argument("--dump", default="")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ.setdefault("HF_HOME", "/data/yequan/huggingface")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score.ttrl_math import reward_func

df = pd.read_parquet(os.path.join(REPO, args.data)).head(args.n_prompts)
tok = AutoTokenizer.from_pretrained(args.model)
prompts = [
    tok.apply_chat_template(list(r), tokenize=False, add_generation_prompt=True)
    for r in df["prompt"]
]
gts = [r["ground_truth"] for r in df["reward_model"]]
srcs = list(df["data_source"])

print(f"[probe] model={args.model} eos={tok.eos_token!r}({tok.eos_token_id}) "
      f"n_prompts={len(prompts)} n={args.n} T={args.temperature} max_tokens={args.max_tokens}")
print(f"[probe] prompt[0] tail: {prompts[0][-200:]!r}")

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
          max_model_len=1024 + args.max_tokens, dtype="bfloat16", seed=0)
sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens)
outs = llm.generate(prompts, sp)

fin = collections.Counter()
lens, correct, has_box = [], [], []
records = []
for o, gt, src in zip(outs, gts, srcs):
    for c in o.outputs:
        fin[c.finish_reason] += 1
        lens.append(len(c.token_ids))
        txt = c.text
        has_box.append("\\boxed" in txt)
        try:
            sc = reward_func(src, txt, gt)
            sc = sc["score"] if isinstance(sc, dict) else float(sc)
        except Exception:
            sc = 0.0
        correct.append(float(sc) > 0)
        records.append({"finish": c.finish_reason, "len": len(c.token_ids),
                        "score": float(sc), "gt": gt, "text": txt})

lens = np.array(lens)
print("\n================ PROBE RESULT " + (args.tag or args.model) + " ================")
print(f"finish_reason: {dict(fin)}   stop-rate={fin['stop']/len(lens):.3f}")
print(f"resp_len: mean={lens.mean():.0f} p50={np.percentile(lens,50):.0f} "
      f"p90={np.percentile(lens,90):.0f} max={lens.max()} min={lens.min()}")
print(f"hit_cap({args.max_tokens}): {(lens>=args.max_tokens).mean():.3f}")
print(f"has_boxed: {np.mean(has_box):.3f}")
print(f"accuracy(avg@{args.n} over {len(outs)} prompts): {np.mean(correct):.4f}")
print("=========================================================\n")
if args.dump:
    with open(args.dump, "w") as f:
        for r in records[:40]:
            f.write(json.dumps(r) + "\n")
    print("dumped", args.dump)
