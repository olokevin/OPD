"""Diagnose why compress_sft MATH/MMLU eval looks low: capture response LENGTH +
box-rate + acc on a small sample, to tell truncation/verbosity from a real gap."""
import re, sys, time
from pathlib import Path
import torch, torch.nn as nn
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "verl"))
sys.path.insert(0, str(REPO / "scripts" / "compress_sft"))
from hetero_load import load_compressed_merged
from eval_vllm import zero_pad_mlps, _render

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MATH_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = load_compressed_merged(CKPT, dtype=torch.bfloat16, device="cpu")
zero_pad_mlps(model)
tmp = "/tmp/" + __import__("os").environ["USER"] + "/diag_pad"
model.save_pretrained(tmp, safe_serialization=True); tok.save_pretrained(tmp); del model
from vllm import LLM, SamplingParams
from verl.utils.reward_score.ttrl_math import compute_score
llm = LLM(model=tmp, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=MATH_TOK + 2048, trust_remote_code=True)

def run(name, df, max_tok, n):
    sub = df.iloc[:n]
    prompts = [_render(tok, list(r["prompt"])) for _, r in sub.iterrows()]
    gts = [r["reward_model"]["ground_truth"] for _, r in sub.iterrows()]
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tok))
    lens = [len(o.outputs[0].token_ids) for o in out]
    resp = [o.outputs[0].text for o in out]
    boxed = [bool(re.search(r"\\boxed", r)) for r in resp]
    finished = [o.outputs[0].finish_reason == "stop" for o in out]  # vs 'length' (truncated)
    acc = [int(compute_score(r, str(g)).get("acc", False)) for r, g in zip(resp, gts)]
    print(f"\n==== {name} (n={n}, max_tok={max_tok}) ====")
    print(f"acc={sum(acc)/n*100:.1f}% | boxed={sum(boxed)/n*100:.0f}% | finished(EOS)={sum(finished)/n*100:.0f}% "
          f"(rest truncated@cap) | resp_len mean={sum(lens)//n} max={max(lens)}")
    for i in range(min(2, n)):
        print(f"  [{i}] len={lens[i]} fin={out[i].outputs[0].finish_reason} boxed={boxed[i]} acc={acc[i]} gt={gts[i]}")
        print(f"      tail: ...{resp[i][-160:].strip()}")

run("MATH-500", pd.read_parquet(REPO/"datasets/test_data/MATH-500/test.parquet"), MATH_TOK, N)
run("MMLU-Pro", pd.read_parquet(REPO/"datasets/test_data/MMLU-Pro/test.parquet"), 512, N)
print("\n[diag] done")
