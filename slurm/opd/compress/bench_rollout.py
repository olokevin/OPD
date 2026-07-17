"""Rollout-engine speed benchmark: vLLM vs HuggingFace generate, in our OPD setting.

Loads Qwen3-4B (the student size) and generates N responses per prompt at our OPD
rollout params (max_new=MAX_RESP_LENGTH, temperature 1.0, top_p 0.95), once via vLLM
and once via HF `model.generate`, reporting wall time and decode tokens/s. This isolates
the generation engine (the dominant rollout cost); it excludes verl's FSDP<->vLLM weight
resharding, which only adds to vLLM's per-step overhead in the full pipeline.

Run in the verl env on 1 GPU.
"""
import json
import os
import time

import torch

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-4B")
N = int(os.environ.get("BENCH_N", "4"))            # responses per prompt (N_RESPONSES)
N_PROMPTS = int(os.environ.get("BENCH_PROMPTS", "8"))
MAX_NEW = int(os.environ.get("BENCH_MAX_NEW", "7168"))   # MAX_RESP_LENGTH
TEMP = float(os.environ.get("BENCH_TEMP", "1.0"))
TOP_P = float(os.environ.get("BENCH_TOP_P", "0.95"))
PARQUET = os.environ.get("BENCH_PARQUET", "datasets/OpenThoughts3_opd.parquet")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL)


def load_prompts(n):
    import pandas as pd
    df = pd.read_parquet(PARQUET).iloc[:n]
    out = []
    for _, row in df.iterrows():
        msgs = list(row["prompt"])
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False))
    return out


prompts = load_prompts(N_PROMPTS)
print(f"[bench] model={MODEL} n_prompts={N_PROMPTS} N={N} max_new={MAX_NEW} temp={TEMP} top_p={TOP_P}",
      flush=True)


def bench_vllm():
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_NEW + 2048, enforce_eager=False)
    sp = SamplingParams(n=N, temperature=TEMP, top_p=TOP_P, max_tokens=MAX_NEW)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    gen_tok = sum(len(o.token_ids) for out in outs for o in out.outputs)
    del llm
    torch.cuda.empty_cache()
    return dt, gen_tok


def bench_hf():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    gen_tok = 0
    with torch.no_grad():
        for p in prompts:
            enc = tok([p], return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=TEMP, top_p=TOP_P, num_return_sequences=N,
                                 pad_token_id=tok.pad_token_id)
            gen_tok += int((out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum().item())
    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return dt, gen_tok


res = {}
which = os.environ.get("BENCH_WHICH", "both")
if which in ("vllm", "both"):
    dt, gt = bench_vllm()
    res["vllm"] = {"wall_s": round(dt, 1), "gen_tokens": gt, "tok_per_s": round(gt / dt, 1)}
    print(f"[bench] VLLM: {dt:.1f}s  {gt} tok  {gt/dt:.1f} tok/s", flush=True)
if which in ("hf", "both"):
    dt, gt = bench_hf()
    res["hf"] = {"wall_s": round(dt, 1), "gen_tokens": gt, "tok_per_s": round(gt / dt, 1)}
    print(f"[bench] HF:   {dt:.1f}s  {gt} tok  {gt/dt:.1f} tok/s", flush=True)
if "vllm" in res and "hf" in res:
    res["vllm_speedup_x"] = round(res["vllm"]["tok_per_s"] / max(res["hf"]["tok_per_s"], 1e-9), 1)
    print(f"[bench] vLLM is {res['vllm_speedup_x']}x faster (tok/s) than HF generate", flush=True)
print("==== BENCH RESULT ====")
print(json.dumps(res, indent=2))
