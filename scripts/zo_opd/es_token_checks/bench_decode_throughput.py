"""es_token decode-throughput profile: clean-only vs clean + N parallel rails.

Measures the packed graphed decode driver (`run_es_decode_packed`) at a fixed
pack width, sweeping the rail count N, plus two reference points:

  stock      -- vLLM `llm.generate` continuous batching (the production decode
                path BP-OPD uses); the number es_token must be compared against.
  N=0        -- the SAME hand-driven packed graph loop with zero rails, i.e.
                clean decode only. Isolates the cost of the serial token loop
                from the cost of the rails.
  N=1,2,...  -- clean + N rank-1-perturbed rows riding the clean KV.

Per-token-step cost is obtained from the SLOPE of wall-clock vs max_tokens over
two lengths, so CUDA-graph capture, prefill and teardown cancel out:
    ms/token-step = (t_long - t_short) / (T_long - T_short)
EOS is disabled (`_all_stop_token_ids = {-1}`, an impossible id) so every run
executes exactly T token-steps.

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<worktree>/verl python \
      scripts/zo_opd/es_token_checks/bench_decode_throughput.py --n-sample 8
"""
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams, TokensPrompt

DEFAULT_MODEL = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                 "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
WEXT = ("verl.workers.rollout.vllm_rollout."
        "es_token_worker_extension.WorkerExtension")
RULES = [r"^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)"
         r"|mlp\.(gate_up_proj|down_proj))$"]

PROMPTS = [
    "Let f(x) = x^3 - 6x^2 + 11x - 6. Find all real roots and explain each step.",
    "A bag has 5 red and 7 blue balls. Two are drawn without replacement. "
    "Find the probability both are red, showing the full computation.",
    "Compute the sum of the first 100 positive integers and prove the formula.",
    "Solve the system 2x + 3y = 12, 4x - y = 5, and verify the solution.",
    "Find the area under y = x^2 from x = 0 to x = 3 using integration.",
    "How many distinct arrangements of the letters in MISSISSIPPI are there?",
    "Prove that the square root of 2 is irrational.",
    "Evaluate the limit of (sin x)/x as x approaches 0 and justify it.",
    "A triangle has sides 7, 24, 25. Find its area and classify it.",
    "Expand (a + b)^5 using the binomial theorem.",
    "Find the derivative of x^x with respect to x.",
    "What is the remainder when 7^100 is divided by 13?",
    "Determine whether the series sum 1/n^2 converges and to what.",
    "Solve x^2 - 5x + 6 < 0 for real x.",
    "Find the inverse of the matrix [[2, 1], [7, 4]].",
    "Compute the 10th Fibonacci number and describe the recurrence.",
]


def make_pids(tok, B):
    """Prompt ids for B slots. Identical to PROMPTS[:B] while B fits the list;
    beyond that the list is cycled with a distinguishing prefix so no two slots
    share a prefix-cache entry."""
    if B <= len(PROMPTS):
        return [tok(p)["input_ids"] for p in PROMPTS[:B]]
    return [tok(f"Problem {i+1}. " + PROMPTS[i % len(PROMPTS)])["input_ids"]
            for i in range(B)]


def no_eos(sp):
    """Disable the driver's EOS gate so every run does exactly T token-steps."""
    try:
        sp._all_stop_token_ids = {-1}
    except Exception:
        pass
    return sp


def time_packed(llm, pids, rollout_ids, n_sample, max_tokens, sigma,
                bucket_list, global_seed=42):
    sp = no_eos(SamplingParams(temperature=0.0, max_tokens=max_tokens))
    cfg = dict(n_sample=n_sample, max_tokens=max_tokens,
               global_seed=global_seed, sigma=sigma, sigma_mode="absolute",
               sample_method="bernoulli", b_pack_buckets=list(bucket_list),
               token_agg="mean")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.collective_rpc("run_es_decode_packed",
                             args=(pids, sp, cfg, rollout_ids, True))[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ntok = [len(c) for c in out["clean_tokens"]]
    return dt, ntok


def time_stock(llm, pids, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, [len(o.outputs[0].token_ids) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-sample", type=int, default=8,
                    help="rail count; 0 = clean decode only through the "
                         "same packed graph driver")
    ap.add_argument("--pack-width", type=int, default=4)
    ap.add_argument("--t-short", type=int, default=64)
    ap.add_argument("--t-long", type=int, default=320)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gmu", type=float, default=0.55)
    ap.add_argument("--stock", action="store_true",
                    help="also measure stock vLLM llm.generate at the same "
                         "batch/length (reference decode path)")
    ap.add_argument("--stock-cudagraph", action="store_true",
                    help="build the engine with enforce_eager=False so stock "
                         "vLLM gets its own decode CUDA graphs (the real "
                         "BP-OPD generation path). Only valid with "
                         "--stock-only: the es driver captures its own graphs "
                         "and requires eager.")
    ap.add_argument("--stock-only", action="store_true",
                    help="measure ONLY stock vLLM generate at --pack-width "
                         "sequences (continuous batching reference; the packed "
                         "driver cannot reach large widths, see scratch-KV)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    N = args.n_sample
    B = args.pack_width
    eager = not (args.stock_cudagraph and args.stock_only)
    llm = LLM(model=args.model, enforce_eager=eager, enable_prefix_caching=True,
              worker_extension_cls=WEXT, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=args.gmu)
    tok = llm.get_tokenizer()
    pids = make_pids(tok, B)
    rollout_ids = list(range(B))
    matched = ([] if args.stock_only else
               llm.collective_rpc("install_es_layers",
                                  args=(RULES, max(N, 1), 42))[0])
    print(f"[cfg] matched_layers={len(matched)} pack_width={B} n_sample={N} "
          f"rows_per_token={B * (1 + N)}", flush=True)

    buckets = [B]
    rec = {"n_sample": N, "pack_width": B, "rows_per_token": B * (1 + N),
           "enforce_eager": eager,
           "matched_layers": len(matched), "t_short": args.t_short,
           "t_long": args.t_long}

    if args.stock_only:
        time_stock(llm, pids, 16)
        s_s, _ = time_stock(llm, pids, args.t_short)
        s_l, ns = time_stock(llm, pids, args.t_long)
        assert set(ns) == {args.t_long}, ns
        ms_stock = (s_l - s_s) / (args.t_long - args.t_short) * 1e3
        rec["stock"] = {"t_short_s": s_s, "t_long_s": s_l,
                        "ms_per_token_step": ms_stock,
                        "tok_per_s": B / (ms_stock / 1e3),
                        "full_call_tok_per_s": B * args.t_long / s_l}
        print(f"[stock B={B} {'eager' if eager else 'cudagraph'}] "
              f"ms/token-step = {ms_stock:.3f}   "
              f"tok/s = {rec['stock']['tok_per_s']:.1f}", flush=True)
        print("RESULT " + json.dumps(rec), flush=True)
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(rec, fh, indent=2)
        return

    # warmup: captures the graph for this bucket, then a throwaway timed call
    time_packed(llm, pids, rollout_ids, N, 16, args.sigma, buckets)
    time_packed(llm, pids, rollout_ids, N, args.t_short, args.sigma, buckets)

    t_s, n_s = time_packed(llm, pids, rollout_ids, N, args.t_short,
                           args.sigma, buckets)
    t_l, n_l = time_packed(llm, pids, rollout_ids, N, args.t_long,
                           args.sigma, buckets)
    assert set(n_s) == {args.t_short}, f"short run stopped early: {n_s}"
    assert set(n_l) == {args.t_long}, f"long run stopped early: {n_l}"
    ms_step = (t_l - t_s) / (args.t_long - args.t_short) * 1e3
    rec.update({
        "t_short_s": t_s, "t_long_s": t_l,
        "ms_per_token_step": ms_step,
        "clean_tok_per_s": B / (ms_step / 1e3),
        "row_steps_per_s": B * (1 + N) / (ms_step / 1e3),
        "full_call_tok_per_s": B * args.t_long / t_l,
    })
    print(f"[packed N={N}] T={args.t_short}: {t_s:.3f}s   "
          f"T={args.t_long}: {t_l:.3f}s", flush=True)
    print(f"[packed N={N}] ms/token-step = {ms_step:.3f}   "
          f"clean tok/s = {rec['clean_tok_per_s']:.1f}   "
          f"row-steps/s = {rec['row_steps_per_s']:.1f}", flush=True)

    if args.stock:
        time_stock(llm, pids, 16)
        s_s, _ = time_stock(llm, pids, args.t_short)
        s_l, ns = time_stock(llm, pids, args.t_long)
        assert set(ns) == {args.t_long}, ns
        ms_stock = (s_l - s_s) / (args.t_long - args.t_short) * 1e3
        rec["stock"] = {
            "t_short_s": s_s, "t_long_s": s_l,
            "ms_per_token_step": ms_stock,
            "tok_per_s": B / (ms_stock / 1e3),
            "full_call_tok_per_s": B * args.t_long / s_l,
        }
        print(f"[stock B={B}] ms/token-step = {ms_stock:.3f}   "
              f"tok/s = {rec['stock']['tok_per_s']:.1f}", flush=True)

    print("RESULT " + json.dumps(rec), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    main()
