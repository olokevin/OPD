"""Phase 5 -- full single-GPU decoder sweep (opd_profile_plan.md §26) and the
Phase 3 kernel audit (§24).

ONE engine; loops clean batch B x rails N x kernel path inside the process
(graphs are captured per (B, N, path) and dropped between points). Per point
the ms/token-step is the SLOPE of wall-clock over T_short -> T_long token
steps (capture/prefill cancel), exactly like bench_decode_throughput.py.

Paths:  rows/full   = shipping es_token (each rail its own FA request, full
                      [rows, V] logits)
        shared/stream = Triton shared-KV rail attention + streaming LM head
        fold/stream   = FA3 GQA-fold rail attention + streaming LM head
        A path is attn/lm[/rail[/step]] with rail in {kernel (default), fused}
        and step in {eager (default), graph} (0902 fused kernels):
        seq/stream/fused/graph = non-causal seqlen_q=R FA3 + streaming head
                      + rail fused into norm/silu/rope + whole token step in
                      the CUDA graph.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_profile/phase5_decode_heatmap.py --Bs 1,4,8,16,64 \
        --Ns 0,1,4,8,16 --paths rows/full,shared/stream --prompt-len 512

--stock-only measures vLLM's own cudagraph decode at the same B (separate
process: the es driver needs enforce_eager). --profile runs the Phase 3
kernel audit (eager step under torch.profiler, kernels bucketed by name).
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams, TokensPrompt

from common import RESULTS_DIR, gpu_info

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

KERNEL_CATS = [
    ("rail_op", r"_rail_fused"),
    ("fused_norm", r"_norm_rail_kernel"),          # 0902: rail + residual + RMSNorm
    ("fused_qkv", r"_qkv_rail_norm_rope"),         # 0902: rail + q/k norm + RoPE
    ("fused_silu", r"_silu_mul_rail"),             # 0902: rail + silu*mul
    ("lm_tail", r"_lm_tail_kernel"),               # 0902: LSE + gather-dot + payload
    ("advance", r"_advance_kernel"),               # 0902: in-graph state advance
    ("noise", r"rademacher|philox"),
    ("attn_shared", r"_rail_attn"),
    ("attn", r"flash|fwd_kernel|attn|fmha"),
    ("lm_head_stream", r"_lm_head_stream|logsumexp"),
    ("gemm", r"gemm|cutlass|nvjet|wgmma|cublas|Sm90|sm90|xmma"),
    ("norm", r"rms_norm|fused_add|layer_norm"),
    ("act", r"silu|act_and_mul|gelu"),
    ("rope", r"rotary|rope"),
    ("kv_cache", r"reshape_and_cache"),
    ("embed", r"embedding|index_select"),
    ("elementwise", r"elementwise|vectorized|copy|fill|reduce|softmax|argmax|"
                    r"gather|scatter|index|cat|arange|Memcpy|Memset|multinomial|cumsum|sort"),
]


def make_pids(tok, B, prompt_len):
    if prompt_len <= 0:
        if B <= len(PROMPTS):
            return [tok(p)["input_ids"] for p in PROMPTS[:B]]
        return [tok(f"Problem {i+1}. " + PROMPTS[i % len(PROMPTS)])["input_ids"]
                for i in range(B)]
    body = tok(" ".join(PROMPTS * 400))["input_ids"]
    out = []
    for i in range(B):
        pre = tok(f"Problem {i+1} of {B}. ")["input_ids"]
        need = prompt_len - len(pre)
        assert need > 0 and need <= len(body)
        out.append(pre + body[:need])
    return out


def no_eos(sp):
    try:
        sp._all_stop_token_ids = {-1}
    except Exception:
        pass
    return sp


def parse_path(p):
    """'attn/lm[/rail[/step]]' -> (attn, lm, rail, step) with defaults."""
    parts = p.split("/")
    parts += ["kernel", "eager"][len(parts) - 2:]
    return tuple(parts[:4])


RAIL_MODE = {"rail_mode": "token", "noise_rank": "1"}   # set from --rail-mode/--noise-rank


def es_cfg_for(N, max_tokens, sigma, bucket, attn_impl, lm_impl, seed=42,
               rail_impl="kernel", step_impl="eager"):
    return dict(n_sample=N, max_tokens=max_tokens, global_seed=seed, sigma=sigma,
                sigma_mode="absolute", sample_method="bernoulli",
                b_pack_buckets=[bucket], token_agg="mean",
                attn_impl=attn_impl, lm_head_impl=lm_impl,
                rail_impl=rail_impl, step_impl=step_impl, **RAIL_MODE)


def time_packed(llm, pids, N, max_tokens, sigma, attn_impl, lm_impl, use_graph=True,
                rail_impl="kernel", step_impl="eager"):
    B = len(pids)
    sp = no_eos(SamplingParams(temperature=0.0, max_tokens=max_tokens))
    cfg = es_cfg_for(N, max_tokens, sigma, B, attn_impl, lm_impl,
                     rail_impl=rail_impl, step_impl=step_impl)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.collective_rpc("run_es_decode_packed",
                             args=(pids, sp, cfg, list(range(B)), use_graph))[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, [len(c) for c in out["clean_tokens"]]


def time_stock(llm, pids, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, [len(o.outputs[0].token_ids) for o in outs]


def slope_ms(t_s, t_l, T_s, T_l):
    return (t_l - t_s) / (T_l - T_s) * 1e3


def profile_point(llm, pids, N, attn_impl, lm_impl, sigma, n_short=2, n_long=10,
                  rail_impl="kernel", step_impl="eager"):
    """Phase 3: eager steps under torch.profiler; CUDA kernel ms per token-step
    bucketed by kernel name. Two runs of different length are differenced so
    the prefill (same in both) cancels: per-step = (long - short) / (n_long - n_short)."""
    from torch.profiler import ProfilerActivity, profile

    def collect(n_tokens):
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            time_packed(llm, pids, N, n_tokens, sigma, attn_impl, lm_impl, use_graph=False,
                        rail_impl=rail_impl, step_impl=step_impl)
        cats = {c: 0.0 for c, _ in KERNEL_CATS}
        cats["other"] = 0.0
        names = {}
        total = 0.0
        for ev in prof.events():
            if ev.device_type.name != "CUDA":
                continue
            dur = ev.time_range.elapsed_us() / 1e3
            total += dur
            names[ev.name] = names.get(ev.name, 0.0) + dur
            for c, pat in KERNEL_CATS:
                if re.search(pat, ev.name, re.I):
                    cats[c] += dur
                    break
            else:
                cats["other"] += dur
        cats["total"] = total
        return cats, names

    time_packed(llm, pids, N, 4, sigma, attn_impl, lm_impl, use_graph=False,
                rail_impl=rail_impl, step_impl=step_impl)   # warm/compile
    c_s, n_s = collect(n_short)
    c_l, n_l = collect(n_long)
    d = n_long - n_short
    per = {k: (c_l[k] - c_s.get(k, 0.0)) / d for k in c_l}
    top = sorted(((k, (n_l[k] - n_s.get(k, 0.0)) / d) for k in n_l), key=lambda kv: -kv[1])[:12]
    return per, [(n[:70], v) for n, v in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--Bs", default="1,4,8,16,64")
    ap.add_argument("--Ns", default="0,1,2,4,8,16,32")
    ap.add_argument("--paths", default="rows/full,shared/stream",
                    help="comma list of attn_impl/lm_head_impl")
    ap.add_argument("--prompt-len", type=int, default=0, help="0 = natural prompts (~40 tok)")
    ap.add_argument("--t-short", type=int, default=64)
    ap.add_argument("--t-long", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=2,
                    help="min over this many runs per length (co-tenant noise)")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gmu", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--kv-cache-gb", type=float, default=None,
                    help="explicit KV budget per rank (bypasses vLLM memory profiling, which\n"
                         "asserts when a co-tenant frees memory mid-profile)")
    ap.add_argument("--stock-only", action="store_true")
    ap.add_argument("--profile", action="store_true", help="Phase 3 kernel audit instead of the sweep")
    ap.add_argument("--rail-mode", default="token", help="token (es-token-decode) | seq (es-decode, held noise)")
    ap.add_argument("--noise-rank", default="1", help="seq mode: int rank or 'full' (packed bits)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    Bs = [int(x) for x in args.Bs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]
    paths = [parse_path(p) for p in args.paths.split(",")]
    eager = not args.stock_only
    kw = dict(model=args.model, enforce_eager=eager, enable_prefix_caching=False,
              worker_extension_cls=WEXT, dtype="bfloat16",
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gmu)
    if args.max_model_len:
        kw["max_model_len"] = args.max_model_len
    if args.kv_cache_gb:
        kw["kv_cache_memory_bytes"] = int(args.kv_cache_gb * 2**30)
    if args.tp > 1:
        # NCCL all-reduce is graph-capturable from a hand-driven capture; the
        # custom IPC all-reduce needs vLLM's own capture path.
        kw["disable_custom_all_reduce"] = True
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    tag = args.tag or f"L{args.prompt_len}{'_stock' if args.stock_only else ''}{'_tp'+str(args.tp) if args.tp>1 else ''}"
    out_path = args.json_out or os.path.join(RESULTS_DIR, f"phase5_{tag}.json")
    rec = dict(gpu=gpu_info(), prompt_len=args.prompt_len, t_short=args.t_short,
               t_long=args.t_long, repeats=args.repeats, tp=args.tp, points=[])

    if args.stock_only:
        for B in Bs:
            pids = make_pids(tok, B, args.prompt_len)
            try:
                time_stock(llm, pids, 8)
                s_s, s_l = float("inf"), float("inf")
                for _ in range(args.repeats):
                    a, _ = time_stock(llm, pids, args.t_short)
                    b, ns = time_stock(llm, pids, args.t_long)
                    s_s, s_l = min(s_s, a), min(s_l, b)
            except Exception as e:
                print(f"[stock B={B}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            assert set(ns) == {args.t_long}, ns
            ms = slope_ms(s_s, s_l, args.t_short, args.t_long)
            p = dict(B=B, N=0, path="stock", ms_per_token_step=ms, clean_tok_per_s=B / (ms / 1e3))
            rec["points"].append(p)
            print(f"[stock B={B:2d} L={args.prompt_len}] ms/token-step={ms:.3f} tok/s={p['clean_tok_per_s']:.0f}", flush=True)
            with open(out_path, "w") as fh:
                json.dump(rec, fh, indent=2)
        return

    n_rails_max = max(max(Ns), 1)
    RAIL_MODE.update(rail_mode=args.rail_mode, noise_rank=str(args.noise_rank))
    rank_arg = "full" if str(args.noise_rank) == "full" else int(args.noise_rank)
    matched = llm.collective_rpc("install_es_layers",
                                 args=(RULES, n_rails_max, 42, args.rail_mode, rank_arg))[0]
    if args.rail_mode == "seq":   # held noise for the whole sweep (antithetic pairs)
        llm.collective_rpc("es_seq_draw", args=([100 + i for i in range(max(1, n_rails_max // 2))], True))
    print(f"[cfg] matched_layers={len(matched)} Bs={Bs} Ns={Ns} paths={paths} L={args.prompt_len}", flush=True)

    for B in Bs:
        pids = make_pids(tok, B, args.prompt_len)
        for attn_impl, lm_impl, rail_impl, step_impl in paths:
            kw = dict(rail_impl=rail_impl, step_impl=step_impl)
            for N in Ns:
                path = f"{attn_impl}/{lm_impl}"
                if (rail_impl, step_impl) != ("kernel", "eager"):
                    path += f"/{rail_impl}/{step_impl}"
                llm.collective_rpc("es_reset_graphs")
                try:
                    if args.profile:
                        per, top = profile_point(llm, pids, N, attn_impl, lm_impl, args.sigma, **kw)
                        p = dict(B=B, N=N, path=path, rows=B * (1 + N), kernel_ms=per, top=top)
                        rec["points"].append(p)
                        print(f"[prof B={B} N={N} {path}] total={per['total']:.3f} ms/step  " +
                              "  ".join(f"{k}={v:.3f}" for k, v in per.items() if k != "total" and v > 0),
                              flush=True)
                        for n, v in top[:6]:
                            print(f"      {v:8.3f}  {n}", flush=True)
                        continue
                    # step_impl=graph pins max_tokens in its graph: warm each length once
                    time_packed(llm, pids, N, args.t_short, args.sigma, attn_impl, lm_impl, **kw)
                    time_packed(llm, pids, N, args.t_long, args.sigma, attn_impl, lm_impl, **kw)
                    t_s, t_l = float("inf"), float("inf")
                    for _ in range(args.repeats):
                        a, n_s = time_packed(llm, pids, N, args.t_short, args.sigma, attn_impl, lm_impl, **kw)
                        b, n_l = time_packed(llm, pids, N, args.t_long, args.sigma, attn_impl, lm_impl, **kw)
                        t_s, t_l = min(t_s, a), min(t_l, b)
                except AssertionError as e:
                    print(f"[B={B} N={N} {path}] SKIP: {str(e)[:160]}", flush=True)
                    continue
                except Exception as e:
                    print(f"[B={B} N={N} {path}] FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                    raise
                assert set(n_s) == {args.t_short} and set(n_l) == {args.t_long}, (n_s, n_l)
                ms = slope_ms(t_s, t_l, args.t_short, args.t_long)
                p = dict(B=B, N=N, path=path, rows=B * (1 + N), ms_per_token_step=ms,
                         clean_tok_per_s=B / (ms / 1e3), row_steps_per_s=B * (1 + N) / (ms / 1e3),
                         t_short_s=t_s, t_long_s=t_l)
                rec["points"].append(p)
                base = [q for q in rec["points"] if q["B"] == B and q["path"] == path and q["N"] == Ns[0]]
                ov = (ms / base[0]["ms_per_token_step"] - 1) if base and Ns[0] == 0 else float("nan")
                print(f"[B={B:2d} N={N:2d} {path:14s} rows={B*(1+N):4d}] ms/token-step={ms:.3f}  "
                      f"clean tok/s={p['clean_tok_per_s']:.0f}  overhead vs N=0: {ov*100:+.1f}%", flush=True)
                with open(out_path, "w") as fh:
                    json.dump(rec, fh, indent=2)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
