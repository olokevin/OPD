"""es_token consolidated GPU parity gate (V2/V3 of the plan's gate ladder).

  Gate (a) -- sigma=0 routing. Graphed packed es decode at sigma=0: each
    prompt's clean tokens must equal stock greedy llm.generate (the
    code-independent oracle). The rail op adds sigma_buf(=0)*..., so the clean
    trajectory must be untouched by the whole rail/graph machinery.

  Gate (b) -- graphed vs eager oracle at sigma>0. SAME rollout_ids -> SAME
    seeded noise on both paths (noise is drawn into the same kind of buffer by
    the same _es_fill_noise). Assert clean tokens BIT-FOR-BIT and the per-rail
    payload (logprob of the clean sampled token) within rtol.

  Gate (c) -- staggered-EOS bucket-padding parity. force_stop_at = [3, 6, 12]:
    a slot finishing mid-bucket must not corrupt its still-active bucket-mates
    (inherited NP C-4 pad rows). Graphed == eager bit-for-bit per prompt.

Usage (one free GPU):
  CUDA_VISIBLE_DEVICES=2 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/es_token_checks/check_es_parity.py
"""
import argparse
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams, TokensPrompt

DEFAULT_MODEL = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                 "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
WEXT = ("verl.workers.rollout.vllm_rollout."
        "es_token_worker_extension.WorkerExtension")


def run_decode(llm, pids, es_cfg, rollout_ids, use_graph, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return llm.collective_rpc(
        "run_es_decode_packed",
        args=(pids, sp, es_cfg, rollout_ids, use_graph))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--payload-rtol", type=float, default=1e-2)
    ap.add_argument("--payload-atol", type=float, default=5e-3)
    ap.add_argument("--rules", nargs="+", default=[
        r"^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)"
        r"|mlp\.(gate_up_proj|down_proj))$"])
    args = ap.parse_args()

    prompts = [
        "Compute 7*8. Answer:",
        "Differentiate x^3. Answer:",
        "What is 10/2? Answer:",
        "Square root of 81? Answer:",
    ]

    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=WEXT, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    matched = llm.collective_rpc(
        "install_es_layers",
        args=(list(args.rules), args.n_sample, args.global_seed))[0]
    print(f"matched {len(matched)} layers; first={matched[:2]}")

    pids = [tok(p)["input_ids"] for p in prompts]
    rollout_ids = list(range(len(pids)))

    def es_cfg(sigma, **kw):
        d = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                 global_seed=args.global_seed, sigma=float(sigma),
                 sigma_mode="absolute", sample_method=args.sample_method,
                 b_pack_buckets=[2, 4], token_agg="sum")
        d.update(kw)
        return d

    # ---- Gate (a): sigma=0 graphed clean == stock greedy ----
    g0 = run_decode(llm, pids, es_cfg(0.0), rollout_ids, True,
                    args.max_tokens)
    sp_ref = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    ref = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids],
                       sp_ref, use_tqdm=False)
    gate_a_ok = True
    print("\n=== Gate (a) sigma=0: graphed clean == stock greedy ===")
    for p in range(len(pids)):
        got = list(g0["clean_tokens"][p])
        exp = list(ref[p].outputs[0].token_ids)
        match = got == exp
        gate_a_ok &= match
        fd = next((i for i in range(min(len(got), len(exp)))
                   if got[i] != exp[i]), None)
        print(f"  prompt {p}: match={match} len(got)={len(got)} "
              f"len(ref)={len(exp)}" + ("" if fd is None else f" first_div={fd}"))
        if not match:
            print(f"    got: {got}\n    ref: {exp}")

    # ---- Gate (b): graphed vs eager oracle at sigma>0 ----
    cfg = es_cfg(args.sigma)
    gg = run_decode(llm, pids, cfg, rollout_ids, True, args.max_tokens)
    ge = run_decode(llm, pids, cfg, rollout_ids, False, args.max_tokens)
    gate_b_tok = all(list(gg["clean_tokens"][p]) == list(ge["clean_tokens"][p])
                     for p in range(len(pids)))
    max_diff = 0.0
    worst = None
    for p in range(len(pids)):
        a, b = gg["payload"][p], ge["payload"][p]
        n = min(a.shape[0], b.shape[0])
        if n == 0:
            continue
        d = (a[:n] - b[:n]).abs()
        tol = args.payload_atol + args.payload_rtol * b[:n].abs()
        viol = (d > tol).float().mean().item()
        md = float(d.max().item())
        if md > max_diff:
            max_diff = md
            worst = (p, viol)
    gate_b_pay = worst is None or worst[1] == 0.0
    print("\n=== Gate (b) graphed vs eager oracle (sigma>0) ===")
    print(f"  clean tokens bit-for-bit : {gate_b_tok}")
    print(f"  payload max |diff|       : {max_diff:.3e} "
          f"(atol {args.payload_atol:.0e} + rtol {args.payload_rtol:.0e}; "
          f"worst prompt {worst})")
    gate_b_ok = gate_b_tok and gate_b_pay

    # ---- Gate (c): staggered EOS, graphed vs eager ----
    stops = [3, 6, 12, 12]
    cfg_c = es_cfg(args.sigma, force_stop_at=stops)
    cg = run_decode(llm, pids, cfg_c, rollout_ids, True, args.max_tokens)
    ce = run_decode(llm, pids, cfg_c, rollout_ids, False, args.max_tokens)
    gate_c_ok = True
    print("\n=== Gate (c) staggered-EOS (force_stop_at) graphed vs eager ===")
    for p in range(len(pids)):
        got, exp = list(cg["clean_tokens"][p]), list(ce["clean_tokens"][p])
        match = got == exp and len(got) == stops[p]
        gate_c_ok &= match
        print(f"  prompt {p} (stop@{stops[p]}): match={match} "
              f"len={len(got)}")
        if not match:
            print(f"    graphed: {got}\n    eager  : {exp}")

    print("\n" + "=" * 60)
    for name, ok in [("a sigma=0 routing", gate_a_ok),
                     ("b graphed==eager", gate_b_ok),
                     ("c staggered-EOS", gate_c_ok)]:
        print(("PASS" if ok else "FAIL") + f" [Gate {name}]")
    all_ok = gate_a_ok and gate_b_ok and gate_c_ok
    print("ALL GATES PASS" if all_ok else "SOME GATES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
