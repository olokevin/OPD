"""End-to-end gates for the 0902 fused-kernel decode paths
(rail_impl=fused, attn_impl=seq, step_impl=graph), mirroring
check_rail_kernels.py:

  G1  sigma=0 greedy vs stock vLLM (identical, or exact bf16 ties only)
  G2  sigma>0 forced tokens: payload vs rows/full, judged against the
      bf16-chaos yardstick (rows/full at bucket 8 vs 4, fold2 vs rows)
  G3  step_impl=graph: graphed vs eager body, bit-parity (same kernels)
  G3b rail fused vs rail kernel on the SAME attention/head (rows/stream):
      payload diff = only the rail update's FMA/reduction order
  G4  multi-wave KV page refresh through the cached graph (longer prompts)
  G5  real EOS in the in-graph step: clean tokens + lengths == eager step

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_token_checks/check_fused_path.py
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams, TokensPrompt

from check_rail_kernels import (DEFAULT_MODEL, LONG_PREFIX, PROMPTS, RULES, WEXT,
                                first_div, payload_diff, stock)

NEW = [("rows", "stream", "fused", "eager"),
       ("seq", "stream", "kernel", "eager"),
       ("seq", "stream", "fused", "eager"),
       ("seq", "stream", "fused", "graph"),
       ("fold", "stream", "fused", "graph")]


def cfg(N, T, sigma, bucket, attn, lm, rail, step, force=None, stop=None):
    c = dict(n_sample=N, max_tokens=T, global_seed=42, sigma=sigma, sigma_mode="absolute",
             sample_method="bernoulli", b_pack_buckets=[bucket], token_agg="mean",
             attn_impl=attn, lm_head_impl=lm, rail_impl=rail, step_impl=step)
    if force is not None:
        c["force_tokens"] = force
    if stop is not None:
        c["force_stop_at"] = stop
    return c


def run(llm, pids, N, T, sigma, path, use_graph=True, force=None, bucket=None, stop=None,
        eos=False):
    sp = SamplingParams(temperature=0.0, max_tokens=T)
    if not eos:
        sp._all_stop_token_ids = {-1}
    return llm.collective_rpc(
        "run_es_decode_packed",
        args=(pids, sp, cfg(N, T, sigma, bucket or len(pids), *path, force=force, stop=stop),
              list(range(len(pids))), use_graph))[0]


def tie_gaps(llm, pids, N, T, ref, ref_out, new_out):
    gaps = []
    forced = run(llm, pids, N, T, 0.0, ("rows", "full", "kernel", "eager"),
                 force=new_out["clean_tokens"])
    for p in range(len(pids)):
        i = first_div(new_out["clean_tokens"][p], ref[p])
        if i is None:
            gaps.append(None)
            continue
        gaps.append(round(float(ref_out["payload"][p][i, 0]) - float(forced["payload"][p][i, 0]), 4))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gmu", type=float, default=0.4)
    args = ap.parse_args()
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=False,
              worker_extension_cls=WEXT, dtype="bfloat16", gpu_memory_utilization=args.gmu)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in PROMPTS]
    pids_long = [tok(LONG_PREFIX + p)["input_ids"] for p in PROMPTS]
    B, T, N = len(pids), args.T, args.N
    llm.collective_rpc("install_es_layers", args=(RULES, N, 42))
    res = {}
    OLD = ("rows", "full", "kernel", "eager")

    def reset():
        llm.collective_rpc("es_reset_graphs")

    ref = stock(llm, pids, T)
    ref_long = stock(llm, pids_long, T)
    # G1 -- sigma=0 greedy vs stock
    reset()
    ref_out = run(llm, pids, N, T, 0.0, OLD)
    for path in NEW:
        reset()
        out = run(llm, pids, N, T, 0.0, path)
        divs = [first_div(out["clean_tokens"][p], ref[p]) for p in range(B)]
        gaps = tie_gaps(llm, pids, N, T, ref, ref_out, out) if any(d is not None for d in divs) else [None] * B
        ok = all(d is None or (g is not None and abs(g) < 0.05) for d, g in zip(divs, gaps))
        res["G1_" + "/".join(path)] = dict(pass_=ok, first_div=divs, tie_gap=gaps)
        print(f"[G1 {'/'.join(path)}] sigma=0 vs stock greedy: "
              f"{'PASS' if all(d is None for d in divs) else ('PASS(near-tie)' if ok else 'FAIL')} "
              f"first_div={divs} tie_gap={gaps}", flush=True)

    # G2 -- forced tokens, sigma>0: payload vs rows/full with the chaos yardstick
    force = ref
    reset()
    old = run(llm, pids, N, T, args.sigma, OLD, force=force)
    reset()
    old8 = run(llm, pids, N, T, args.sigma, OLD, force=force, bucket=8)
    y_mx, y_mn, _ = payload_diff(old8, old)
    reset()
    old_fa2 = run(llm, pids, N, T, args.sigma, ("fold2", "full", "kernel", "eager"), force=force)
    y2_mx, y2_mn, _ = payload_diff(old_fa2, old)
    # The yardstick that bites (es_profile_results.md 6): the SAME fused kernels
    # with a different alpha-reduction tiling -> ulp-level alpha differences ->
    # bf16 chaos through 28 layers. Judge the fused paths against that scale.
    from verl.trainer.es_token.fused_rail_kernels import set_alpha_tiling
    reset()
    f_ref = run(llm, pids, N, T, args.sigma, ("rows", "stream", "fused", "eager"), force=force)
    set_alpha_tiling(1024, 8)
    reset()
    f_alt = run(llm, pids, N, T, args.sigma, ("rows", "stream", "fused", "eager"), force=force)
    set_alpha_tiling(4096, 16)
    y3_mx, y3_mn, _ = payload_diff(f_alt, f_ref)
    print(f"[G2 yardstick fused BLOCK_IN 1024/8w vs 4096/16w] max|d|={y3_mx:.4f} mean={y3_mn:.5f}", flush=True)
    y_mx, y_mn = max(y_mx, y2_mx, y3_mx), max(y_mn, y2_mn, y3_mn)
    print(f"[G2 yardstick] max over (bucket8-vs-4, fold2-vs-rows, fused-tiling): max|d|={y_mx:.4f} mean={y_mn:.5f}", flush=True)
    res["G2_yardstick"] = dict(max_abs=y_mx, mean_abs=y_mn, fused_tiling=dict(max_abs=y3_mx, mean_abs=y3_mn))
    for path in NEW:
        reset()
        new = run(llm, pids, N, T, args.sigma, path, force=force)
        assert new["clean_tokens"] == old["clean_tokens"], "forced tokens differ?!"
        d, dm, dd = payload_diff(new, old)
        ok = d <= max(0.15, 1.5 * y_mx) and dm <= max(0.01, 1.5 * y_mn)
        res["G2_" + "/".join(path)] = dict(pass_=ok, max_abs=d, mean_abs=dm, max_abs_raildiff=dd)
        print(f"[G2 {'/'.join(path)}] payload vs rows/full (forced): max|d|={d:.4f} mean={dm:.5f} "
              f"max|d(l_n-l_0)|={dd:.4f} {'PASS' if ok else 'FAIL'}", flush=True)
    # G3b -- fused rail vs kernel rail, same attention/head: the pure rail-op delta
    reset()
    a = run(llm, pids, N, T, args.sigma, ("rows", "stream", "kernel", "eager"), force=force)
    reset()
    b = run(llm, pids, N, T, args.sigma, ("rows", "stream", "fused", "eager"), force=force)
    d, dm, dd = payload_diff(a, b)
    ok = d <= max(0.15, 1.5 * y_mx)
    res["G3b_rail_fused_vs_kernel"] = dict(pass_=ok, max_abs=d, mean_abs=dm)
    print(f"[G3b rows/stream fused-vs-kernel rail] max|d|={d:.4f} mean={dm:.5f} {'PASS' if ok else 'FAIL'}", flush=True)

    # G3 -- in-graph step: graphed vs eager body, bit parity
    for path in [p for p in NEW if p[3] == "graph"]:
        reset()
        g = run(llm, pids, N, T, args.sigma, path, use_graph=True, force=force)
        reset()
        e = run(llm, pids, N, T, args.sigma, path, use_graph=False, force=force)
        d = max((g["payload"][p] - e["payload"][p]).abs().max().item() for p in range(B))
        ok = d == 0.0 and g["clean_tokens"] == e["clean_tokens"]
        res["G3_" + "/".join(path)] = dict(pass_=ok, max_abs=d)
        print(f"[G3 {'/'.join(path)}] graphed vs eager payload max|d|={d:.3e} {'PASS' if ok else 'FAIL'}", flush=True)

    # G4 -- multi-wave page refresh through the cached graph
    reset()
    ref_long_out = run(llm, pids_long, N, T, 0.0, OLD)
    for path in NEW:
        reset()
        _ = run(llm, pids, N, T, 0.0, path)
        outB = run(llm, pids_long, N, T, 0.0, path)
        divs = [first_div(outB["clean_tokens"][p], ref_long[p]) for p in range(B)]
        gaps = tie_gaps(llm, pids_long, N, T, ref_long, ref_long_out, outB) if any(d is not None for d in divs) else [None] * B
        ok = all(d is None or (g is not None and abs(g) < 0.05) for d, g in zip(divs, gaps))
        res["G4_" + "/".join(path)] = dict(pass_=ok, first_div=divs, tie_gap=gaps)
        print(f"[G4 {'/'.join(path)}] wave-B (longer prompts) vs stock: "
              f"{'PASS' if all(d is None for d in divs) else ('PASS(near-tie)' if ok else 'FAIL')} "
              f"first_div={divs} tie_gap={gaps}", flush=True)

    # G5 -- real EOS + staggered forced stops in the in-graph step vs the eager step
    T5 = 48
    stop_at = [10, 20, T5 + 5, 30]
    for path in [p for p in NEW if p[3] == "graph"]:
        reset()
        e = run(llm, pids, N, T5, 0.0, ("rows", "stream", "kernel", "eager"), stop=stop_at, eos=True)
        reset()
        g = run(llm, pids, N, T5, 0.0, path, stop=stop_at, eos=True)
        lens_e = [len(t) for t in e["clean_tokens"]]
        lens_g = [len(t) for t in g["clean_tokens"]]
        same = [first_div(g["clean_tokens"][p], e["clean_tokens"][p]) for p in range(B)]
        pd = max((g["payload"][p][:min(lens_e[p], lens_g[p])] - e["payload"][p][:min(lens_e[p], lens_g[p])]).abs().max().item() for p in range(B))
        ok = lens_e == lens_g and all(x is None for x in same)
        res["G5_" + "/".join(path)] = dict(pass_=ok, lens_eager=lens_e, lens_graph=lens_g, first_div=same, payload_max_abs=pd)
        print(f"[G5 {'/'.join(path)}] EOS/forced-stop lengths eager={lens_e} graph={lens_g} first_div={same} "
              f"payload max|d|={pd:.4f} {'PASS' if ok else 'FAIL'}", flush=True)

    print("RESULT " + json.dumps(res))
    allok = all(v.get("pass_", True) for v in res.values())
    print("ALL_PASS" if allok else "SOME_FAIL")


if __name__ == "__main__":
    main()
