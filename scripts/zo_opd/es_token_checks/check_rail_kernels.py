"""Gates for the rail-aware kernels wired into the es_token decode
(attn_impl in {shared, fold}, lm_head_impl=stream) and for the per-wave KV
page refresh.

  G1  sigma=0, new path: clean greedy tokens vs stock vLLM greedy (first
      divergence position + logit gap if any; kernels differ, so bit-parity
      is not guaranteed -- near-ties are reported, not hidden)
  G2  sigma>0, teacher-forced tokens: payload (per-rail logp of the clean
      token) old path vs new path, |diff| bound (old path rounds logits to
      bf16, so ~1e-2 is expected; >0.1 is a bug)
  G3  graphed vs eager oracle on the NEW path, same forced tokens: payload
      bit-parity (same kernels either way)
  G4  multi-wave page refresh: wave A (short prompts) then wave B (longer
      prompts) through the cached graph, sigma=0: wave-B tokens vs stock. With
      ES_NO_KV_REFRESH=1 the old behaviour is reproduced (expected FAIL).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_token_checks/check_rail_kernels.py
"""
import argparse
import json
import os

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
]
LONG_PREFIX = ("Carefully read the following background before answering. "
               "In mathematics, a proof is a deductive argument for a statement, "
               "showing that the stated assumptions logically guarantee the conclusion. ") * 6


def cfg(N, T, sigma, bucket, attn_impl, lm_impl, force=None):
    """bucket > len(pids) pads the wave (used as the bf16-chaos yardstick)."""
    c = dict(n_sample=N, max_tokens=T, global_seed=42, sigma=sigma, sigma_mode="absolute",
             sample_method="bernoulli", b_pack_buckets=[bucket], token_agg="mean",
             attn_impl=attn_impl, lm_head_impl=lm_impl)
    if force is not None:
        c["force_tokens"] = force
    return c


def run(llm, pids, N, T, sigma, attn_impl, lm_impl, use_graph=True, force=None, bucket=None):
    sp = SamplingParams(temperature=0.0, max_tokens=T)
    return llm.collective_rpc("run_es_decode_packed",
                              args=(pids, sp, cfg(N, T, sigma, bucket or len(pids), attn_impl, lm_impl, force),
                                    list(range(len(pids))), use_graph))[0]


def payload_diff(a, b):
    B = len(a["payload"])
    mx = max((a["payload"][p] - b["payload"][p]).abs().max().item() for p in range(B))
    mn = sum((a["payload"][p] - b["payload"][p]).abs().mean().item() for p in range(B)) / B
    dd = max(((a["payload"][p][:, 1:] - a["payload"][p][:, :1])
              - (b["payload"][p][:, 1:] - b["payload"][p][:, :1])).abs().max().item() for p in range(B))
    return mx, mn, dd


def tie_gaps(llm, pids, N, T, ref, ref_out, new_out):
    """For every greedy divergence: how far (in the OLD path's own logp) the
    new token was from the old token at that position. <~0.05 = near-tie."""
    gaps = []
    forced = run(llm, pids, N, T, 0.0, "rows", "full", force=new_out["clean_tokens"])
    for p in range(len(pids)):
        i = first_div(new_out["clean_tokens"][p], ref[p])
        if i is None:
            gaps.append(None)
            continue
        old_own = float(ref_out["payload"][p][i, 0])       # old path logp of its own token
        old_new = float(forced["payload"][p][i, 0])        # old path logp of the new token
        gaps.append(round(old_own - old_new, 4))
    return gaps


def stock(llm, pids, T):
    sp = SamplingParams(temperature=0.0, max_tokens=T)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
    return [list(o.outputs[0].token_ids) for o in outs]


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


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

    ref = stock(llm, pids, T)
    ref_long = stock(llm, pids_long, T)

    # G1 -- sigma=0 greedy vs stock, each new path. The new kernels are not
    # bit-identical to FA/cuBLAS, so a greedy trajectory may flip at a near-tie;
    # PASS = identical, or every divergence is a near-tie (gap < 0.05 nats).
    llm.collective_rpc("es_reset_graphs")
    ref_out = run(llm, pids, N, T, 0.0, "rows", "full")
    for attn_impl, lm_impl in [("rows", "full"), ("rows", "stream"), ("shared", "stream"),
                               ("fold", "stream")]:
        llm.collective_rpc("es_reset_graphs")
        out = run(llm, pids, N, T, 0.0, attn_impl, lm_impl)
        divs = [first_div(out["clean_tokens"][p], ref[p]) for p in range(B)]
        gaps = tie_gaps(llm, pids, N, T, ref, ref_out, out) if any(d is not None for d in divs) else [None] * B
        ok = all(d is None or (g is not None and abs(g) < 0.05) for d, g in zip(divs, gaps))
        res[f"G1_{attn_impl}_{lm_impl}"] = dict(pass_=ok, first_div=divs, tie_gap=gaps)
        print(f"[G1 {attn_impl}/{lm_impl}] sigma=0 vs stock greedy: "
              f"{'PASS' if all(d is None for d in divs) else ('PASS(near-tie)' if ok else 'FAIL')} "
              f"first_div={divs} tie_gap={gaps}", flush=True)

    # G2 -- sigma>0, forced tokens: old vs new payload. Yardstick = the OLD
    # path against itself at a different wave width (bucket 8 = 4 pad slots):
    # the GEMM shapes change, cuBLAS reduction order changes, and bf16 chaos
    # through 28 layers sets the intrinsic |d| scale (wiki es_token §9). A new
    # kernel passes if its deviation is within ~2x that scale.
    force = ref
    llm.collective_rpc("es_reset_graphs")
    old = run(llm, pids, N, T, args.sigma, "rows", "full", force=force)
    llm.collective_rpc("es_reset_graphs")
    old8 = run(llm, pids, N, T, args.sigma, "rows", "full", force=force, bucket=8)
    y_mx, y_mn, y_dd = payload_diff(old8, old)
    res["G2_yardstick_bucket8"] = dict(max_abs=y_mx, mean_abs=y_mn, max_abs_raildiff=y_dd)
    print(f"[G2 yardstick rows/full bucket8 vs bucket4] max|d|={y_mx:.4f} mean={y_mn:.5f} "
          f"max|d(l_n-l_0)|={y_dd:.4f}", flush=True)
    # Second yardstick: FA2 instead of FA3 for the attention (both correct,
    # not bit-identical) -- the honest scale of bf16 chaos from a sub-ulp
    # attention difference propagated through 28 layers.
    llm.collective_rpc("es_reset_graphs")
    old_fa2 = run(llm, pids, N, T, args.sigma, "fold2", "full", force=force)
    y2_mx, y2_mn, y2_dd = payload_diff(old_fa2, old)
    res["G2_yardstick_fa2"] = dict(max_abs=y2_mx, mean_abs=y2_mn, max_abs_raildiff=y2_dd)
    print(f"[G2 yardstick fold2(FA2)/full vs rows(FA3)/full] max|d|={y2_mx:.4f} mean={y2_mn:.5f} "
          f"max|d(l_n-l_0)|={y2_dd:.4f}", flush=True)
    # Third yardstick (the one that bites): the shared Triton kernel against
    # ITSELF at a different KV tile (BLOCK_N 32 vs 64). Same code, different
    # reduction order -> ulp-level attention differences -> the same chaos.
    # Measured 2026-08-31: max 2.19 / mean 0.019, vs 1.70 / 0.022 against FA.
    os.environ["ES_RAIL_ATTN_BLOCK_N"] = "32"
    llm.collective_rpc("es_reset_graphs")
    sh32 = run(llm, pids, N, T, args.sigma, "shared", "full", force=force, use_graph=False)
    os.environ["ES_RAIL_ATTN_BLOCK_N"] = "64"
    llm.collective_rpc("es_reset_graphs")
    sh64 = run(llm, pids, N, T, args.sigma, "shared", "full", force=force, use_graph=False)
    y3_mx, y3_mn, y3_dd = payload_diff(sh32, sh64)
    res["G2_yardstick_tile"] = dict(max_abs=y3_mx, mean_abs=y3_mn, max_abs_raildiff=y3_dd)
    print(f"[G2 yardstick shared BLOCK_N=32 vs 64] max|d|={y3_mx:.4f} mean={y3_mn:.5f} "
          f"max|d(l_n-l_0)|={y3_dd:.4f}", flush=True)
    y_mx, y_mn = max(y_mx, y2_mx, y3_mx), max(y_mn, y2_mn, y3_mn)
    for attn_impl, lm_impl in [("rows", "stream"), ("shared", "stream"), ("fold", "stream"),
                               ("shared", "full")]:
        llm.collective_rpc("es_reset_graphs")
        new = run(llm, pids, N, T, args.sigma, attn_impl, lm_impl, force=force)
        assert new["clean_tokens"] == old["clean_tokens"], "forced tokens differ?!"
        d, dm, dd = payload_diff(new, old)
        ok = d <= max(0.15, 1.5 * y_mx) and dm <= max(0.01, 1.5 * y_mn)
        res[f"G2_{attn_impl}_{lm_impl}"] = dict(pass_=ok, max_abs=d, mean_abs=dm, max_abs_raildiff=dd)
        print(f"[G2 {attn_impl}/{lm_impl}] payload vs rows/full (forced): max|d|={d:.4f} mean={dm:.5f} "
              f"max|d(l_n-l_0)|={dd:.4f} {'PASS' if ok else 'FAIL'}", flush=True)

    # G3 -- graphed vs eager on the new path
    for attn_impl in ("shared", "fold"):
        llm.collective_rpc("es_reset_graphs")
        g = run(llm, pids, N, T, args.sigma, attn_impl, "stream", use_graph=True, force=force)
        e = run(llm, pids, N, T, args.sigma, attn_impl, "stream", use_graph=False, force=force)
        d = max((g["payload"][p] - e["payload"][p]).abs().max().item() for p in range(B))
        ok = d == 0.0
        res[f"G3_{attn_impl}"] = dict(pass_=ok, max_abs=d)
        print(f"[G3 {attn_impl}/stream] graphed vs eager payload max|d|={d:.3e} {'PASS' if ok else 'FAIL'}", flush=True)

    # G4 -- multi-wave KV page refresh (cached graph, different longest prompt).
    # Stale pages give garbage (large tie gaps / early divergence); near-tie
    # flips from the non-bit-identical kernels are tolerated like G1.
    llm.collective_rpc("es_reset_graphs")
    ref_long_out = run(llm, pids_long, N, T, 0.0, "rows", "full")
    for attn_impl, lm_impl in [("rows", "full"), ("fold", "stream"), ("shared", "stream")]:
        llm.collective_rpc("es_reset_graphs")
        _ = run(llm, pids, N, T, 0.0, attn_impl, lm_impl)             # wave A captures
        outB = run(llm, pids_long, N, T, 0.0, attn_impl, lm_impl)      # wave B reuses graph
        divs = [first_div(outB["clean_tokens"][p], ref_long[p]) for p in range(B)]
        gaps = tie_gaps(llm, pids_long, N, T, ref_long, ref_long_out, outB) if any(d is not None for d in divs) else [None] * B
        ok = all(d is None or (g is not None and abs(g) < 0.05) for d, g in zip(divs, gaps))
        res[f"G4_{attn_impl}_{lm_impl}"] = dict(pass_=ok, first_div=divs, tie_gap=gaps,
                                                 no_refresh=bool(os.environ.get("ES_NO_KV_REFRESH")))
        print(f"[G4 {attn_impl}/{lm_impl}] wave-B (longer prompts) vs stock: "
              f"{'PASS' if all(d is None for d in divs) else ('PASS(near-tie)' if ok else 'FAIL')} "
              f"first_div={divs} tie_gap={gaps} (ES_NO_KV_REFRESH={os.environ.get('ES_NO_KV_REFRESH')})", flush=True)

    print("RESULT " + json.dumps(res))
    allok = all(v.get("pass_", True) for k, v in res.items())
    print("ALL_PASS" if allok else "SOME_FAIL")


if __name__ == "__main__":
    main()
