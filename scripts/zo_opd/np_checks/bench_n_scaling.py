"""M3 micro-benchmark (spec §7.5): does the 1+N decode step stay ~flat in s/tok
as N grows, confirming the user's "N rails are ~free" premise (decode is
memory-bound; the N extra rails share weights+KV, adding compute not memory
traffic)?

Reports per-token wall-clock for N in {1,8,16,32,64} on a single prompt, for the
chosen driver. Expectation if the premise holds: near-flat s/tok until the 1+N
GEMM saturates, then a gentler-than-linear rise. V1 eager would inflate every
point with per-token Python dispatch; the graphed driver isolates the FLOPs.

Usage (1 GPU + small model):
  conda run -n verl python scripts/zo_opd/np_checks/bench_n_scaling.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' \
      --driver graphed_cuda --max-tokens 64
"""
import argparse
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def _run(llm, driver, prompt_ids, layer, np_cfg):
    sp = SamplingParams(temperature=0.0)
    if driver == "v1":
        return llm.collective_rpc(
            "run_np_decode", args=(prompt_ids, sp, layer, np_cfg, 0))[0]
    use_cuda = (driver == "graphed_cuda")
    return llm.collective_rpc(
        "run_np_decode_graphed",
        args=(prompt_ids, sp, layer, np_cfg, 0, use_cuda))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--driver",
                    choices=["v1", "graphed_eager", "graphed_cuda"],
                    default="graphed_cuda")
    ap.add_argument("--prompt", default="Compute the integral of x^2. Answer:")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--ns", default="1,8,16,32,64")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    prompt_ids = tok(args.prompt)["input_ids"]
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    print(f"driver={args.driver}  max_tokens={args.max_tokens}  layer={args.layer}")
    print(f"{'N':>5} {'s/step':>10} {'ms/tok':>10} {'rel(N=1)':>10}")
    base_ms = None
    for n_sample in ns:
        np_cfg = dict(n_sample=n_sample, max_tokens=args.max_tokens,
                      global_seed=42, sigma=args.sigma,
                      sample_method=args.sample_method)
        # warmup (esp. graphed_cuda: pays the one-time capture)
        _run(llm, args.driver, prompt_ids, args.layer, np_cfg)
        torch.cuda.synchronize()
        t0 = time.time()
        out = _run(llm, args.driver, prompt_ids, args.layer, np_cfg)
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_tok = max(len(out["clean_tokens"]), 1)
        ms_tok = dt / n_tok * 1e3
        if base_ms is None:
            base_ms = ms_tok
        print(f"{n_sample:>5} {dt:>10.3f} {ms_tok:>10.3f} {ms_tok/base_ms:>10.2f}x")


if __name__ == "__main__":
    main()
