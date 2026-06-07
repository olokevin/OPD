"""Timing isolation: how much of the packed_graphed all-layer DECODE cost is the
per-token noise refill (896 host-orchestrated draw_noise calls/token for 28 layers
x pack_width x n_sample)?

Runs run_np_decode_packed_graphed TWICE on the SAME worker, same prompts, same
config, all 28 down_proj layers:
  (A) refill ON  -- the production path (per-token _np_fill_u_buf_all_layers_packed).
  (B) refill OFF -- NP_BENCH_SKIP_NOISE=1: noise pre-filled ONCE before the loop,
      per-token refill skipped. Everything else identical (replay, per-token sync,
      D2H u/x captures, compute_logits, topk). Isolates the noise-refill cost.

Reports per-token decode ms for each and the speedup, so we can see how much of
the measured 1368s decode is the 896-kernel/token noise refill vs the rest of the
per-token host glue (sync + 28 D2H captures + full-vocab logits/topk).

NP_BENCH_SKIP_NOISE BREAKS gradient correctness (stale noise) -- this is a
wall-clock attribution harness ONLY.

Usage (one free GPU among 1/2/3):
  cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
  PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=1 NP_KEEP_CUDA_VISIBLE=1 \
      /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/bench_noise_refill_isolation.py \
      --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""
import argparse
import os
import re

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: F401
from vllm import LLM, SamplingParams


def _time_decode_on_worker(self, prompt_ids_list, layer_names, n_sample,
                           max_tokens, sigma, global_seed, sample_method,
                           rollout_ids, skip_noise):
    """Run ONE run_np_decode_packed_graphed and return wall-clock + token counts.
    skip_noise toggles NP_BENCH_SKIP_NOISE on the worker for this call."""
    import time
    if skip_noise:
        os.environ["NP_BENCH_SKIP_NOISE"] = "1"
    else:
        os.environ.pop("NP_BENCH_SKIP_NOISE", None)

    sp = SamplingParams(temperature=0.0)
    np_cfg = dict(n_sample=int(n_sample), max_tokens=int(max_tokens),
                  global_seed=int(global_seed), sigma=float(sigma),
                  sample_method=sample_method)

    # warm up one short decode so capture + cuBLAS settle (not timed).
    _ = self.run_np_decode_packed_graphed(
        prompt_ids_list, sp, list(layer_names), np_cfg, list(rollout_ids))

    torch.cuda.synchronize()
    t0 = time.time()
    out = self.run_np_decode_packed_graphed(
        prompt_ids_list, sp, list(layer_names), np_cfg, list(rollout_ids))
    torch.cuda.synchronize()
    dt = time.time() - t0

    B = len(prompt_ids_list)
    tok_counts = [len(out["clean_tokens"][p]) for p in range(B)]
    total_tokens = sum(tok_counts)
    return {
        "dt": dt,
        "tok_counts": tok_counts,
        "total_tokens": total_tokens,
        "skip_noise": bool(skip_noise),
        "n_layers": len(layer_names),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)       # one wave at pack_width=4
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--n-layers", type=int, default=28,
                    help="number of down_proj layers to perturb (all-layer = 28)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.7)
    args = ap.parse_args()

    import verl.workers.rollout.vllm_rollout.np_worker_extension as npx
    print("np_worker_extension.__file__ =", npx.__file__)
    assert "np-alllayer-graphed" in npx.__file__, (
        "STALE verl loaded -- prepend PYTHONPATH=$PWD/verl. "
        f"got {npx.__file__}")

    prompts = [
        "Compute 7*8 step by step. Answer:",
        "Differentiate x^3 + 2x with steps. Answer:",
        "What is 100/4? Show work. Answer:",
        "Square root of 144 explained? Answer:",
    ][: args.n_prompts]
    B = len(prompts)

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem_util)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]

    # Resolve the first n_layers down_proj layer names from the real model.
    names = llm.collective_rpc(
        lambda self: [n for n, _ in self.model_runner.model.named_modules()
                      if re.fullmatch(r"model\.layers\.\d+\.mlp\.down_proj", n)])[0]
    names = sorted(names, key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1)))
    layers = names[: args.n_layers]
    print(f"perturbing {len(layers)} layers: {layers[0]} .. {layers[-1]}")
    llm.collective_rpc("install_perturb_layers", args=(layers,))

    rollout_ids = list(range(B))

    # (A) refill ON, then (B) refill OFF, same worker/prompts/config.
    on = llm.collective_rpc(
        _time_decode_on_worker,
        args=(pids, layers, args.n_sample, args.max_tokens, args.sigma,
              args.global_seed, args.sample_method, rollout_ids, False))[0]
    off = llm.collective_rpc(
        _time_decode_on_worker,
        args=(pids, layers, args.n_sample, args.max_tokens, args.sigma,
              args.global_seed, args.sample_method, rollout_ids, True))[0]

    # max token count drives the #decode-steps (one graph replay per step).
    steps_on = max(on["tok_counts"])
    steps_off = max(off["tok_counts"])
    ms_tok_on = on["dt"] / max(steps_on, 1) * 1e3
    ms_tok_off = off["dt"] / max(steps_off, 1) * 1e3
    refills_per_tok = len(layers) * B * args.n_sample

    print("\n========================================================================")
    print(" NOISE-REFILL ISOLATION  (packed_graphed all-layer decode)")
    print("========================================================================")
    print(f" config: {len(layers)} layers x {B} prompts x n_sample={args.n_sample}, "
          f"max_tokens={args.max_tokens}, one wave (pack_width={B})")
    print(f" per-token noise refill does {len(layers)}*{B}*{args.n_sample} "
          f"= {refills_per_tok} draw_noise calls/token")
    print("")
    print(f" (A) refill ON  : decode {on['dt']:.3f}s  over {steps_on} steps "
          f"-> {ms_tok_on:.2f} ms/token-step")
    print(f" (B) refill OFF : decode {off['dt']:.3f}s  over {steps_off} steps "
          f"-> {ms_tok_off:.2f} ms/token-step")
    if ms_tok_off > 0:
        print(f" speedup (ON/OFF)        : {ms_tok_on / ms_tok_off:.1f}x")
    refill_ms = ms_tok_on - ms_tok_off
    if ms_tok_on > 0:
        print(f" noise-refill share      : {refill_ms:.2f} ms/token "
              f"({100*refill_ms/ms_tok_on:.0f}% of refill-ON token time)")
    print(f" residual (non-refill)   : {ms_tok_off:.2f} ms/token "
          f"(replay + per-token sync + {len(layers)} D2H u/x captures + compute_logits + topk)")
    print("========================================================================")
    print(" INTERPRETATION: refill-OFF is the decode cost WITHOUT the per-token")
    print(" noise regeneration; the gap (A-B) is the noise-refill tax. If A>>B the")
    print(" 1368s production decode is dominated by the 896-kernel/token refill and")
    print(" batching/pre-generating noise is the fix.")
    print("========================================================================")


if __name__ == "__main__":
    main()
