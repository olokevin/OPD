"""Throughput/memory/util grid for the packed NP decode driver.

For each (batch_size, n_sample) cell, decode `batch_size` prompts via the packed
driver in waves of `pack_width`, at max_tokens, and report:
  s/step (whole batch, one layer), tok/s (clean tokens decoded / s),
  peak GPU mem (torch.cuda.max_memory_allocated), and mean SM util sampled via
  nvidia-smi during the run.

This measures the STUDENT decode only (no teacher scoring / no assemble) so the
packing effect is isolated. End-to-end step timing incl. teacher+assemble is
measured separately by bench_np_vs_bp.sh (Task 11).

Usage:
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/bench_packed_grid.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --max-tokens 1024 \
      --batches 1,2,4,8,16 --rails 8,16,64 --pack-width 8
"""
import argparse
import os
import subprocess
import threading
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


class _UtilSampler(threading.Thread):
    """Polls nvidia-smi for this process's visible GPU SM-util while running."""
    def __init__(self, gpu_index, interval=0.1):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                    text=True, timeout=2).strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def _make_prompts(tok, n):
    base = [
        "Solve for x: 2x+3=7. Answer:", "Compute the integral of x^2. Answer:",
        "What is 12*13? Answer:", "Differentiate sin(x). Answer:",
        "Factor x^2-9. Answer:", "What is 100/4? Answer:",
        "Sum 1..10. Answer:", "Square root of 144? Answer:",
    ]
    out = []
    for i in range(n):
        out.append(tok(base[i % len(base)] + f" ({i})")["input_ids"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--rails", default="8,16,64")
    ap.add_argument("--pack-width", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    args = ap.parse_args()

    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    batches = [int(x) for x in args.batches.split(",")]
    rails = [int(x) for x in args.rails.split(",")]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem_util)
    tok = llm.get_tokenizer()
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    print(f"# packed grid: model={args.model} layer={args.layer} "
          f"max_tokens={args.max_tokens} pack_width={args.pack_width}")
    print(f"{'batch':>6} {'rails':>6} {'s/step':>9} {'tok/s':>9} "
          f"{'peakGB':>8} {'SM%':>6}")
    for n_sample in rails:
        for batch in batches:
            pids = _make_prompts(tok, batch)
            rollout_ids = list(range(batch))
            np_cfg = dict(n_sample=n_sample, max_tokens=args.max_tokens,
                          global_seed=42, sigma=args.sigma,
                          sample_method="bernoulli")
            try:
                # warmup one small wave
                _ = llm.collective_rpc(
                    "run_np_decode_packed",
                    args=(pids[: min(args.pack_width, batch)],
                          SamplingParams(temperature=0.0), args.layer,
                          dict(np_cfg, max_tokens=8),
                          rollout_ids[: args.pack_width]))[0]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                sampler = _UtilSampler(gpu_index)
                sampler.start()
                t0 = time.time()
                total_tok = 0
                for w0 in range(0, batch, args.pack_width):
                    out = llm.collective_rpc(
                        "run_np_decode_packed",
                        args=(pids[w0:w0 + args.pack_width],
                              SamplingParams(temperature=0.0), args.layer, np_cfg,
                              rollout_ids[w0:w0 + args.pack_width]))[0]
                    total_tok += sum(len(ct) for ct in out["clean_tokens"])
                torch.cuda.synchronize()
                dt = time.time() - t0
                sampler.stop()
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                tok_s = total_tok / dt if dt > 0 else 0.0
                print(f"{batch:>6} {n_sample:>6} {dt:>9.3f} {tok_s:>9.1f} "
                      f"{peak_gb:>8.2f} {sampler.mean():>6.1f}")
            except AssertionError as e:
                print(f"{batch:>6} {n_sample:>6} {'OOM/KV-fit':>9} "
                      f"(pack_width={args.pack_width}: {str(e)[:60]})")


if __name__ == "__main__":
    main()
