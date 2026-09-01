"""Shared helpers for the es_token parallel-rail efficiency profile
(docs/results/ZO_OPD/opd_profile_plan.md).

Timing: CUDA events around a callable, `warmup` untimed iterations then
`iters` timed ones; returns the MIN and MEDIAN per-call ms. On a shared GPU
(other jobs resident) the min is the robust kernel-time estimate; the median
shows the contention. Every script records both.
"""
import json
import os
import platform
import subprocess
import time

import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Qwen3-1.7B decoder linears (vLLM fused names) -- (name, d_in, d_out).
QWEN3_1P7B_LINEARS = [
    ("qkv_proj", 2048, 4096),
    ("o_proj", 2048, 2048),
    ("gate_up_proj", 2048, 12288),
    ("down_proj", 6144, 2048),
]
QWEN3_1P7B_ATTN = dict(num_layers=28, hidden=2048, n_q=16, n_kv=8, head_dim=128,
                       vocab=151936)

# H100 NVL datasheet (per GPU): dense BF16 tensor 835 TFLOP/s (1671 w/ sparsity),
# HBM3 3.94 TB/s. Ridge ~212 FLOP/byte. Phase 0 MEASURES both; these are only
# the nominal reference printed next to the measurement.
H100_NVL_SPEC = dict(bf16_tflops=835.0, hbm_tbps=3.94)


def cuda_time(fn, warmup=10, iters=100, sync_each=False):
    """Return (min_ms, median_ms) of fn() over `iters` timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[0], times[len(times) // 2]


def cuda_time_graphed(fn, warmup=3, iters=50):
    """Capture fn() into a CUDA graph and time replays: removes launch
    overhead so the number is GPU time only (what the decode graph sees)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    del g
    return times[0], times[len(times) // 2]


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    info = dict(
        name=p.name, sms=p.multi_processor_count,
        total_mem_gb=round(p.total_memory / 2**30, 1),
        cc=f"{p.major}.{p.minor}", torch=torch.__version__,
        cuda=torch.version.cuda, host=platform.node(),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.max.sm,"
             "clocks.max.mem,power.limit,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = q.stdout.strip().splitlines()
    except Exception:
        pass
    return info


def save_json(name, rec):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[saved] {path}", flush=True)
    return path


def md_table(headers, rows, fmt=None):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            if isinstance(c, float):
                f = (fmt or {}).get(headers[i], "{:.3f}")
                cells.append(f.format(c))
            else:
                cells.append(str(c))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def next_pow2(x):
    return 1 << (int(x) - 1).bit_length()
