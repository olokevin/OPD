"""Phase 0 -- hardware calibration (opd_profile_plan.md §21).

Measures, on the current GPU:
  * sustainable HBM bandwidth (device copy, read-only reduction)
  * BF16 GEMM TFLOP/s at the Qwen3-1.7B linear shapes vs M (= rows in flight)
    -> where each shape leaves the weight-bandwidth floor (the measured ridge)
  * decode-attention KV bandwidth (FA3 paged decode, B=64, L=4096)

    CUDA_VISIBLE_DEVICES=0 python scripts/zo_opd/es_profile/phase0_calibrate.py
"""
import argparse
import math

import torch

from common import (H100_NVL_SPEC, QWEN3_1P7B_ATTN, QWEN3_1P7B_LINEARS,
                    cuda_time, gpu_info, md_table, save_json)


def bench_hbm(dev):
    out = {}
    n = 2 * 2**30 // 2  # 2 GiB of bf16
    x = torch.empty(n, dtype=torch.bfloat16, device=dev).normal_()
    y = torch.empty_like(x)
    mn, md = cuda_time(lambda: y.copy_(x), warmup=5, iters=30)
    out["copy_gbps"] = 2 * n * 2 / (mn * 1e-3) / 1e9
    out["copy_gbps_median"] = 2 * n * 2 / (md * 1e-3) / 1e9
    xf = x.view(torch.float32) if False else x
    mn, md = cuda_time(lambda: xf.float().sum() if False else torch.sum(xf, dtype=torch.float32),
                       warmup=5, iters=30)
    out["read_gbps"] = n * 2 / (mn * 1e-3) / 1e9
    out["read_gbps_median"] = n * 2 / (md * 1e-3) / 1e9
    del x, y
    return out


def bench_gemm(dev, shapes, Ms):
    rows = []
    for name, d_in, d_out in shapes:
        W = torch.randn(d_out, d_in, dtype=torch.bfloat16, device=dev)
        w_bytes = W.numel() * 2
        for M in Ms:
            X = torch.randn(M, d_in, dtype=torch.bfloat16, device=dev)
            mn, md = cuda_time(lambda: torch.matmul(X, W.t()), warmup=10,
                               iters=100)
            flops = 2.0 * M * d_in * d_out
            act_bytes = (M * d_in + M * d_out) * 2
            rows.append(dict(shape=name, d_in=d_in, d_out=d_out, M=M,
                             ms_min=mn, ms_median=md,
                             tflops=flops / (mn * 1e-3) / 1e12,
                             gbps=(w_bytes + act_bytes) / (mn * 1e-3) / 1e9,
                             ai=flops / (w_bytes + act_bytes)))
        del W
    return rows


def bench_attn_decode(dev, B=64, L=4096, block_size=16):
    """FA3 paged decode as vLLM calls it: q [B, n_q, D] vs KV cache pages."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    a = QWEN3_1P7B_ATTN
    n_q, n_kv, D = a["n_q"], a["n_kv"], a["head_dim"]
    blocks_per_seq = math.ceil(L / block_size)
    nblk = B * blocks_per_seq + 1
    k_cache = torch.randn(nblk, block_size, n_kv, D, dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn_like(k_cache)
    bt = torch.arange(1, nblk, dtype=torch.int32, device=dev).view(B, blocks_per_seq)
    q = torch.randn(B, n_q, D, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(q)
    cu = torch.arange(B + 1, dtype=torch.int32, device=dev)
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    res = {}
    for fav in (3, 2):
        def fn():
            flash_attn_varlen_func(q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu,
                                   max_seqlen_q=1, seqused_k=sl, max_seqlen_k=L,
                                   softmax_scale=D ** -0.5, causal=True, block_table=bt,
                                   fa_version=fav)
        mn, md = cuda_time(fn, warmup=5, iters=50)
        kv_bytes = B * L * n_kv * D * 2 * 2
        res[f"fa{fav}"] = dict(ms_min=mn, ms_median=md,
                               kv_gbps=kv_bytes / (mn * 1e-3) / 1e9)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", default="1,8,16,32,64,96,128,192,256,384,512,768,1024,2048,4096")
    args = ap.parse_args()
    dev = torch.device("cuda")
    Ms = [int(m) for m in args.Ms.split(",")]
    info = gpu_info()
    print("[gpu]", info, flush=True)

    hbm = bench_hbm(dev)
    print(f"[hbm] copy {hbm['copy_gbps']:.0f} GB/s (median {hbm['copy_gbps_median']:.0f}), "
          f"read {hbm['read_gbps']:.0f} GB/s (median {hbm['read_gbps_median']:.0f}); "
          f"spec {H100_NVL_SPEC['hbm_tbps']*1e3:.0f}", flush=True)

    gemm = bench_gemm(dev, QWEN3_1P7B_LINEARS, Ms)
    peak = max(r["tflops"] for r in gemm)
    bw = hbm["copy_gbps"]
    ridge = peak * 1e12 / (bw * 1e9)
    print(f"[gemm] peak measured {peak:.0f} TFLOP/s -> measured ridge "
          f"{ridge:.0f} FLOP/byte (spec {H100_NVL_SPEC['bf16_tflops']*1e12/(H100_NVL_SPEC['hbm_tbps']*1e12):.0f})",
          flush=True)
    for name, _, _ in QWEN3_1P7B_LINEARS:
        rs = [r for r in gemm if r["shape"] == name]
        print(f"  {name}: " + "  ".join(f"M={r['M']}:{r['ms_min']:.3f}ms/{r['tflops']:.0f}TF"
                                       for r in rs), flush=True)

    attn = bench_attn_decode(dev)
    print(f"[attn decode B=64 L=4096] fa3 {attn['fa3']['ms_min']:.3f} ms "
          f"({attn['fa3']['kv_gbps']:.0f} GB/s KV)  fa2 {attn['fa2']['ms_min']:.3f} ms "
          f"({attn['fa2']['kv_gbps']:.0f} GB/s)", flush=True)

    rec = dict(gpu=info, hbm=hbm, gemm=gemm, gemm_peak_tflops=peak,
               ridge_measured=ridge, attn_decode=attn, spec=H100_NVL_SPEC)
    save_json("phase0_calibrate.json", rec)

    # md summary
    hdr = ["shape"] + [f"M={m}" for m in Ms]
    rows = []
    for name, d_in, d_out in QWEN3_1P7B_LINEARS:
        rs = {r["M"]: r for r in gemm if r["shape"] == name}
        rows.append([f"{name} {d_in}->{d_out}"] + [f"{rs[m]['ms_min']:.3f} / {rs[m]['tflops']:.0f}" for m in Ms])
    print("\n" + md_table(hdr, rows))


if __name__ == "__main__":
    main()
