"""Kernel-level gates for rail_seq_kernels.py (es-decode held rails):
  bits generation (antithetic inversion, determinism), full-rank bit rail vs a
  torch +-1 GEMV, standalone low-rank rail vs torch, both update kernels vs
  torch, and timing of the full-rank rail at Qwen2-1.5B / Qwen3-1.7B shapes.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_token_checks/check_seq_kernels.py
"""
import argparse
import math
import time

import torch

from verl.trainer.es_token.rail_seq_kernels import (
    SeqNoise, bits_update, lowrank_update, rademacher_bits, rail_bits, rail_lowrank,
    unpack_bits)


def stats(a, b):
    d = (a.float() - b.float()).abs()
    return f"max|d|={d.max().item():.3e} mean={d.mean().item():.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--bucket", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    N, bucket = args.N, args.bucket
    width = 1 + N
    R = bucket * width
    ok_all = True

    def gate(name, out, ref, tol):
        nonlocal ok_all
        d = (out.float() - ref.float()).abs().max().item()
        ok = d <= tol
        ok_all &= ok
        print(f"[{name}] {stats(out, ref)} {'PASS' if ok else 'FAIL'} (tol {tol})", flush=True)

    # Qwen2-1.5B down_proj / gate_up shapes + Qwen3-1.7B qkv
    layout = {}
    off = 0
    for name, d_out, d_in in [("down", 1536, 8960), ("gate_up", 17920, 1536), ("qkv", 4096, 2048)]:
        layout[name] = (off, d_out, off + d_out, d_in)
        off += d_out + d_in
    sigma = torch.full((1,), 1e-3, device=dev, dtype=dt)

    # ---- 1. bits: determinism + antithetic ---------------------------------
    sn = SeqNoise(layout, N, "full", dev, dt)
    seeds = [1234 + i for i in range(N // 2)]
    sn.draw(seeds, antithetic=True)
    b0 = sn.bits["down"].clone()
    sn.draw(seeds, antithetic=True)
    gate("bits deterministic", sn.bits["down"], b0, 0.0)
    inv = (sn.bits["down"][0::2] ^ -1)
    gate("bits antithetic (odd = ~even)", sn.bits["down"][1::2], inv, 0.0)
    e0 = unpack_bits(sn.bits["down"][0], 1536, 8960)
    print(f"[bits] mean sign rail0 = {e0.mean().item():+.4f} (expect ~0), frac +1 = {(e0 > 0).float().mean().item():.4f}", flush=True)
    sn2 = SeqNoise(layout, N, "full", dev, dt)
    sn2.draw(seeds, antithetic=False)
    print(f"[bits] non-antithetic rails 0/1 differ: {(sn2.bits['down'][0] != sn2.bits['down'][1]).float().mean().item():.3f} of words", flush=True)

    # ---- 2. full-rank rail vs torch ----------------------------------------
    for name in ("down", "gate_up"):
        _, d_out, _, d_in = layout[name]
        x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
        y = (torch.randn(R, d_out, device=dev) * 0.3).to(dt)
        y_ref = y.clone().float()
        for p in range(bucket):
            for n in range(N):
                row = p * width + 1 + n
                eps = unpack_bits(sn.bits[name][n], d_out, d_in)          # [d_out, d_in]
                y_ref[row] += float(sigma) * (eps @ x[row].float())
        y_ref = y_ref.to(dt)
        y_out = y.clone()
        sn.apply_rail(name, x, y_out, sigma, width, bucket)
        gate(f"rail_bits {name} (rail rows)", y_out[[r for r in range(R) if r % width]], y_ref[[r for r in range(R) if r % width]], 2e-2)
        gate(f"rail_bits {name} (clean rows untouched)", y_out[::width], y[::width], 0.0)

    # ---- 3. low-rank rail vs torch -----------------------------------------
    for rank in (1, 4, 16):
        snl = SeqNoise(layout, N, rank, dev, dt)
        snl.draw(seeds, antithetic=True)
        _, d_out, _, d_in = layout["down"]
        off_a, off_b = snl.offsets("down")
        x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
        y = (torch.randn(R, d_out, device=dev) * 0.3).to(dt)
        y_ref = y.clone().float()
        for p in range(bucket):
            for n in range(N):
                row = p * width + 1 + n
                A = snl.noise[n, off_a:off_a + rank * d_out].view(rank, d_out).float()
                B = snl.noise[n, off_b:off_b + rank * d_in].view(rank, d_in).float()
                y_ref[row] += float(sigma) / math.sqrt(rank) * (A.t() @ (B @ x[row].float()))
        y_ref = y_ref.to(dt)
        y_out = y.clone()
        snl.apply_rail("down", x, y_out, sigma, width, bucket)
        gate(f"rail_lowrank r={rank}", y_out, y_ref, 2e-2)
        # antithetic: rail 1's A == -rail 0's A, B identical
        A0 = snl.noise[0, off_a:off_a + rank * d_out]
        A1 = snl.noise[1, off_a:off_a + rank * d_out]
        B0 = snl.noise[0, off_b:off_b + rank * d_in]
        B1 = snl.noise[1, off_b:off_b + rank * d_in]
        gate(f"lowrank r={rank} antithetic", torch.cat([A1 + A0, B1 - B0]), torch.zeros(rank * (d_out + d_in), device=dev, dtype=dt), 0.0)

    # ---- 4. updates vs torch -----------------------------------------------
    coef = torch.randn(N, device=dev) * 1e-3
    _, d_out, _, d_in = layout["down"]
    ref = torch.zeros(d_out, d_in, device=dev)
    for n in range(N):
        ref += coef[n] * unpack_bits(sn.bits["down"][n], d_out, d_in)
    out = sn.update("down", coef)
    gate("bits_update", out, ref, 1e-6)
    snl = SeqNoise(layout, N, 4, dev, dt)
    snl.draw(seeds, antithetic=True)
    off_a, off_b = snl.offsets("down")
    ref = torch.zeros(d_out, d_in, device=dev)
    for n in range(N):
        A = snl.noise[n, off_a:off_a + 4 * d_out].view(4, d_out).float()
        B = snl.noise[n, off_b:off_b + 4 * d_in].view(4, d_in).float()
        ref += coef[n] / 2.0 * (A.t() @ B)
    out = snl.update("down", coef)
    gate("lowrank_update r=4", out, ref, 1e-4)

    # ---- 5. timing of the full-rank rail (one token, all rails) ------------
    for N_t in (8, 32):
        for bucket_t in (1, 8):
            width_t = 1 + N_t
            R_t = bucket_t * width_t
            snt = SeqNoise({"down": layout["down"], "gate_up": layout["gate_up"]}, N_t, "full", dev, dt)
            snt.draw([7 + i for i in range(N_t // 2)], antithetic=True)
            xs = {nm: (torch.randn(R_t, layout[nm][3], device=dev) * 0.5).to(dt) for nm in ("down", "gate_up")}
            ys = {nm: torch.zeros(R_t, layout[nm][1], device=dev, dtype=dt) for nm in ("down", "gate_up")}
            def run():
                for nm in ("down", "gate_up"):
                    snt.apply_rail(nm, xs[nm], ys[nm], sigma, width_t, bucket_t)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                run()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            n_bytes = sum(snt.bits[nm].numel() * 4 for nm in snt.bits)
            params = sum(layout[nm][1] * layout[nm][3] for nm in ("down", "gate_up"))
            print(f"[timing full-rank rail] N={N_t:2d} bucket={bucket_t}: down+gate_up {ms:.3f} ms/token "
                  f"({n_bytes/1e6:.0f} MB bits -> {n_bytes/ms/1e6:.0f} GB/s; {params*N_t*bucket_t*2/ms/1e9:.1f} TFLOP/s eff)", flush=True)
            del snt
    # ---- 6. fused consumer kernels in held rank-r mode vs standalone + vLLM op --
    from vllm import _custom_ops as ops
    from verl.trainer.es_token.fused_rail_kernels import (
        RailArgs, norm_rail, qkv_rail_norm_rope, silu_mul_rail)
    EPS = 1e-6
    for rank in (1, 4, 8):
        lay = {"o": (0, 1536, 1536, 1536), "gu": (3072, 2 * 8960, 3072 + 2 * 8960, 1536),
               "qkv3": (3072 + 2 * 8960 + 1536, 4096, 3072 + 2 * 8960 + 1536 + 4096, 2048),
               "qkv2": (3072 + 2 * 8960 + 1536 + 4096 + 2048, 2048, 3072 + 2 * 8960 + 1536 + 4096 + 2048 + 2048, 1536)}
        snf = SeqNoise(lay, N, rank, dev, dt)
        snf.draw(seeds, antithetic=True)
        # norm (o_proj -> post_attention_layernorm), H=1536 (Qwen2, non-pow2)
        _, d_out, _, d_in = lay["o"]
        off_a, off_b = snf.offsets("o")
        x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
        y = (torch.randn(R, d_out, device=dev) * 0.3).to(dt)
        res = (torch.randn(R, d_out, device=dev) * 2.0).to(dt)
        w = (1 + 0.1 * torch.randn(d_out, device=dev)).to(dt)
        y_ref, res_ref = y.clone(), res.clone()
        rail_lowrank(x, y_ref, snf.noise, sigma, off_a, off_b, rank, width)
        ops.fused_add_rms_norm(y_ref, res_ref, w, EPS)
        y_out, res_out = y.clone(), res.clone()
        norm_rail(y_out, res_out, w, EPS, RailArgs(x, snf.noise, snf.noise, sigma, off_a, off_b, d_in, width, rank=rank))
        gate(f"fused norm rank={rank}: residual", res_out, res_ref, 0.0)
        gate(f"fused norm rank={rank}: normed", y_out, y_ref, 2e-2)
        # silu (gate_up), I = 8960
        _, d_out, _, d_in = lay["gu"]
        off_a, off_b = snf.offsets("gu")
        I = d_out // 2
        x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
        gu = (torch.randn(R, d_out, device=dev) * 1.0).to(dt)
        gu_ref = gu.clone()
        rail_lowrank(x, gu_ref, snf.noise, sigma, off_a, off_b, rank, width)
        out_ref = torch.empty(R, I, device=dev, dtype=dt)
        torch.ops._C.silu_and_mul(out_ref, gu_ref)
        out = torch.empty(R, I, device=dev, dtype=dt)
        silu_mul_rail(gu, out, RailArgs(x, snf.noise, snf.noise, sigma, off_a, off_b, d_in, width, rank=rank))
        gate(f"fused silu rank={rank}", out, out_ref, 3e-2)
        # qkv, Qwen3 dims (16/8/128, q/k norm) and Qwen2 dims (12/2/128, no norm)
        for tag, HQ, HKV, D, has_norm in (("qkv3", 16, 8, 128, True), ("qkv2", 12, 2, 128, False)):
            _, d_out, _, d_in = lay[tag]
            off_a, off_b = snf.offsets(tag)
            x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
            qkv = (torch.randn(R, d_out, device=dev) * 0.8).to(dt)
            qw = (1 + 0.1 * torch.randn(D, device=dev)).to(dt)
            kw = (1 + 0.1 * torch.randn(D, device=dev)).to(dt)
            pos = torch.randint(0, 4000, (R,), device=dev, dtype=torch.long)
            inv_freq = 1.0 / (1000000 ** (torch.arange(0, D, 2, device=dev).float() / D))
            fr = torch.outer(torch.arange(4096, device=dev).float(), inv_freq)
            cs = torch.cat([fr.cos(), fr.sin()], -1).to(dt)
            qkv_ref = qkv.clone()
            rail_lowrank(x, qkv_ref, snf.noise, sigma, off_a, off_b, rank, width)
            q, k, v = qkv_ref.split([HQ * D, HKV * D, HKV * D], dim=-1)
            q = q.contiguous(); k = k.contiguous()
            if has_norm:
                qn = torch.empty_like(q.view(-1, D)); ops.rms_norm(qn, q.view(-1, D), qw, EPS); q = qn.view(R, HQ * D)
                kn = torch.empty_like(k.view(-1, D)); ops.rms_norm(kn, k.view(-1, D), kw, EPS); k = kn.view(R, HKV * D)
            ops.rotary_embedding(pos, q, k, D, cs, True)
            qkv_out = qkv.clone()
            qkv_rail_norm_rope(qkv_out, pos, qw if has_norm else None, kw if has_norm else None, EPS, cs, HQ, HKV, D,
                               RailArgs(x, snf.noise, snf.noise, sigma, off_a, off_b, d_in, width, rank=rank), has_norm=has_norm)
            qo, ko, _ = qkv_out.split([HQ * D, HKV * D, HKV * D], dim=-1)
            gate(f"fused qkv rank={rank} {tag}: q", qo, q, 3e-2)
            gate(f"fused qkv rank={rank} {tag}: k", ko, k, 3e-2)
    print("ALL_PASS" if ok_all else "SOME_FAIL")


if __name__ == "__main__":
    main()
