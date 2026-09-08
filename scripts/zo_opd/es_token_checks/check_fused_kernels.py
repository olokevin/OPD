"""Kernel-level gates for fused_rail_kernels.py: each fused kernel against the
exact chain it replaces (shipping rail op `apply_rail` -> the vLLM CUDA op),
on random data in Qwen3-1.7B shapes, plus the in-graph step helpers against
torch references.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_token_checks/check_fused_kernels.py
"""
import argparse

import torch
from vllm import _custom_ops as ops

from verl.trainer.es_token.fused_rail_kernels import (
    RailArgs, advance, fill_rademacher_rows_t, lm_tail, norm_rail,
    qkv_rail_norm_rope, silu_mul_rail)
from verl.trainer.es_token.lm_head_kernel import LMHeadWorkspace, lm_head_stream
from verl.trainer.es_token.noise_kernel import fill_rademacher_rows
from verl.trainer.es_token.rail_kernel import apply_rail

H, HQ, HKV, D, I, V = 2048, 16, 8, 128, 6144, 151936
EPS = 1e-6


def rail_idx(bucket, width, device):
    R = bucket * width
    rows = torch.arange(R, device=device)
    pert = rows[rows % width != 0]
    return pert, (pert % width - 1), (pert // width)


def ref_rail(x, y, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in):
    y = y.clone()
    apply_rail(x, y, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in,
               noise.stride(0), signs.stride(0))
    return y


def stats(a, b):
    d = (a.float() - b.float()).abs()
    return f"max|d|={d.max().item():.3e} mean={d.mean().item():.2e} ulp-mismatch={(d > 0).float().mean().item()*100:.3f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=2)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    bucket, N = args.bucket, args.N
    width = 1 + N
    R = bucket * width
    pri, rail, pidx = rail_idx(bucket, width, dev)
    # flat noise / sign buffers covering (u, v) for one o_proj-like layer and
    # a down_proj-like layer and a gate_up-like and qkv-like layer
    layout = {}
    off = 0
    for name, d_out, d_in in [("qkv", HQ * D + 2 * HKV * D, H), ("o", H, H),
                              ("gu", 2 * I, H), ("down", H, I)]:
        layout[name] = (off, d_out, off + d_out, d_in)
        off += d_out + d_in
    d_total = off
    noise = torch.empty(bucket, d_total, device=dev, dtype=dt)
    fill_rademacher_rows(noise, [11, 22, 33, 44][:bucket])
    signs = torch.empty(N, d_total, device=dev, dtype=dt)
    fill_rademacher_rows(signs, [100 + n for n in range(N)])
    sigma = torch.full((1,), args.sigma, device=dev, dtype=dt)
    ok_all = True

    def gate(name, out, ref, tol):
        nonlocal ok_all
        d = (out.float() - ref.float()).abs().max().item()
        ok = d <= tol
        ok_all &= ok
        print(f"[{name}] {stats(out, ref)} {'PASS' if ok else 'FAIL'} (tol {tol})", flush=True)

    # ---- 1. norm_rail (o_proj -> post_attention_layernorm) --------------- #
    off_u, d_out, off_v, d_in = layout["o"]
    x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
    y = (torch.randn(R, d_out, device=dev) * 0.3).to(dt)
    res = (torch.randn(R, H, device=dev) * 2.0).to(dt)
    w = (1 + 0.1 * torch.randn(H, device=dev)).to(dt)
    y_ref = ref_rail(x, y, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in)
    res_ref = res.clone()
    ops.fused_add_rms_norm(y_ref, res_ref, w, EPS)
    y_out, res_out = y.clone(), res.clone()
    norm_rail(y_out, res_out, w, EPS, RailArgs(x, noise, signs, sigma, off_u, off_v, d_in, width))
    gate("norm_rail o_proj: residual", res_out, res_ref, 0.0)
    gate("norm_rail o_proj: normed", y_out, y_ref, 2e-2)
    # down_proj (d_in = 6144) into the next input_layernorm
    off_u, d_out, off_v, d_in = layout["down"]
    x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
    y_ref = ref_rail(x, y, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in)
    res_ref = res.clone()
    ops.fused_add_rms_norm(y_ref, res_ref, w, EPS)
    y_out, res_out = y.clone(), res.clone()
    norm_rail(y_out, res_out, w, EPS, RailArgs(x, noise, signs, sigma, off_u, off_v, d_in, width))
    gate("norm_rail down_proj: residual", res_out, res_ref, 0.0)
    gate("norm_rail down_proj: normed", y_out, y_ref, 2e-2)
    # no rail, with residual (N=0 path) and clean rows equal the rail-free op
    y_ref, res_ref = y.clone(), res.clone()
    ops.fused_add_rms_norm(y_ref, res_ref, w, EPS)
    y_out, res_out = y.clone(), res.clone()
    norm_rail(y_out, res_out, w, EPS, None)
    gate("norm no-rail: residual", res_out, res_ref, 0.0)
    gate("norm no-rail: normed", y_out, y_ref, 2e-2)

    # ---- 2. qkv rail + q/k norm + rope ---------------------------------- #
    off_u, d_out, off_v, d_in = layout["qkv"]
    x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
    qkv = (torch.randn(R, d_out, device=dev) * 0.8).to(dt)
    qw = (1 + 0.1 * torch.randn(D, device=dev)).to(dt)
    kw = (1 + 0.1 * torch.randn(D, device=dev)).to(dt)
    pos = torch.randint(0, 4000, (R,), device=dev, dtype=torch.long)
    # vLLM cos_sin_cache layout [max_pos, D] = [cos(D/2) | sin(D/2)], bf16
    inv_freq = 1.0 / (1000000 ** (torch.arange(0, D, 2, device=dev).float() / D))
    t = torch.arange(4096, device=dev).float()
    freqs = torch.outer(t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).to(dt)
    qkv_ref = ref_rail(x, qkv, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in)
    q, k, v = qkv_ref.split([HQ * D, HKV * D, HKV * D], dim=-1)
    qn = torch.empty_like(q.contiguous().view(-1, D))
    ops.rms_norm(qn, q.contiguous().view(-1, D), qw, EPS)
    kn = torch.empty_like(k.contiguous().view(-1, D))
    ops.rms_norm(kn, k.contiguous().view(-1, D), kw, EPS)
    qn = qn.view(R, HQ * D).contiguous()
    kn = kn.view(R, HKV * D).contiguous()
    ops.rotary_embedding(pos, qn, kn, D, cos_sin, True)
    qkv_out = qkv.clone()
    qkv_rail_norm_rope(qkv_out, pos, qw, kw, EPS, cos_sin, HQ, HKV, D,
                       RailArgs(x, noise, signs, sigma, off_u, off_v, d_in, width))
    qo, ko, vo = qkv_out.split([HQ * D, HKV * D, HKV * D], dim=-1)
    gate("qkv fused: q", qo, qn, 3e-2)
    gate("qkv fused: k", ko, kn, 3e-2)
    gate("qkv fused: v untouched (clean rows)", vo[::width], qkv[::width, HQ * D + HKV * D:], 0.0)

    # ---- 3. silu_mul rail ---------------------------------------------- #
    off_u, d_out, off_v, d_in = layout["gu"]
    x = (torch.randn(R, d_in, device=dev) * 0.5).to(dt)
    gu = (torch.randn(R, d_out, device=dev) * 1.0).to(dt)
    gu_ref = ref_rail(x, gu, noise, signs, sigma, pri, rail, pidx, off_u, d_out, off_v, d_in)
    out_ref = torch.empty(R, I, device=dev, dtype=dt)
    torch.ops._C.silu_and_mul(out_ref, gu_ref)
    out = torch.empty(R, I, device=dev, dtype=dt)
    silu_mul_rail(gu, out, RailArgs(x, noise, signs, sigma, off_u, off_v, d_in, width))
    gate("silu_mul_rail", out, out_ref, 3e-2)
    out2 = torch.empty(R, I, device=dev, dtype=dt)
    silu_mul_rail(gu, out2, None)
    out2_ref = torch.empty(R, I, device=dev, dtype=dt)
    torch.ops._C.silu_and_mul(out2_ref, gu)
    gate("silu_mul no-rail", out2, out2_ref, 3e-2)

    # ---- 4. lm tail: LSE + gather-dot + payload store -------------------- #
    hid = (torch.randn(R, H, device=dev) * 0.5).to(dt)
    W = (torch.randn(V, H, device=dev) * 0.02).to(dt)
    clean_idx = torch.full((R,), -1, dtype=torch.int32, device=dev)
    clean_idx[::width] = torch.arange(bucket, dtype=torch.int32, device=dev)
    ws = LMHeadWorkspace(R, V, bucket, 128, dev)
    clean_logits, lse_ref, _ = lm_head_stream(hid, W, clean_idx, ws=ws)
    am = clean_logits.argmax(-1)
    force = torch.full((bucket, 8), -1, dtype=torch.long, device=dev)
    force[0, 3] = 12345
    tcnt = torch.tensor([3], dtype=torch.long, device=dev)
    payload = torch.zeros(R, 8, device=dev)
    lm_tail(hid, W, ws.mp, ws.sp, am, force, tcnt, payload, width)
    chosen = am.clone()
    chosen[0] = 12345
    chosen_rows = chosen.repeat_interleave(width)
    logit_ref = (hid.float() * W[chosen_rows].float()).sum(-1)
    gate("lm_tail payload col t", payload[:, 3], logit_ref - lse_ref, 1e-3)
    gate("lm_tail other cols untouched", payload[:, [0, 1, 2, 4, 5, 6, 7]], torch.zeros(R, 7, device=dev), 0.0)

    # ---- 5. advance kernel vs python ------------------------------------ #
    max_blocks, bs = 16, 16
    bt = torch.randint(100, 200, (bucket, max_blocks), dtype=torch.int32, device=dev)
    ids = torch.zeros(R, dtype=torch.long, device=dev)
    posb = torch.tensor([37] * width + [61] * width, dtype=torch.long, device=dev)[:R]
    sl = posb + 1
    slb = sl[::width].clone().to(torch.int32)
    sm = torch.full((R,), -1, dtype=torch.long, device=dev)
    active = torch.ones(bucket, dtype=torch.int32, device=dev)
    ngen = torch.zeros(bucket, dtype=torch.int32, device=dev)
    tokens = torch.zeros(bucket, 8, dtype=torch.long, device=dev)
    eos = torch.tensor([151645, 151643, -1, -1], dtype=torch.long, device=dev)
    fstop = torch.full((bucket,), 10 ** 9, dtype=torch.long, device=dev)
    am2 = torch.tensor([777, 151645], dtype=torch.long, device=dev)[:bucket]   # slot 1 hits EOS
    force2 = torch.full((bucket, 8), -1, dtype=torch.long, device=dev)
    advance(am2, force2, active, ngen, tokens, ids, posb, sl, slb, sm, bt, fstop, eos, tcnt, width, bs)
    exp_ids = ids.clone()
    ok = (tokens[0, 3] == 777).item() and (ngen[0] == 4).item() and (active[0] == 1).item()
    ok &= (ids[:width] == 777).all().item() and (posb[:width] == 38).all().item() and (sl[:width] == 39).all().item()
    ok &= (slb[0] == 39).item() and (sm[0] == bt[0, 38 // bs].item() * bs + 38 % bs).item()
    if bucket > 1:
        ok &= (tokens[1, 3] == 151645).item() and (active[1] == 0).item() and (sm[width] == -1).item()
        ok &= (posb[width] == 61).item()   # frozen
    ok_all &= ok
    print(f"[advance] {'PASS' if ok else 'FAIL'} ids={ids.tolist()[:width]} pos={posb.tolist()[:width]} "
          f"sm={sm.tolist()[:width]} active={active.tolist()} ngen={ngen.tolist()}", flush=True)

    # ---- 6. noise from a device counter == host-seeded noise ------------ #
    seed_tbl = torch.randint(1, 2 ** 40, (8, bucket), dtype=torch.long, device=dev)
    o1 = torch.empty(bucket, d_total, device=dev, dtype=dt)
    fill_rademacher_rows_t(o1, seed_tbl, tcnt)
    o2 = torch.empty_like(o1)
    fill_rademacher_rows(o2, seed_tbl[3].tolist(), seed_tbl[3])
    gate("rademacher_t == rademacher(seeds[t])", o1, o2, 0.0)

    print("ALL_PASS" if ok_all else "SOME_FAIL")


if __name__ == "__main__":
    main()
