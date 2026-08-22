"""Isolate and optimise the es_token per-layer rank-1 rail op.

The end-to-end profile (results/zo_opd.md §2) shows the packed decode costs
2.94 ms/token-step with rails OFF (N=0) and 6.35 ms with a SINGLE rail (N=1),
then only +0.10 ms per additional rail. That +3.41 ms step is therefore a fixed
cost paid the moment the rail path becomes active -- it cannot be FLOPs (N=1
touches 4 rows of at most 6144 elements per layer). This benchmark reproduces
the rail op alone, at the real Qwen3-1.7B shapes for all 112 matched linears,
inside a CUDA graph, and compares implementations.

Variants
  v0_current   exactly the shipping ESTokenLinear branch
  v1_flat      ONE broadcast kernel per token builds the sign*noise product for
               ALL layers into a flat [P, d_total] buffer; each layer then reads
               u_eff / v_eff as VIEWS (removes 6 kernels/layer)
  v2_fused     v1 + per-layer math fused (vecdot + addcmul instead of
               mul/sum/mul/mul/add)
  v3_contig    v2 + perturbed rows contiguous, so x_p / y_p are slices rather
               than gather/scatter (removes 2 more kernels/layer)

  CUDA_VISIBLE_DEVICES=6 python scripts/zo_opd/es_token_checks/bench_rail_op.py
"""
import argparse
import time

import torch

# Qwen3-1.7B: 28 decoder layers x (qkv, o, gate_up, down)
LAYER_SHAPES = [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
N_DECODER = 28


def build_layers(n_decoder=N_DECODER):
    dims = []
    for i in range(n_decoder):
        for (d_out, d_in) in LAYER_SHAPES:
            dims.append((d_out, d_in))
    layout, off = [], 0
    for (d_out, d_in) in dims:
        layout.append((off, d_out, off + d_out, d_in))
        off += d_out + d_in
    return dims, layout, off


class State:
    """Everything the rail op reads, allocated once (as in the real worker)."""

    def __init__(self, bucket, n_sample, dtype, device, contig=False):
        self.dims, self.layout, self.d_total = build_layers()
        self.bucket, self.n_sample = bucket, n_sample
        P = bucket * n_sample
        self.P = P
        self.width = 1 + n_sample
        R_rows = bucket * self.width

        self.noise_buf = torch.randn(bucket, self.d_total, device=device,
                                     dtype=dtype).sign_()
        # per-layer signs, exactly as install_es_layers builds them
        self.S = [torch.randn(n_sample, d_out, device=device, dtype=dtype).sign_()
                  for (d_out, _) in self.dims]
        self.R = [torch.randn(n_sample, d_in, device=device, dtype=dtype).sign_()
                  for (_, d_in) in self.dims]
        # flat [n_sample, d_total] signs in the SAME layout as noise_buf
        self.signs_flat = torch.empty(n_sample, self.d_total, device=device,
                                      dtype=dtype)
        for li, (off_u, d_out, off_v, d_in) in enumerate(self.layout):
            self.signs_flat[:, off_u:off_u + d_out] = self.S[li]
            self.signs_flat[:, off_v:off_v + d_in] = self.R[li]
        self.noise_eff = torch.empty(P, self.d_total, device=device, dtype=dtype)
        # sign*sigma flat (u-side scaled by sigma, v-side by 1)
        self.signs_sig = self.signs_flat  # sigma folded in below
        # per-layer CONTIGUOUS [P, d] noise blocks (v5)
        self.blocked = torch.empty(P * self.d_total, device=device, dtype=dtype)
        self.blocks = []
        for (off_u, d_out, off_v, d_in) in self.layout:
            bu = self.blocked[P * off_u: P * off_u + P * d_out].view(P, d_out)
            bv = self.blocked[P * off_v: P * off_v + P * d_in].view(P, d_in)
            self.blocks.append((bu, bv))

        if contig:
            # [all clean rows][all perturbed rows]
            clean = list(range(bucket))
            pert = list(range(bucket, bucket + P))
            self.rail = torch.arange(P, device=device) % n_sample
            self.pidx = torch.arange(P, device=device) // n_sample
        else:
            # shipping layout: per prompt [clean, rail0..railN-1]
            clean = [p * self.width for p in range(bucket)]
            pert = [p * self.width + 1 + n for p in range(bucket)
                    for n in range(n_sample)]
            self.rail = torch.tensor([n for _ in range(bucket)
                                      for n in range(n_sample)], device=device)
            self.pidx = torch.tensor([p for p in range(bucket)
                                      for _ in range(n_sample)], device=device)
        self.pri = torch.tensor(pert, dtype=torch.long, device=device)
        self.clean_row_idx = torch.tensor(clean, dtype=torch.long, device=device)
        self.contig = contig

        # per-layer activation / output buffers (stand-ins for the vLLM tensors)
        self.x = [torch.randn(R_rows, d_in, device=device, dtype=dtype)
                  for (_, d_in) in self.dims]
        self.y = [torch.randn(R_rows, d_out, device=device, dtype=dtype)
                  for (d_out, _) in self.dims]
        self.sigma = [torch.full((1,), 0.01, device=device, dtype=dtype)
                      for _ in self.dims]
        # sigma folded into the u-side of a flat scale vector (v-side = 1)
        self.sigma_flat = torch.ones(self.d_total, device=device, dtype=dtype)
        for li, (off_u, d_out, off_v, d_in) in enumerate(self.layout):
            self.sigma_flat[off_u:off_u + d_out] = 0.01
        self.signs_sig = (self.signs_flat * self.sigma_flat).contiguous()


def v0_current(st):
    """The shipping ESTokenLinear branch, per layer."""
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        x, y = st.x[li], st.y[li]
        u = st.noise_buf[:, off_u:off_u + d_out]
        v = st.noise_buf[:, off_v:off_v + d_in]
        S, R, sigma = st.S[li], st.R[li], st.sigma[li]
        x_p = x[st.pri]
        v_eff = R[st.rail] * v[st.pidx]
        alpha = (x_p * v_eff).sum(dim=-1, keepdim=True)
        u_eff = S[st.rail] * u[st.pidx]
        y[st.pri] = y[st.pri] + sigma * alpha * u_eff


def _fill_noise_eff(st, fold_sigma):
    """ONE broadcast kernel for every layer's sign*noise (and optionally sigma)."""
    sf = st.signs_flat * st.sigma_flat if fold_sigma else st.signs_flat
    torch.mul(st.noise_buf[:, None, :], sf[None, :, :],
              out=st.noise_eff.view(st.bucket, st.n_sample, st.d_total))


def v1_flat(st):
    _fill_noise_eff(st, fold_sigma=False)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        x, y = st.x[li], st.y[li]
        u_eff = st.noise_eff[:, off_u:off_u + d_out]
        v_eff = st.noise_eff[:, off_v:off_v + d_in]
        x_p = x[st.pri]
        alpha = (x_p * v_eff).sum(dim=-1, keepdim=True)
        y[st.pri] = y[st.pri] + st.sigma[li] * alpha * u_eff


def v2_fused(st):
    _fill_noise_eff(st, fold_sigma=True)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        x, y = st.x[li], st.y[li]
        u_eff = st.noise_eff[:, off_u:off_u + d_out]
        v_eff = st.noise_eff[:, off_v:off_v + d_in]
        alpha = torch.linalg.vecdot(x[st.pri], v_eff)[:, None]
        y_p = y[st.pri]
        y_p.addcmul_(alpha, u_eff)
        y[st.pri] = y_p


def v3_contig(st):
    """Requires the [clean | perturbed] row layout so x_p / y_p are slices."""
    assert st.contig
    b = st.bucket
    _fill_noise_eff(st, fold_sigma=True)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        u_eff = st.noise_eff[:, off_u:off_u + d_out]
        v_eff = st.noise_eff[:, off_v:off_v + d_in]
        alpha = torch.linalg.vecdot(st.x[li][b:], v_eff)[:, None]
        st.y[li][b:].addcmul_(alpha, u_eff)




# ---------------------------------------------------------------- v4 / v5 ---
def v4_bmm(st):
    """v3 but alpha via a batched GEMV (one kernel) instead of vecdot (two)."""
    assert st.contig
    b, P = st.bucket, st.P
    _fill_noise_eff(st, fold_sigma=True)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        u_eff = st.noise_eff[:, off_u:off_u + d_out]
        v_eff = st.noise_eff[:, off_v:off_v + d_in]
        alpha = torch.bmm(st.x[li][b:].view(P, 1, d_in),
                          v_eff.reshape(P, d_in, 1)).view(P, 1)
        st.y[li][b:].addcmul_(alpha, u_eff)


def _fill_noise_eff_blocked(st):
    """Per-layer CONTIGUOUS [P, d] blocks, so downstream ops stay vectorized.
    Costs 2 kernels/layer to fill but makes every consumer contiguous."""
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        bu, bv = st.blocks[li]
        torch.mul(st.noise_buf[:, None, off_u:off_u + d_out],
                  st.signs_sig[None, :, off_u:off_u + d_out],
                  out=bu.view(st.bucket, st.n_sample, d_out))
        torch.mul(st.noise_buf[:, None, off_v:off_v + d_in],
                  st.signs_sig[None, :, off_v:off_v + d_in],
                  out=bv.view(st.bucket, st.n_sample, d_in))


def v5_blocked(st):
    """Contiguous per-layer noise blocks + contiguous rows -> vectorized ops."""
    assert st.contig
    b = st.bucket
    _fill_noise_eff_blocked(st)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        bu, bv = st.blocks[li]
        alpha = torch.linalg.vecdot(st.x[li][b:], bv)[:, None]
        st.y[li][b:].addcmul_(alpha, bu)


# ------------------------------------------------------------------- v6 -----
import triton
import triton.language as tl


@triton.jit
def _rail_fused(X, V, U, Y, SIGMA, d_in, d_out,
                sx, sv, su, sy,
                BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr):
    """One program per perturbed row: alpha = <x, v>, then y += sigma*alpha*u.

    Fuses the whole rank-1 rail op of ONE layer into a single launch, so the
    per-layer cost is one CUDA-graph node instead of 3-14."""
    p = tl.program_id(0)
    sigma = tl.load(SIGMA).to(tl.float32)
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, d_in, BLOCK_IN):
        idx = off + tl.arange(0, BLOCK_IN)
        m = idx < d_in
        x = tl.load(X + p * sx + idx, mask=m, other=0.0).to(tl.float32)
        v = tl.load(V + p * sv + idx, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x * v, axis=0)
    a = acc * sigma
    for off in range(0, d_out, BLOCK_OUT):
        idx = off + tl.arange(0, BLOCK_OUT)
        m = idx < d_out
        u = tl.load(U + p * su + idx, mask=m, other=0.0).to(tl.float32)
        y = tl.load(Y + p * sy + idx, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + p * sy + idx, (y + a * u).to(Y.dtype.element_ty), mask=m)


def v6_triton(st):
    """One fused Triton launch per layer (needs the flat sign*noise buffer)."""
    assert st.contig
    b, P = st.bucket, st.P
    _fill_noise_eff(st, fold_sigma=False)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        u_eff = st.noise_eff[:, off_u:off_u + d_out]
        v_eff = st.noise_eff[:, off_v:off_v + d_in]
        xs, ys = st.x[li][b:], st.y[li][b:]
        _rail_fused[(P,)](xs, v_eff, u_eff, ys, st.sigma[li], d_in, d_out,
                          xs.stride(0), v_eff.stride(0), u_eff.stride(0),
                          ys.stride(0), BLOCK_IN=1024, BLOCK_OUT=1024,
                          num_warps=4)




@triton.jit
def _rail_fused_rowidx(X, Y, NOISE, SIGNS, SIGMA, PRI, RAIL, PIDX,
                       off_u, off_v, d_in, d_out, snoise, ssign, sx, sy,
                       BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr):
    """Fully fused rail op for ONE layer, arbitrary packed-row layout.

    Program p handles perturbed row PRI[p], which belongs to rail RAIL[p] of
    slot PIDX[p]. It forms  sigma * <x, r_n (.) v_slot> * (s_n (.) u_slot)  on
    the fly, reading the flat per-slot noise buffer and the flat per-rail sign
    buffer at this layer's offsets -- so there is no [P, d_total] sign*noise
    materialisation and no constraint on how the packed rows are ordered."""
    p = tl.program_id(0)
    row = tl.load(PRI + p)
    nb = tl.load(PIDX + p) * snoise
    sb = tl.load(RAIL + p) * ssign
    sigma = tl.load(SIGMA).to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, d_in, BLOCK_IN):
        idx = off + tl.arange(0, BLOCK_IN)
        m = idx < d_in
        x = tl.load(X + row * sx + idx, mask=m, other=0.0).to(tl.float32)
        v = tl.load(NOISE + nb + off_v + idx, mask=m, other=0.0).to(tl.float32)
        r = tl.load(SIGNS + sb + off_v + idx, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x * v * r, axis=0)
    a = acc * sigma

    for off in range(0, d_out, BLOCK_OUT):
        idx = off + tl.arange(0, BLOCK_OUT)
        m = idx < d_out
        u = tl.load(NOISE + nb + off_u + idx, mask=m, other=0.0).to(tl.float32)
        sg = tl.load(SIGNS + sb + off_u + idx, mask=m, other=0.0).to(tl.float32)
        y = tl.load(Y + row * sy + idx, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + row * sy + idx, (y + a * u * sg).to(Y.dtype.element_ty),
                 mask=m)


def v7_triton_rowidx(st):
    """One fused Triton launch per layer; no flat sign*noise buffer, no row
    layout constraint (works with the shipping interleaved packing)."""
    P = st.P
    nflat = st.noise_buf.view(-1)
    sflat = st.signs_flat.view(-1)
    for li, (off_u, d_out, off_v, d_in) in enumerate(st.layout):
        x, y = st.x[li], st.y[li]
        _rail_fused_rowidx[(P,)](
            x, y, nflat, sflat, st.sigma[li], st.pri, st.rail, st.pidx,
            off_u, off_v, d_in, d_out, st.d_total, st.d_total,
            x.stride(0), y.stride(0),
            BLOCK_IN=1024, BLOCK_OUT=1024, num_warps=4)


VARIANTS = {"v0_current": (v0_current, False), "v1_flat": (v1_flat, False),
            "v2_fused": (v2_fused, False), "v3_contig": (v3_contig, True), "v4_bmm": (v4_bmm, True),
            "v5_blocked": (v5_blocked, True),
            "v6_triton": (v6_triton, True),
            "v7_triton_rowidx": (v7_triton_rowidx, False)}


def time_graphed(fn, st, iters=200):
    """Capture one call in a CUDA graph and time steady-state replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(st)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, default=4)
    ap.add_argument("--n-sample", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    dev = torch.device("cuda")
    dt = torch.bfloat16

    dims, layout, d_total = build_layers()
    print(f"layers={len(dims)}  d_total={d_total}  bucket={args.bucket}  "
          f"dtype={dt}")
    print(f"{'variant':<14s}" + "".join(f"  N={n:<3d} ms" for n in args.n_sample))
    base = {}
    for name in args.variants:
        fn, need_contig = VARIANTS[name]
        row = []
        for n in args.n_sample:
            st = State(args.bucket, n, dt, dev, contig=need_contig)
            ms = time_graphed(fn, st, args.iters)
            row.append(ms)
            base.setdefault(n, {})[name] = ms
            del st
            torch.cuda.empty_cache()
        print(f"{name:<14s}" + "".join(f"  {m:8.3f}" for m in row))

    print("\nspeedup vs v0_current (rail-op time only):")
    for n in args.n_sample:
        b = base[n].get("v0_current")
        if b is None:
            continue
        parts = [f"{k}={b / v:.2f}x" for k, v in base[n].items()
                 if k != "v0_current"]
        print(f"  N={n}: v0={b:.3f} ms  " + "  ".join(parts))


if __name__ == "__main__":
    main()
