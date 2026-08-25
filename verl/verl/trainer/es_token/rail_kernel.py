"""Fused Triton kernel for the es_token per-token rank-1 rail op.

Motivation (results/zo_opd.md §2/§6). The PyTorch formulation of

    y[i] += sigma_l * ((r_n (.) v_p)^T x[i]) * (s_n (.) u_p)

issues ~14 kernels per matched linear (two gathers and a mul for each of the
u/v sides, a gather of x, a mul+reduce for alpha, then index/mul/mul/add/
index_put for the update). With 112 matched linears that is ~1570 CUDA-graph
nodes per decode token, every one of them operating on at most a few thousand
elements -- so the decode is bound by node count, not by arithmetic. Measured:
turning rails on at all cost +3.41 ms/token-step while going from 1 rail to 32
cost only a further +3.08 ms.

This kernel collapses the whole per-layer op into ONE launch. Each program
handles one perturbed row, reads that row's (rail, slot) from the index tensors
so the packed row layout is unconstrained, and forms the sign-modulated noise on
the fly -- so no [P, d_total] sign*noise buffer is ever materialised.

Everything it touches is a persistent buffer or a fixed index tensor and the
shapes are static, so it stays CUDA-graph capturable exactly like the PyTorch
path it replaces. Accumulation is fp32.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _rail_fused(X, Y, NOISE, SIGNS, SIGMA, PRI, RAIL, PIDX,
                    off_u, off_v, d_in, d_out, snoise, ssign, sx, sy,
                    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr):
        """One program per perturbed row.

        PRI[p]  -- the packed row this program writes
        PIDX[p] -- its slot, selecting the row of the flat per-slot noise buffer
        RAIL[p] -- its rail, selecting the row of the flat per-rail sign buffer
        off_u / off_v -- this layer's u and v offsets inside those flat buffers
        """
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
            tl.store(Y + row * sy + idx,
                     (y + a * u * sg).to(Y.dtype.element_ty), mask=m)


def rail_supported(x: torch.Tensor, y: torch.Tensor) -> bool:
    """The kernel indexes rows by stride, so it needs a contiguous last dim."""
    return (HAVE_TRITON and x.is_cuda and y.is_cuda
            and x.stride(-1) == 1 and y.stride(-1) == 1
            and x.dim() == 2 and y.dim() == 2)


# Tuned on H100 NVL over BLOCK_IN x BLOCK_OUT x num_warps for the four
# Qwen3-1.7B linear shapes (see scripts/zo_opd/results/es_token_rail_op.txt).
# Large blocks win because the grid is tiny (P = bucket * n_sample programs), so
# the kernel is latency-bound and fewer reduction iterations beat occupancy.
BLOCK_IN_DEFAULT = 4096
BLOCK_OUT_DEFAULT = 4096
NUM_WARPS_DEFAULT = 16


def apply_rail(x, y, noise_flat, signs_flat, sigma, pri, rail, pidx,
               off_u, d_out, off_v, d_in, d_total_noise, d_total_sign,
               block_in=BLOCK_IN_DEFAULT, block_out=BLOCK_OUT_DEFAULT,
               num_warps=NUM_WARPS_DEFAULT):
    """y[pri] += sigma * <x[pri], r(.)v> * (s(.)u), in one launch. In place."""
    P = pri.numel()
    if P == 0:
        return
    _rail_fused[(P,)](
        x, y, noise_flat, signs_flat, sigma, pri, rail, pidx,
        off_u, off_v, d_in, d_out, d_total_noise, d_total_sign,
        x.stride(0), y.stride(0),
        BLOCK_IN=block_in, BLOCK_OUT=block_out, num_warps=num_warps)
