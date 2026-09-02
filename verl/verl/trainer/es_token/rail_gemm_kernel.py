"""Triton GEMM with the es_token rank-1 rail correction FUSED into the epilogue
(opd_profile_plan.md §17, implementation 3).

    Y = X W^T                                  all rows
    Y[i] += sigma * <x_i, r_n (.) v_p> * (s_n (.) u_p)      perturbed rows

The shipping path is cuBLAS GEMM + one separate Triton pass (rail_kernel.py)
that re-reads x and read-modify-writes y. Here the X tile is already in
registers for the matmul, so the per-row scalar

    alpha_i = <x_i, r_n (.) v_p>

is accumulated in the SAME K loop (an extra [BLOCK_M, BLOCK_K] load of the
noise/sign slices per K step), and the rank-1 add is applied to the output
tile before the single store. No second pass over X or Y.

Row -> (rail, slot) is read from two index tensors (rail = -1 marks a clean
row, which then gets alpha = 0 and u = 0 via masking), so the packed row layout
is unconstrained, exactly like rail_kernel.py.

alpha is recomputed by every N-tile of the same M-tile (the noise/sign K-slices
are re-read from L2, d_in*2B*2 per row per N-tile). The benchmark in
scripts/zo_opd/es_profile/phase1_linear_rails.py decides whether this beats
cuBLAS + separate pass; it is NOT wired into the trainer unless it wins.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _rail_gemm(X, W, Y, NOISE, SIGNS, SIGMA, RAIL, SLOT,
                   M, N, K, off_u, off_v,
                   sxm, sxk, swn, swk, sym, syn, snoise, ssign,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, FUSE: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        m_mask = rm < M
        n_mask = rn < N

        if FUSE:
            rail = tl.load(RAIL + rm, mask=m_mask, other=-1)
            slot = tl.load(SLOT + rm, mask=m_mask, other=0)
            pert = rail >= 0
            nb = slot.to(tl.int64) * snoise
            sb = tl.where(pert, rail, 0).to(tl.int64) * ssign
            alpha = tl.zeros((BLOCK_M,), dtype=tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_ptrs = X + rm[:, None] * sxm + rk[None, :] * sxk
        w_ptrs = W + rn[:, None] * swn + rk[None, :] * swk
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + rk) < K
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
            if FUSE:
                vm = pert[:, None] & k_mask[None, :]
                v = tl.load(NOISE + nb[:, None] + off_v + k0 + rk[None, :], mask=vm, other=0.0)
                r = tl.load(SIGNS + sb[:, None] + off_v + k0 + rk[None, :], mask=vm, other=0.0)
                alpha += tl.sum(x.to(tl.float32) * v.to(tl.float32) * r.to(tl.float32), axis=1)
            x_ptrs += BLOCK_K * sxk
            w_ptrs += BLOCK_K * swk

        if FUSE:
            sigma = tl.load(SIGMA).to(tl.float32)
            um = pert[:, None] & n_mask[None, :]
            u = tl.load(NOISE + nb[:, None] + off_u + rn[None, :], mask=um, other=0.0)
            s = tl.load(SIGNS + sb[:, None] + off_u + rn[None, :], mask=um, other=0.0)
            acc += (alpha * sigma)[:, None] * u.to(tl.float32) * s.to(tl.float32)

        y_ptrs = Y + rm[:, None] * sym + rn[None, :] * syn
        tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _pick_block_m(M):
    return max(16, min(128, 1 << (int(M) - 1).bit_length()))


def rail_gemm(x, w, noise_flat, signs_flat, sigma, rail_of_row, slot_of_row,
              off_u, off_v, out=None, fuse=True, block_n=128, block_k=64,
              num_warps=None, num_stages=3):
    """out = x @ w.T (+ fused rank-1 rail correction on rows with rail >= 0).

    x [M, K] bf16, w [N, K] bf16, noise_flat [bucket, d_total], signs_flat
    [n_rails, d_total], sigma [1] tensor, rail_of_row/slot_of_row [M] int32/64.
    """
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM = _pick_block_m(M)
    if num_warps is None:
        num_warps = 4 if BM <= 64 else 8
    grid = (triton.cdiv(M, BM), triton.cdiv(N, block_n))
    _rail_gemm[grid](
        x, w, out, noise_flat, signs_flat, sigma, rail_of_row, slot_of_row,
        M, N, K, off_u, off_v,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        noise_flat.stride(0), signs_flat.stride(0),
        BLOCK_M=BM, BLOCK_N=block_n, BLOCK_K=block_k, FUSE=bool(fuse),
        num_warps=num_warps, num_stages=num_stages)
    return out
