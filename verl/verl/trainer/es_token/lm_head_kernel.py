"""Streaming rail LM head (opd_profile_plan.md §9/§25).

The shipping decode calls model.compute_logits(hidden) on ALL (1+N) rows per
slot -> a [rows, V] bf16 tensor (V = 151,936), then .float() -> a second
[rows, V] fp32 copy, then logsumexp + a one-element gather per row. Only the
clean rows ever need full logits (to sample); a rail needs just its
log-normaliser and the logit of the clean token.

lm_head_stream() tiles the vocabulary: every program computes one
[BLOCK_M, BLOCK_V] logits tile in registers (fp32 accumulation over K), emits
the tile's (running max, sum-exp) per row, writes the tile to the clean-logit
buffer ONLY for clean rows, and drops it. The [rows, V] tensor is never
materialised; per-row LSE = logsumexp over the per-tile (m, s) pairs (a
[rows, V/BLOCK_V] fp32 reduction). The clean token's rail logit is then a
[rows, d] gather-dot against W[chosen] (lm_head_gather_logit).

The grid is ordered so consecutive programs share the same W tile (different
M tiles), so for rows > BLOCK_M the weight is still read from HBM ~once and
re-read from L2.
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
    def _lm_head_stream(X, W, CIDX, LOGITS, MP, SP, M, V, K, N_MT,
                        s_xm, s_wv, s_lm, s_mm,
                        BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        pid_m = pid % N_MT
        pid_v = pid // N_MT
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rv = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        rk = tl.arange(0, BLOCK_K)
        m_mask = rm < M
        v_mask = rv < V
        acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        x_ptrs = X + rm[:, None] * s_xm + rk[None, :]
        w_ptrs = W + rv[:, None] * s_wv + rk[None, :]
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + rk) < K
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        accm = tl.where(v_mask[None, :], acc, float("-inf"))
        mx = tl.max(accm, 1)
        se = tl.sum(tl.exp(accm - mx[:, None]), 1)
        tl.store(MP + rm * s_mm + pid_v, mx, mask=m_mask)
        tl.store(SP + rm * s_mm + pid_v, se, mask=m_mask)
        cidx = tl.load(CIDX + rm, mask=m_mask, other=-1)
        cmask = (cidx >= 0)[:, None] & v_mask[None, :]
        tl.store(LOGITS + cidx[:, None].to(tl.int64) * s_lm + rv[None, :], acc, mask=cmask)


class LMHeadWorkspace:
    def __init__(self, M, V, n_clean, block_v, device):
        self.n_vt = triton.cdiv(V, block_v)
        self.mp = torch.empty(M, self.n_vt, dtype=torch.float32, device=device)
        self.sp = torch.empty_like(self.mp)
        self.logits = torch.empty(n_clean, V, dtype=torch.float32, device=device)
        self.lse = torch.empty(M, dtype=torch.float32, device=device)
        self.block_v = block_v


def _block_m(M):
    return max(16, min(64, 1 << (int(M) - 1).bit_length()))


def lm_head_stream(x, w, clean_idx, ws=None, n_clean=None, block_v=128, block_k=64,
                   num_warps=None, num_stages=3):
    """x [M, K] bf16 (all rows), w [V, K] bf16, clean_idx [M] int32 (row's index
    into the clean-logit buffer, -1 for rails). Returns (clean_logits [n_clean, V]
    fp32, lse [M] fp32, ws)."""
    M, K = x.shape
    V = w.shape[0]
    if ws is None:
        ws = LMHeadWorkspace(M, V, n_clean, block_v, x.device)
    BM = _block_m(M)
    n_mt = triton.cdiv(M, BM)
    if num_warps is None:
        num_warps = 4 if BM <= 32 else 8
    _lm_head_stream[(n_mt * ws.n_vt,)](
        x, w, clean_idx, ws.logits, ws.mp, ws.sp, M, V, K, n_mt,
        x.stride(0), w.stride(0), ws.logits.stride(0), ws.mp.stride(0),
        BLOCK_M=BM, BLOCK_V=ws.block_v, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages)
    torch.logsumexp(ws.mp + torch.log(ws.sp), dim=1, out=ws.lse)
    return ws.logits, ws.lse, ws


def lm_head_gather_logit(x, w, chosen):
    """logit of token chosen[i] for row i: <x_i, W[chosen_i]> in fp32. [M]."""
    wc = w.index_select(0, chosen)                    # [M, K] bf16
    return (x.float() * wc.float()).sum(dim=-1)
