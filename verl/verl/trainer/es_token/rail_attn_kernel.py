"""Shared-KV rail decode attention (opd_profile_plan.md §6/§18).

Problem. In the shipping es_token decode every rail row is its own "request"
in vLLM's FlashAttention call, so the clean sequence's historical KV pages
are re-read once per rail: (1+N) x the KV traffic of the clean decode. That
is the naive reference of §18 and it silently erases the memory-bound
argument for attention.

Two rail-aware implementations, both reading each KV page ONCE per slot:

  rail_attention_shared(...)  Triton kernel. One program per (slot b, kv head
      h, KV split). It loads the [R*G, D] query tile -- ALL rails x the GQA
      group of head h -- once, then streams the slot's KV pages in BLOCK_N
      tiles; each tile is used by every rail query through one tl.dot, with an
      online softmax per query row. Split-KV partials (m, l, acc) are merged by
      a second tiny kernel. Rails are, to the kernel, extra query heads that
      share a KV head: the "rail as an additional query-group dimension" of
      §18. Grid and shapes are static, seq_len is read from a device tensor,
      so the kernel replays inside the decode CUDA graph.

  rail_attention_fold(...)  the same idea through the stock FA3 kernel: fold
      the rail axis into the head axis in (kv-head, rail, group) order so FA3's
      own GQA packing loads each KV tile once for R*G query heads. Costs two
      permute copies per layer; needs no custom kernel.

Semantics match the shipping path exactly: rails attend the clean history
INCLUDING the clean current-token K/V (which the clean row wrote to the cache
before attention; rails never write KV). Only clean current-token KV exists in
the cache, so no rail-specific k_t/v_t is used -- same as today.

Layouts (vLLM FlashAttention backend, NHD): key_cache / value_cache
[num_blocks, block_size, H_kv, D]; q [B, R, H_q, D] (any strides, D
contiguous); block_table [B(strided), max_blocks] int32; seq_lens [B(strided)]
int32 = seqused_k (history + current token).
"""
import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _rail_attn_split(Q, KC, VC, BT, SL, OP, MP, LP, O, sm_scale,
                         s_qb, s_qr, s_qh,
                         s_kb, s_kt, s_kh,
                         s_vb, s_vt, s_vh,
                         s_btb, s_slb,
                         s_ob, s_os, s_oh, s_og,      # OP [B, S, Hkv, RG_tiles*BLOCK_RG, D]
                         s_mb, s_ms, s_mh,            # MP/LP [B, S, Hkv, RG_tiles*BLOCK_RG]
                         s_fb, s_fr, s_fh,            # O [B, R, Hq, D] (SINGLE_SPLIT only)
                         R, G, NUM_SPLITS, N_RG_TILES,
                         BLOCK_RG: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                         SINGLE_SPLIT: tl.constexpr):
        b = tl.program_id(0)
        hg = tl.program_id(1)
        s = tl.program_id(2)
        h = hg // N_RG_TILES
        rgt = hg % N_RG_TILES

        seq_len = tl.load(SL + b * s_slb)
        n_tiles = (seq_len + BLOCK_N - 1) // BLOCK_N
        per = (n_tiles + NUM_SPLITS - 1) // NUM_SPLITS
        t0 = s * per
        t1 = tl.minimum(t0 + per, n_tiles)

        rg = rgt * BLOCK_RG + tl.arange(0, BLOCK_RG)
        rg_mask = rg < R * G
        r = rg // G
        j = rg % G
        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = (Q + b * s_qb + r[:, None] * s_qr + (h * G + j)[:, None] * s_qh
                  + offs_d[None, :])
        q = tl.load(q_ptrs, mask=rg_mask[:, None], other=0.0)

        m_i = tl.full((BLOCK_RG,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_RG,), tl.float32)
        acc = tl.zeros((BLOCK_RG, BLOCK_D), tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for t in range(t0, t1):
            pos = t * BLOCK_N + offs_n
            pmask = pos < seq_len
            blk = tl.load(BT + b * s_btb + pos // BLOCK_SIZE, mask=pmask, other=0)
            intra = pos % BLOCK_SIZE
            koff = blk.to(tl.int64) * s_kb + intra * s_kt + h * s_kh
            voff = blk.to(tl.int64) * s_vb + intra * s_vt + h * s_vh
            k = tl.load(KC + koff[:, None] + offs_d[None, :], mask=pmask[:, None], other=0.0)
            v = tl.load(VC + voff[:, None] + offs_d[None, :], mask=pmask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(pmask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        if SINGLE_SPLIT:
            o = acc / l_i[:, None]
            o_ptrs = (O + b * s_fb + r[:, None] * s_fr + (h * G + j)[:, None] * s_fh
                      + offs_d[None, :])
            tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=rg_mask[:, None])
        else:
            op = OP + b * s_ob + s * s_os + h * s_oh + rg[:, None] * s_og + offs_d[None, :]
            tl.store(op, acc, mask=rg_mask[:, None])
            tl.store(MP + b * s_mb + s * s_ms + h * s_mh + rg, m_i, mask=rg_mask)
            tl.store(LP + b * s_mb + s * s_ms + h * s_mh + rg, l_i, mask=rg_mask)

    @triton.jit
    def _rail_attn_combine(OP, MP, LP, O,
                           s_ob, s_os, s_oh, s_og,
                           s_mb, s_ms, s_mh,
                           s_fb, s_fr, s_fh,
                           R, G, NUM_SPLITS,
                           BLOCK_RG: tl.constexpr, BLOCK_D: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        rgt = tl.program_id(2)
        rg = rgt * BLOCK_RG + tl.arange(0, BLOCK_RG)
        rg_mask = rg < R * G
        r = rg // G
        j = rg % G
        offs_d = tl.arange(0, BLOCK_D)

        m_max = tl.full((BLOCK_RG,), float("-inf"), tl.float32)
        for s in range(0, NUM_SPLITS):
            m_s = tl.load(MP + b * s_mb + s * s_ms + h * s_mh + rg, mask=rg_mask, other=float("-inf"))
            m_max = tl.maximum(m_max, m_s)
        l = tl.zeros((BLOCK_RG,), tl.float32)
        acc = tl.zeros((BLOCK_RG, BLOCK_D), tl.float32)
        for s in range(0, NUM_SPLITS):
            m_s = tl.load(MP + b * s_mb + s * s_ms + h * s_mh + rg, mask=rg_mask, other=float("-inf"))
            l_s = tl.load(LP + b * s_mb + s * s_ms + h * s_mh + rg, mask=rg_mask, other=0.0)
            w = tl.exp(m_s - m_max)
            o_s = tl.load(OP + b * s_ob + s * s_os + h * s_oh + rg[:, None] * s_og + offs_d[None, :],
                          mask=rg_mask[:, None], other=0.0)
            l += w * l_s
            acc += w[:, None] * o_s
        o = acc / l[:, None]
        o_ptrs = O + b * s_fb + r[:, None] * s_fr + (h * G + j)[:, None] * s_fh + offs_d[None, :]
        tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=rg_mask[:, None])


def pick_num_splits(B, n_kv, target_programs=None):
    """Static split count for a (B, H_kv) grid: enough programs to fill the
    SMs, capped so the partial buffers stay small. Fixed per captured graph."""
    if target_programs is None:
        target_programs = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    s = math.ceil(target_programs / max(1, B * n_kv))
    return int(min(32, max(1, s)))


class RailAttnWorkspace:
    """Persistent split-KV partial buffers (graph-stable addresses)."""

    def __init__(self, B, n_kv, R, G, num_splits, D, device):
        self.block_rg = max(16, min(128, 1 << (R * G - 1).bit_length()))
        self.n_rg_tiles = math.ceil(R * G / self.block_rg)
        rg_pad = self.n_rg_tiles * self.block_rg
        self.num_splits = num_splits
        self.op = torch.empty(B, num_splits, n_kv, rg_pad, D, dtype=torch.float32, device=device)
        self.mp = torch.empty(B, num_splits, n_kv, rg_pad, dtype=torch.float32, device=device)
        self.lp = torch.empty_like(self.mp)


def rail_attention_shared(q, key_cache, value_cache, block_table, seq_lens, sm_scale,
                          out=None, ws=None, num_splits=None, block_n=64, num_warps=4,
                          num_stages=2):
    """q [B, R, Hq, D] -> out [B, R, Hq, D]. block_table [B, max_blocks] (row-strided
    OK), seq_lens [B] int32 (strided OK). key/value_cache [nblk, bs, Hkv, D]."""
    B, R, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    bs = key_cache.shape[1]
    G = Hq // Hkv
    if out is None:
        out = torch.empty_like(q)
    if num_splits is None:
        num_splits = pick_num_splits(B, Hkv)
    if ws is None or ws.num_splits != num_splits:
        ws = RailAttnWorkspace(B, Hkv, R, G, num_splits, D, q.device)
    single = num_splits == 1
    grid = (B, Hkv * ws.n_rg_tiles, num_splits)
    _rail_attn_split[grid](
        q, key_cache, value_cache, block_table, seq_lens, ws.op, ws.mp, ws.lp, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0), seq_lens.stride(0),
        ws.op.stride(0), ws.op.stride(1), ws.op.stride(2), ws.op.stride(3),
        ws.mp.stride(0), ws.mp.stride(1), ws.mp.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        R, G, num_splits, ws.n_rg_tiles,
        BLOCK_RG=ws.block_rg, BLOCK_N=block_n, BLOCK_D=D, BLOCK_SIZE=bs,
        SINGLE_SPLIT=single, num_warps=num_warps, num_stages=num_stages)
    if not single:
        _rail_attn_combine[(B, Hkv, ws.n_rg_tiles)](
            ws.op, ws.mp, ws.lp, out,
            ws.op.stride(0), ws.op.stride(1), ws.op.stride(2), ws.op.stride(3),
            ws.mp.stride(0), ws.mp.stride(1), ws.mp.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            R, G, num_splits, BLOCK_RG=ws.block_rg, BLOCK_D=D, num_warps=4)
    return out, ws


def rail_attention_fold(q, key_cache, value_cache, block_table, seq_lens, sm_scale,
                        max_seqlen_k, out=None, fa_version=3, cu_seqlens_q=None,
                        q_fold_buf=None, o_fold_buf=None):
    """Same contract through stock FA: fold rails into the head axis in
    (kv-head, rail, group) order so heads sharing a KV head are contiguous and
    FA's GQA packing reuses each KV tile across all R*G of them."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    B, R, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    G = Hq // Hkv
    qf = q.view(B, R, Hkv, G, D).permute(0, 2, 1, 3, 4)          # [B, Hkv, R, G, D]
    if q_fold_buf is None:
        q_fold_buf = torch.empty(B, Hkv * R * G, D, dtype=q.dtype, device=q.device)
    q_fold_buf.view(B, Hkv, R, G, D).copy_(qf)
    if o_fold_buf is None:
        o_fold_buf = torch.empty_like(q_fold_buf)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=q.device)
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    flash_attn_varlen_func(
        q=q_fold_buf, k=key_cache, v=value_cache, out=o_fold_buf,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1, seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k, softmax_scale=sm_scale, causal=True,
        block_table=block_table, fa_version=fa_version)
    if out is None:
        out = torch.empty_like(q)
    out.view(B, R, Hkv, G, D).copy_(o_fold_buf.view(B, Hkv, R, G, D).permute(0, 2, 1, 3, 4))
    return out


def rail_attention_seq(q, key_cache, value_cache, block_table, seq_lens, sm_scale,
                       max_seqlen_k, out=None, fa_version=3, cu_seqlens_q=None):
    """The fold's KV reuse with NO permute and no custom kernel: one FA request
    per slot with seqlen_q = R and causal=False, so all R rail rows attend the
    slot's full seqused_k history (which already holds the clean current
    token's K/V). FA3's GQA packing puts the R*G query rows of a KV head in one
    M tile, i.e. each KV tile is read once per slot. q [B, R, Hq, D] with any
    row/rail strides (D contiguous) -- the qkv split view is used as is."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    B, R, Hq, D = q.shape
    if out is None:
        out = torch.empty_like(q)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=q.device) * R
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    flash_attn_varlen_func(
        q=q.view(B * R, Hq, D), k=key_cache, v=value_cache, out=out.view(B * R, Hq, D),
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=R, seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k, softmax_scale=sm_scale, causal=False,
        block_table=block_table, fa_version=fa_version)
    return out


def rail_attention_rows(q, key_cache, value_cache, block_table_rows, seq_lens_rows,
                        sm_scale, max_seqlen_k, out=None, fa_version=3, cu_seqlens_q=None):
    """The shipping path: every rail row is its own FA request (KV re-read per
    rail). block_table_rows [B*R, max_blocks], seq_lens_rows [B*R]."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    B, R, Hq, D = q.shape
    qr = q.reshape(B * R, Hq, D)
    if out is None:
        out = torch.empty_like(q)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(B * R + 1, dtype=torch.int32, device=q.device)
    flash_attn_varlen_func(
        q=qr, k=key_cache, v=value_cache, out=out.view(B * R, Hq, D),
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1, seqused_k=seq_lens_rows,
        max_seqlen_k=max_seqlen_k, softmax_scale=sm_scale, causal=True,
        block_table=block_table_rows, fa_version=fa_version)
    return out


def rail_attention_reference(q, key_cache, value_cache, block_table, seq_lens, sm_scale):
    """fp32 torch reference (gathers the pages), for correctness gates."""
    B, R, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    bs = key_cache.shape[1]
    G = Hq // Hkv
    outs = []
    for b in range(B):
        L = int(seq_lens[b])
        nb = (L + bs - 1) // bs
        blks = block_table[b, :nb].long()
        K = key_cache[blks].reshape(nb * bs, Hkv, D)[:L].float()   # [L, Hkv, D]
        V = value_cache[blks].reshape(nb * bs, Hkv, D)[:L].float()
        qb = q[b].float().view(R, Hkv, G, D)                         # [R, Hkv, G, D]
        s = torch.einsum("rhgd,lhd->rhgl", qb, K) * sm_scale
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("rhgl,lhd->rhgd", p, V).reshape(R, Hq, D)
        outs.append(o)
    return torch.stack(outs).to(q.dtype)
