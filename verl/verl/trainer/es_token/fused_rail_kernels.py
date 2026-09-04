"""Zero-launch rail op: the es_token rank-1 rail perturbation fused into the
kernels that already read each linear's output (es_profile_results.md 0902).

Phase 3 of the profile measured the shipping rail op (rail_kernel.py) at a
FIXED 0.47-0.50 ms per token-step at every N: 112 latency-bound launches of
~4.3 us. Nothing about the arithmetic is expensive -- the launch is. So the
op is moved into the consumer of every perturbed linear's output, which is
one of three kernels in a Qwen3 decoder layer:

    qkv_proj   -> q_norm + k_norm + RoPE         (_qkv_rail_norm_rope_kernel)
    o_proj     -> post_attention_layernorm         (_norm_rail_kernel, residual)
    gate_up    -> silu_and_mul                     (_silu_mul_rail_kernel)
    down_proj  -> next layer's input_layernorm     (_norm_rail_kernel, residual)
                  / the final model.norm

Each kernel is one program per packed row. For a perturbed row it first
reduces  a = sigma * <x, r_n (.) v_p>  over the linear's INPUT x (stashed by
the ESTokenLinear wrapper), applies  y += a * (s_n (.) u_p)  to the output it
was going to read anyway, then does its normal job. Clean rows (rail < 0)
skip the reduction. Per layer this replaces 6 launches (4 rail ops + q_norm +
k_norm, RoPE folded too) with 0 extra ones: 10 kernels/layer instead of 16.

Rounding mirrors the kernels it replaces (vLLM csrc + rail_kernel.py) so a
sigma=0 decode stays as close to stock as reduction order allows:
  rail update   y = bf16(y + a*u*s)                    (rail_kernel.py)
  fused_add_rms z = bf16(y + res); var = mean(z^2); out = bf16(bf16(z*rsqrt)*w)
  rms_norm      out = bf16(bf16(x*rsqrt)*w)             (q_norm / k_norm)
  rope (neox)   o1 = bf16(bf16(x1*cos) - bf16(x2*sin)); o2 = bf16(bf16(x2*cos) + bf16(x1*sin))
  silu_and_mul  out = bf16(bf16(silu(g)) * h)

Also here, for the fully in-graph token step (step_impl="graph"):
  _lm_tail_kernel   per row: LSE from the streaming head's (m, s) tiles, the
                    clean token's logit as a gather-dot, payload[row, t] store
  _advance_kernel   per slot: EOS/stop test, token record, next input id /
                    position / seq_len / KV slot -- the host's per-token
                    buffer refill, on device, indexed by a device counter t
  _rademacher_rows_t  the noise fill reading its seeds from seed_tbl[t]
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:
    try:  # vLLM's silu is expf + IEEE '/'; tl.exp is ex2.approx-based and
        # Triton's '/' lowers to the approximate div.full -- use libdevice.
        from triton.language.extra import libdevice as _ld
        _exp = _ld.exp
        _div = _ld.div_rn
    except Exception:  # pragma: no cover
        _exp = tl.exp

        @triton.jit
        def _div(a, b):
            return a / b

    @triton.jit
    def _norm_rail_kernel(Y, RES, W, X, NOISE, SIGNS, SIGMA,
                          off_u, off_v, d_in, width,
                          sy, sres, sx, snoise, ssign, eps,
                          HAS_RAIL: tl.constexpr, HAS_RES: tl.constexpr,
                          H: tl.constexpr, BLOCK_IN: tl.constexpr,
                          HP: tl.constexpr = None):
        """[rail on Y] -> RES += Y -> Y = rmsnorm(RES) * W   (in place, like
        vLLM's fused_add_rms_norm). HAS_RES=False: plain rms_norm of Y."""
        row = tl.program_id(0)
        idx = tl.arange(0, HP)
        hm = idx < H
        y = tl.load(Y + row * sy + idx, mask=hm, other=0.0).to(tl.float32)
        if HAS_RAIL:
            slot = row // width
            rail = row % width - 1
            if rail >= 0:
                acc = tl.zeros((), dtype=tl.float32)
                for off in range(0, d_in, BLOCK_IN):
                    j = off + tl.arange(0, BLOCK_IN)
                    m = j < d_in
                    x = tl.load(X + row * sx + j, mask=m, other=0.0).to(tl.float32)
                    v = tl.load(NOISE + slot * snoise + off_v + j, mask=m, other=0.0).to(tl.float32)
                    r = tl.load(SIGNS + rail * ssign + off_v + j, mask=m, other=0.0).to(tl.float32)
                    acc += tl.sum(x * v * r, axis=0)
                a = acc * tl.load(SIGMA).to(tl.float32)
                u = tl.load(NOISE + slot * snoise + off_u + idx, mask=hm, other=0.0).to(tl.float32)
                s = tl.load(SIGNS + rail * ssign + off_u + idx, mask=hm, other=0.0).to(tl.float32)
                y = (y + a * u * s).to(Y.dtype.element_ty).to(tl.float32)
        if HAS_RES:
            res = tl.load(RES + row * sres + idx, mask=hm, other=0.0).to(tl.float32)
            z = (y + res).to(Y.dtype.element_ty).to(tl.float32)
            tl.store(RES + row * sres + idx, z.to(RES.dtype.element_ty), mask=hm)
        else:
            z = y
        var = tl.sum(z * z, axis=0) / H
        inv = tl.rsqrt(var + eps)
        w = tl.load(W + idx, mask=hm, other=0.0).to(tl.float32)
        out = ((z * inv).to(Y.dtype.element_ty).to(tl.float32) * w).to(Y.dtype.element_ty)
        tl.store(Y + row * sy + idx, out, mask=hm)

    @triton.jit
    def _qkv_rail_norm_rope_kernel(QKV, X, NOISE, SIGNS, SIGMA, QW, KW, CS, POS,
                                   off_u, off_v, d_in, width,
                                   sq, sx, snoise, ssign, scs, eps,
                                   HAS_RAIL: tl.constexpr, HQ: tl.constexpr,
                                   HKV: tl.constexpr, D: tl.constexpr,
                                   BLOCK_IN: tl.constexpr,
                                   HAS_NORM: tl.constexpr = True,
                                   HQP: tl.constexpr = None,
                                   HKVP: tl.constexpr = None):
        """[rail on the qkv row] -> per-head RMSNorm of q and k -> neox RoPE at
        POS[row], all in place in QKV. v is left alone: a perturbed v of a rail
        row is never cached (slot -1) and clean rows have a = 0."""
        row = tl.program_id(0)
        HD: tl.constexpr = D // 2
        hq = tl.arange(0, HQP)
        hk = tl.arange(0, HKVP)
        mq = (hq < HQ)[:, None]
        mk = (hk < HKV)[:, None]
        dh = tl.arange(0, HD)
        base = QKV + row * sq
        q1o = hq[:, None] * D + dh[None, :]
        q2o = q1o + HD
        k1o = HQ * D + hk[:, None] * D + dh[None, :]
        k2o = k1o + HD
        q1 = tl.load(base + q1o, mask=mq, other=0.0).to(tl.float32)
        q2 = tl.load(base + q2o, mask=mq, other=0.0).to(tl.float32)
        k1 = tl.load(base + k1o, mask=mk, other=0.0).to(tl.float32)
        k2 = tl.load(base + k2o, mask=mk, other=0.0).to(tl.float32)
        if HAS_RAIL:
            slot = row // width
            rail = row % width - 1
            if rail >= 0:
                acc = tl.zeros((), dtype=tl.float32)
                for off in range(0, d_in, BLOCK_IN):
                    j = off + tl.arange(0, BLOCK_IN)
                    m = j < d_in
                    x = tl.load(X + row * sx + j, mask=m, other=0.0).to(tl.float32)
                    v = tl.load(NOISE + slot * snoise + off_v + j, mask=m, other=0.0).to(tl.float32)
                    r = tl.load(SIGNS + rail * ssign + off_v + j, mask=m, other=0.0).to(tl.float32)
                    acc += tl.sum(x * v * r, axis=0)
                a = acc * tl.load(SIGMA).to(tl.float32)
                nb = NOISE + slot * snoise + off_u
                sb = SIGNS + rail * ssign + off_u
                q1 = (q1 + a * tl.load(nb + q1o, mask=mq, other=0.0).to(tl.float32)
                      * tl.load(sb + q1o, mask=mq, other=0.0).to(tl.float32)
                      ).to(QKV.dtype.element_ty).to(tl.float32)
                q2 = (q2 + a * tl.load(nb + q2o, mask=mq, other=0.0).to(tl.float32)
                      * tl.load(sb + q2o, mask=mq, other=0.0).to(tl.float32)
                      ).to(QKV.dtype.element_ty).to(tl.float32)
                k1 = (k1 + a * tl.load(nb + k1o, mask=mk, other=0.0).to(tl.float32)
                      * tl.load(sb + k1o, mask=mk, other=0.0).to(tl.float32)
                      ).to(QKV.dtype.element_ty).to(tl.float32)
                k2 = (k2 + a * tl.load(nb + k2o, mask=mk, other=0.0).to(tl.float32)
                      * tl.load(sb + k2o, mask=mk, other=0.0).to(tl.float32)
                      ).to(QKV.dtype.element_ty).to(tl.float32)
        # per-head RMSNorm (vLLM rms_norm on [rows*heads, D]); Qwen2 has no q/k norm
        if HAS_NORM:
            qw1 = tl.load(QW + dh).to(tl.float32)
            qw2 = tl.load(QW + HD + dh).to(tl.float32)
            kw1 = tl.load(KW + dh).to(tl.float32)
            kw2 = tl.load(KW + HD + dh).to(tl.float32)
            iq = tl.rsqrt((tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) / D + eps)
            ik = tl.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / D + eps)
            q1 = ((q1 * iq[:, None]).to(QKV.dtype.element_ty).to(tl.float32) * qw1[None, :]
                  ).to(QKV.dtype.element_ty).to(tl.float32)
            q2 = ((q2 * iq[:, None]).to(QKV.dtype.element_ty).to(tl.float32) * qw2[None, :]
                  ).to(QKV.dtype.element_ty).to(tl.float32)
            k1 = ((k1 * ik[:, None]).to(QKV.dtype.element_ty).to(tl.float32) * kw1[None, :]
                  ).to(QKV.dtype.element_ty).to(tl.float32)
            k2 = ((k2 * ik[:, None]).to(QKV.dtype.element_ty).to(tl.float32) * kw2[None, :]
                  ).to(QKV.dtype.element_ty).to(tl.float32)
        # neox RoPE with vLLM's bf16 arithmetic order
        pos = tl.load(POS + row)
        cos = tl.load(CS + pos * scs + dh).to(tl.float32)
        sin = tl.load(CS + pos * scs + HD + dh).to(tl.float32)
        T = QKV.dtype.element_ty
        o1 = ((q1 * cos[None, :]).to(T).to(tl.float32) - (q2 * sin[None, :]).to(T).to(tl.float32)).to(T)
        o2 = ((q2 * cos[None, :]).to(T).to(tl.float32) + (q1 * sin[None, :]).to(T).to(tl.float32)).to(T)
        tl.store(base + q1o, o1, mask=mq)
        tl.store(base + q2o, o2, mask=mq)
        o1 = ((k1 * cos[None, :]).to(T).to(tl.float32) - (k2 * sin[None, :]).to(T).to(tl.float32)).to(T)
        o2 = ((k2 * cos[None, :]).to(T).to(tl.float32) + (k1 * sin[None, :]).to(T).to(tl.float32)).to(T)
        tl.store(base + k1o, o1, mask=mk)
        tl.store(base + k2o, o2, mask=mk)

    @triton.jit
    def _silu_mul_rail_kernel(GU, OUT, X, NOISE, SIGNS, SIGMA,
                              off_u, off_v, d_in, width, I,
                              sgu, sout, sx, snoise, ssign,
                              HAS_RAIL: tl.constexpr, BLOCK: tl.constexpr,
                              BLOCK_IN: tl.constexpr):
        """OUT = silu(GU[:, :I] (+rail)) * (GU[:, I:] (+rail)); one program/row."""
        row = tl.program_id(0)
        a = tl.zeros((), dtype=tl.float32)
        slot = row // width
        rail = row % width - 1
        if HAS_RAIL:
            if rail >= 0:
                acc = tl.zeros((), dtype=tl.float32)
                for off in range(0, d_in, BLOCK_IN):
                    j = off + tl.arange(0, BLOCK_IN)
                    m = j < d_in
                    x = tl.load(X + row * sx + j, mask=m, other=0.0).to(tl.float32)
                    v = tl.load(NOISE + slot * snoise + off_v + j, mask=m, other=0.0).to(tl.float32)
                    r = tl.load(SIGNS + rail * ssign + off_v + j, mask=m, other=0.0).to(tl.float32)
                    acc += tl.sum(x * v * r, axis=0)
                a = acc * tl.load(SIGMA).to(tl.float32)
        rail_eff = tl.maximum(rail, 0)   # clean rows: a == 0, address stays valid
        T = GU.dtype.element_ty
        for off in range(0, I, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < I
            g = tl.load(GU + row * sgu + j, mask=m, other=0.0).to(tl.float32)
            h = tl.load(GU + row * sgu + I + j, mask=m, other=0.0).to(tl.float32)
            if HAS_RAIL:
                nb = NOISE + slot * snoise + off_u
                sb = SIGNS + rail_eff * ssign + off_u
                g = (g + a * tl.load(nb + j, mask=m, other=0.0).to(tl.float32)
                     * tl.load(sb + j, mask=m, other=0.0).to(tl.float32)).to(T).to(tl.float32)
                h = (h + a * tl.load(nb + I + j, mask=m, other=0.0).to(tl.float32)
                     * tl.load(sb + I + j, mask=m, other=0.0).to(tl.float32)).to(T).to(tl.float32)
            sg = _div(g, 1.0 + _exp(-g)).to(T).to(tl.float32)
            tl.store(OUT + row * sout + j, (sg * h).to(T), mask=m)

    @triton.jit
    def _lm_tail_kernel(HID, W, MP, SP, ARGMAX, FORCE, TCNT, PAYLOAD,
                        width, n_vt, K, sh, sw, smp, sforce, spay,
                        BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
        """payload[row, t] = <hid[row], W[tok(slot)]> - LSE[row], with
        tok = FORCE[slot, t] if >= 0 else ARGMAX[slot]. One program per row."""
        row = tl.program_id(0)
        slot = row // width
        t = tl.load(TCNT)
        jv = tl.arange(0, BLOCK_V)
        mv = jv < n_vt
        mp = tl.load(MP + row * smp + jv, mask=mv, other=float("-inf"))
        sp = tl.load(SP + row * smp + jv, mask=mv, other=1.0)
        lg = tl.where(mv, mp + tl.log(sp), float("-inf"))
        mx = tl.max(lg, axis=0)
        lse = mx + tl.log(tl.sum(tl.exp(lg - mx), axis=0))
        tok = tl.load(FORCE + slot * sforce + t)
        am = tl.load(ARGMAX + slot)
        tok = tl.where(tok >= 0, tok, am)
        acc = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            jk = k0 + tl.arange(0, BLOCK_K)
            mk = jk < K
            h = tl.load(HID + row * sh + jk, mask=mk, other=0.0).to(tl.float32)
            w = tl.load(W + tok * sw + jk, mask=mk, other=0.0).to(tl.float32)
            acc += tl.sum(h * w, axis=0)
        tl.store(PAYLOAD + row * spay + t, acc - lse)

    @triton.jit
    def _advance_kernel(ARGMAX, FORCE, ACTIVE, NGEN, TOKENS, IDS, POS, SL, SLB, SM,
                        BT, FORCE_STOP, EOS, TCNT,
                        width, block_size, sforce, stok, sbt,
                        N_EOS: tl.constexpr, BLOCK_W: tl.constexpr):
        """Per slot: record this token, stop on EOS / forced stop, else advance
        the slot's rows to the next position (ids, positions, seq_lens, the
        clean row's KV slot from the block table). Finished/pad slots keep
        their last-valid metadata and write no KV (C-4 semantics)."""
        p = tl.program_id(0)
        t = tl.load(TCNT)
        act = tl.load(ACTIVE + p)
        if act > 0:
            tok = tl.load(FORCE + p * sforce + t)
            am = tl.load(ARGMAX + p)
            tok = tl.where(tok >= 0, tok, am)
            tl.store(TOKENS + p * stok + t, tok)
            tl.store(NGEN + p, t + 1)
            e = tl.arange(0, N_EOS)
            eos = tl.load(EOS + e)
            hit = tl.sum(tl.where(eos == tok, 1, 0), axis=0)
            fs = tl.load(FORCE_STOP + p)
            if (hit > 0) | (t + 1 >= fs):
                tl.store(ACTIVE + p, 0)
                tl.store(SM + p * width, -1)
            else:
                r = tl.arange(0, BLOCK_W)
                mr = r < width
                rows = p * width + r
                pos = tl.load(POS + p * width) + 1
                tl.store(IDS + rows, tok.to(IDS.dtype.element_ty), mask=mr)
                tl.store(POS + rows, pos.to(POS.dtype.element_ty), mask=mr)
                tl.store(SL + rows, (pos + 1).to(SL.dtype.element_ty), mask=mr)
                tl.store(SLB + p, (pos + 1).to(SLB.dtype.element_ty))
                blk = tl.load(BT + p * sbt + pos // block_size)
                tl.store(SM + p * width, (blk.to(tl.int64) * block_size + pos % block_size))

    @triton.jit
    def _rademacher_rows_t(OUT, SEEDS, TCNT, d_total, stride_r, stride_t,
                           BLOCK: tl.constexpr):
        """noise_kernel._rademacher_rows with the seed row picked by the device
        counter: OUT[r, j] = +-1 from Philox(SEEDS[t, r], j). Same bits."""
        r = tl.program_id(0)
        b = tl.program_id(1)
        t = tl.load(TCNT)
        seed = tl.load(SEEDS + t * stride_t + r)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        rnd = tl.randint(seed, offs)
        val = tl.where((rnd & 1) == 1, 1.0, -1.0)
        tl.store(OUT + r * stride_r + offs, val.to(OUT.dtype.element_ty),
                 mask=offs < d_total)


# ---------------------------------------------------------------- wrappers --
# The alpha reduction uses the SAME tiling as rail_kernel.apply_rail
# (BLOCK_IN 4096, 16 warps): with +-1 noise/signs the update itself is exact,
# so an identical fp32 summation order makes the fused rail bit-identical to
# the shipping kernel (gate G3b). ES_FUSED_BLOCK_IN=1024 is the chaos yardstick.
_BLOCK_IN = int(os.environ.get("ES_FUSED_BLOCK_IN", 4096))
_WARPS = int(os.environ.get("ES_FUSED_WARPS", 16))
_NULL = None


def set_alpha_tiling(block_in=None, warps=None):
    """Gate hook: change the alpha-reduction tiling in-process (chaos yardstick)."""
    global _BLOCK_IN, _WARPS
    if block_in is not None:
        _BLOCK_IN = int(block_in)
    if warps is not None:
        _WARPS = int(warps)
    return _BLOCK_IN, _WARPS


def _null(device):
    """A 1-element placeholder for pointer args a constexpr-disabled path
    never dereferences (Triton still wants a tensor)."""
    global _NULL
    if _NULL is None or _NULL.device != device:
        _NULL = torch.zeros(1, device=device, dtype=torch.float32)
    return _NULL


class RailArgs:
    """The rail operands of ONE perturbed linear, as its consumer kernel needs
    them: x (the linear's input, [rows, d_in]), the flat noise/sign buffers,
    the [1] sigma tensor, this layer's (u, v) offsets and the packed width."""
    __slots__ = ("x", "noise", "signs", "sigma", "off_u", "off_v", "d_in", "width")

    def __init__(self, x, noise, signs, sigma, off_u, off_v, d_in, width):
        self.x, self.noise, self.signs, self.sigma = x, noise, signs, sigma
        self.off_u, self.off_v, self.d_in, self.width = off_u, off_v, d_in, width


def norm_rail(y, residual, weight, eps, rail=None, num_warps=None):
    """In place: [y += rail]; residual += y; y = rmsnorm(residual) * weight.
    residual=None -> plain rms_norm of y (rail still applied if given)."""
    R, H = y.shape
    assert y.stride(-1) == 1 and (residual is None or residual.stride(-1) == 1)
    dev = y.device
    num_warps = num_warps or _WARPS
    if rail is None:
        z = _null(dev)
        _norm_rail_kernel[(R,)](
            y, residual if residual is not None else z, weight, z, z, z, z,
            0, 0, 0, 1, y.stride(0), residual.stride(0) if residual is not None else 0,
            0, 0, 0, eps, HAS_RAIL=False, HAS_RES=residual is not None,
            H=H, BLOCK_IN=_BLOCK_IN, num_warps=num_warps,
            HP=triton.next_power_of_2(H))
    else:
        _norm_rail_kernel[(R,)](
            y, residual if residual is not None else _null(dev), weight,
            rail.x, rail.noise, rail.signs, rail.sigma,
            rail.off_u, rail.off_v, rail.d_in, rail.width,
            y.stride(0), residual.stride(0) if residual is not None else 0,
            rail.x.stride(0), rail.noise.stride(0), rail.signs.stride(0), eps,
            HAS_RAIL=True, HAS_RES=residual is not None,
            H=H, BLOCK_IN=_BLOCK_IN, num_warps=num_warps,
            HP=triton.next_power_of_2(H))
    return y, residual


def qkv_rail_norm_rope(qkv, positions, q_weight, k_weight, eps, cos_sin, hq, hkv, d,
                       rail=None, num_warps=None, has_norm=None):
    """In place on qkv [rows, (hq+2*hkv)*d]: rail, q/k per-head RMSNorm, RoPE.
    enable_fp_fusion=False: vLLM's RoPE rounds x*cos and y*sin to bf16 before
    combining; an FMA would skip one rounding (measured 18% ulp mismatches)."""
    R = qkv.shape[0]
    assert qkv.stride(-1) == 1 and cos_sin.dtype == qkv.dtype
    dev = qkv.device
    num_warps = num_warps or _WARPS
    z = _null(dev)
    if has_norm is None:
        has_norm = q_weight is not None
    if not has_norm:
        q_weight = k_weight = z
        eps = 0.0
    if rail is None:
        _qkv_rail_norm_rope_kernel[(R,)](
            qkv, z, z, z, z, q_weight, k_weight, cos_sin, positions,
            0, 0, 0, 1, qkv.stride(0), 0, 0, 0, cos_sin.stride(0), eps,
            HAS_RAIL=False, HQ=hq, HKV=hkv, D=d, BLOCK_IN=_BLOCK_IN, num_warps=num_warps,
            HAS_NORM=bool(has_norm), HQP=triton.next_power_of_2(hq),
            HKVP=triton.next_power_of_2(hkv), enable_fp_fusion=False)
    else:
        _qkv_rail_norm_rope_kernel[(R,)](
            qkv, rail.x, rail.noise, rail.signs, rail.sigma, q_weight, k_weight,
            cos_sin, positions,
            rail.off_u, rail.off_v, rail.d_in, rail.width,
            qkv.stride(0), rail.x.stride(0), rail.noise.stride(0), rail.signs.stride(0),
            cos_sin.stride(0), eps,
            HAS_RAIL=True, HQ=hq, HKV=hkv, D=d,
            BLOCK_IN=_BLOCK_IN, num_warps=num_warps,
            HAS_NORM=bool(has_norm), HQP=triton.next_power_of_2(hq),
            HKVP=triton.next_power_of_2(hkv), enable_fp_fusion=False)
    return qkv


def silu_mul_rail(gu, out, rail=None, block=2048, num_warps=None):
    """out [rows, I] = silu(gu[:, :I]) * gu[:, I:], rail applied to gu first."""
    R, two_i = gu.shape
    I = two_i // 2
    assert gu.stride(-1) == 1 and out.stride(-1) == 1 and out.shape == (R, I)
    dev = gu.device
    num_warps = num_warps or _WARPS
    z = _null(dev)
    if rail is None:
        _silu_mul_rail_kernel[(R,)](
            gu, out, z, z, z, z, 0, 0, 0, 1, I,
            gu.stride(0), out.stride(0), 0, 0, 0,
            HAS_RAIL=False, BLOCK=block, BLOCK_IN=_BLOCK_IN, num_warps=num_warps)
    else:
        _silu_mul_rail_kernel[(R,)](
            gu, out, rail.x, rail.noise, rail.signs, rail.sigma,
            rail.off_u, rail.off_v, rail.d_in, rail.width, I,
            gu.stride(0), out.stride(0), rail.x.stride(0), rail.noise.stride(0),
            rail.signs.stride(0),
            HAS_RAIL=True, BLOCK=block, BLOCK_IN=_BLOCK_IN, num_warps=num_warps)
    return out


def lm_tail(hidden, w_lm, mp, sp, argmax, force, t_cnt, payload, width, num_warps=4):
    """payload[:, t] = logit of the chosen token - LSE, per packed row."""
    R, K = hidden.shape
    n_vt = mp.shape[1]
    _lm_tail_kernel[(R,)](
        hidden, w_lm, mp, sp, argmax, force, t_cnt, payload,
        width, n_vt, K, hidden.stride(0), w_lm.stride(0), mp.stride(0),
        force.stride(0), payload.stride(0),
        BLOCK_V=triton.next_power_of_2(n_vt), BLOCK_K=1024, num_warps=num_warps)


def advance(argmax, force, active, ngen, tokens, ids, pos, sl, sl_b, sm, bt,
            force_stop, eos, t_cnt, width, block_size):
    bucket = active.shape[0]
    _advance_kernel[(bucket,)](
        argmax, force, active, ngen, tokens, ids, pos, sl, sl_b, sm, bt,
        force_stop, eos, t_cnt, width, block_size,
        force.stride(0), tokens.stride(0), bt.stride(0),
        N_EOS=eos.shape[0], BLOCK_W=triton.next_power_of_2(width), num_warps=1)


def fill_rademacher_rows_t(out, seed_tbl, t_cnt, block=4096):
    rows, d_total = out.shape
    grid = (rows, triton.cdiv(d_total, block))
    _rademacher_rows_t[grid](out, seed_tbl, t_cnt, d_total, out.stride(0),
                             seed_tbl.stride(0), BLOCK=block, num_warps=8)
