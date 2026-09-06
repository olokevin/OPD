"""es-decode rails: ONE perturbation per rail, HELD for every generated token
(zo_opd_short.md "es-decode"; es-token-decode draws fresh noise per token).

Rail n of every perturbed linear l carries  dW_{n,l} = sigma * eps_{n,l}  with
eps of unit per-element RMS, in one of two forms:

  low-rank (noise_rank = r):  eps = (1/sqrt(r)) * sum_k a_k b_k^T,
      a_k in {+-1}^{d_out}, b_k in {+-1}^{d_in}  (Rademacher, seeded Philox)
      -> per rail a flat [r * d_total] bf16 row; the rail op is r reductions
         <b_k, x> and r axpys, in the consumer kernels (fused_rail_kernels,
         RANK > 0) or the standalone _rail_lowrank_kernel here.

  full-rank (noise_rank = "full"):  eps in {+-1}^{d_out x d_in}, stored as
      PACKED BITS: one int32 word = 32 signs of one output row, i.e. 1/16 of
      bf16 storage (32 rails x 1.31 B perturbed params = 5.2 GB, not 84 GB).
      The rail op (_rail_bits_kernel) streams the bits once per (rail, 64-row
      block), unpacks them to a bf16 +-1 tile and uses ONE tl.dot against the
      rail's x rows of every slot (exact products, fp32 accumulate) -- so the
      per-token cost is the bit traffic (164 MB/rail for R1-Distill-1.5B) plus
      a tensor-core GEMV, instead of N extra weight-sized reads.

Antithetic pairs: rail 2i+1 is the exact negation of rail 2i (a_k negated /
every word inverted), so the seeds list has N/2 entries.

Update: W += sum_n coef_n * eps_n with the trainer's OpenAI-ES coefficients
(coef already contains alpha; eps is the SAME unit-scale noise the rails
used, still resident on the GPU):
  low-rank: one fp32 GEMM per layer (A^T diag(coef/sqrt r) B),
  full:     _bits_update_kernel accumulates coef-weighted signs into an fp32
            [d_out, d_in] tile, one program per (m, k) block.
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
    def _rademacher_bits_kernel(BITS, SEEDS, n_words, wbase, s_n,
                                NEG_ODD: tl.constexpr, BLOCK: tl.constexpr):
        """BITS[n, w] = 32 Philox bits of (seed[pair(n)], wbase + w); odd rails
        inverted when NEG_ODD (antithetic). wbase makes every layer's words a
        distinct slice of one per-rail stream."""
        n = tl.program_id(0)
        b = tl.program_id(1)
        if NEG_ODD:
            seed = tl.load(SEEDS + n // 2)
        else:
            seed = tl.load(SEEDS + n)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        w = tl.randint(seed, wbase + offs).to(tl.int32, bitcast=True)
        if NEG_ODD:
            w = tl.where((n % 2) == 1, w ^ -1, w)
        tl.store(BITS + n * s_n + offs, w, mask=offs < n_words)

    @triton.jit
    def _rail_bits_kernel(X, Y, BITS, SIGMA, width, bucket, M, K,
                          sx, sy, sb_n, sb_m,
                          BM: tl.constexpr, BK: tl.constexpr, BP: tl.constexpr):
        """y[rows of rail n, m-block] += sigma * sign_n[m-block, :] @ x[rows, :]^T.
        Grid (rail, m-block); the bits of a block are read once for all slots.
        Unpacking is a static loop over the 32 bit positions of each word: bit b
        of word j is column k = k0 + 32 j + b, so every bit position gives a
        [BM, BK/32] +-1 tile and one tl.dot against the matching (strided) x
        columns -- no 3-D reshape through shared memory (1.4x the reshape form)."""
        n = tl.program_id(0)
        mb = tl.program_id(1)
        rm = mb * BM + tl.arange(0, BM)
        mm = rm < M
        rp = tl.arange(0, BP)
        mp = rp < bucket
        rows = rp * width + 1 + n
        NW: tl.constexpr = BK // 32
        rw = tl.arange(0, NW)
        acc = tl.zeros((BP, BM), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kw = k0 // 32 + rw
            w = tl.load(BITS + n * sb_n + rm[:, None] * sb_m + kw[None, :],
                        mask=mm[:, None] & (kw * 32 < K)[None, :], other=0)   # [BM, NW] int32
            for b in tl.static_range(32):
                sgn = (((w >> b) & 1) * 2 - 1).to(X.dtype.element_ty)       # [BM, NW]
                kk = k0 + rw * 32 + b
                x = tl.load(X + rows[:, None] * sx + kk[None, :],
                            mask=mp[:, None] & (kk < K)[None, :], other=0.0)  # [BP, NW]
                acc = tl.dot(x, tl.trans(sgn), acc)
        sigma = tl.load(SIGMA).to(tl.float32)
        yp = Y + rows[:, None] * sy + rm[None, :]
        msk = mp[:, None] & mm[None, :]
        y = tl.load(yp, mask=msk, other=0.0).to(tl.float32)
        tl.store(yp, (y + sigma * acc).to(Y.dtype.element_ty), mask=msk)

    @triton.jit
    def _rail_lowrank_kernel(X, Y, NOISE, SIGMA, off_a, off_b, d_in, d_out, width,
                             rank, inv_sqrt_r, snoise, sx, sy,
                             RB: tl.constexpr, BLOCK: tl.constexpr, BLOCK_OUT: tl.constexpr):
        """One program per packed row: y += (sigma/sqrt r) sum_k <b_k, x> a_k
        with (a_k, b_k) = rail `rail`'s held rank-r factors. Clean rows exit."""
        row = tl.program_id(0)
        rail = row % width - 1
        if rail >= 0:
            rk = tl.arange(0, RB)
            mk = rk < rank
            nb = NOISE + rail * snoise
            acc = tl.zeros((RB,), dtype=tl.float32)
            for off in range(0, d_in, BLOCK):
                j = off + tl.arange(0, BLOCK)
                mj = j < d_in
                x = tl.load(X + row * sx + j, mask=mj, other=0.0).to(tl.float32)
                b = tl.load(nb + off_b + rk[:, None] * d_in + j[None, :],
                            mask=mk[:, None] & mj[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(b * x[None, :], axis=1)
            coef = acc * (tl.load(SIGMA).to(tl.float32) * inv_sqrt_r)
            for off in range(0, d_out, BLOCK_OUT):
                i = off + tl.arange(0, BLOCK_OUT)
                mi = i < d_out
                a = tl.load(nb + off_a + rk[:, None] * d_out + i[None, :],
                            mask=mk[:, None] & mi[None, :], other=0.0).to(tl.float32)
                upd = tl.sum(coef[:, None] * a, axis=0)
                y = tl.load(Y + row * sy + i, mask=mi, other=0.0).to(tl.float32)
                tl.store(Y + row * sy + i, (y + upd).to(Y.dtype.element_ty), mask=mi)

    @triton.jit
    def _bits_update_kernel(BITS, COEF, OUT, N, M, K, sb_n, sb_m, so_m,
                            BM: tl.constexpr, BK: tl.constexpr):
        """OUT[m, k] = sum_n coef[n] * sign_n[m, k]  (fp32), one program per tile."""
        mb = tl.program_id(0)
        kb = tl.program_id(1)
        rm = mb * BM + tl.arange(0, BM)
        mm = rm < M
        rw = tl.arange(0, BK // 32)
        rb = tl.arange(0, 32)
        rk = kb * BK + tl.arange(0, BK)
        kw = kb * (BK // 32) + rw
        acc = tl.zeros((BM, BK), dtype=tl.float32)
        for n in range(0, N):
            c = tl.load(COEF + n)
            w = tl.load(BITS + n * sb_n + rm[:, None] * sb_m + kw[None, :],
                        mask=mm[:, None] & (kw * 32 < K)[None, :], other=0)
            bits = (w[:, :, None] >> rb[None, None, :]) & 1
            sgn = tl.reshape(bits, (BM, BK)).to(tl.float32) * 2.0 - 1.0
            acc += c * sgn
        tl.store(OUT + rm[:, None] * so_m + rk[None, :], acc,
                 mask=mm[:, None] & (rk < K)[None, :])


# ------------------------------------------------------------------- python --
def rademacher_bits(bits, seeds_dev, wbase, antithetic, block=4096):
    """Fill bits [N, n_words] int32 from per-rail (or per-pair) seeds."""
    N, n_words = bits.shape
    grid = (N, triton.cdiv(n_words, block))
    _rademacher_bits_kernel[grid](bits, seeds_dev, n_words, int(wbase), bits.stride(0),
                                  NEG_ODD=bool(antithetic), BLOCK=block, num_warps=4)


def rail_bits(x, y, bits, sigma, width, bucket, block_m=None, block_k=None, num_warps=8):
    """In place: y[rail rows] += sigma * unpack(bits[n]) @ x[rail rows]. bits
    [N_alloc, M * K/32] int32 (row m's words contiguous), x [R, K], y [R, M].
    The grid covers the ACTIVE rails (width - 1), which may be fewer than the
    rails allocated (a sweep installs N_max once) -- never index rows past R."""
    N = int(width) - 1
    if N <= 0:
        return
    assert N <= bits.shape[0], (N, bits.shape)
    M, K = y.shape[1], x.shape[1]
    # Few rails = few programs and a serial chain of K/16 small dots per program:
    # halve the chain (BK 1024) and double the programs (BM 64). Many rails are
    # throughput-bound and prefer the wider tile (measured 2026-09-05).
    if block_m is None:
        block_m = 64 if N <= 4 else 128
    if block_k is None:
        block_k = 1024 if N <= 4 else 512
    block_k = min(block_k, max(512, (K // 32) * 32)) if K >= 512 else 512
    assert K % 32 == 0 and x.stride(-1) == 1 and y.stride(-1) == 1
    assert block_k % 32 == 0 and block_k // 32 >= 16   # tl.dot K-dim
    BP = max(16, triton.next_power_of_2(int(bucket)))
    grid = (N, triton.cdiv(M, block_m))
    _rail_bits_kernel[grid](x, y, bits, sigma, int(width), int(bucket), M, K,
                            x.stride(0), y.stride(0), bits.stride(0), K // 32,
                            BM=block_m, BK=block_k, BP=BP, num_warps=num_warps)


def rail_lowrank(x, y, noise, sigma, off_a, off_b, rank, width, num_warps=8):
    """In place, standalone (rail_impl=kernel) rank-r held rail on all rows."""
    R, d_in = x.shape
    d_out = y.shape[1]
    RB = max(2, triton.next_power_of_2(int(rank)))
    block = max(64, min(2048, 8192 // RB))
    _rail_lowrank_kernel[(R,)](
        x, y, noise, sigma, int(off_a), int(off_b), d_in, d_out, int(width),
        int(rank), 1.0 / math.sqrt(rank), noise.stride(0), x.stride(0), y.stride(0),
        RB=RB, BLOCK=block, BLOCK_OUT=block, num_warps=num_warps)


def bits_update(bits, coef, M, K, out=None, block_m=64, block_k=256):
    """out [M, K] fp32 = sum_n coef[n] * sign_n."""
    N = bits.shape[0]
    if out is None:
        out = torch.empty(M, K, dtype=torch.float32, device=bits.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    _bits_update_kernel[grid](bits, coef, out, N, M, K, bits.stride(0), K // 32,
                              out.stride(0), BM=block_m, BK=block_k, num_warps=4)
    return out


def lowrank_update(noise, coef, off_a, off_b, d_out, d_in, rank):
    """[d_out, d_in] fp32 = sum_n coef[n]/sqrt(r) sum_k a_{n,k} b_{n,k}^T."""
    N = noise.shape[0]
    A = noise[:, off_a:off_a + rank * d_out].reshape(N * rank, d_out).float()
    B = noise[:, off_b:off_b + rank * d_in].reshape(N * rank, d_in).float()
    c = (coef.float() / math.sqrt(rank)).repeat_interleave(rank)          # [N*r]
    return (A * c[:, None]).t() @ B


# ------------------------------------------------------------ references ----
def unpack_bits(bits_row, M, K):
    """int32 [M*K/32] -> fp32 [M, K] of +-1 (torch reference)."""
    w = bits_row.view(M, K // 32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, device=w.device, dtype=torch.int64)
    b = (w[:, :, None] >> shifts) & 1
    return (b.reshape(M, K).float() * 2.0 - 1.0)


class SeqNoise:
    """Per-step held noise for every perturbed linear. `layout` is the es_token
    flat layout {name: (off_u, d_out, off_v, d_in)} (rank-1 units)."""

    def __init__(self, layout, n_rails, rank, device, dtype):
        self.layout = layout
        self.n = int(n_rails)
        self.rank = rank                      # int or "full"
        self.device = device
        self.dtype = dtype
        self.antithetic = False
        if rank == "full":
            self.bits = {}
            self.wbase = {}
            wb = 0
            for ln, (_, d_out, _, d_in) in layout.items():
                assert d_in % 32 == 0, (ln, d_in)
                nw = d_out * (d_in // 32)
                self.bits[ln] = torch.empty(self.n, nw, dtype=torch.int32, device=device)
                self.wbase[ln] = wb
                wb += nw
            self.n_bytes = wb * 4 * self.n
        else:
            r = int(rank)
            d_total = max(off_v + d_in for (_, _, off_v, d_in) in layout.values())
            self.noise = torch.empty(self.n, r * d_total, dtype=dtype, device=device)
            self.n_bytes = self.noise.numel() * self.noise.element_size()

    def offsets(self, name):
        """(off_a, off_b) of layer `name` in the rank-r flat row."""
        off_u, d_out, off_v, d_in = self.layout[name]
        r = int(self.rank)
        return r * off_u, r * off_v

    def draw(self, seeds, antithetic):
        """seeds: N ints (N/2 when antithetic). Same bits every call with the
        same seeds -> a step's rails and its update see identical noise."""
        self.antithetic = bool(antithetic)
        seeds_dev = torch.as_tensor(list(seeds), dtype=torch.int64, device=self.device)
        if self.rank == "full":
            for ln, b in self.bits.items():
                rademacher_bits(b, seeds_dev, self.wbase[ln], self.antithetic)
        else:
            from verl.trainer.es_token.noise_kernel import fill_rademacher_rows
            if self.antithetic:
                half = self.noise[0::2]
                fill_rademacher_rows(half, list(seeds), seeds_dev)
                self.noise[1::2].copy_(half)
                # negate the A side only (dW = sigma/sqrt r * sum a b^T)
                for ln, (off_u, d_out, _, _) in self.layout.items():
                    r = int(self.rank)
                    self.noise[1::2, r * off_u:r * off_u + r * d_out].neg_()
            else:
                fill_rademacher_rows(self.noise, list(seeds), seeds_dev)

    def apply_rail(self, name, x, y, sigma_buf, width, bucket):
        """Standalone rail op for layer `name` (after its GEMM), in place on y."""
        _, d_out, _, d_in = self.layout[name]
        if self.rank == "full":
            rail_bits(x, y, self.bits[name], sigma_buf, width, bucket)
        else:
            off_a, off_b = self.offsets(name)
            rail_lowrank(x, y, self.noise, sigma_buf, off_a, off_b, int(self.rank), width)

    def update(self, name, coef):
        """fp32 [d_out, d_in] = sum_n coef_n eps_{n, name}."""
        _, d_out, _, d_in = self.layout[name]
        if self.rank == "full":
            return bits_update(self.bits[name], coef, d_out, d_in)
        off_a, off_b = self.offsets(name)
        return lowrank_update(self.noise, coef, off_a, off_b, d_out, d_in, int(self.rank))
