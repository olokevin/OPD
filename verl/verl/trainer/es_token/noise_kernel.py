"""Direct Rademacher noise fill for es_token.

The shipping path went through `verl.trainer.np.seeding.draw_noise`, which for
`method="bernoulli"` builds an **int64** `randint` buffer, casts it to fp32,
scales it to +-1, casts to bf16 and copies into the destination:

    bits = torch.randint(0, 2, shape, generator=gen, dtype=torch.int64)  # 8 B/elt
    n    = bits.to(torch.float32) * 2.0 - 1.0
    out.copy_(n.to(torch.bfloat16))

At d_total = 917,504 that is ~42 MB of traffic and ~6 kernels **per slot per
token** for 1.8 MB of useful output -- measured at 0.199 ms/token-step in decode,
and paid again for every one of the 65,536 token records during assembly.

This module draws +-1 **directly, in the destination dtype**, in one launch for a
whole batch of rows. Values come from Triton's counter-based Philox
(`tl.randint`), so they are a pure function of (seed, position): no generator
state, no host RNG, and the assembly can regenerate exactly what the decode drew.

INVARIANT: decode and assembly must produce bit-identical noise, so BOTH go
through `fill_rademacher_rows` here. The implementation is selected once at
import (Triton if importable, else the torch fallback) and never varies within a
process, so a run is always self-consistent.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

# "triton" (default when available) or "torch". Set ES_NOISE_IMPL=torch to A/B.
_IMPL = os.environ.get("ES_NOISE_IMPL", "triton" if HAVE_TRITON else "torch")


if HAVE_TRITON:

    @triton.jit
    def _rademacher_rows(OUT, SEEDS, d_total, stride_r, BLOCK: tl.constexpr):
        """OUT[r, j] = +-1 from Philox(SEEDS[r], j). One program per (row, block)."""
        r = tl.program_id(0)
        b = tl.program_id(1)
        seed = tl.load(SEEDS + r)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        rnd = tl.randint(seed, offs)
        val = tl.where((rnd & 1) == 1, 1.0, -1.0)
        tl.store(OUT + r * stride_r + offs, val.to(OUT.dtype.element_ty),
                 mask=offs < d_total)


def _fill_triton(out, seeds_dev, block=4096):
    rows = out.shape[0]
    d_total = out.shape[1]
    grid = (rows, triton.cdiv(d_total, block))
    _rademacher_rows[grid](out, seeds_dev, d_total, out.stride(0),
                           BLOCK=block, num_warps=8)


def _fill_torch(out, seeds_host):
    """Fallback: still avoids the int64 buffer and the cast chain -- draws {0,1}
    straight into the destination dtype, then maps to +-1 in place."""
    for r, s in enumerate(seeds_host):
        row = out[r]
        gen = torch.Generator(device=row.device)
        gen.manual_seed(int(s))
        row.random_(0, 2, generator=gen)
        row.mul_(2).sub_(1)


def impl_name():
    return _IMPL if (HAVE_TRITON or _IMPL == "torch") else "torch"


def fill_rademacher_rows(out, seeds_host, seeds_dev=None):
    """Fill `out` [rows, d_total] with +-1, row r seeded by seeds_host[r].

    seeds_dev: optional pre-uploaded int64 device copy of seeds_host (decode
    hoists this out of the token loop). Ignored by the torch fallback.
    """
    assert out.dim() == 2, f"expected [rows, d_total], got {tuple(out.shape)}"
    if _IMPL == "triton" and HAVE_TRITON and out.is_cuda:
        if seeds_dev is None:
            seeds_dev = torch.as_tensor(list(seeds_host), dtype=torch.int64,
                                        device=out.device)
        _fill_triton(out, seeds_dev)
    else:
        _fill_torch(out, seeds_host)
