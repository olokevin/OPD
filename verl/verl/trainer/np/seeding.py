"""Deterministic perturbation seeds and noise draws.

Invariant: no perturbation tensor is ever stored. Callers regenerate noise
on demand from a seed produced by noise_seed(...). Only integer seeds cross
RPC/Ray boundaries. See spec §2 "Never store u_q".
"""
import hashlib
from typing import Tuple

import torch

_MASK_63 = (1 << 63) - 1


def noise_seed(global_seed: int, step: int, layer: str, rollout: int, q: int) -> int:
    """Stable 63-bit seed for the perturbation of sample q, rollout, layer, step.

    Uses blake2b over the field tuple so the namespace can grow without
    collision and is stable across processes (Python's hash() is salted).
    """
    key = f"{int(global_seed)}|{int(step)}|{layer}|{int(rollout)}|{int(q)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK_63


def draw_noise(
    seed: int,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    method: str = "gaussian",
) -> torch.Tensor:
    """Regenerate the noise tensor for a seed. Deterministic per (seed, shape, method).

    method:
      - "gaussian":  N(0, 1)
      - "bernoulli": Rademacher, values in {-1, +1} (symmetric two-point)
      - "uniform":   U(-1, 1)
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if method == "gaussian":
        n = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    elif method == "bernoulli":
        bits = torch.randint(0, 2, shape, generator=gen, device=device, dtype=torch.int64)
        n = bits.to(torch.float32) * 2.0 - 1.0
    elif method == "uniform":
        n = torch.rand(shape, generator=gen, device=device, dtype=torch.float32) * 2.0 - 1.0
    else:
        raise ValueError(f"unknown sample_method: {method!r}")
    return n.to(dtype)
