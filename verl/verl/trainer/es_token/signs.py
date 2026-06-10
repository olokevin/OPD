"""Fixed Hadamard sign rails for es_token (pure math; no vLLM/GPU coupling).

Rail n's rank-1 weight perturbation at a layer is
    delta_W_n = sigma * (s_n (.) u_t) (r_n (.) v_t)^T
where s_n in {+-1}^{d_out}, r_n in {+-1}^{d_in} are FIXED for the whole run and
(u_t, v_t) is the shared per-token base noise. With Rademacher (u, v) and the
sign rows built here, the N rail directions at a token are EXACTLY
Frobenius-orthogonal:
    <dW_m, dW_n> = (sum_i s_m[i] s_n[i] u_i^2) * (sum_j r_m[j] r_n[j] v_j^2)
                 = (sum_i s_m[i] s_n[i]) * (...)   since u_i^2 = 1
                 = 0                                for m != n (Hadamard rows).

Construction: rows 1..N of the Sylvester Hadamard matrix H_M (M = smallest
power of two STRICTLY greater than N, so the all-ones row 0 is skipped and all
used rows are zero-sum), tiled across the dimension (s_n[i] = H_M[n+1, i % M]),
then multiplied by a FIXED random column flip c (the same flip for every rail,
so pairwise products are unchanged and orthogonality is preserved exactly when
dim % M == 0; for non-multiple dims the residual inner product is bounded by
(dim % M)/dim).
"""
from typing import Tuple

import torch

from verl.trainer.np.seeding import draw_noise, noise_seed


def next_pow2_above(n: int) -> int:
    """Smallest power of two STRICTLY greater than n (so row 0 can be skipped)."""
    m = 1
    while m <= int(n):
        m *= 2
    return m


def hadamard_matrix(m: int) -> torch.Tensor:
    """Sylvester Hadamard matrix H_m (m a power of two), entries in {-1, +1}."""
    assert m >= 1 and (m & (m - 1)) == 0, f"m must be a power of two, got {m}"
    H = torch.ones(1, 1)
    while H.shape[0] < m:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def flip_seed(global_seed: int, layer: str, side: str) -> int:
    """Deterministic seed for a layer's fixed column-flip vector (one per side).

    Reuses the NP noise_seed hash with a reserved namespace so flips can be
    regenerated anywhere (decode worker, assembly, tests) bit-identically.
    """
    return noise_seed(int(global_seed), 0, f"es_token_flip|{layer}|{side}", 0, 0)


def sign_rows(n_rails: int, dim: int, seed: int,
              dtype: torch.dtype = torch.float32,
              device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """[n_rails, dim] sign rows: tiled Hadamard rows 1..n_rails * fixed column flip.

    seed: the flip_seed(...) for this (layer, side); the flip is drawn via the
    same draw_noise("bernoulli") regenerator the noise path uses.
    """
    m = next_pow2_above(n_rails)
    H = hadamard_matrix(m)
    reps = (int(dim) + m - 1) // m
    rows = H[1:1 + int(n_rails)].repeat(1, reps)[:, : int(dim)]  # [N, dim]
    flip = draw_noise(int(seed), (int(dim),), torch.device("cpu"),
                      torch.float32, "bernoulli")                # {-1,+1}
    out = rows * flip[None, :]
    return out.to(device=device, dtype=dtype)


def build_layer_signs(layer_name: str, n_rails: int, d_out: int, d_in: int,
                      global_seed: int, dtype: torch.dtype,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(S [n_rails, d_out], R [n_rails, d_in]) for one layer, deterministic in
    (global_seed, layer_name) so every worker / the assembly / tests agree."""
    S = sign_rows(n_rails, d_out, flip_seed(global_seed, layer_name, "out"),
                  dtype=dtype, device=device)
    R = sign_rows(n_rails, d_in, flip_seed(global_seed, layer_name, "in"),
                  dtype=dtype, device=device)
    return S, R
