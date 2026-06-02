"""Node-perturbation gradient estimation (pure math; no vLLM/GPU coupling).

Per step t, given per-sample losses L_t^(q), the clean baseline L_t, and the
regenerated perturbations u_q (rows of `u`), form the per-sample scalar scale,
then the per-token gradient g_t = (1/n) sum_q scale_q * u_q, and accumulate the
rank-1 outer product g_t (x) x_t into the layer's delta_W. See spec §1, §3.
"""
from typing import Optional

import torch


def sample_scale(
    L_q: torch.Tensor,            # [n_sample] per-sample loss at this token
    L_clean: Optional[float],     # baseline (clean) loss; required for "average"
    sigma: float,
    mode: str,                    # "average" | "grpo"
) -> torch.Tensor:
    """Per-sample scalar weighting of u_q. Lower L is better (minimization).

    Both modes are one-sided finite differences and MUST keep the 1/sigma factor
    so the scale carries the units of a directional derivative (dL/dy ~ (L_q-L0)/sigma).
    Without 1/sigma the resulting delta_W is off the true-gradient scale by ~sigma,
    which (combined with the dropped 1/sigma) is what made the old grpo path need an
    absurd lr. See scripts/zo_opd/results/ANALYSIS.md.
      - average: (L_q - L_clean) / sigma         (baseline = clean row's loss)
      - grpo:    ((L_q - mean_q) / std_q) / sigma (z-scored advantage, then /sigma)
                 keeps BOTH the grpo standardization (1/std) and the finite-
                 difference scale (1/sigma).
    """
    if mode == "average":
        if L_clean is None:
            raise ValueError("average mode requires L_clean baseline")
        return (L_q - float(L_clean)) / sigma
    if mode == "grpo":
        # ((L_q - mean_q) / std_q) / sigma  -- BOTH scalings:
        #   * 1/std   : grpo standardization (per-token, unit-variance advantage)
        #   * 1/sigma : finite-difference scale (so the result has dL/dy units)
        # Keeping both means the per-token weighting is the z-scored advantage but
        # still carries the 1/sigma directional-derivative scale.
        mean = L_q.mean()
        std = L_q.std(unbiased=False) + 1e-8
        return ((L_q - mean) / std) / sigma
    raise ValueError(f"unknown grad_estimate_sample mode: {mode!r}")


def accumulate_delta_w(
    delta_w: torch.Tensor,        # [d_out, d_in] in/out accumulator
    scales: torch.Tensor,         # [n_sample]
    u: torch.Tensor,              # [n_sample, d_out] regenerated perturbations
    x_t: torch.Tensor,            # [d_in] captured clean input to the layer
    normalize: bool = False,
    eps: float = 1e-6,
) -> None:
    """delta_w += outer( (1/n) sum_q scales_q * (u_q [/ ||u_q||^2]), x_t )."""
    u_eff = u
    if normalize:
        sq = (u * u).sum(dim=-1, keepdim=True).clamp_min(eps)  # [n_sample,1]
        u_eff = u / sq
    g_t = (scales[:, None] * u_eff).mean(dim=0)                # [d_out]
    delta_w.add_(torch.outer(g_t.to(delta_w.dtype), x_t.to(delta_w.dtype)))
