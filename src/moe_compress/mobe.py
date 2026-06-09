"""MoBE — Mixture-of-Basis-Experts (arXiv 2508.05257) for OLMoE experts.

Idea: experts within a layer are highly redundant, so factor each expert weight
matrix as a per-expert low-rank map composed with a SMALL SHARED BASIS:

    W^i  ≈  A^i @ ( Σ_j alpha^{ij} B^j )

  - B^j ∈ R^{r×d_in}  : `m` basis matrices SHARED across all experts in the layer
  - A^i ∈ R^{d_out×r} : per-expert left factor (the specialization)
  - alpha^i ∈ R^m     : per-expert nonneg coefficients combining the basis

This is TRAINING-FREE in the study sense: it never fine-tunes the LLM and uses no
data. It solves a post-hoc WEIGHT-RECONSTRUCTION problem min Σ_i ‖W^i − Ŵ^i‖_F^2
with Adam on the factors themselves (paper uses lr≈0.07, z-score weight norm).
We apply it per layer to the stacked gate_proj and up_proj of the layer's experts
(the two d_out×d_in matrices); down_proj is left to the SVD/Nystrom-style structured
path is NOT used here — MoBE replaces gate+up with the factored form materialized
back to a single equivalent dense Linear so the per-Linear OLMoE checkpoint reloads
unchanged in shape... EXCEPT the stored weight is now low-rank (A@B), i.e. the saved
matrix is dense but rank-deficient. To realise a real PARAM reduction we instead keep
the model's forward identical by replacing each expert's gate/up Linear with the
reconstructed (rank-limited) weight; storage_retain is then measured by the EFFECTIVE
factor params (A^i + shared B + alpha) / original, reported via budget.compute_budget
using a sparsity-equivalent mask is not applicable, so MoBE reports its OWN factor
budget (see compress_olmoe handling).

Budget control: choose (r, m) per layer so factor params hit the retain target.
  factor params (per layer, for ONE of {gate,up}) = E·(d_out·r) + m·(r·d_in) + E·m
  original (per layer, ONE matrix)                = E·(d_out·d_in)
We solve for r given a chosen m (default m = max(2, E//8)) to match `retain`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _solve_rank(E, d_out, d_in, m, retain):
    """Largest r such that E*d_out*r + m*r*d_in + E*m <= retain*E*d_out*d_in."""
    budget = retain * E * d_out * d_in - E * m
    denom = E * d_out + m * d_in
    r = int(budget // denom)
    return max(1, r)


@torch.no_grad()
def _init_factors(W, m, r, device):
    """W: (E, d_out, d_in) stacked expert matrices. Init shared basis B from the
    top-r right singular vectors of the mean expert, A^i by projection, alpha uniform."""
    E, d_out, d_in = W.shape
    Wf = W.float()
    mean_w = Wf.mean(0)                                   # (d_out, d_in)
    # top-r right singular directions of the mean expert -> shared basis seed
    _, _, Vh = torch.linalg.svd(mean_w, full_matrices=False)
    B0 = Vh[:r].to(device)                                # (r, d_in)
    B = B0.unsqueeze(0).repeat(m, 1, 1).clone()           # (m, r, d_in)
    # small noise so bases differentiate
    B += 0.01 * B.std() * torch.randn_like(B)
    # store pre-activation alpha so softplus(alpha) ~= 1/m at init
    import torch.nn.functional as _F
    target = torch.tensor(1.0 / m)
    alpha_pre = torch.log(torch.expm1(target).clamp_min(1e-6))  # inverse-softplus
    alpha = torch.full((E, m), float(alpha_pre), device=device)  # (E, m), pre-activation
    # A^i = W^i @ B_eff^i^+  (least squares init), B_eff^i = sum_j softplus(alpha_ij) B_j
    Beff = torch.einsum("em,mrd->erd", _F.softplus(alpha), B)   # (E, r, d_in)
    A = torch.empty(E, d_out, r, device=device)
    for i in range(E):
        # solve A_i (d_out x r): W_i (d_out x d_in) ~ A_i @ Beff_i (r x d_in)
        sol = torch.linalg.lstsq(Beff[i].T, Wf[i].T.to(device)).solution  # (r, d_out)
        A[i] = sol.T
    return A.contiguous(), B.contiguous(), alpha.contiguous()


def _reconstruct(A, B, alpha):
    Beff = torch.einsum("em,mrd->erd", alpha, B)          # (E, r, d_in)
    return torch.einsum("eor,erd->eod", A, Beff)          # (E, d_out, d_in)


@torch.no_grad()
def _materialize_into(experts, attr, W_hat):
    """Write reconstructed weights W_hat (E,d_out,d_in) back into each expert's
    `attr` Linear (.weight). Shape unchanged (dense but low-rank)."""
    for i, ex in enumerate(experts):
        getattr(ex, attr).weight.data.copy_(W_hat[i].to(getattr(ex, attr).weight.dtype))


def mobe_compress_layer(experts, m, retain, device, iters=400, lr=0.02):
    """Factor the layer's gate_proj and up_proj across experts with a shared basis,
    optimize reconstruction, write the rank-limited reconstruction back in place.
    Returns the effective factor-param retain for this layer (avg over gate/up)."""
    retains = []
    for attr in ("gate_proj", "up_proj"):
        W = torch.stack([getattr(ex, attr).weight.data for ex in experts]).to(device)  # (E,dout,din)
        E, d_out, d_in = W.shape
        r = _solve_rank(E, d_out, d_in, m, retain)
        A0, B0, alpha0 = _init_factors(W, m, r, device)
        # fresh leaf parameters so Adam can update them (init tensors are non-leaf)
        A = A0.detach().clone().requires_grad_(True)
        B = B0.detach().clone().requires_grad_(True)
        alpha = alpha0.detach().clone().requires_grad_(True)
        Wf = W.float()
        scale = Wf.std() + 1e-8
        opt = torch.optim.Adam([A, B, alpha], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            # softplus keeps coeffs positive with a SMOOTH always-on gradient
            # (plain relu collapsed: a single big step pushed alpha<0 -> dead unit).
            Wh = _reconstruct(A, B, F.softplus(alpha))
            loss = ((Wh - Wf) / scale).pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            Wh = _reconstruct(A, B, F.softplus(alpha))
        _materialize_into(experts, attr, Wh)
        factor_params = E * d_out * r + m * r * d_in + E * m
        orig = E * d_out * d_in
        retains.append(factor_params / orig)
        logger.info(f"    {attr}: m={m} r={r} factor_retain={factor_params/orig:.3f} "
                    f"final_recon_loss={loss.item():.4e}")
    return sum(retains) / len(retains)


def mobe_compress_model(model, *, retain, device, m=None, iters=400, lr=0.02):
    """Apply MoBE to every layer's experts. Leaves down_proj untouched so the
    storage budget is realised on gate+up (2/3 of expert params); the effective
    factor-retain on gate+up is set so the OVERALL expert retain ~= `retain`.
    (down_proj is 1/3 of params and kept full, so to hit overall `retain` we
    compress gate+up to retain' = (3*retain - 1)/2.)"""
    # solve gate+up target so overall (incl. full down) hits `retain`
    gu_retain = max(0.05, (3.0 * retain - 1.0) / 2.0)
    if model.config.num_experts is None:
        raise RuntimeError("num_experts missing")
    logger.info(f"mobe: overall retain={retain} -> gate+up factor_retain={gu_retain:.3f} "
                f"(down_proj kept full)")
    eff = []
    for li in range(model.config.num_hidden_layers):
        experts = model.model.layers[li].mlp.experts
        E = len(experts)
        mm = m if m is not None else max(2, E // 8)
        r = mobe_compress_layer(experts, mm, gu_retain, device, iters=iters, lr=lr)
        eff.append(r)
        if li == 0 or li == model.config.num_hidden_layers - 1:
            logger.info(f"  layer {li}: gate+up factor_retain≈{r:.3f}")
    logger.info(f"mobe done: mean gate+up factor_retain={sum(eff)/len(eff):.3f}")
