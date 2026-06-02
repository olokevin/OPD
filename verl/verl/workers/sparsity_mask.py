"""SparseGPT sparsity-mask preservation hook for the verl actor.

When training a SparseGPT-pruned model with verl's vanilla Adam actor update,
zero entries become nonzero after the very first step (the gradient on a zero
weight is generally nonzero; Adam moments amplify any small grad). This module
provides a *post-optimizer* re-zero hook that snapshots the pruning mask once
at model build time and re-zeros those entries after every ``optimizer.step``.

Activation is gated by the environment variable
``SPARSEGPT_PRESERVE_MASK=1`` (default off — verl's other PEFT paths are
unaffected). Mask granularity: every ``nn.Linear`` module **except** any whose
name contains a token from ``SPARSEGPT_PRESERVE_SKIP`` (default
``lm_head,embed``). Only entries that are *already exactly zero* at build time
are masked, so this is a no-op for non-sparse runs.

The mask is stored on the same device as the parameter and is kept around for
the lifetime of the actor module. For FSDP1 (full_shard) and FSDP2 the actor
runs single-GPU here so we only deal with full local tensors.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


_ATTR = "_sparsegpt_zero_mask"


def is_enabled() -> bool:
    return os.environ.get("SPARSEGPT_PRESERVE_MASK", "0") not in ("0", "false", "False", "")


def _skip_tokens() -> Tuple[str, ...]:
    raw = os.environ.get("SPARSEGPT_PRESERVE_SKIP", "lm_head,embed")
    return tuple(tok for tok in (s.strip() for s in raw.split(",")) if tok)


def attach_masks(model: nn.Module) -> int:
    """Build a zero-mask for every Linear weight that already has any zeros.

    Returns the number of masked Linear weights.
    """
    skip = _skip_tokens()
    n_masked = 0
    n_zeros = 0
    n_total = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(tok in name for tok in skip):
            continue
        w = mod.weight
        if w is None:
            continue
        with torch.no_grad():
            zero_mask = (w == 0)
            local_zero = int(zero_mask.sum().item())
            local_total = w.numel()
        if local_zero == 0:
            continue
        # Store as the same dtype/device as the weight; bool is fine and 8x smaller.
        mod.register_buffer(_ATTR, zero_mask, persistent=False)
        n_masked += 1
        n_zeros += local_zero
        n_total += local_total
    if n_masked == 0:
        print("[sparsity_mask] no zero-weight Linear modules found; nothing to preserve.")
    else:
        frac = n_zeros / max(n_total, 1)
        print(f"[sparsity_mask] preserving zeros in {n_masked} Linear modules "
              f"({n_zeros / 1e9:.3f}B / {n_total / 1e9:.3f}B = {frac * 100:.2f}% zeros)")
    return n_masked


@torch.no_grad()
def reapply_masks(model: nn.Module) -> None:
    """Re-zero masked weights. Call AFTER optimizer.step()."""
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        mask = getattr(mod, _ATTR, None)
        if mask is None:
            continue
        w = mod.weight
        # DTensor (FSDP2) and FlatParameter (FSDP1) both expose .data; the
        # zero in-place is safe because we masked at the same shape.
        w.data.masked_fill_(mask, 0.0)


@torch.no_grad()
def mask_gradients(model: nn.Module) -> None:
    """Zero gradients at masked positions BEFORE clip / optimizer.step.

    Optional companion to reapply_masks. Without this, grad_norm reflects the
    full-dense grad and would be clipped too aggressively. With it, the
    optimizer never updates masked entries, so Adam moments stay zero — both
    hooks together guarantee strict mask preservation.
    """
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        mask = getattr(mod, _ATTR, None)
        if mask is None:
            continue
        w = mod.weight
        if w.grad is None:
            continue
        w.grad.data.masked_fill_(mask, 0.0)


def has_any_masks(model: nn.Module) -> bool:
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and getattr(mod, _ATTR, None) is not None:
            return True
    return False


@torch.no_grad()
def report_realised_sparsity(model: nn.Module) -> Dict[str, float]:
    """For debugging: total nonzero fraction over masked Linear weights."""
    n_zero_now = 0
    n_total = 0
    n_should_be_zero = 0
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        mask = getattr(mod, _ATTR, None)
        if mask is None:
            continue
        w = mod.weight.data
        n_total += w.numel()
        n_zero_now += (w == 0).sum().item()
        n_should_be_zero += int(mask.sum().item())
    return {
        "linear_nz_total": float(n_total),
        "linear_z_now": float(n_zero_now),
        "linear_z_expected": float(n_should_be_zero),
        "ratio_z_now_vs_expected": (n_zero_now / max(n_should_be_zero, 1)),
    }
