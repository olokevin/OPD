"""Dual-axis compression-budget accounting for the recovery atlas.

RESEARCH_REVIEW fix #2: "retain 0.75" is NOT a common budget across families in a
top-k MoE. We therefore report TWO axes for every compressed checkpoint so a
family-level inversion can be checked against each (an inversion that holds under
both axes is the strong result):

  - storage_retain   : nonzero expert params / original expert params.
                       The honest "how much smaller is it on disk" number.
                       Whole-expert drop and width-shrink both shrink storage;
                       unstructured SparseGPT zeroes weights (storage only if
                       stored sparse), so storage_retain counts nonzeros.
  - active_retain    : expected active expert-MLP nonzeros PER TOKEN under top-k
                       routing / original active nonzeros per token.
                       This is the capacity a token actually sees. Expert-drop
                       leaves it ~unchanged (still top-8 over the survivors);
                       width-shrink cuts it; unstructured cuts it by sparsity.

Only EXPERT params count (attention + router are frozen / untouched).
"""
from __future__ import annotations

import torch
from loguru import logger


def _is_expert_linear(name: str) -> bool:
    return ".experts." in name and name.endswith(
        ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))


@torch.no_grad()
def expert_param_stats(model) -> dict:
    """Total and nonzero EXPERT params, per-layer expert counts, and the
    baseline-time num_experts (captured from the live ModuleList, NOT config,
    so it is robust to later config.num_experts edits)."""
    total = nonzero = 0
    per_layer_experts: dict[int, int] = {}
    for name, p in model.named_parameters():
        if not _is_expert_linear(name):
            continue
        total += p.numel()
        nonzero += int((p != 0).sum().item())
    for li in range(model.config.num_hidden_layers):
        blk = model.model.layers[li].mlp
        per_layer_experts[li] = len(blk.experts)
    n_layers = model.config.num_hidden_layers
    n_experts0 = per_layer_experts[0]
    return {"total": total, "nonzero": nonzero, "per_layer_experts": per_layer_experts,
            "n_layers": n_layers, "n_experts": n_experts0,
            "nz_per_expert": total / (n_layers * n_experts0) if n_experts0 else 0}


@torch.no_grad()
def compute_budget(model, baseline: dict, *, top_k: int | None = None) -> dict:
    """Both budget axes for `model`, relative to `baseline` = expert_param_stats(uncompressed).

    `top_k` defaults to the model's num_experts_per_tok (active experts per token).
    """
    # ORIGINAL top_k (a token still activates this many experts post-drop),
    # taken from the baseline, not the possibly-clamped current config.
    if top_k is None:
        top_k = baseline.get("top_k", model.config.num_experts_per_tok)
    cur = expert_param_stats(model)
    base_total = baseline["total"]
    base_nz_per_expert = baseline["nz_per_expert"]   # baseline-time, robust to config edits

    storage_retain = cur["nonzero"] / base_total if base_total else 0.0

    # Active retain: per layer, a token activates top_k experts. The expected
    # active nonzeros per token = top_k * (mean nonzeros per surviving expert),
    # but expert COUNT also shrank (drop), which does NOT reduce per-token active
    # capacity as long as >= top_k experts survive. So active capacity per token
    # ~= top_k * (nonzeros-per-expert / original-nonzeros-per-expert).
    # We compute it layer-wise then average weighted by original layer params.
    import re
    # nonzeros per layer (current) and per-expert avg
    cur_layer_nz: dict[int, int] = {}
    cur_layer_ne: dict[int, int] = {}
    for li in range(model.config.num_hidden_layers):
        blk = model.model.layers[li].mlp
        cur_layer_ne[li] = len(blk.experts)
        nz = 0
        for e in blk.experts:
            for m in (e.gate_proj, e.up_proj, e.down_proj):
                nz += int((m.weight != 0).sum().item())
        cur_layer_nz[li] = nz

    active_ratios = []
    for li in range(model.config.num_hidden_layers):
        ne = max(cur_layer_ne[li], 1)
        cur_nz_per_expert = cur_layer_nz[li] / ne
        # tokens still pick min(top_k, ne) experts; capacity/token relative to base
        eff_topk = min(top_k, ne)
        active = eff_topk * cur_nz_per_expert
        base_active = top_k * base_nz_per_expert
        active_ratios.append(active / base_active if base_active else 0.0)
    active_retain = sum(active_ratios) / len(active_ratios) if active_ratios else 0.0

    return {
        "storage_retain": round(storage_retain, 4),
        "active_retain": round(active_retain, 4),
        "expert_params_total": base_total,
        "expert_params_nonzero": cur["nonzero"],
        "experts_per_layer": cur["per_layer_experts"],
    }


def log_budget(b: dict, tag: str = "") -> None:
    logger.info(f"[budget{(' ' + tag) if tag else ''}] storage_retain="
                f"{b['storage_retain']} active_retain={b['active_retain']} "
                f"nonzero={b['expert_params_nonzero']:,}/{b['expert_params_total']:,}")
