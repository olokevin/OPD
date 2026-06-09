"""Shared helpers for whole-expert pruning and merging on OLMoE.

OLMoE layout (transformers 4.56, verl env): each decoder layer has
``mlp = OlmoeSparseMoeBlock`` with ``.gate`` (nn.Linear(hidden, 64), the router)
and ``.experts`` (ModuleList of 64 OlmoeMLP, each gate/up/down nn.Linear).

PROTOCOL: compression NEVER edits router weights (``.gate``). When we drop or
merge experts we resize the ``.experts`` ModuleList AND the corresponding rows of
``.gate.weight`` so the router still outputs one logit per surviving expert — but
the surviving rows are COPIED verbatim from the original router (a pure
bookkeeping reindex, not a learned change). The router then re-adapts during
recovery SFT (RESEARCH_REVIEW fix #3). ``norm_topk_prob`` and ``top_k`` are
preserved; if a layer is reduced below top_k experts we clamp top_k for that
layer (logged).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


@torch.no_grad()
def collect_expert_activation_stats(model, loader, device):
    """Per-(layer,expert): routed-token count and sum of output L2 norms.

    Returns {(layer, expert): {"count": int, "sum_norm": float}} — enough for
    frequency, soft-logits-free REAP-style saliency, and importance weighting.
    """
    from collections import defaultdict
    stats = defaultdict(lambda: {"count": 0, "sum_norm": 0.0})
    handles = []

    def _make_hook(layer, expert):
        def _hook(mod, inputs, output):
            out = output if isinstance(output, torch.Tensor) else output[0]
            stats[(layer, expert)]["count"] += int(out.shape[0])
            stats[(layer, expert)]["sum_norm"] += float(out.float().norm(dim=-1).sum())
        return _hook

    for li in range(model.config.num_hidden_layers):
        for ei, ex in enumerate(model.model.layers[li].mlp.experts):
            handles.append(ex.register_forward_hook(_make_hook(li, ei)))
    model.eval()
    try:
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch.get("attention_mask")
            am = am.to(device) if am is not None else None
            model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return dict(stats)


def reap_saliency(stats, layer, expert):
    """REAP-style saliency = mean over routed tokens of ||E(x)||_2 (gate folded
    into the routed-token selection). Higher = more important."""
    s = stats.get((layer, expert), {"count": 0, "sum_norm": 0.0})
    return (s["sum_norm"] / s["count"]) if s["count"] else 0.0


def frequency(stats, layer, expert):
    return stats.get((layer, expert), {"count": 0})["count"]


@torch.no_grad()
def rebuild_layer_experts(model, layer: int, keep_idx: list[int],
                          new_experts: list[nn.Module] | None = None):
    """Replace layer's experts with `new_experts` (or the kept originals) and
    reindex the router rows to match. keep_idx maps new slot -> original expert.
    """
    blk = model.model.layers[layer].mlp
    device = blk.gate.weight.device
    dtype = blk.gate.weight.dtype
    old_gate_w = blk.gate.weight.data  # (num_experts, hidden)

    experts = new_experts if new_experts is not None else [blk.experts[i] for i in keep_idx]
    n_new = len(experts)

    # new router: copy the rows of the experts we keep (verbatim reindex)
    new_gate = nn.Linear(blk.gate.in_features, n_new, bias=blk.gate.bias is not None)
    new_gate = new_gate.to(device=device, dtype=dtype)
    new_gate.weight.data.copy_(old_gate_w[keep_idx])
    if blk.gate.bias is not None:
        new_gate.bias.data.copy_(blk.gate.bias.data[keep_idx])

    blk.gate = new_gate
    blk.experts = nn.ModuleList(experts)
    blk.num_experts = n_new
    if blk.top_k > n_new:
        logger.warning(f"layer {layer}: top_k {blk.top_k} > {n_new} experts; "
                       f"clamping top_k -> {n_new}")
        blk.top_k = n_new


@torch.no_grad()
def clone_expert(expert) -> nn.Module:
    import copy
    return copy.deepcopy(expert)
