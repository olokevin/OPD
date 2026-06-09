"""The 6 focal expert-compression methods (3 families x 2) + magnitude control.

Shared interface (all in-place on the model; ROUTER WEIGHTS UNTOUCHED at compress
time except the verbatim reindex done by rebuild_layer_experts when experts are
removed/merged):

    compress(model, *, retain, calib_loader, seed, device) -> None

`retain` is the TARGET storage-retain fraction of expert params (0.75, 0.50, ...).
Families:
  - expert-removal:  random_drop, reap_drop          (drop whole experts)
  - merge:           slimqwen_merge, hcsmoe_merge     (fuse experts)
  - weight-approx:   svd_llm_v2, sparsegpt            (shrink/sparsify each expert)
  - control:         magnitude                        (global magnitude prune)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from . import expert_ops as eo  # noqa: E402

FAMILIES = {
    "random_drop": "expert-removal",
    "reap_drop": "expert-removal",
    "slimqwen_merge": "merge",
    "hcsmoe_merge": "merge",
    "nystrom": "weight-approx",
    "nystrom_combined": "weight-approx",
    "nystrom_combined_fwdnat_bwdot3": "weight-approx",
    "nystrom_combined_ot3": "weight-approx",
    "svd_llm_v2": "weight-approx",
    "sparsegpt": "weight-approx",
    "mobe": "weight-approx",
    "magnitude": "control",
}


# ----------------------------------------------------------------------------
# Family 1: whole-expert removal (drop). retain 0.75 -> keep 48/64 experts/layer.
# ----------------------------------------------------------------------------
def _n_keep(model, retain):
    return max(1, int(round(retain * model.config.num_experts)))


def random_drop(model, *, retain, calib_loader, seed, device):
    g = torch.Generator().manual_seed(seed)
    n_keep = _n_keep(model, retain)
    for li in range(model.config.num_hidden_layers):
        ne = len(model.model.layers[li].mlp.experts)
        keep = torch.randperm(ne, generator=g)[:n_keep].sort().values.tolist()
        eo.rebuild_layer_experts(model, li, keep)
    logger.info(f"random_drop: kept {n_keep}/{model.config.num_experts} experts/layer (seed={seed})")


def reap_drop(model, *, retain, calib_loader, seed, device):
    stats = eo.collect_expert_activation_stats(model, calib_loader, device)
    n_keep = _n_keep(model, retain)
    for li in range(model.config.num_hidden_layers):
        ne = len(model.model.layers[li].mlp.experts)
        sal = [(eo.reap_saliency(stats, li, ei), ei) for ei in range(ne)]
        keep = sorted([ei for _, ei in sorted(sal, reverse=True)[:n_keep]])
        eo.rebuild_layer_experts(model, li, keep)
    logger.info(f"reap_drop: kept top-{n_keep} experts/layer by REAP saliency")


# ----------------------------------------------------------------------------
# Family 2: expert merging. retain 0.75 -> consolidate to 48 experts/layer.
# SlimQwen partial-preservation: keep top-half intact, build the rest by merging
# discarded experts into their most-similar surviving base (importance-weighted).
# ----------------------------------------------------------------------------
@torch.no_grad()
def _expert_vec(expert):
    """Flat concat of an expert's weights, for similarity."""
    return torch.cat([expert.gate_proj.weight.flatten(),
                      expert.up_proj.weight.flatten(),
                      expert.down_proj.weight.flatten()]).float()


@torch.no_grad()
def _convex_merge(base, other, w_base, w_other):
    """In-place: base <- (w_base*base + w_other*other)/(w_base+w_other)."""
    z = w_base + w_other + 1e-9
    a, b = w_base / z, w_other / z
    for mb, mo in ((base.gate_proj, other.gate_proj),
                   (base.up_proj, other.up_proj),
                   (base.down_proj, other.down_proj)):
        mb.weight.data.mul_(a).add_(mo.weight.data, alpha=b)
    return base


def slimqwen_merge(model, *, retain, calib_loader, seed, device):
    stats = eo.collect_expert_activation_stats(model, calib_loader, device)
    n_keep = _n_keep(model, retain)
    for li in range(model.config.num_hidden_layers):
        blk = model.model.layers[li].mlp
        ne = len(blk.experts)
        imp = {ei: eo.reap_saliency(stats, li, ei) for ei in range(ne)}
        ranked = sorted(range(ne), key=lambda e: imp[e], reverse=True)
        n_intact = max(1, n_keep // 2)
        intact = set(ranked[:n_intact])
        bases = ranked[:n_keep]                 # all survivors are merge bases
        discarded = ranked[n_keep:]
        new_experts = [eo.clone_expert(blk.experts[e]) for e in bases]
        base_vecs = [_expert_vec(blk.experts[e]) for e in bases]
        for d in discarded:
            dv = _expert_vec(blk.experts[d])
            best, best_sim = None, -2.0
            for slot, e in enumerate(bases):
                if e in intact:
                    continue
                sim = torch.nn.functional.cosine_similarity(dv, base_vecs[slot], dim=0).item()
                if sim > best_sim:
                    best_sim, best = sim, slot
            if best is None:  # all survivors intact-preserved -> global best-sim
                best = max(range(len(bases)),
                           key=lambda s: torch.nn.functional.cosine_similarity(
                               dv, base_vecs[s], dim=0).item())
            _convex_merge(new_experts[best], blk.experts[d], imp[bases[best]], imp[d])
        eo.rebuild_layer_experts(model, li, bases, new_experts=new_experts)
    logger.info(f"slimqwen_merge: {model.config.num_experts}->{n_keep} experts/layer "
                f"(partial-preservation, importance-weighted cosine merge)")


def hcsmoe_merge(model, *, retain, calib_loader, seed, device):
    """HC-SMoE: agglomerative clustering on mean expert OUTPUTS, frequency-weighted
    merge within each cluster (router-independent grouping)."""
    from collections import defaultdict
    import numpy as np
    sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    handles = []

    @torch.no_grad()
    def _mk(li, ei):
        def _h(mod, inp, out):
            o = out if isinstance(out, torch.Tensor) else out[0]
            s = o.float().sum(0)
            sums[(li, ei)] = s if sums[(li, ei)] is None else sums[(li, ei)] + s
            counts[(li, ei)] += int(o.shape[0])
        return _h

    for li in range(model.config.num_hidden_layers):
        for ei, ex in enumerate(model.model.layers[li].mlp.experts):
            handles.append(ex.register_forward_hook(_mk(li, ei)))
    model.eval()
    with torch.no_grad():
        for batch in calib_loader:
            ids = batch["input_ids"].to(device)
            am = batch.get("attention_mask")
            am = am.to(device) if am is not None else None
            model(input_ids=ids, attention_mask=am, use_cache=False)
    for h in handles:
        h.remove()

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        have_scipy = True
    except Exception:  # noqa: BLE001
        have_scipy = False
        logger.warning("scipy unavailable; hcsmoe degenerates to no-merge fallback")

    n_keep = _n_keep(model, retain)
    for li in range(model.config.num_hidden_layers):
        blk = model.model.layers[li].mlp
        ne = len(blk.experts)
        means = []
        for ei in range(ne):
            c = counts[(li, ei)]
            v = (sums[(li, ei)] / c) if (c and sums[(li, ei)] is not None) \
                else torch.zeros(model.config.hidden_size, device=device)
            means.append(v.cpu().numpy())
        X = np.stack(means)
        if have_scipy and ne > n_keep:
            Z = linkage(pdist(X, metric="cosine"), method="average")
            labels = fcluster(Z, t=n_keep, criterion="maxclust")
        else:
            labels = np.arange(ne)
        clusters = defaultdict(list)
        for ei, lab in enumerate(labels):
            clusters[int(lab)].append(ei)
        new_experts, keep_repr = [], []
        for _, members in sorted(clusters.items()):
            members_sorted = sorted(members, key=lambda e: counts[(li, e)], reverse=True)
            rep = members_sorted[0]
            merged = eo.clone_expert(blk.experts[rep])
            acc_w = max(counts[(li, rep)], 1)
            for m in members_sorted[1:]:
                _convex_merge(merged, blk.experts[m], acc_w, max(counts[(li, m)], 1))
                acc_w += max(counts[(li, m)], 1)
            new_experts.append(merged)
            keep_repr.append(rep)
        eo.rebuild_layer_experts(model, li, keep_repr, new_experts=new_experts)
    logger.info(f"hcsmoe_merge: clustered -> ~{n_keep} experts/layer "
                f"(scipy={have_scipy}, freq-weighted output-cluster merge)")


# ----------------------------------------------------------------------------
# Family 3: per-expert weight approximation (reuse src/compress).
# retain 0.75 -> intermediate 1024->768 (SVD/Nystrom) / 25% sparse (SparseGPT).
# ----------------------------------------------------------------------------
def nystrom(model, *, retain, calib_loader, seed, device):
    """Per-expert structured intermediate-dim shrink via forward-only Nystrom on
    each expert gate/up/down triplet (reuses src/compress/structured/nystrom.py);
    attn + router skipped. retain 0.75 -> intermediate 1024->768."""
    from compress.calibration import collect_covariances_reweighted
    from compress.structured.nystrom import find_mlp_triplets, nystrom_compress_model
    skip = ("lm_head", "gate")
    triplets = find_mlp_triplets(model, skip)
    down_keys = {d for (_p, _g, _u, d) in triplets}
    cov = collect_covariances_reweighted(model, calib_loader, device=device,
                                         skip_layers=skip, reweight="sequence")
    down_cov = {k: v for k, v in cov.items() if k in down_keys}
    nystrom_compress_model(model, {k: v.clone() for k, v in down_cov.items()},
                           sparsity=1.0 - retain, skip_layers=skip, device=device)
    logger.info(f"nystrom: per-expert forward-only Nystrom intermediate shrink, retain={retain} "
                f"({len(triplets)} expert triplets)")


def svd_llm_v2(model, *, retain, calib_loader, seed, device):
    """Per-expert-MATRIX whitening SVD (SVD-LLM-V2): forward input-covariance
    whitening + truncated SVD on EACH expert gate/up/down linear independently
    (vs nystrom which factors the gate/up/down triplet jointly). Reuses
    src/compress/svd/svd_llm_v2.py. Like MoBE, the on-disk weight is dense-but-
    low-rank (factored U_r@V_r materialized back), so the FACTOR retain is the
    compression_ratio, not the nonzero count."""
    from compress.calibration import collect_covariances_reweighted
    from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model
    from compress.integration import materialize_svd_to_linear
    skip = ("lm_head", "gate")
    cov = collect_covariances_reweighted(model, calib_loader, device=device,
                                         skip_layers=skip, reweight="sequence")
    # only expert linears (drop attn + router)
    exp_cov = {k: v for k, v in cov.items() if ".experts." in k}
    svd_llm_v2_compress_model(model, {k: v.clone() for k, v in exp_cov.items()},
                              compression_ratio=retain, skip_layers=skip,
                              device=device, objective="forward")
    materialize_svd_to_linear(model)
    model._mobe_factor_retain = retain  # reuse the factor-budget reporting path
    logger.info(f"svd_llm_v2: per-expert-matrix whitening SVD, factor_retain={retain} "
                f"({len(exp_cov)} expert linears)")


def nystrom_combined(model, *, retain, calib_loader, seed, device):
    """Forward+backward calibrated Nystrom (joint kernel C_f & C_b): like `nystrom`
    (structured intermediate shrink on the gate/up/down triplet) but the Nystrom
    column selection / kernel uses BOTH the forward hidden-activation covariance
    C_f AND the backward hidden-gradient covariance C_b. Needs a CE backward pass
    for the gradients. Reuses src/compress/structured/nystrom.py."""
    from compress.calibration import collect_nystrom_combined_statistics
    from compress.structured.nystrom import find_mlp_triplets, nystrom_combined_compress_model
    skip = ("lm_head", "gate")
    triplets = find_mlp_triplets(model, skip)
    # {down_name: (C_f, C_b)} pairs over expert down_proj (CE-loss fwd+bwd)
    stats = collect_nystrom_combined_statistics(model, calib_loader, device=device,
                                                skip_layers=skip, reweight="sequence")
    stats = {k: v for k, v in stats.items() if ".experts." in k}
    nystrom_combined_compress_model(model, stats, sparsity=1.0 - retain,
                                    skip_layers=skip, device=device)
    logger.info(f"nystrom_combined: per-expert fwd+bwd Nystrom shrink, retain={retain} "
                f"({len(triplets)} expert triplets)")


def _combined_stats_on(model, loader, device, skip):
    """{down_name: (C_f, C_b)} from one fwd+bwd pass over `loader`."""
    from compress.calibration import collect_nystrom_combined_statistics
    return collect_nystrom_combined_statistics(model, loader, device=device,
                                               skip_layers=skip, reweight="sequence")


def _run_combined_split(model, *, retain, device, fwd_jsonl, bwd_jsonl, label):
    """Nystrom-combined where C_f comes from one data source and C_b from another.
    Collects combined stats on EACH loader (the fwd-only run uses its C_f, the
    bwd run uses its C_b), then zips them into (C_f, C_b) pairs. (Two passes since
    fwd & bwd are coupled within a single collect call.)"""
    from transformers import AutoTokenizer
    from compress.structured.nystrom import find_mlp_triplets, nystrom_combined_compress_model
    from moe_compress.calib import build_calib_loader_from, NATIVE_JSONL, OT3_JSONL
    skip = ("lm_head", "gate")
    tok = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True) \
        if getattr(model.config, "_name_or_path", None) else None
    if tok is None:
        tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct",
                                            trust_remote_code=True)
    triplets = find_mlp_triplets(model, skip)
    fwd_loader = build_calib_loader_from(tok, fwd_jsonl)
    cf_stats = _combined_stats_on(model, fwd_loader, device, skip)   # keep C_f
    if bwd_jsonl == fwd_jsonl:
        cb_stats = cf_stats                                          # recipe 2: same source
    else:
        bwd_loader = build_calib_loader_from(tok, bwd_jsonl)
        cb_stats = _combined_stats_on(model, bwd_loader, device, skip)  # keep C_b
    # zip: C_f from cf_stats, C_b from cb_stats (only experts seen in both)
    stats = {}
    for k in cf_stats:
        if k in cb_stats:
            stats[k] = (cf_stats[k][0], cb_stats[k][1])
    n_pairs = len(stats)  # capture BEFORE compress (it del's entries as it consumes)
    logger.info(f"{label}: fwd<-{Path(fwd_jsonl).name} bwd<-{Path(bwd_jsonl).name}, "
                f"retain={retain} ({len(triplets)} triplets, {n_pairs} stat pairs)")
    nystrom_combined_compress_model(model, stats, sparsity=1.0 - retain,
                                    skip_layers=skip, device=device)


def nystrom_combined_fwdnat_bwdot3(model, *, retain, calib_loader, seed, device):
    """Recipe 1 (split): forward C_f from prompt+OLMoE-native response; backward
    C_b from prompt+OpenThoughts3 training response."""
    from moe_compress.calib import NATIVE_JSONL, OT3_JSONL
    _run_combined_split(model, retain=retain, device=device,
                        fwd_jsonl=NATIVE_JSONL, bwd_jsonl=OT3_JSONL,
                        label="nystrom_combined[fwd=native,bwd=ot3]")


def nystrom_combined_ot3(model, *, retain, calib_loader, seed, device):
    """Recipe 2 (both target): forward AND backward from prompt+OpenThoughts3
    training response."""
    from moe_compress.calib import OT3_JSONL
    _run_combined_split(model, retain=retain, device=device,
                        fwd_jsonl=OT3_JSONL, bwd_jsonl=OT3_JSONL,
                        label="nystrom_combined[fwd=ot3,bwd=ot3]")


def mobe(model, *, retain, calib_loader, seed, device):
    """MoBE (Mixture-of-Basis-Experts, arXiv 2508.05257): factor each layer's
    expert gate/up matrices as A^i @ (Σ_j alpha^ij B^j) with a small SHARED basis
    {B^j}, exploiting cross-expert redundancy. TRAINING-FREE in the study sense
    (no model FT, no data): a post-hoc weight-reconstruction solved with Adam.

    NOTE: MoBE writes back a dense-but-low-rank weight (same shape), so the nonzero
    storage budget can't see its compression. It stashes the achieved FACTOR retain
    on `model._mobe_factor_retain` for compress_olmoe to report instead."""
    from moe_compress.mobe import mobe_compress_model
    mobe_compress_model(model, retain=retain, device=device)
    # effective factor retain (gate+up compressed, down full = 1/3 of params)
    gu = max(0.05, (3.0 * retain - 1.0) / 2.0)
    model._mobe_factor_retain = retain  # target overall retain (gate+up at gu, down full)
    logger.info(f"mobe: overall expert factor_retain≈{retain} (gate+up at {gu:.3f}, down full)")


def sparsegpt(model, *, retain, calib_loader, seed, device):
    """Per-expert SparseGPT unstructured pruning at sparsity (1-retain).

    scope='mlp' targets every expert linear (mlp.experts.*); skip_layers=('gate',)
    excludes the router 'gate' leaf (also under .mlp), and 'self_attn' never matches
    the mlp scope so attention is left untouched. All layers pruned (thirds 1,2,3).

    memory_limit_gb=2.5 forces ONE decoder layer's experts (192 linears ~= 2.25GB
    of Hessians) per group -> 16 groups / 16 forward passes. The default 30GB packs
    ~2900 hooks into a single forward and hangs (36GB total Hessian footprint over
    3072 expert linears)."""
    from compress.unstructured import sparsegpt_prune
    sparsity = 1.0 - retain
    sparsegpt_prune(model, calib_loader, sparsity=sparsity, device=device,
                    scope="mlp", skip_layers=("lm_head", "gate"),
                    thirds_to_prune=(1, 2, 3), memory_limit_gb=2.5)
    logger.info(f"sparsegpt: per-expert unstructured sparsity={sparsity} (storage-only)")


def magnitude(model, *, retain, calib_loader, seed, device):
    """Control: global magnitude prune of expert weights to sparsity (1-retain).

    Threshold = global magnitude quantile estimated from a fp32 SAMPLE of the
    6.4B expert weights (torch.kthvalue on the full bf16 tensor is unreliable /
    memory-heavy and silently zeroed nothing — see budget check)."""
    import torch as _t
    sparsity = 1.0 - retain
    names = [n for n, _ in model.named_parameters()
             if ".experts." in n and n.endswith(("gate_proj.weight", "up_proj.weight",
                                                  "down_proj.weight"))]
    with torch.no_grad():
        params = dict(model.named_parameters())
        # sample up to ~5M abs-weights in fp32 to estimate the global threshold
        samp = []
        budget = 5_000_000
        per = max(1, budget // len(names))
        for n in names:
            f = params[n].detach().abs().flatten().float()
            if f.numel() > per:
                idx = _t.randint(0, f.numel(), (per,), device=f.device)
                f = f[idx]
            samp.append(f.cpu())
        allw = _t.cat(samp)
        thresh = _t.quantile(allw, sparsity).item()
        masked = 0
        total = 0
        for n in names:
            p = params[n]
            mask = (p.detach().abs() > thresh)
            p.data.mul_(mask.to(p.dtype))
            masked += int((~mask).sum().item())
            total += p.numel()
    logger.info(f"magnitude: global expert magnitude prune sparsity={sparsity} "
                f"thresh={thresh:.5f} actually_zeroed={masked/total:.3f}")


REGISTRY = {
    "random_drop": random_drop,
    "reap_drop": reap_drop,
    "slimqwen_merge": slimqwen_merge,
    "hcsmoe_merge": hcsmoe_merge,
    "nystrom": nystrom,
    "nystrom_combined": nystrom_combined,
    "nystrom_combined_fwdnat_bwdot3": nystrom_combined_fwdnat_bwdot3,
    "nystrom_combined_ot3": nystrom_combined_ot3,
    "svd_llm_v2": svd_llm_v2,
    "sparsegpt": sparsegpt,
    "mobe": mobe,
    "magnitude": magnitude,
}


def get(method: str):
    if method not in REGISTRY:
        raise KeyError(f"unknown method {method!r}; have {sorted(REGISTRY)}")
    return REGISTRY[method]
