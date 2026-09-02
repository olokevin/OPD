"""Diff a bf16 HF checkpoint against its base model -> coordinate mask for PERTURB_MODE=sgdmask.

    python3 scripts/es/build_sgd_mask.py \
        --base Qwen/Qwen2.5-Math-7B \
        --ckpt /data/yequan/bp/BP-q2p5-7b/<run>/global_step_10/actor/huggingface \
        --out datasets/es_math/sgd_mask_qwen2p5_math_7b_st10.pt [--threshold 1e-5] [--compare other.pt]

Mask = |W_ckpt - W_base| > threshold, entrywise (arXiv:2602.07729 uses 1e-5 on bf16 weights).
HF q/k/v and gate/up are concatenated along dim 0 into vLLM's fused qkv_proj / gate_up_proj so
the flat indices address the live vLLM parameter that es_worker_extension perturbs.  Only the
2-D linear weights go into the mask (embed/lm_head/norms/biases are reported, not perturbed --
the same convention as every other structured ES mode).  A JSON summary lands next to `--out`.
"""
import argparse, json, os, re, sys, time
from collections import defaultdict

import torch
from safetensors import safe_open

FUSE = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}


def resolve(path):
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)


def weight_map(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(d, v) for k, v in json.load(open(idx))["weight_map"].items()}
    single = os.path.join(d, "model.safetensors")
    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


def module_type(name):
    if "embed_tokens" in name: return "embed"
    if "lm_head" in name: return "lm_head"
    if name.endswith("norm.weight") or "layernorm" in name: return "norm"
    m = re.search(r"\.(\w+_proj)\.(weight|bias)$", name)
    return f"{m.group(1)}.{m.group(2)}" if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=1e-5)
    ap.add_argument("--compare", default=None, help="another mask .pt: report overlap")
    a = ap.parse_args()
    t0 = time.time()
    bmap, cmap = weight_map(resolve(a.base)), weight_map(resolve(a.ckpt))
    assert set(bmap) == set(cmap), (set(bmap) ^ set(cmap))
    handles = {}
    def get(m, k):
        f = m[k]
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)

    layers, pending = {}, defaultdict(dict)       # vllm name -> flat idx ; fused parts
    by_type = defaultdict(lambda: [0, 0, 0])      # numel, n>thr, n!=0
    per_layer = defaultdict(lambda: [0, 0])
    rows_cols = {}
    dw_all, w_changed, w_all_med = [], [], []
    tot = tot_changed = tot_nonzero = mask_total = 0
    for name in sorted(bmap):
        w0 = get(bmap, name)
        w1 = get(cmap, name)
        assert w0.shape == w1.shape and w0.dtype == w1.dtype == torch.bfloat16, (name, w0.dtype, w1.dtype)
        d = (w1.float() - w0.float()).abs_()
        changed = d > a.threshold
        n, nc, nz = w0.numel(), int(changed.sum()), int((d > 0).sum())
        t = module_type(name)
        by_type[t][0] += n; by_type[t][1] += nc; by_type[t][2] += nz
        tot += n; tot_changed += nc; tot_nonzero += nz
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            per_layer[int(m.group(1))][0] += n; per_layer[int(m.group(1))][1] += nc
        if nc:
            dw_all.append(d[changed]); w_changed.append(w0.float()[changed].abs())
        if w0.ndim == 2 and t not in ("embed", "lm_head"):
            w_all_med.append(w0.float().abs().flatten()[::997])   # strided sample of |w|
            rows = int(changed.any(1).sum()); cols = int(changed.any(0).sum())
            rows_cols[name] = (rows, w0.shape[0], cols, w0.shape[1], nc)
            mm = re.search(r"\.(\w+_proj)\.weight$", name)
            if mm and mm.group(1) in FUSE:
                fused, pos = FUSE[mm.group(1)]
                vname = name.replace(mm.group(1), fused)
                pending[vname][pos] = (changed, w0.shape[0])
            else:
                layers[name] = changed.flatten().nonzero().squeeze(1).to(torch.int64)
        del w0, w1, d, changed
    for vname, parts in pending.items():
        assert sorted(parts) == list(range(len(parts))), (vname, sorted(parts))
        cat = torch.cat([parts[i][0] for i in range(len(parts))], dim=0)   # vLLM: cat along out dim
        layers[vname] = cat.flatten().nonzero().squeeze(1).to(torch.int64)
    layers = {k: v for k, v in layers.items() if v.numel()}
    mask_total = sum(int(v.numel()) for v in layers.values())

    dw = torch.cat(dw_all) if dw_all else torch.zeros(1)
    wc = torch.cat(w_changed) if w_changed else torch.zeros(1)
    wa = torch.cat(w_all_med)
    q = lambda x, p: float(torch.quantile(x[torch.randperm(x.numel())[:2_000_000]], p)) if x.numel() else float("nan")
    stats = {
        "base": a.base, "ckpt": a.ckpt, "threshold": a.threshold,
        "n_params": tot, "n_changed": tot_changed, "n_nonzero_diff": tot_nonzero,
        "density_pct": 100.0 * tot_changed / tot, "sparsity_pct": 100.0 * (1 - tot_changed / tot),
        "mask_entries": mask_total, "mask_layers": len(layers),
        "mask_density_pct_of_model": 100.0 * mask_total / tot,
        "by_type": {t: {"numel": v[0], "changed": v[1], "nonzero": v[2], "density_pct": 100.0 * v[1] / v[0]}
                    for t, v in sorted(by_type.items())},
        "per_layer_density_pct": {l: 100.0 * v[1] / v[0] for l, v in sorted(per_layer.items())},
        "dw_changed": {"min": float(dw.min()), "p50": q(dw, 0.5), "p90": q(dw, 0.9), "max": float(dw.max())},
        "abs_w_changed": {"p10": q(wc, 0.1), "p50": q(wc, 0.5), "p90": q(wc, 0.9)},
        "abs_w_all_sample": {"p10": q(wa, 0.1), "p50": q(wa, 0.5), "p90": q(wa, 0.9)},
        "rows_cols": {k: {"rows_touched": v[0], "rows": v[1], "cols_touched": v[2], "cols": v[3], "n": v[4]}
                      for k, v in rows_cols.items()},
    }
    if a.compare:
        other = torch.load(a.compare, map_location="cpu", weights_only=False)["layers"]
        inter = sum(len(set(layers[k].tolist()) & set(other[k].tolist())) for k in layers if k in other)
        n_other = sum(int(v.numel()) for v in other.values())
        stats["compare"] = {"path": a.compare, "other_entries": n_other, "intersection": inter,
                            "jaccard": inter / max(1, mask_total + n_other - inter),
                            "frac_of_other_in_this": inter / max(1, n_other)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"meta": {k: stats[k] for k in ("base", "ckpt", "threshold", "n_params", "n_changed", "mask_entries")},
                "layers": layers}, a.out)
    json.dump(stats, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"[mask] {a.ckpt}\n  changed {tot_changed:,} / {tot:,} = {stats['density_pct']:.4f}% "
          f"(nonzero diff {tot_nonzero:,}); mask {mask_total:,} entries over {len(layers)} linear weights "
          f"= {stats['mask_density_pct_of_model']:.4f}% of the model")
    for t, v in stats["by_type"].items():
        print(f"  {t:<18} {v['changed']:>12,} / {v['numel']:>14,}  {v['density_pct']:.4f}%")
    print(f"  |dw| changed: p50 {stats['dw_changed']['p50']:.2e} max {stats['dw_changed']['max']:.2e}; "
          f"|w| changed p50 {stats['abs_w_changed']['p50']:.2e} vs all p50 {stats['abs_w_all_sample']['p50']:.2e}")
    if a.compare:
        print(f"  overlap with {a.compare}: {stats['compare']}")
    print(f"  wrote {a.out} (+.json) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
