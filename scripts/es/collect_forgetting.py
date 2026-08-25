"""Turn the per-arm forgetting evals + delta statistics into the markdown tables
pasted into docs/results/ES/es_results.md.

    python scripts/es/collect_forgetting.py [--root /data/yequan/es]
"""

import argparse
import json
import os

PRIOR = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge",
         "openbookqa", "boolq", "mmlu"]
SHORT = {"hellaswag": "HellaSw", "piqa": "PIQA", "winogrande": "WinoG",
         "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "openbookqa": "OBQA",
         "boolq": "BoolQ", "mmlu": "MMLU"}
# order, label, sigma, MATH-500 best-@step from section 7 of the results page
ARMS = [
    ("base",     "base Qwen2.5-Math-7B", "-",       "-"),
    ("dense",    "dense (paper ES)",     "1e-3",    "40"),
    ("zoact",    "zoact r=1",            "1e-3",    "130"),
    ("insparse", "insparse d=1%",        "1e-3",    "80"),
    ("fura",     "fura small-core",      "1.25e-2", "30"),
    ("iso",      "iso fixed-spectrum",   "5e-2",    "60"),
    ("isobtt",   "isobtt fixed-spec",    "5e-2",    "120"),
]


def _pick(d, task):
    for k in (f"{task}/acc_norm", f"{task}/acc"):
        if k in d:
            return d[k]
    return None


def load(root, tag):
    fdir = os.path.join(root, "forgetting", tag)
    out = {"tag": tag}
    p = os.path.join(fdir, "lm_eval.json")
    if os.path.exists(p):
        raw = json.load(open(p))
        for t in PRIOR:
            v = _pick(raw, t)
            if v is not None:
                out[t] = v
    p = os.path.join(fdir, "math500.json")
    if os.path.exists(p):
        out["math500"] = json.load(open(p))["math500_acc"]
    p = os.path.join(root, "materialized", tag, "delta_stats.json")
    if os.path.exists(p):
        s = json.load(open(p))
        out["rel"] = s["global_rel"]
        out["sparsity"] = 100 * s["global_sparsity"]
        out["untouched"] = 100 * s["global_exact_zero"]
        out["per_tensor"] = s["per_tensor"]
    return out


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/yequan/es")
    args = ap.parse_args()

    data = {tag: load(args.root, tag) for tag, *_ in ARMS}
    base = data.get("base", {})
    present = [t for t in PRIOR if t in base]

    def mean_prior(d):
        vs = [d[t] for t in present if t in d]
        return sum(vs) / len(vs) if len(vs) == len(present) else None

    print("### Prior-knowledge retention at each arm's best-MATH-500 checkpoint\n")
    hdr = ["Arm", "σ", "best @", "MATH-500"] + [SHORT[t] for t in present] + ["**Prior mean**", "Δ prior"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    bm = mean_prior(base)
    for tag, label, sig, step in ARMS:
        d = data.get(tag, {})
        if not d.get("math500") and tag != "base":
            continue
        m = mean_prior(d)
        row = [label, sig, step, fmt(d.get("math500"))]
        row += [fmt(d.get(t)) for t in present]
        row += [f"**{fmt(m, 2)}**",
                "—" if (m is None or bm is None) else f"{m - bm:+.2f}"]
        print("| " + " | ".join(row) + " |")

    print("\n### Update geometry (the mechanism arXiv:2601.20861 proposes)\n")
    hdr = ["Arm", "‖ΔW‖_F/‖W‖_F", "sparsity(τ=1e-6)", "untouched (Δ=0)", "MATH-500 Δ", "prior Δ"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for tag, label, sig, step in ARMS:
        d = data.get(tag, {})
        if tag == "base" or "rel" not in d or "math500" not in d:
            continue
        m = mean_prior(d)
        print("| " + " | ".join([
            label,
            f"{d['rel']:.2e}",
            f"{d['sparsity']:.1f}%",
            f"{d['untouched']:.1f}%",
            "—" if not base.get("math500") else f"{d['math500'] - base['math500']:+.1f}",
            "—" if (m is None or bm is None) else f"{m - bm:+.2f}",
        ]) + " |")

    # layerwise sparsity by parameter group (the paper's Figure 4)
    print("\n### Update sparsity by parameter group (τ=1e-6, mean over layers)\n")
    groups = ["Q", "K", "V", "WO", "MLP", "LayerNorm", "Embed"]
    print("| Arm | " + " | ".join(groups) + " |")
    print("|" + "---|" * (len(groups) + 1))
    for tag, label, sig, step in ARMS:
        d = data.get(tag, {})
        pt = d.get("per_tensor")
        if not pt or "math500" not in d:
            continue
        cells = []
        for g in groups:
            vals = [v["sparsity"] for v in pt.values() if v["group"] == g]
            cells.append(f"{100*sum(vals)/len(vals):.0f}%" if vals else "—")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
