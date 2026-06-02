"""Plot per-layer compression-sensitivity curves from layer_sensitivity.py shards.

Reads every results/layer_sens_*.json shard, merges cells, and renders one
figure per model: two panels (self_attn | mlp), x = decoder layer index,
y = MATH-500 accuracy, one curve per retain ratio, with a dashed horizontal
line at the uncompressed-model accuracy.

Usage:
  python3 plot_layer_sensitivity.py \
      --results-dir results --out-dir figures
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_TITLES = {"self_attn": "self_attn (SVD-LLM-V2, q/k/v/o)",
                 "mlp": "mlp (Nystrom, gate/up/down)"}
MODEL_TITLES = {"qwen3-4b-base": "Qwen3-4B (non-thinking, base)",
                "qwen3-4b-rlmath": "Qwen3-4B-Non-Thinking-RL-Math (Step500)"}


def load_shards(results_dir: Path):
    """Return (cells, baselines): cells is a list of cell dicts;
    baselines maps model_tag -> baseline acc (first non-null seen)."""
    cells = []
    baselines: dict[str, float] = {}
    for fp in sorted(glob.glob(str(results_dir / "layer_sens_*.json"))):
        with open(fp) as f:
            data = json.load(f)
        tag = data.get("model_tag")
        if data.get("baseline_acc") is not None and tag not in baselines:
            baselines[tag] = data["baseline_acc"]
        cells.extend(data.get("cells", []))
    return cells, baselines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, baselines = load_shards(results_dir)
    if not cells:
        raise SystemExit(f"No cells found under {results_dir}/layer_sens_*.json")

    # index: acc[model_tag][module][ratio] = {layer: acc}
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ratios_seen = set()
    for c in cells:
        if c.get("math500_acc") is None:
            continue
        acc[c["model_tag"]][c["module"]][c["ratio"]][c["layer"]] = c["math500_acc"]
        ratios_seen.add(c["ratio"])
    ratios = sorted(ratios_seen, reverse=True)  # 0.9, 0.8, 0.6, 0.5

    # Swept layers are non-contiguous (e.g. 30->20 gap); plot against a compact
    # categorical x-axis (one tick per swept layer, ascending) so no line is
    # drawn across un-swept layers. Fixed 0-100 y-axis so the (flat, near-
    # baseline) reality is honest rather than zoomed into noise.
    written = []
    for model_tag in sorted(acc):
        modules = ["self_attn", "mlp"]
        # union of swept layers for this model, ascending
        layers = sorted({l for m in modules for r in ratios
                         for l in acc[model_tag][m].get(r, {})})
        xpos = {l: i for i, l in enumerate(layers)}
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        for ax, module in zip(axes, modules):
            for ratio in ratios:
                series = acc[model_tag][module].get(ratio, {})
                if not series:
                    continue
                ls = sorted(series)
                xs = [xpos[l] for l in ls]
                ys = [100.0 * series[l] for l in ls]
                ax.plot(xs, ys, marker="o", ms=4, lw=1.4,
                        label=f"retain {ratio:g}")
            base = baselines.get(model_tag)
            if base is not None:
                ax.axhline(100.0 * base, ls="--", color="k", lw=1.3,
                           label=f"uncompressed ({100*base:.0f}%)")
            ax.set_title(MODULE_TITLES.get(module, module))
            ax.set_xlabel("decoder layer index (swept, non-contiguous)")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([str(l) for l in layers], fontsize=7)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower left", ncol=2)
        axes[0].set_ylabel("MATH-500 accuracy (%)")
        fig.suptitle(
            f"Per-layer compression sensitivity — {MODEL_TITLES.get(model_tag, model_tag)}\n"
            f"(compress ONE module per layer; MATH-500 first 100; OpenThought3 calib)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = out_dir / f"layer_sensitivity_{model_tag}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(str(out))
        print(f"wrote {out}")

    # also dump a compact merged CSV for the docs/appendix
    csv = out_dir / "layer_sensitivity_merged.csv"
    with open(csv, "w") as f:
        f.write("model_tag,module,ratio,layer,math500_acc\n")
        for c in sorted(cells, key=lambda c: (c["model_tag"], c["module"],
                                              -c["ratio"], c["layer"])):
            if c.get("math500_acc") is None:
                continue
            f.write(f"{c['model_tag']},{c['module']},{c['ratio']},"
                    f"{c['layer']},{c['math500_acc']:.4f}\n")
    print(f"wrote {csv}")
    print("baselines:", baselines)
    return written


if __name__ == "__main__":
    main()
