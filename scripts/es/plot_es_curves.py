"""MATH-500 curves for a set of ES runs.

    python3 scripts/es/collect_es_curves.py --json /tmp/es.json logs/es/*.log
    python3 scripts/es/plot_es_curves.py /tmp/es.json --pick 'name=label' ... --out fig.png

Single panel on purpose: at N=10 every mode costs ~120 s/iteration (the rollout dominates and
none of the perturbation kernels is measurable against it), so ES iteration and wall-clock are
the same axis.  The per-iteration cost is annotated rather than given a redundant second panel.
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--pick", nargs="+", required=True, help="run_name=Legend label")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="ES on Qwen2.5-Math-7B, MATH lvl3-5 -> MATH-500 (N=10)")
ap.add_argument("--ref", type=float, default=71.82)
ap.add_argument("--ref-label", default="dense N=30 plateau (14.8 GPU-h)")
a = ap.parse_args()

rows = {r["name"]: r for r in json.load(open(a.json))}
fig, ax = plt.subplots(figsize=(7.6, 4.8))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
iters = []

for i, spec in enumerate(a.pick):
    name, _, label = spec.partition("=")
    r = rows[name]
    steps = sorted(int(s) for s in r["evals"])
    acc = [r["evals"].get(str(s), r["evals"].get(s)) for s in steps]
    ax.plot(steps, acc, "-o", ms=4, lw=1.8, color=colors[i % len(colors)], label=label)
    if r.get("iter_s"):
        iters.append(r["iter_s"])

ax.axhline(a.ref, color="gray", ls="--", lw=1, zorder=0)
ax.text(0.99, a.ref, a.ref_label, ha="right", va="bottom", fontsize=8, color="gray",
        transform=ax.get_yaxis_transform())
ax.set_xlabel("ES iteration")
ax.set_ylabel("MATH-500 greedy accuracy (%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
if iters:
    s = sum(iters) / len(iters)
    ax.set_title(f"{a.title}\nall arms {s:.0f} s/iteration — 10 iterations = {10*s/3600:.2f} GPU-h",
                 fontsize=10)
else:
    ax.set_title(a.title, fontsize=10)
fig.tight_layout()
fig.savefig(a.out, dpi=160)
print("wrote", a.out)
