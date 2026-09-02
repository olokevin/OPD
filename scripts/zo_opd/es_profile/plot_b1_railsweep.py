"""Plot the single-batch (B=1) decode rail sweep: ms/token-step and rail-probe
throughput vs total rails R = 1+N, three kernel paths + the stock reference.

    python scripts/zo_opd/es_profile/plot_b1_railsweep.py \
        --out docs/results/ZO_OPD/figs/es_profile_b1_railsweep.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

from common import RESULTS_DIR, md_table

# categorical slots 1-3 (validated), ink + surface tokens
C = {"rows/full": "#2a78d6", "shared/stream": "#eb6834", "fold/stream": "#1baf7a"}
LABEL = {"rows/full": "rows (shipping)", "shared/stream": "shared (Triton)",
         "fold/stream": "fold (FA3 GQA)"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9b9a95"
SURFACE = "#fcfcfb"
RIDGE = 161  # measured H100 NVL ridge, FLOP/byte == free rows for weight-bound GEMMs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1_railsweep")
    ap.add_argument("--stock-tag", default="b1_stock")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "b1_railsweep.png"))
    args = ap.parse_args()
    d = json.load(open(os.path.join(RESULTS_DIR, f"phase5_{args.tag}.json")))
    pts = [p for p in d["points"] if p["B"] == 1]
    try:
        ds = json.load(open(os.path.join(RESULTS_DIR, f"phase5_{args.stock_tag}.json")))
        stock_ms = [p["ms_per_token_step"] for p in ds["points"] if p["B"] == 1][0]
    except Exception:
        stock_ms = None

    paths = [p for p in C if any(q["path"] == p for q in pts)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", axis="y", color="#e8e7e3", lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xscale("log", base=2)
        ax.axvline(RIDGE, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

    a1, a2 = axes
    rows_tbl = []
    for path in paths:
        rs = sorted((q for q in pts if q["path"] == path), key=lambda q: q["N"])
        R = [1 + q["N"] for q in rs]
        ms = [q["ms_per_token_step"] for q in rs]
        a1.plot(R, ms, "-o", color=C[path], lw=2, ms=4.5, mec=SURFACE, mew=0.8, zorder=3)
        probe = [(q["N"] / q["ms_per_token_step"] * 1e3) for q in rs]  # rail evals/s
        a2.plot([r for r in R if r > 1], [p for r, p in zip(R, probe) if r > 1],
                "-o", color=C[path], lw=2, ms=4.5, mec=SURFACE, mew=0.8, zorder=3)
        ends = locals().setdefault("_ends", [])
        ends.append((path, R[-1], ms[-1]))
        rows_tbl.append([path] + [f"{q['ms_per_token_step']:.2f}" for q in rs])
    for rank, (path, xe, ye) in enumerate(sorted(locals().get("_ends", []), key=lambda t: -t[2])):
        a1.annotate(LABEL[path], (xe, ye), xytext=(6, (1 - rank) * 11), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    # throughput panel: what literally-free rails would deliver (N / best N=0 latency)
    base0 = min(q["ms_per_token_step"] for q in pts if q["N"] == 0)
    Ns_all = sorted({q["N"] for q in pts if q["N"] > 0})
    a2.plot([1 + n for n in Ns_all], [n / base0 * 1e3 for n in Ns_all], color=MUTED, lw=1.2,
            ls=(0, (4, 3)), zorder=2)
    a2.annotate("ideal: rails free at N=0 latency", (1 + Ns_all[-1], Ns_all[-1] / base0 * 1e3),
                xytext=(-2, 9), textcoords="offset points", color=INK2, fontsize=8, ha="right")
    if stock_ms:
        a1.axhline(stock_ms, color=MUTED, lw=1.4, ls=(0, (1, 2)), zorder=1)
        a1.annotate(f"stock vLLM B=1 ({stock_ms:.2f} ms)", (1.05, stock_ms),
                    xytext=(0, -11), textcoords="offset points", color=INK2, fontsize=8.5)
    for ax, ttl, yl in ((a1, "clean-token latency", "ms per token-step"),
                        (a2, "rail-probe throughput", "rail evaluations / s")):
        ax.set_title(ttl, color=INK, fontsize=11, loc="left")
        ax.set_ylabel(yl, color=INK2, fontsize=9.5)
        ax.set_xlabel("total rails R = 1 + N   (log₂)", color=INK2, fontsize=9.5)
        ax.annotate(f"ridge ≈{RIDGE} rows", (RIDGE, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8)
    Ns = sorted({q["N"] for q in pts})
    ticks = [1 + n for n in Ns if n not in (48, 96, 192)]
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    a1.set_ylim(bottom=0)
    a2.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}"))
    a1.legend(handles=[plt.Line2D([], [], color=C[p], lw=2, label=LABEL[p]) for p in paths],
              loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("es_token single-batch decode (B=1, ctx 512+64…384): rails vs latency",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {args.out}")
    hdr = ["path"] + [f"R={1+n}" for n in Ns]
    print(md_table(hdr, rows_tbl))


if __name__ == "__main__":
    main()
