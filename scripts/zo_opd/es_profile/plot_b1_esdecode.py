"""es-decode decode throughput at B=1 on the ds15b student
(DeepSeek-R1-Distill-Qwen-1.5B): ms per token-step vs total rails R = 1+N for
es-token-decode (fresh rank-1 per token) and es-decode with held rank-1 /
rank-4 / full-rank (packed bits) perturbations, all on the seq/stream/fused/
graph path; stock vLLM clean decode as the floor.

    python scripts/zo_opd/es_profile/plot_b1_esdecode.py \
        --out docs/results/ZO_OPD/figs/es_decode_b1_throughput.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import RESULTS_DIR, md_table

TAGS = [("esdecode_token", "es-token-decode (fresh rank-1 / token)", "#2a78d6"),
        ("esdecode_r1", "es-decode, held rank-1", "#d63a2a"),
        ("esdecode_r4", "es-decode, held rank-4", "#d9a400"),
        ("esdecode_full", "es-decode, held full-rank (packed bits)", "#1baf7a")]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9b9a95", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def load(tag):
    p = os.path.join(RESULTS_DIR, f"phase5_ds15b_b1_{tag}.json")
    if not os.path.exists(p):
        return []
    return sorted((q for q in json.load(open(p))["points"] if q["B"] == 1), key=lambda q: q["N"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "b1_esdecode.png"))
    args = ap.parse_args()
    stock = load("stock")
    stock_ms = stock[0]["ms_per_token_step"] if stock else None
    data = {t: load(t) for t, _, _ in TAGS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(a1)
    style(a2)
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
    Ns_all = sorted({q["N"] for pts in data.values() for q in pts})
    tbl = []
    for tag, label, col in TAGS:
        pts = data[tag]
        if not pts:
            continue
        R = [1 + q["N"] for q in pts]
        ms = [q["ms_per_token_step"] for q in pts]
        a1.plot(R, ms, "-o", color=col, lw=2, ms=4, mec=SURFACE, mew=0.8, zorder=3, label=label)
        a2.plot([r for r in R if r > 1], [q["N"] / q["ms_per_token_step"] * 1e3 for q in pts if q["N"] > 0],
                "-o", color=col, lw=2, ms=4, mec=SURFACE, mew=0.8, zorder=3)
        by = {q["N"]: q["ms_per_token_step"] for q in pts}
        tbl.append([label] + [f"{by[n]:.2f}" if n in by else "" for n in Ns_all])
    if stock_ms:
        a1.axhline(stock_ms, color=MUTED, lw=1.4, ls=(0, (1, 2)), zorder=1)
        a1.annotate(f"stock vLLM B=1, clean decode ({stock_ms:.2f} ms)", (1 + Ns_all[-1], stock_ms),
                    xytext=(0, -11), textcoords="offset points", color=INK2, fontsize=8.5, ha="right")
        a2.plot([1 + n for n in Ns_all if n > 0], [n / stock_ms * 1e3 for n in Ns_all if n > 0],
                color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        a2.annotate("ideal: rails free at stock latency", (1 + Ns_all[-1], Ns_all[-1] / stock_ms * 1e3),
                    xytext=(-2, 9), textcoords="offset points", color=INK2, fontsize=8, ha="right")
    a1.set_yscale("log", base=2)
    yt = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    a1.yaxis.set_major_locator(FixedLocator(yt))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.set_title("decode: clean-token latency vs rails", color=INK, fontsize=11, loc="left")
    a1.set_ylabel("ms per token-step", color=INK2, fontsize=9.5)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}"))
    a2.set_title("rail-probe throughput", color=INK, fontsize=11, loc="left")
    a2.set_ylabel("rail evaluations / s", color=INK2, fontsize=9.5)
    ticks = [1 + n for n in Ns_all]
    for ax in (a1, a2):
        ax.set_xlabel("total rails R = 1 + N   (log₂)", color=INK2, fontsize=9.5)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    a1.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("es-decode vs es-token-decode, B=1, DeepSeek-R1-Distill-Qwen-1.5B (ctx 512), H100 NVL",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {args.out}")
    print(md_table(["path"] + [f"R={1+n}" for n in Ns_all], tbl))


if __name__ == "__main__":
    main()
