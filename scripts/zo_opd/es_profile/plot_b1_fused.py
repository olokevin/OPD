"""0902 B=1 plot: (left) decode ms/token-step vs total rails R = 1+N for the
shipping path, the 0831 rail-aware path and the 0902 fused/in-graph path,
against stock vLLM; (right) the full one-prompt training step (decode +
teacher + assemble/apply) vs N for the shipping and fused paths, against
clean generation of the same 512 tokens and a BP-OPD batch-1 step.

    python scripts/zo_opd/es_profile/plot_b1_fused.py \
        --out docs/results/ZO_OPD/figs/es_profile_b1_fused.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import RESULTS_DIR, md_table

C = {"rows/full": "#2a78d6", "fold/stream": "#1baf7a", "shared/stream": "#eb6834",
     "rows/stream/fused/eager": "#b48ef2", "seq/stream/fused/eager": "#d9a400",
     "seq/stream/fused/graph": "#d63a2a", "fold/stream/fused/graph": "#8a6d3b"}
LABEL = {"rows/full": "rows/full (shipping)", "fold/stream": "fold/stream (0831)",
         "shared/stream": "shared/stream (0831)",
         "rows/stream/fused/eager": "rows + fused rail",
         "seq/stream/fused/eager": "seq + fused rail (eager tail)",
         "seq/stream/fused/graph": "seq + fused rail + in-graph step (0902)",
         "fold/stream/fused/graph": "fold + fused rail + in-graph step"}
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9b9a95", "#fcfcfb"
RIDGE = 161


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def load_points(tag):
    d = json.load(open(os.path.join(RESULTS_DIR, f"phase5_{tag}.json")))
    return [p for p in d["points"] if p["B"] == 1]


def load_steps():
    """{path: {N: (step, decode, teacher, assemble)}} from es_b1_step_*.tsv,
    mean over steps >= 1 (step 0 carries graph capture)."""
    out = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "es_b1_step_*.tsv")):
        rows = [l.rstrip("\n").split("\t") for l in open(f)][1:]
        acc = {}
        for r in rows:
            if len(r) < 7 or int(r[2]) < 1:
                continue
            path = r[0][:-len("/kernel/eager")] if r[0].endswith("/kernel/eager") else r[0]
            acc.setdefault((path, int(r[1])), []).append([float(x) for x in r[3:7]])
        for (path, N), v in acc.items():
            m = [sum(c) / len(c) for c in zip(*v)]
            out.setdefault(path, {})[N] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1_fused")
    ap.add_argument("--stock-tag", default="b1_stock")
    ap.add_argument("--old-tag", default="b1_railsweep", help="0831 sweep for rows/full + fold/stream")
    ap.add_argument("--bp-json", default=os.path.join(RESULTS_DIR, "bp_b1.json"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "b1_fused.png"))
    args = ap.parse_args()

    pts = load_points(args.tag)
    try:
        old = load_points(args.old_tag)
    except Exception:
        old = []
    have = {p["path"] for p in pts}
    for p in old:   # fill 0831 paths not re-measured today
        if p["path"] not in have:
            pts.append(dict(p, from_old=True))
    try:
        stock_ms = [p["ms_per_token_step"] for p in load_points(args.stock_tag)][0]
    except Exception:
        stock_ms = None
    steps = load_steps()
    bp = json.load(open(args.bp_json)) if os.path.exists(args.bp_json) else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(a1)
    style(a2)
    a1.set_xscale("log", base=2)
    a1.axvline(RIDGE, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    paths = [p for p in C if any(q["path"] == p for q in pts)]
    # the figure keeps the story to four lines; the table below prints all
    PLOT = [p for p in paths if p not in ("shared/stream", "rows/stream/fused/eager")]
    tbl = []
    Ns_all = sorted({q["N"] for q in pts})
    for path in paths:
        rs = sorted((q for q in pts if q["path"] == path), key=lambda q: q["N"])
        R = [1 + q["N"] for q in rs]
        ms = [q["ms_per_token_step"] for q in rs]
        ls = (0, (5, 2)) if rs[0].get("from_old") else "-"
        if path in PLOT:
            a1.plot(R, ms, ls=ls, marker="o", color=C[path], lw=2, ms=4, mec=SURFACE, mew=0.8, zorder=3)
        by = {q["N"]: q["ms_per_token_step"] for q in rs}
        tbl.append([path] + [f"{by[n]:.2f}" if n in by else "" for n in Ns_all])
    if stock_ms:
        a1.axhline(stock_ms, color=MUTED, lw=1.4, ls=(0, (1, 2)), zorder=1)
        a1.annotate(f"stock vLLM B=1, clean decode ({stock_ms:.2f} ms)", (385, stock_ms), xytext=(0, -11),
                    textcoords="offset points", color=INK2, fontsize=8.5, ha="right")
    a1.set_title("decode: clean-token latency vs rails", color=INK, fontsize=11, loc="left")
    a1.set_ylabel("ms per token-step", color=INK2, fontsize=9.5)
    a1.set_xlabel("total rails R = 1 + N   (log₂)", color=INK2, fontsize=9.5)
    a1.annotate(f"ridge ≈{RIDGE} rows", (RIDGE, 0.03), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8)
    ticks = [1 + n for n in Ns_all if n not in (2, 48, 96, 192)]
    a1.xaxis.set_major_locator(FixedLocator(ticks))
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xticklabels([str(t) for t in ticks], fontsize=8)
    # log y: the fused paths live within 2.3-4 ms while the shipping path runs to 19 ms
    a1.set_yscale("log", base=2)
    yt = [2, 2.5, 3, 4, 5, 7, 10, 15, 20]
    a1.set_ylim(2.0, 21)
    a1.yaxis.set_major_locator(FixedLocator(yt))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.grid(True, which="major", axis="y", color="#e8e7e3", lw=0.8, zorder=0)
    a1.legend(handles=[plt.Line2D([], [], color=C[p], lw=2, label=LABEL[p]) for p in PLOT],
              loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)

    # ---- right: one-prompt step time -------------------------------------
    a2.set_xscale("log", base=2)
    step_paths = [p for p in C if p in steps]
    for path in step_paths:
        Ns = sorted(steps[path])
        tot = [steps[path][n][0] for n in Ns]
        dec = [steps[path][n][1] for n in Ns]
        a2.plot([1 + n for n in Ns], tot, "-o", color=C[path], lw=2, ms=4, mec=SURFACE, mew=0.8, zorder=3)
        a2.plot([1 + n for n in Ns], dec, ls=(0, (2, 2)), color=C[path], lw=1.2, zorder=2)
    x_lo = min((1 + n for p in step_paths for n in steps[p]), default=1)
    if stock_ms:
        gen_s = stock_ms * args.max_tokens / 1e3
        a2.axhline(gen_s, color=MUTED, lw=1.4, ls=(0, (1, 2)), zorder=1)
        a2.annotate(f"stock vLLM clean generation, {args.max_tokens} tok ({gen_s:.2f} s)", (x_lo, gen_s),
                    xytext=(2, -12), textcoords="offset points", color=INK2, fontsize=8.5)
    if bp:
        a2.axhline(bp["step_s"], color="#333", lw=1.6, ls="-.", zorder=1)
        a2.annotate(f"BP-OPD step, batch 1, steady ({bp['step_s']:.1f} s; generation {bp['gen_s']:.1f} s)",
                    (x_lo, bp["step_s"]), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=8.5)
    a2.set_title("one-prompt training step vs rails", color=INK, fontsize=11, loc="left")
    a2.set_ylabel("seconds per step (solid = total, dotted = decode)", color=INK2, fontsize=9.5)
    a2.set_xlabel("total rails R = 1 + N   (log₂)", color=INK2, fontsize=9.5)
    if step_paths:
        Ns_s = sorted({n for p in step_paths for n in steps[p]})
        a2.xaxis.set_major_locator(FixedLocator([1 + n for n in Ns_s]))
        a2.xaxis.set_minor_formatter(NullFormatter())
        a2.set_xticklabels([str(1 + n) for n in Ns_s], fontsize=8)
        a2.legend(handles=[plt.Line2D([], [], color=C[p], lw=2, label=LABEL[p]) for p in step_paths],
                  loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    a2.set_ylim(bottom=0)
    fig.suptitle("es_token single-prompt (B=1, ctx 512): decode and full step vs rails, Qwen3-1.7B on H100 NVL",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print(f"[saved] {args.out}")
    print(md_table(["path"] + [f"R={1+n}" for n in Ns_all], tbl))
    if steps:
        Ns_s = sorted({n for p in step_paths for n in steps[p]})
        rows = []
        for p in step_paths:
            rows.append([p] + [("%.2f (%.2f/%.2f/%.2f)" % tuple(steps[p][n])) if n in steps[p] else "" for n in Ns_s])
        print(md_table(["path: step s (decode/teacher/assemble)"] + [f"N={n}" for n in Ns_s], rows))


if __name__ == "__main__":
    main()
