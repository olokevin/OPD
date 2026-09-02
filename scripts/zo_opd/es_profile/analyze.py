"""Turn the phase JSONs into the markdown tables + figures used by
docs/results/ZO_OPD/es_profile_results.md.

    python scripts/zo_opd/es_profile/analyze.py [--figs docs/results/ZO_OPD/figs]
"""
import argparse
import glob
import json
import os

import numpy as np

from common import RESULTS_DIR, md_table

PATH_LABEL = {"rows/full": "rows/full (shipping)", "shared/stream": "shared/stream (Triton)",
              "fold/stream": "fold/stream (FA3 GQA-fold)", "stock": "stock vLLM"}


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def phase1_tables():
    d = load("phase1_linear_rails.json")
    if not d:
        return ""
    out = ["### Phase 1 -- flattened cuBLAS GEMM alone, T(B,R)/T(B,1) (rail op excluded)\n"]
    Rs, Bs = d["Rs"], d["Bs"]
    for shape in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        rows = []
        for B in Bs:
            rs = {r["R"]: r for r in d["recs"] if r["shape"] == shape and r["B"] == B}
            rows.append([B] + [rs[R]["rel_flat"] for R in Rs])
        out.append(f"\n**{shape}**\n\n" + md_table(["B \\ R"] + [str(R) for R in Rs], rows,
                                                  fmt={str(R): "{:.2f}" for R in Rs}))
    # N_free from the flat GEMM alone
    rows = []
    for shape in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        for B in Bs:
            rs = [r for r in d["recs"] if r["shape"] == shape and r["B"] == B]
            cells = []
            for eps in (0.05, 0.10, 0.25):
                ok = [r["R"] for r in rs if r["rel_flat"] <= 1 + eps]
                cells.append((max(ok) - 1) if ok else 0)
            rows.append([shape, B] + cells)
    out.append("\n**N_free from the GEMM alone (extra rails within eps of the clean GEMM)**\n\n" +
               md_table(["shape", "B", "N5%", "N10%", "N25%"], rows))
    # rail-op fixed cost and fused-epilogue verdict
    rows = []
    for shape in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        for B in (1, 8, 64):
            rs = {r["R"]: r for r in d["recs"] if r["shape"] == shape and r["B"] == B}
            r8 = rs[8]
            rows.append([shape, B, r8["flat"] * 1e3, (r8["flat+rail"] - r8["flat"]) * 1e3,
                         r8["fused"] / r8["flat+rail"], r8["triton"] / r8["flat"], r8["serial"] / r8["flat"]])
    out.append("\n**R=8: rail-op cost on top of the GEMM, and the fused-epilogue / plain-Triton / serial ratios**\n\n" +
               md_table(["shape", "B", "flat GEMM us", "+rail op us", "fused/flat+rail", "triton/cublas", "serial/flat"], rows,
                        fmt={"flat GEMM us": "{:.1f}", "+rail op us": "{:.1f}", "fused/flat+rail": "{:.2f}",
                             "triton/cublas": "{:.2f}", "serial/flat": "{:.2f}"}))
    return "\n".join(out)


def phase2_tables():
    d = load("phase2_shared_kv_attn.json")
    if not d:
        return ""
    out = ["### Phase 2 -- attention T(B,R)/T(B,1): rows / fold / shared, and unique-KV GB/s at R=32\n"]
    for L in d["Ls"]:
        rows = []
        for B in d["Bs"]:
            rs = {r["R"]: r for r in d["recs"] if r["B"] == B and r["L"] == L}
            if not rs:
                continue
            r32 = rs[max(rs)]
            rows.append([B] + [f"{rs[R]['rel_rows']:.1f} / {rs[R]['rel_fold']:.2f} / {rs[R]['rel_shared']:.2f}"
                               for R in (2, 8, 16, 32) if R in rs] +
                        [f"{r32['rows']:.3f} / {r32['fold']:.3f} / {r32['shared']:.3f}",
                         f"{r32['gbps_rows_actual']:.0f} / {r32['gbps_unique_fold']:.0f} / {r32['gbps_unique_shared']:.0f}"])
        out.append(f"\n**L={L}**\n\n" + md_table(["B", "R=2", "R=8", "R=16", "R=32", "ms @R=32", "GB/s @R=32 (rows actual / fold uniq / shared uniq)"], rows))
    return "\n".join(out)


def phase4_tables():
    d = load("phase4_lm_head.json")
    if not d:
        return ""
    Bs = sorted({r["B"] for r in d["recs"]})
    Rs = sorted({r["R"] for r in d["recs"]})
    rows = []
    for B in Bs:
        rs = {r["R"]: r for r in d["recs"] if r["B"] == B}
        rows.append([B] + [f"{rs[R]['full']:.2f} / {rs[R]['stream']:.2f} ({rs[R]['full']/rs[R]['stream']:.2f}x)" if R in rs else "-" for R in Rs])
    return ("### Phase 4 -- LM head ms: full (shipping) / stream (Triton), speedup\n\n" +
            md_table(["B \\ R"] + [str(R) for R in Rs], rows))


def phase5(figs_dir):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "phase5_L*.json")))
    if not files:
        return ""
    out = []
    by_ctx = {}
    for f in files:
        d = json.load(open(f))
        L = d["prompt_len"]
        by_ctx.setdefault(L, []).extend(d["points"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    for L in sorted(by_ctx):
        pts = by_ctx[L]
        paths = [p for p in ("rows/full", "shared/stream", "fold/stream") if any(q["path"] == p for q in pts)]
        Bs = sorted({q["B"] for q in pts if q["path"] != "stock"})
        Ns = sorted({q["N"] for q in pts if q["path"] != "stock"})
        stock = {q["B"]: q["ms_per_token_step"] for q in pts if q["path"] == "stock"}
        out.append(f"\n### Phase 5 -- prompt_len={L} (context ~{L}+64..320): ms/token-step, rows = B, cols = N\n")
        for path in paths:
            rows, over_rows, nfree = [], [], []
            for B in Bs:
                rs = {q["N"]: q for q in pts if q["path"] == path and q["B"] == B}
                base = rs.get(0)
                rows.append([B] + [f"{rs[N]['ms_per_token_step']:.2f}" if N in rs else "-" for N in Ns] +
                            [f"{stock[B]:.2f}" if B in stock else "-"])
                if base:
                    ov = {N: rs[N]["ms_per_token_step"] / base["ms_per_token_step"] - 1 for N in rs}
                    over_rows.append([B] + [f"{ov[N]*100:+.0f}%" if N in ov else "-" for N in Ns] +
                                     [f"{(base['ms_per_token_step']/stock[B]-1)*100:+.0f}%" if B in stock else "-"])
                    nf = []
                    for eps in (0.05, 0.10, 0.25):
                        ok = [N for N in ov if ov[N] <= eps]
                        nf.append(max(ok) if ok else 0)
                    nfree.append([B] + nf + [f"{rs[max(rs)]['row_steps_per_s']:.0f}" if rs else "-"])
            out.append(f"\n**{PATH_LABEL[path]}** -- ms/token-step (last col: stock vLLM at the same B)\n\n" +
                       md_table(["B \\ N"] + [str(N) for N in Ns] + ["stock"], rows))
            out.append(f"\n**{PATH_LABEL[path]}** -- clean-token overhead vs N=0 (last col: N=0 driver vs stock)\n\n" +
                       md_table(["B \\ N"] + [str(N) for N in Ns] + ["N=0 vs stock"], over_rows))
            out.append(f"\n**{PATH_LABEL[path]}** -- N_free at 5% / 10% / 25%, and row-steps/s at max N\n\n" +
                       md_table(["B", "N5%", "N10%", "N25%", "row-steps/s @Nmax"], nfree))
            # marginal view: overhead relative to N=1 (the rails-on fixed cost removed)
            m_rows = []
            for B in Bs:
                rs = {q["N"]: q for q in pts if q["path"] == path and q["B"] == B}
                if 1 not in rs or 0 not in rs:
                    continue
                b1 = rs[1]["ms_per_token_step"]
                ov1 = {N: rs[N]["ms_per_token_step"] / b1 - 1 for N in rs if N >= 1}
                nf = []
                for eps in (0.05, 0.10, 0.25):
                    ok = [N for N in ov1 if ov1[N] <= eps]
                    nf.append(max(ok) if ok else 1)
                m_rows.append([B, f"{(b1 - rs[0]['ms_per_token_step']):.2f} ({(b1/rs[0]['ms_per_token_step']-1)*100:+.0f}%)"] +
                              [f"{ov1[N]*100:+.0f}%" if N in ov1 else "-" for N in Ns if N >= 1] + nf)
            out.append(f"\n**{PATH_LABEL[path]}** -- rails-on fixed cost (N=1 minus N=0, ms) and overhead vs N=1; N_free vs N=1 at 5/10/25%\n\n" +
                       md_table(["B", "fixed ms"] + [f"N={N}" for N in Ns if N >= 1] + ["N5%", "N10%", "N25%"], m_rows))
        if plt is not None and figs_dir:
            os.makedirs(figs_dir, exist_ok=True)
            fig, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.2), squeeze=False)
            for ax, path in zip(axes[0], paths):
                M = np.full((len(Bs), len(Ns)), np.nan)
                for i, B in enumerate(Bs):
                    rs = {q["N"]: q for q in pts if q["path"] == path and q["B"] == B}
                    if 0 not in rs:
                        continue
                    for j, N in enumerate(Ns):
                        if N in rs:
                            M[i, j] = (rs[N]["ms_per_token_step"] / rs[0]["ms_per_token_step"] - 1) * 100
                im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=200, aspect="auto")
                ax.set_xticks(range(len(Ns)), [str(n) for n in Ns])
                ax.set_yticks(range(len(Bs)), [str(b) for b in Bs])
                ax.set_xlabel("extra rails N")
                ax.set_ylabel("clean batch B")
                ax.set_title(f"{PATH_LABEL[path]}\nclean-token overhead % (prompt_len={L})", fontsize=9)
                for i in range(len(Bs)):
                    for j in range(len(Ns)):
                        if not np.isnan(M[i, j]):
                            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7,
                                    color="white" if M[i, j] < 120 else "black")
                if not np.all(np.isnan(M)):
                    try:
                        ax.contour(np.nan_to_num(M, nan=1e9), levels=[5, 10, 25], colors=["w", "orange", "r"],
                                   linewidths=1.0)
                    except Exception:
                        pass
            fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="overhead % vs N=0")
            fp = os.path.join(figs_dir, f"es_profile_heatmap_L{L}.png")
            fig.savefig(fp, dpi=130, bbox_inches="tight")
            plt.close(fig)
            out.append(f"\n![heatmap L={L}](figs/{os.path.basename(fp)})\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "analysis.md"))
    args = ap.parse_args()
    parts = [phase1_tables(), phase2_tables(), phase4_tables(), phase5(args.figs)]
    txt = "\n\n".join(p for p in parts if p)
    with open(args.out, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
