"""Regenerate the MATH-500 curve + summary tables in docs/results/ES/es_results.md
from the raw run logs. Idempotent: rewrites everything between the AUTO markers.

    python3 scripts/es/collect_es_results.py [--print]
"""

import argparse
import os
import re
import sys

REPO = "/home/yequan/Project/compression/OPD"
DOC = os.path.join(REPO, "docs/results/ES/es_results.md")
BEGIN, END = "<!-- AUTO:RESULTS BEGIN -->", "<!-- AUTO:RESULTS END -->"

# (log stem, table label, ||dW||/||W||, trainable count)
# Runs 5/6 are the fixed-spectrum (ISO) modes: the perturbation is a group action, not an
# additive coefficient, so the count is the *manifold dimension searched per step*.
RUNS = [
    ("run1_dense", "dense (paper ES)", "5.0e-2", 7_615_616_512),
    ("run2_zoact", "zoact r=1", "4.2e-3", 1_390_592),
    ("run3_insparse", "insparse d=1%", "1.6e-2*", 65_415_168),
    ("run4_fura", "fura small-core", "4.0e-3", 97_771_520),
    ("run5_iso", "iso fixed-spectrum", "5.0e-2", 141_102_080),
    ("run6_isobtt", "isobtt fixed-spec small-core", "5.0e-2", 48_470_016),
]
ISO_RUNS = {"run5_iso", "run6_isobtt"}

EVAL_RE = re.compile(r"\[Eval @ step (\d+)\] avg_reward=([\d.]+).*?acc=([\d.]+)%")
STEP_RE = re.compile(r"training/global_step:(\d+)")
ITER_RE = re.compile(r"train/iteration_time:([\d.]+)")
STD_RE = re.compile(r"train/reward_std:([\d.eE+-]+)")
# fixed-spectrum health: ||W||_F drift (iso) / max|R^T R - I| (isobtt)
HEALTH_RE = re.compile(r"iso/(?:frob_drift|orth_err):([\d.eE+-]+)")


def parse(log):
    if not os.path.exists(log):
        return None
    evals, steps, iters, stds, health = {}, [], [], [], []
    with open(log, errors="ignore") as f:
        for line in f:
            for m in EVAL_RE.finditer(line):
                evals[int(m.group(1))] = float(m.group(3))
            for m in STEP_RE.finditer(line):
                steps.append(int(m.group(1)))
            for m in ITER_RE.finditer(line):
                iters.append(float(m.group(1)))
            for m in STD_RE.finditer(line):
                stds.append(float(m.group(1)))
            for m in HEALTH_RE.finditer(line):
                health.append(float(m.group(1)))
    return {
        "evals": evals,
        "last_step": max(steps) if steps else 0,
        "iter_s": sum(iters[-10:]) / len(iters[-10:]) if iters else None,
        "rstd": sum(stds[-10:]) / len(stds[-10:]) if stds else None,
        "health": max(health) if health else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    data = {name: parse(os.path.join(REPO, "logs/es", name + ".log")) for name, *_ in RUNS}
    live = [(n, lbl, dw, npar) for (n, lbl, dw, npar) in RUNS if data[n] and data[n]["evals"]]

    out = [BEGIN, ""]
    out.append("| # | Run | trainable coeffs | ‖ΔW‖/‖W‖ | mean reward σ (last 10 it) | "
               "MATH-500 best | best @ step | steps done |")
    out.append("|---|---|---|---|---|---|---|---|")
    for i, (n, lbl, dw, npar) in enumerate(RUNS, 1):
        d = data[n]
        if not d or not d["evals"]:
            mark = "†" if n in ISO_RUNS else ""
            out.append(f"| {i} | {lbl} | {npar:,}{mark} | {dw} | — | _queued_ | — | — |")
            continue
        best_step = max(d["evals"], key=lambda k: d["evals"][k])
        best = d["evals"][best_step]
        rstd = f"{d['rstd']:.3f}" if d["rstd"] is not None else "—"
        out.append(f"| {i} | {lbl} | {npar:,}{'†' if n in ISO_RUNS else ''} | {dw} | {rstd} | "
                   f"**{best:.1f}** | {best_step} | {d['last_step']} |")
    out.append("")
    out.append("\\* insparse ‖ΔW‖/‖W‖ measured at the 10% test density, not the 1% run density.")
    out.append("")
    out.append("† manifold dimension searched per step, not a coefficient count — the ISO modes "
               "perturb by a group action, not an additive coefficient. They run at "
               "σ = 5e-2 / α = 2.5e-2 (footprint-matched to run 1, *not* the paper's nominal σ); "
               "see [§10](#10-iso-fixed-spectrum-es).")
    out.append("")
    hz = [(lbl, data[n]["health"]) for n, lbl, _, _ in RUNS
          if n in ISO_RUNS and data[n] and data[n]["health"] is not None]
    if hz:
        out.append("Fixed-spectrum constraint health (worst value seen; ‖W‖_F drift for `iso`, "
                   "max|RᵀR − I| for `isobtt` — both should stay at fp32 round-off): "
                   + ", ".join(f"**{lbl}** {v:.1e}" for lbl, v in hz) + ".")
        out.append("")

    if live:
        out.append("### MATH-500 curve (eval every 10 iterations, 3,000-token budget)")
        out.append("")
        hdr = "| step | " + " | ".join(lbl for _, lbl, _, _ in live) + " |"
        out.append(hdr)
        out.append("|" + "---|" * (len(live) + 1))
        allsteps = sorted({s for n, *_ in live for s in data[n]["evals"]})
        for s in allsteps:
            cells = []
            for n, *_ in live:
                v = data[n]["evals"].get(s)
                cells.append(f"{v:.1f}" if v is not None else "")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")
        times = [d["iter_s"] for d in data.values() if d and d["iter_s"]]
        if times:
            out.append(f"Timing: ~{sum(times)/len(times):.0f} s/iteration "
                       f"(30 perturbations x 64-prompt rollout + grading).")
            out.append("")
    out.append(END)
    block = "\n".join(out)

    if args.show:
        print(block)
        return 0

    doc = open(DOC).read()
    if BEGIN in doc and END in doc:
        pre, rest = doc.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        doc = pre + block + post
    else:
        print(f"markers not found in {DOC}", file=sys.stderr)
        return 1
    open(DOC, "w").write(doc)
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
