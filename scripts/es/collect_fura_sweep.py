"""Fill the FuRA LR-sweep table in docs/results/ES/es_results.md.

Kept separate from collect_es_results.py so the two never fight over the same file
region: this one only rewrites between the AUTO:FURASWEEP markers.
"""
import os, re, sys

REPO = "/home/yequan/Project/compression/OPD"
DOC = os.path.join(REPO, "docs/results/ES/es_results.md")
BEGIN, END = "<!-- AUTO:FURASWEEP BEGIN -->", "<!-- AUTO:FURASWEEP END -->"

# label, log, sigma, alpha, multiple of the paper alpha=5e-4
SWEEP = [
    ("1x (paper α)",   "run4_fura_alpha5e-4_baseline", "1e-3",    "5e-4",    "1×"),
    ("4x",             "sweep_fura_a4x",               "1e-3",    "2e-3",    "4×"),
    ("12.5x (matched)","sweep_fura_a12.5x",            "1e-3",    "6.25e-3", "12.5×"),
    ("40x",            "sweep_fura_a40x",              "1e-3",    "2e-2",    "40×"),
    ("σ+α matched",    "sweep_fura_match-sig",         "1.25e-2", "6.25e-3", "12.5×"),
]
# reference curves from the completed runs, for the "did it match full ES" question
EVAL_RE = re.compile(r"\[Eval @ step (\d+)\] avg_reward=([\d.]+).*?acc=([\d.]+)%")
STD_RE = re.compile(r"train/reward_std:([\d.eE+-]+)")
ACC_RE = re.compile(r"train/accuracy:([\d.]+)")


def parse(log):
    p = os.path.join(REPO, "logs/es", log + ".log")
    if not os.path.exists(p):
        return None
    ev, std, acc = {}, [], []
    for line in open(p, errors="ignore"):
        for m in EVAL_RE.finditer(line):
            ev[int(m.group(1))] = float(m.group(3))
        for m in STD_RE.finditer(line):
            std.append(float(m.group(1)))
        for m in ACC_RE.finditer(line):
            acc.append(float(m.group(1)))
    return {"ev": ev, "std": std, "acc": acc}


def main():
    rows, data = [], {}
    for label, log, sig, alp, mult in SWEEP:
        data[label] = parse(log)

    out = [BEGIN, ""]
    out.append("| α | ×paper | σ | MATH-500 @5 | @10 | @15 | @20 | train acc @20 | reward σ (mean) | status |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, log, sig, alp, mult in SWEEP:
        d = data[label]
        if d is None:
            out.append(f"| {alp} | {mult} | {sig} | | | | | | | _queued_ |")
            continue
        ev = d["ev"]
        cells = [f"{ev[s]:.1f}" if s in ev else "" for s in (5, 10, 15, 20)]
        tacc = f"{d['acc'][19]:.1f}" if len(d["acc"]) >= 20 else (
            f"{d['acc'][-1]:.1f}*" if d["acc"] else "")
        rstd = f"{sum(d['std'])/len(d['std']):.3f}" if d["std"] else ""
        status = "done" if 20 in ev else f"running (step {len(d['acc'])})"
        out.append(f"| {alp} | {mult} | {sig} | " + " | ".join(cells) +
                   f" | {tacc} | {rstd} | {status} |")
    # dense reference, restricted to its first 20 iterations so it is apples-to-apples
    # with the 20-iteration sweep (its full-run averages are lower: the spread decays).
    dn = parse("run1_dense")
    if dn:
        ev = dn["ev"]
        c = [f"{ev.get(10, 0):.1f}†", f"{ev.get(10, 0):.1f}",
             f"{ev.get(20, 0):.1f}†", f"**{ev.get(20, 0):.1f}**"]
        std20 = dn["std"][:20]
        out.append(f"| 5e-4 | — | 1e-3 | " + " | ".join(c) +
                   f" | {dn['acc'][19]:.1f} | {sum(std20)/len(std20):.3f} | "
                   "**dense reference** (first 20 it) |")
    out.append("")
    out.append("† dense was evaluated every 10 steps, so its @5/@15 cells repeat the "
               "neighbouring eval; the sweep uses every 5.")
    out.append("")
    out.append(END)
    block = "\n".join(out)

    doc = open(DOC).read()
    if BEGIN not in doc or END not in doc:
        print("markers missing; add them to the doc first", file=sys.stderr)
        print(block)
        return 1
    pre, rest = doc.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    open(DOC, "w").write(pre + block + post)
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
