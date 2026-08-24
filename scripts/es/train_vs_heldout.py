"""Pair each MATH-500 eval with the train accuracy at the same ES step.

The fixed 64-problem training batch is small enough to overfit, so "how far along is
this run" is better read as train accuracy than as step count.  Comparing methods at
*matched train accuracy* separates two things the step-indexed curve conflates:
  - a slower run that is simply earlier on the same trajectory, vs
  - a run that genuinely generalises worse per unit of training progress.

    python3 scripts/es/train_vs_heldout.py [--runs run1_dense sweep_fura_a12.5x ...]
"""
import argparse
import os
import re

REPO = "/home/yequan/Project/compression/OPD"
EV = re.compile(r"\[Eval @ step (\d+)\].*?acc=([\d.]+)%")
AC = re.compile(r"train/accuracy:([\d.]+)")

DEFAULT = [
    ("dense (paper α)", "run1_dense"),
    ("zoact r=1", "run2_zoact"),
    ("insparse d=1%", "run3_insparse"),
    ("fura 1× α", "run4_fura_alpha5e-4_baseline"),
    ("fura 12.5× α", "sweep_fura_a12.5x"),
    ("fura 40× α", "sweep_fura_a40x"),
    ("fura σ+α matched", "sweep_fura_match-sig"),
    ("fura 4× α", "sweep_fura_a4x"),
]


def load(log):
    p = os.path.join(REPO, "logs/es", log + ".log")
    if not os.path.exists(p):
        return None
    ev, ac = {}, []
    for line in open(p, errors="ignore"):
        for m in EV.finditer(line):
            ev[int(m.group(1))] = float(m.group(2))
        for m in AC.finditer(line):
            ac.append(float(m.group(1)))
    return ev, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--band", nargs=2, type=float, default=[73.5, 76.5],
                    help="train-accuracy band to average held-out over")
    args = ap.parse_args()
    runs = [(n, n) for n in args.runs] if args.runs else DEFAULT

    print(f"{'run':20s} {'step':>5s} {'train':>7s} {'MATH-500':>9s} {'gap':>7s}")
    band = {}
    for name, log in runs:
        r = load(log)
        if not r:
            continue
        ev, ac = r
        for s in sorted(ev):
            if s == 0 or s > len(ac):
                continue
            t = ac[s - 1]
            print(f"{name:20s} {s:5d} {t:7.1f} {ev[s]:9.1f} {ev[s]-t:+7.1f}")
            if args.band[0] <= t <= args.band[1]:
                band.setdefault(name, []).append(ev[s])
        print()

    lo, hi = args.band
    print(f"Held-out MATH-500 averaged over evals with train accuracy in [{lo}, {hi}]:")
    for name, vs in band.items():
        print(f"  {name:20s} {sum(vs)/len(vs):5.1f}  (n={len(vs)}: "
              + ", ".join(f"{v:.1f}" for v in vs) + ")")


if __name__ == "__main__":
    main()
