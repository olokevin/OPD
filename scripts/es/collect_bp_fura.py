#!/usr/bin/env python3
"""Per-step curve of a BP GRPO run (verl console log) as a markdown table.

Usage:  python scripts/es/collect_bp_fura.py logs/es/bp_fura_lr1.0_gpu4.log [more logs...]

Prints, per log: step | greedy MATH-500 | train score | grad norm | entropy | s/step,
plus the trainable-parameter line if the adapter printed one.
"""
import re
import sys

_KV = re.compile(r"([\w/@.\-]+):(?:np\.float64\()?([-+\d.eE]+)\)?")


def parse(path):
    rows = {}
    n_params = None
    with open(path, errors="replace") as f:
        for line in f:
            if "trainable" in line.lower() and "param" in line.lower() and n_params is None:
                n_params = line.strip()[:200]
            m = re.search(r"step:(\d+) - ", line)
            if not m:
                continue
            step = int(m.group(1))
            d = rows.setdefault(step, {})
            for k, v in _KV.findall(line[m.end():]):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
    return rows, n_params


def fmt(v, scale=1.0, nd=1):
    return "" if v is None else f"{v * scale:.{nd}f}"


def main():
    for path in sys.argv[1:]:
        rows, n_params = parse(path)
        print(f"\n### {path}")
        if n_params:
            print(n_params)
        print("| step | MATH-500 (greedy) | train score | grad norm | entropy | s/step |")
        print("|---|---|---|---|---|---|")
        for step in sorted(rows):
            d = rows[step]
            print(
                f"| {step} | {fmt(d.get('val-core/MATH-500/acc/mean@1'), 100, 1)} "
                f"| {fmt(d.get('critic/score/mean'), 1, 3)} "
                f"| {fmt(d.get('actor/grad_norm'), 1, 3)} "
                f"| {fmt(d.get('actor/entropy'), 1, 3)} "
                f"| {fmt(d.get('timing_s/step'), 1, 0)} |"
            )


if __name__ == "__main__":
    main()
