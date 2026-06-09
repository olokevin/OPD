"""Aggregate moe_compress training-free atlas metrics into a step-0 ranking table,
grouped by compression FAMILY (the headline analysis axis). Reads every
<method>_r<ret>_s<seed>.json under the metrics dir.

This is the step-0 (training-free) half of the study. The post-recovery half
(AURC, inversion test) is added once recovery runs land — this script already
prints the step-0 family ranking so we can see whether families separate
training-free (the baseline the inversion test compares against).

Usage:
  python scripts/moe_compress/analyze_atlas.py [metrics_dir]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS_DIR = Path(sys.argv[1] if len(sys.argv) > 1
                   else "/data/yequan/moe_compress/metrics")

TASKS = ["mmlu", "gsm8k", "arc_challenge", "hellaswag"]


def load():
    rows = []
    for jf in sorted(METRICS_DIR.glob("*_r0.*_s*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:  # noqa: BLE001 - partial/in-progress file
            continue
        if "step0" not in d:
            continue
        row = {
            "method": d["method"], "family": d.get("family", "?"),
            "retain": d["retain_target"], "seed": d["seed"],
            "storage": d["budget"]["storage_retain"],
            "active": d["budget"]["active_retain"],
        }
        for t in TASKS:
            row[t] = (d["step0"].get(t) or {}).get("value")
        rows.append(row)
    return rows


def main():
    rows = load()
    if not rows:
        print(f"no completed metrics in {METRICS_DIR} yet")
        return
    print(f"# Training-free atlas — {len(rows)} runs from {METRICS_DIR}\n")
    hdr = f"{'method':18s} {'family':14s} {'ret':4s} {'stor':5s} {'act':5s} " + \
          " ".join(f"{t[:8]:>8s}" for t in TASKS)
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["retain"], x["family"], x["method"])):
        vals = " ".join(f"{(r[t] if r[t] is not None else float('nan')):8.3f}" for t in TASKS)
        print(f"{r['method']:18s} {r['family']:14s} {r['retain']:<4} "
              f"{r['storage']:<5} {r['active']:<5} {vals}")

    # family-level step-0 means (per retain)
    print("\n## Family step-0 means (avg over tasks present, per retain)")
    by = defaultdict(list)
    for r in rows:
        present = [r[t] for t in TASKS if r[t] is not None]
        if present:
            by[(r["retain"], r["family"])].append(sum(present) / len(present))
    for (ret, fam), vals in sorted(by.items()):
        print(f"  retain={ret} {fam:14s} mean={sum(vals)/len(vals):.3f}  (n={len(vals)})")

    # who is the step-0 winner family per retain (the baseline for the inversion test)
    print("\n## Step-0 winner FAMILY per retain (to compare vs post-recovery)")
    fam_means = defaultdict(dict)
    for (ret, fam), vals in by.items():
        fam_means[ret][fam] = sum(vals) / len(vals)
    for ret, fams in sorted(fam_means.items()):
        win = max(fams, key=fams.get)
        print(f"  retain={ret}: winner={win} ({fams[win]:.3f}) | all: "
              + ", ".join(f"{f}={v:.3f}" for f, v in sorted(fams.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
