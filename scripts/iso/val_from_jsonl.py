"""Per-benchmark avg@16 + 5-benchmark mean from verl val_generations/<step>.jsonl
(used for the final step, whose console line is lost when the run exits).

  python3 scripts/iso/val_from_jsonl.py <run_dir>/val_generations/50.jsonl
"""
import json
import sys

import numpy as np
import pandas as pd

TD = "/home/yequan/Project/compression/OPD/datasets/test_data"
BENCH = ["AIME24", "AIME25", "AMC23", "Minerva", "Olympiad-Bench"]
key2b = {}
for b in BENCH:
    for p in pd.read_parquet(f"{TD}/{b}/test.parquet").prompt:
        key2b[p[0]["content"][:200]] = b
for path in sys.argv[1:]:
    acc = {b: [] for b in BENCH}
    for line in open(path):
        r = json.loads(line)
        b = next(v for k, v in key2b.items() if k in r["input"])
        acc[b].append(float(r["acc"]))
    pb = [100 * np.mean(acc[b]) for b in BENCH]
    print(f"{path}: " + " | ".join(f"{b} {v:.2f}" for b, v in zip(BENCH, pb)) + f" | avg {np.mean(pb):.2f}")
