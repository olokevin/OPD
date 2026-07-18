"""Grade a JSON list of {response, ground_truth} with the ttrl_math grader and write
back {accuracy, n_correct, n_total}. Invoked as a subprocess (in the `verl` conda env,
which has ttrl_math + latex2sympy2_extended + math_verify) by the in-trainer
MathMMLUEvalCallback, so the `sft` training env stays free of verl/ray/grading deps.

Usage (verl env):
  PYTHONPATH=src:verl /home/yequan/miniconda3/envs/verl/bin/python \\
    scripts/compress_sft/grade_responses.py --in <responses.json> --out <result.json>

Input JSON: [{"response": "...", "ground_truth": "..."}, ...]
Output JSON: {"accuracy": float, "n_correct": int, "n_total": int}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    from verl.utils.reward_score.ttrl_math import compute_score

    items = json.loads(Path(args.inp).read_text())
    n_correct = 0
    for it in items:
        try:
            res = compute_score(it["response"], str(it["ground_truth"]))
            n_correct += int(res.get("acc", False))
        except Exception:
            pass  # grading failure counts as wrong
    n_total = len(items)
    out = {
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
    }
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
