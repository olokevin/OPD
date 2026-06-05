"""Per-ratio 'length to first CORRECT answer' for the sweep trace probes.

The plain comp_len column conflates two failures: (a) the model reaches the right
answer then loops past the token cap (a TERMINATION failure), and (b) the model
never reaches the right answer (a REASONING failure). To pave away (a), we measure
the char index at which the compressed trace FIRST emits a \\boxed{...} whose
content the ttrl_math grader scores correct — i.e. how far in the model first
*reaches* the answer, ignoring whatever degenerate repetition follows.

Output: per ratio, n_reached/5 (probes that ever produce the correct boxed answer)
and the median len-to-first-correct over those that do.

Usage:
  PYTHONPATH=src:verl python scripts/opd/math/compressed_opd/analyze_len_to_correct.py \
    --sweep-dir scripts/opd/math/compressed_opd/results/sweep \
    --probe-set scripts/opd/math/compressed_opd/results/blockT/trace_probe_set.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "verl"))
from verl.utils.reward_score.ttrl_math import compute_score  # noqa: E402


def len_to_first_correct(text: str, gold: str):
    """Char index just past the first \\boxed{...} whose prefix grades correct,
    or None if the text never produces a correct boxed answer."""
    i = 0
    while True:
        m = re.search(r'\\boxed\s*{', text[i:])
        if not m:
            return None
        j = i + m.end()
        depth = 1
        while j < len(text) and depth:
            depth += (text[j] == '{') - (text[j] == '}')
            j += 1
        if compute_score(text[:j], str(gold)).get('acc', False):
            return j
        i = j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--probe-set", required=True)
    args = ap.parse_args()

    ps = {p['problem_id']: p for p in json.load(open(args.probe_set))['probes']}
    sweep_dir = Path(args.sweep_dir)
    files = sorted(sweep_dir.glob("traces_r*.json"),
                   key=lambda p: float(p.stem.split("_r")[1]))
    rows = []
    for f in files:
        r = float(f.stem.split("_r")[1])
        traces = json.load(open(f))['traces']
        lens = {}
        for t in traces:
            lens[t['problem_id']] = len_to_first_correct(
                t['comp_text'], ps[t['problem_id']]['gold'])
        reached = [L for L in lens.values() if L is not None]
        med = sorted(reached)[len(reached) // 2] if reached else None
        rows.append({"ratio": r, "n_reached": len(reached), "n_probes": len(traces),
                     "median_len_to_correct": med, "per_probe": lens})

    out = sweep_dir / "len_to_correct.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"{'ratio':>6} {'reached':>9} {'median len-to-correct':>22}")
    for r in rows:
        med = "—" if r['median_len_to_correct'] is None else str(r['median_len_to_correct'])
        print(f"{r['ratio']:>6} {r['n_reached']:>5}/{r['n_probes']:<3} {med:>22}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
