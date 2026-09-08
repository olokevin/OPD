"""progress.py -- one-screen status of the ds15b arms (BP vs ES) from their run logs.

  python scripts/zo_opd/ds15b/progress.py            # latest step + evals per arm
  python scripts/zo_opd/ds15b/progress.py --every 10 # step table every 10 steps
"""
import argparse, glob, os, re

ARMS = {
    "BP":   "logs/ds15b/run_*.log",
    "ES-A": "logs/ds15b/es/run_*.log",
    "ES-B": "logs/ds15b/es/B/run_*.log",
    "ES-C": "logs/ds15b/es/C/run_*.log",
    "ES-D": "logs/ds15b/es/D/run_*.log",
    "ES-F(b64n1-N16)": "logs/ds15b/es/F/run_*.log",
}
KEYS = ["critic/rewards/mean", "response_length/mean", "es/post_update_gain", "es/cum_footprint",
        "es/d_snr", "timing_s/step", "val-core/MATH-500/acc/mean@2", "val-core/AIME24/acc/mean@2"]


def parse(path):
    steps, evals = {}, {}
    for line in open(path, errors="ignore"):
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        st, rest = int(m.group(1)), m.group(2)
        d = {}
        for tok in rest.split(" - "):
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            v = re.sub(r"np\.float64\(|\)", "", v)
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if "val-core/MATH-500/acc/mean@2" in d:
            evals[st] = d
        else:
            steps[st] = d
    return steps, evals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=0); a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    for arm, pat in ARMS.items():
        logs = sorted(glob.glob(pat), key=os.path.getmtime)
        if not logs:
            continue
        steps, evals = parse(logs[-1])
        print(f"=== {arm}  ({logs[-1]}; {len(steps)} steps, {len(evals)} evals)")
        rows = sorted(steps)
        if a.every:
            rows = [s for s in rows if s == 1 or s % a.every == 0]
        else:
            rows = rows[-1:]
        for s in rows:
            d = steps[s]
            print(f"  step {s:4d}  " + "  ".join(f"{'/'.join(k.split('/')[-2:])}={d[k]:.4g}" for k in KEYS if k in d))
        for s in sorted(evals):
            d = evals[s]
            print(f"  eval @{s:4d}  MATH-500 {d.get('val-core/MATH-500/acc/mean@2', float('nan')):.3f}"
                  f"  AIME24 {d.get('val-core/AIME24/acc/mean@2', float('nan')):.3f}")


if __name__ == "__main__":
    main()
