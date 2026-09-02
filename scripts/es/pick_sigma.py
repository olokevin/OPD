"""From a `probe_reward_std.sh` log, pick the sigma whose train/reward_std lands at TARGET.

log-log fit of reward_std vs sigma (it is close to a power law in the useful range), then
solve for the target.  Prints `sigma_star sigma_hi` -- the target and a 2x bracket.

    python3 scripts/es/pick_sigma.py logs/es/probe_furazoact_gpu6.log [--target 0.050]
"""
import argparse, math, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--target", type=float, default=0.050)
a = ap.parse_args()

pts, cur = [], None
for line in open(a.log, errors="ignore"):
    m = re.search(r"probe \S+ r=\d+ sigma=(\S+) ", line)
    if m:
        cur = (float(m.group(1)), [])
        pts.append(cur)
        continue
    m = re.search(r"train/reward_std:([\d.eE+-]+)", line)
    if m and cur:
        cur[1].append(float(m.group(1)))

pts = [(s, sum(v) / len(v)) for s, v in pts if v]
# reward_std == 0 means every population member scored identically -- the perturbation
# destroyed the model, so that sigma carries no information and cannot be log-fitted.
dead = [s for s, v in pts if v <= 0]
pts = [(s, v) for s, v in pts if v > 0]
for s, v in pts:
    print(f"# sigma={s:<10g} reward_std={v:.4f}", file=sys.stderr)
if dead:
    print(f"# dead (reward_std=0, model destroyed): sigma in {dead}", file=sys.stderr)
if len(pts) < 2:
    sys.exit("need >=2 live probe points")
# never recommend at or above the smallest sigma that killed the model
ceil_ = min(dead) / 2 if dead else float("inf")

# least-squares line through (log sigma, log std); reward_std saturates at large sigma, so
# fit only the points below 1.5x target where the power law still holds, if there are two.
usable = [p for p in pts if p[1] <= 1.5 * a.target] or pts
if len(usable) < 2:
    usable = sorted(pts, key=lambda p: p[1])[:2]
xs = [math.log(s) for s, _ in usable]
ys = [math.log(v) for _, v in usable]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
den = sum((x - mx) ** 2 for x in xs) or 1e-30
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
star = math.exp(mx + (math.log(a.target) - my) / b) if b else usable[-1][0]
lo, hi = min(s for s, _ in pts), max(s for s, _ in pts)
star = min(max(star, lo / 3), hi * 3)          # do not extrapolate wildly
star = min(star, ceil_)
print(f"{star:.6g} {min(star * 2, ceil_):.6g}")
