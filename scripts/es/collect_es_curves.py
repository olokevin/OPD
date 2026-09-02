"""Parse ES chain logs into per-run MATH-500 curves, plateaus and cost.

A chain script (`chain_*.sh`) runs several configs back-to-back into one log, so this
splits on the `trainer.experiment_name=` line that `set -x` echoes at each launch.

    python3 scripts/es/collect_es_curves.py logs/es/*pop10*.log            # table
    python3 scripts/es/collect_es_curves.py --json out.json logs/es/*.log  # machine-readable
"""
import argparse, json, math, os, re, sys

NAME_RE = re.compile(r"trainer\.experiment_name=(\S+)")
EVAL_RE = re.compile(r"\[Eval @ step (\d+)\] avg_reward=[\d.]+ . [\d.]+ acc=([\d.]+)% time=([\d.]+)s")
ITER_RE = re.compile(r"train/iteration_time:([\d.]+)")
STD_RE = re.compile(r"train/reward_std:([\d.eE+-]+)")


def split_runs(paths):
    runs, cur = [], None
    for path in paths:
        for line in open(path, errors="ignore"):
            m = NAME_RE.search(line)
            if m:
                cur = {"name": m.group(1), "log": os.path.basename(path),
                       "evals": {}, "eval_s": [], "iters": [], "stds": []}
                runs.append(cur)
                continue
            if cur is None:
                continue
            m = EVAL_RE.search(line)
            if m:
                cur["evals"][int(m.group(1))] = float(m.group(2))
                cur["eval_s"].append(float(m.group(3)))
            for m in ITER_RE.finditer(line):
                cur["iters"].append(float(m.group(1)))
            for m in STD_RE.finditer(line):
                cur["stds"].append(float(m.group(1)))
    # the launch line is echoed more than once per job (set -x plus Hydra's own dump), and a
    # re-launch of the same config appends -- so MERGE same-named blocks, later wins per step.
    seen = {}
    for r in runs:
        cur = seen.setdefault(r["name"], dict(r, evals={}, eval_s=[], iters=[], stds=[]))
        cur["evals"].update(r["evals"])
        for k in ("eval_s", "iters", "stds"):
            cur[k].extend(r[k])
    return list(seen.values())


def med(xs):
    return sorted(xs)[len(xs) // 2] if xs else None


def summarize(r, plateau_from=40):
    ev = r["evals"]
    pl = [v for s, v in ev.items() if s >= plateau_from]
    it = med(r["iters"])
    n_it = max(ev) if ev else 0
    out = {
        "name": r["name"],
        "steps": n_it,
        "base": ev.get(0),
        "plateau": sum(pl) / len(pl) if pl else None,
        "plateau_se": (
            (sum((v - sum(pl) / len(pl)) ** 2 for v in pl) / (len(pl) * max(len(pl) - 1, 1))) ** 0.5
            if len(pl) > 1 else None),
        "n_plateau": len(pl),
        "best": max(ev.values()) if ev else None,
        "best_step": max(ev, key=ev.get) if ev else None,
        "iter_s": it,
        "eval_s": med(r["eval_s"]),
        "reward_std": (sum(r["stds"]) / len(r["stds"])) if r["stds"] else None,
        "gpu_h": (it * n_it + (med(r["eval_s"]) or 0) * len(ev)) / 3600 if it else None,
        "evals": ev,
    }
    # cheapest point at which the run first reaches 70% held-out
    hit = [s for s in sorted(ev) if ev[s] >= 70.0]
    out["gpu_h_to_70"] = (it * hit[0] / 3600) if (hit and it) else None
    return out


ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--json")
ap.add_argument("--plateau-from", type=int, default=40)
a = ap.parse_args()

rows = [summarize(r, a.plateau_from) for r in split_runs(a.logs)]
rows = [r for r in rows if r["evals"]]

f = lambda v, p=2: "—" if v is None else f"{v:.{p}f}"
print(f"| run | steps | base | plateau (>={a.plateau_from}) | best @ step | s/iter | GPU-h | GPU-h to 70% | reward sd |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    pl = f(r["plateau"]) + (f" ± {r['plateau_se']:.2f}" if r["plateau_se"] else "")
    print(f"| `{r['name']}` | {r['steps']} | {f(r['base'],1)} | {pl} | "
          f"{f(r['best'],1)} @ {r['best_step']} | {f(r['iter_s'],0)} | {f(r['gpu_h'])} | "
          f"{f(r['gpu_h_to_70'])} | {f(r['reward_std'],4)} |")

print()
allsteps = sorted({s for r in rows for s in r["evals"]})
print("| step | " + " | ".join(r["name"] for r in rows) + " |")
print("|" + "---|" * (len(rows) + 1))
for s in allsteps:
    print(f"| {s} | " + " | ".join(f"{r['evals'][s]:.1f}" if s in r["evals"] else "" for r in rows) + " |")

if a.json:
    json.dump(rows, open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}", file=sys.stderr)
