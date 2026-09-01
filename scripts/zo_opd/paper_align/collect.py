"""Tabulate the paper-aligned BP-OPD and ES runs into one comparable view."""
import glob, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
os.chdir(REPO)

def num(line, key):
    m = re.search(re.escape(key) + r":(?:np\.float64\()?([-0-9.e+]+)", line)
    return float(m.group(1)) if m else None

bp_logs = sorted(glob.glob("logs/train/bp_paper_*.log"), key=os.path.getmtime)
for bp in bp_logs[-1:]:
    print(f"=== BP-OPD  ({bp}) ===")
    prog, steps = None, {}
    with open(bp, errors="ignore") as f:
        for line in f:
            if "Training Progress:" in line:
                m = re.search(r"Training Progress: *\d+%[^]]*\]", line)
                if m: prog = m.group(0)
            m = re.search(r"step:(\d+) - ", line)
            if not m: continue
            s = int(m.group(1))
            d = steps.setdefault(s, {})
            for k, key in [("math", "val-core/MATH-500/acc/mean@4"),
                           ("amc", "val-core/AMC23/acc/mean@4"),
                           ("aime", "val-core/AIME24/acc/mean@4"),
                           ("ovl", "val-topk/overlap_ratio"),
                           ("len", "response_length/mean"),
                           ("clip", "response_length/clip_ratio"),
                           ("gn", "actor/grad_norm"),
                           ("t", "timing_s/step")]:
                v = num(line, key)
                if v is not None and k not in d: d[k] = v
    if prog: print(" ", prog)
    print(f"  {'step':>5} {'MATH-500@4':>11} {'AMC23@4':>8} {'AIME24@4':>9} "
          f"{'overlap':>8} {'resp_len':>9} {'clip':>6} {'grad_nrm':>9} {'s/step':>7}")
    for s in sorted(steps):
        d = steps[s]
        if "math" not in d and s % 20 != 0 and s > 0:
            continue
        fmt = lambda k, p=4: (f"{d[k]:.{p}f}" if k in d else "-")
        print(f"  {s:>5} {fmt('math'):>11} {fmt('amc'):>8} {fmt('aime'):>9} "
              f"{fmt('ovl'):>8} {fmt('len',1):>9} {fmt('clip',3):>6} "
              f"{fmt('gn',3):>9} {fmt('t',1):>7}")

for f in sorted(glob.glob("logs/es_lr_sweep/lr_*.log")) + sorted(glob.glob("logs/es_paper/*.log")):
    txt = open(f, errors="ignore").read()
    probes = re.findall(r"\[Probe @ step (\d+)\] heldout_clean_loss=([0-9.]+)", txt)
    accs = re.findall(r"eval/accuracy:([0-9.]+)", txt)
    steps_t = re.findall(r"train/step_time:([0-9.]+)", txt)
    dws = re.findall(r"train/dW_norm_mean:([0-9.]+)", txt)
    Ls = re.findall(r"train/L_clean_mean:([0-9.]+)", txt)
    if not (probes or accs): continue
    print(f"\n=== ES  ({f}) ===")
    print("  probe (64 fixed, GREEDY, lower=better): " +
          " ".join(f"s{s}:{v}" for s, v in probes))
    print("  MATH-500 greedy acc (%):                " + " ".join(accs))
    if steps_t:
        st = [float(x) for x in steps_t]
        print(f"  steps={len(st)}  median step {sorted(st)[len(st)//2]:.1f}s  "
              f"dW_norm_mean {dws[0] if dws else '-'} -> {dws[-1] if dws else '-'}")
    if Ls:
        print(f"  L_clean_mean (per-batch, NOT a curve): {Ls[0]} ... {Ls[-1]}")
