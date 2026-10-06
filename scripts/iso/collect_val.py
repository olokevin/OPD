"""Per-step eval table from a dapo_math_4b.sh log: avg@16 per benchmark + the
paper's aggregate (unweighted mean of the five benchmarks), plus key train stats.

  python3 scripts/iso/collect_val.py logs/iso/iso_lr7.5e-7.log [more logs...]
"""
import re
import sys

BENCH = ["AIME24", "AIME25", "AMC23", "Minerva", "Olympiad-Bench"]
TRAIN = ["critic/score/mean", "actor/grad_norm", "actor/kl_loss", "actor/entropy",
         "response_length/mean", "response_length/clip_ratio", "train/num_gen_batches", "timing_s/step"]


def parse(path):
    rows = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        step, kv = int(m.group(1)), m.group(2)
        vals = dict(re.findall(r"([\w/@\-\.]+):(?:np\.float64\()?([-\d\.e]+)", kv))
        rows.setdefault(step, {}).update(vals)
    return rows


def main():
    for path in sys.argv[1:]:
        rows = parse(path)
        print(f"\n## {path}")
        print("| step | " + " | ".join(BENCH) + " | **avg** |")
        print("|" + "---|" * (len(BENCH) + 2))
        for s in sorted(rows):
            r = rows[s]
            acc = [r.get(f"val-core/{b}/acc/mean@16") for b in BENCH]
            if all(a is not None for a in acc):
                a = [float(x) for x in acc]
                print(f"| {s} | " + " | ".join(f"{100*x:.2f}" for x in a) + f" | **{100*sum(a)/len(a):.2f}** |")
        print("\n| step | " + " | ".join(TRAIN) + " |")
        print("|" + "---|" * (len(TRAIN) + 1))
        for s in sorted(rows):
            r = rows[s]
            if "actor/grad_norm" in r:
                print(f"| {s} | " + " | ".join(f"{float(r[k]):.4g}" if k in r else "-" for k in TRAIN) + " |")


if __name__ == "__main__":
    main()
