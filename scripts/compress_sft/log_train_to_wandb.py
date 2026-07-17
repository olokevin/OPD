"""Single login-node ONLINE logger that drives ONE wandb run per objective with
BOTH the training curve and the benchmark-eval points.

Compute nodes have no outbound internet, so the training process logs wandb
offline (those dirs are never synced) and the eval jobs only write metric JSONs.
This daemon, running on a login node, is the canonical writer:

  * train/*  — read from LlamaFactory's trainer_log.jsonl (the COMPLETE step
    history; robust to the broken offline+resume consolidation across the many
    4h re-allocations). Only logs global_steps not already present; fills gaps.
  * eval/{math500,aime24,mmlu_pro}_acc — read from the per-step eval JSONs that
    compress_sft_eval_job.sh writes under <metrics-root>/<obj>_r<ratio>/step<N>/
    (gated by an EVAL_DONE marker so partial writes aren't logged).

Both are logged into the SAME run id, against train/global_step as the x-axis.

  python log_train_to_wandb.py --objective forward --ratio 0.7
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

DATA_ROOT = os.environ.get("DATA_ROOT", "/pscratch/sd/y/yequan/opd")

# eval json filename -> (metric key in json, wandb key)
EVAL_METRICS = [
    ("math500.json", "math500_acc", "eval/math500_acc"),
    ("math500.json", "aime24_acc", "eval/aime24_acc"),
    ("mmlu_pro.json", "mmlu_pro_acc", "eval/mmlu_pro_acc"),
]


def history_steps(api, path, key):
    """Steps already present in the run for a given metric key (dedup on restart)."""
    try:
        r = api.run(path)
        return {row.get("train/global_step") for row in r.scan_history(keys=[key, "train/global_step"])
                if row.get(key) is not None and row.get("train/global_step") is not None}
    except Exception:
        return set()


def read_eval(step_dir: Path):
    """Return {wandb_key: value} for a completed step dir, or None if not done."""
    if not (step_dir / "EVAL_DONE").exists():
        return None
    out = {}
    cache: dict = {}
    for fname, jkey, wkey in EVAL_METRICS:
        if fname not in cache:
            p = step_dir / fname
            try:
                cache[fname] = json.load(open(p)) if p.exists() else {}
            except Exception:
                cache[fname] = {}
        v = cache[fname].get(jkey)
        if v is not None:
            out[wkey] = v
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["forward", "combined"])
    ap.add_argument("--ratio", default="0.7", help="retain ratio tag, e.g. 0.7")
    ap.add_argument("--project", default="nersc_compress_sft_qwen4b")
    ap.add_argument("--entity", default="yequan_zhao-university-of-california-santa-barbara")
    ap.add_argument("--metrics-root", default=f"{DATA_ROOT}/compress_sft/metrics")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    import wandb

    tag = f"{args.objective}_r{args.ratio}"
    rid = f"qwen3_4b_nersc_compress_{args.objective}_r{args.ratio}_sft"
    logf = f"{DATA_ROOT}/compress_sft/sft/qwen3_4b/{tag}/trainer_log.jsonl"
    eval_dir = Path(args.metrics_root) / tag

    api = wandb.Api()
    run_path = f"{args.entity}/{args.project}/{rid}"
    done = history_steps(api, run_path, "train/loss")
    eval_done = history_steps(api, run_path, "eval/math500_acc")
    print(f"[wandb {tag}] resuming {rid}; {len(done)} train steps, "
          f"{len(eval_done)} eval steps already present; eval_dir={eval_dir}")

    os.environ["WANDB_MODE"] = "online"
    run = wandb.init(project=args.project, entity=args.entity, id=rid, name=rid, resume="allow")
    run.define_metric("train/global_step")
    for k in ("train/loss", "train/learning_rate", "train/epoch",
              "eval/math500_acc", "eval/aime24_acc", "eval/mmlu_pro_acc"):
        run.define_metric(k, step_metric="train/global_step")

    while True:
        # --- train curve ---
        if os.path.exists(logf):
            new = 0
            with open(logf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    s = d.get("current_steps")
                    if s is None or s in done or "loss" not in d:
                        continue
                    run.log({"train/loss": d["loss"], "train/learning_rate": d.get("lr"),
                             "train/epoch": d.get("epoch"), "train/global_step": s})
                    done.add(s)
                    new += 1
            if new:
                print(f"[wandb {tag}] +{new} train steps (through {max(done)})")

        # --- benchmark eval points ---
        if eval_dir.is_dir():
            for sd in sorted(eval_dir.glob("step*"), key=lambda p: int(p.name[4:] or -1)):
                try:
                    step = int(sd.name[4:])
                except ValueError:
                    continue
                if step in eval_done:
                    continue
                m = read_eval(sd)
                if m is None:
                    continue
                run.log({**m, "train/global_step": step})
                eval_done.add(step)
                print(f"[wandb {tag}] eval@step{step}: {m}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
