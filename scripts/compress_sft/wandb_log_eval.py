"""Log post-hoc benchmark metrics to a wandb run at a given checkpoint step.

Reads the math500/mmlu metrics JSONs written by eval_opd_ckpt.py / eval_mmlu_pro.py
and logs eval/math500_acc + eval/mmlu_pro_acc to a SEPARATE per-objective eval run
(`..._eval`) in the project, with the checkpoint global_step as the x-axis. Offline
mode -> picked up by the wandb sync daemon (compute nodes have no internet).
"""
from __future__ import annotations
import argparse, json, os


def _load(path, key):
    if path and os.path.exists(path):
        try:
            v = json.load(open(path)).get(key)
            return v
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-json")
    ap.add_argument("--mmlu-json")
    ap.add_argument("--run-id", required=True, help="wandb run id (use the _eval run)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--step", type=int, required=True, help="checkpoint global_step (x-axis)")
    ap.add_argument("--wandb-dir", default=None)
    args = ap.parse_args()

    metrics = {}
    m = _load(args.math_json, "math500_acc")
    if m is not None:
        metrics["eval/math500_acc"] = m
    mm = _load(args.mmlu_json, "mmlu_pro_acc")
    if mm is not None:
        metrics["eval/mmlu_pro_acc"] = mm
    if not metrics:
        print("wandb_log_eval: no metrics found; nothing logged")
        return

    os.environ.setdefault("WANDB_MODE", "offline")
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir
    import wandb
    wandb.init(project=args.project, id=args.run_id, name=args.run_id, resume="allow")
    wandb.log(metrics, step=args.step)
    wandb.finish()
    print(f"wandb_log_eval: logged {metrics} at step {args.step} -> {args.project}/{args.run_id}")


if __name__ == "__main__":
    main()
