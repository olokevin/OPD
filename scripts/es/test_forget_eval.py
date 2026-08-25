"""Gate for the in-loop prior-task probe (`verl.trainer.es.forget_eval`).

The probe scores lm-eval multiple-choice tasks through the ES trainer's own vLLM
engine.  This checks that the log-likelihood it computes agrees with lm-eval's
reference vLLM backend on the same model/task/limit -- i.e. that the token span and
the prompt_logprobs alignment are right.  Run before spending a 15 h ES arm on it.

    python scripts/es/test_forget_eval.py --model Qwen/Qwen2.5-0.5B --limit 200
"""

import argparse
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "verl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tasks", default="hellaswag,arc_easy")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    import ray
    from transformers import AutoTokenizer
    from vllm import LLM

    tasks = [t for t in args.tasks.split(",") if t]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, enforce_eager=True, seed=0)

    # The probe talks to a Ray actor handle (`engine.generate.remote(...)` + `ray.get`).
    # Present the local engine through the same surface instead of standing up Ray.
    fake_engine = SimpleNamespace(generate=SimpleNamespace(remote=llm.generate))
    real_get = ray.get
    ray.get = lambda x: x
    try:
        from verl.trainer.es.forget_eval import evaluate_prior_tasks
        ours = evaluate_prior_tasks(fake_engine, tok, tasks, limit=args.limit,
                                    batch_size=256, max_len=2048)
    finally:
        ray.get = real_get

    import lm_eval
    ref = lm_eval.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={args.model},dtype=bfloat16,gpu_memory_utilization={args.gpu_mem},"
                   f"max_model_len=2048,enforce_eager=True",
        tasks=tasks, limit=args.limit, num_fewshot=0, bootstrap_iters=0,
    )["results"]

    ok = True
    print("\n%-16s %10s %10s %8s" % ("task", "in-loop", "lm_eval", "delta"))
    for t in tasks:
        m = ref[t]
        key = "acc_norm,none" if "acc_norm,none" in m else "acc,none"
        r = 100.0 * float(m[key])
        o = ours[f"forget/{t}"]
        good = abs(o - r) <= args.tol
        ok &= good
        print("%-16s %10.4f %10.4f %8.4f  %s" % (t, o, r, o - r, "PASS" if good else "FAIL"))
    print("\n" + ("ALL PASS" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
