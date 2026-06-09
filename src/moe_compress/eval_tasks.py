"""Training-free eval of a compressed OLMoE via lm-eval-harness, matching the
OLMoE-paper task settings as closely as the harness allows:

  - MMLU            : 5-shot, acc (multiple-choice log-likelihood)
  - GSM8K (CoT)     : 8-shot, exact-match on the final answer (generation)
  - ARC-Challenge   : 25-shot acc_norm (standard)
  - HellaSwag       : 10-shot acc_norm (standard)

We wrap an ALREADY-LOADED HF model (the in-memory compressed model) with lm-eval's
HFLM so eval reuses the same weights — no save/reload round-trip needed at step 0.
Ground truth comes from each task's own dataset (lm-eval handles this); we never
use another model's output as ground truth.
"""
from __future__ import annotations

from loguru import logger

# (task, num_fewshot, primary_metric_substr)
TASK_SPEC = [
    ("mmlu", 5, "acc"),
    ("gsm8k", 8, "exact_match"),
    ("arc_challenge", 25, "acc_norm"),
    ("hellaswag", 10, "acc_norm"),
]


def eval_all(model, tokenizer, device, *, limit: int | None = None,
             tasks: list[str] | None = None, batch_size: int = 16) -> dict:
    """Run the 4-task suite on an in-memory HF model. `limit` caps examples/task
    (use a small number for smoke; None for full)."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    spec = [s for s in TASK_SPEC if (tasks is None or s[0] in tasks)]
    out = {}
    for task, nshot, metric_key in spec:
        logger.info(f"eval {task} ({nshot}-shot, limit={limit}) ...")
        res = simple_evaluate(model=lm, tasks=[task], num_fewshot=nshot,
                              limit=limit, bootstrap_iters=0, verbosity="ERROR")
        row = res["results"][task]
        # find the primary metric (key like 'acc,none' / 'exact_match,strict-match')
        val = None
        for k, v in row.items():
            if k.startswith(metric_key) and isinstance(v, (int, float)):
                val = float(v)
                break
        if val is None:  # fallback: first numeric metric
            val = next((float(v) for v in row.values() if isinstance(v, (int, float))), None)
        out[task] = {"metric": metric_key, "value": val, "nshot": nshot, "raw": row}
        logger.info(f"  {task}: {metric_key}={val}")
    return out


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                             trust_remote_code=True).to(a.device)
    r = eval_all(m, tok, a.device, limit=a.limit, tasks=a.tasks)
    import json
    print(json.dumps(r, indent=2, default=str))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2, default=str)
