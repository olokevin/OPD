"""Generative HellaSwag — the format-sensitive counterpart of the log-likelihood probe.

Section 14.5 of docs/results/ES/es_results.md leaves one hypothesis open: the forgetting
arXiv:2601.20861 reports may be *output-format drift* rather than knowledge loss. An ES
run that teaches a model to always emit `<think>...</think><answer>...</answer>` cannot
hurt a log-likelihood ranking over the four endings (which never asks the model to
generate), but it would wreck any protocol that asks the model to *say* which ending it
picks.

This scores the same HellaSwag items generatively: multiple choice A-D, greedy decode,
parse the letter.  The diagnostic is `unparsed` — the fraction of items where the model
did not emit a usable choice at all.

    python scripts/es/eval_hellaswag_gen.py --model <path> [--limit 1000]
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "verl"))

LETTERS = "ABCD"


def build_prompt(doc, tok):
    body = "\n".join(f"{LETTERS[i]}. {c.strip()}" for i, c in enumerate(doc["choices"]))
    user = (f"{doc['query'].strip()}\n{body}\n\n"
            "Which ending best completes the passage? Reply with a single letter (A, B, C, or D).")
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    return user + "\nAnswer:"


def parse_letter(text):
    m = re.search(r"\b([ABCD])\b", text.upper())
    return LETTERS.index(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lm_eval.tasks as lmt
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    task = lmt.TaskManager()
    from lm_eval.tasks import get_task_dict
    docs = list(get_task_dict(["hellaswag"], task)["hellaswag"].validation_docs())[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, seed=0)
    outs = llm.generate([build_prompt(d, tok) for d in docs],
                        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0))

    n_ok = n_unparsed = 0
    samples = []
    for o, d in zip(outs, docs):
        txt = o.outputs[0].text
        pred = parse_letter(txt)
        if pred is None:
            n_unparsed += 1
        elif pred == int(d["gold"]):
            n_ok += 1
        if len(samples) < 5:
            samples.append(txt[:160])

    res = {"model": args.model, "n": len(docs),
           "hellaswag_gen_acc": 100.0 * n_ok / len(docs),
           "unparsed": 100.0 * n_unparsed / len(docs),
           "samples": samples}
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}))
    for t in samples:
        print("  sample:", repr(t))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
