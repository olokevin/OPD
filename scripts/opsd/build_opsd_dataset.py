#!/usr/bin/env python3
"""Build the verl parquet for OPSD (On-Policy Self-Distillation, arXiv:2601.18734).

Source: `siyanzhao/Openthoughts_math_30k_opsd` -- the authors' own 30k OpenThoughts
math slice, with `problem`, `solution` (the concise reference solution used as the
teacher's privileged context) and `Answer` (the boxed final answer).

Two prompts per row, verbatim from the reference implementation's `data_collator.py`
(`reason_first=False` branch, which is what `scripts/run_opsd_1b.sh` uses):

  student  = problem only,  chat template with enable_thinking=False
  teacher  = problem + reference solution + transition prompt,
             chat template with enable_thinking=True

The student prompt goes into `prompt` (verl applies the chat template itself, so we
store the raw user message).  The teacher prompt is stored FULLY TEMPLATED as a string
in `extra_info["teacher_prompt"]` -- `extra_info` is one of the four non-tensor keys
verl keeps on the batch through generation, so it reaches the reward-model worker.

    python scripts/opsd/build_opsd_dataset.py --out datasets/opsd_openthoughts_math_30k.parquet
"""
import argparse
import os

# Verbatim from https://github.com/siyan-zhao/OPSD/blob/main/data_collator.py
TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def student_user_message(problem: str) -> str:
    return f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def teacher_user_message(problem: str, solution: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="tokenizer whose chat template templates the teacher prompt")
    ap.add_argument("--out", default="datasets/opsd_openthoughts_math_30k.parquet")
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-teacher-tokens", type=int, default=3072,
                    help="drop rows whose templated teacher prompt exceeds this")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import pandas as pd

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(args.dataset)["train"]
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"loaded {len(ds)} rows from {args.dataset}")

    rows, dropped = [], 0
    for i, r in enumerate(ds):
        problem, solution, answer = r["problem"], r["solution"], r["Answer"]
        t_msg = teacher_user_message(problem, solution)
        # enable_thinking=True for the teacher (paper's TM-on teacher / TM-off student)
        t_prompt = tok.apply_chat_template(
            [{"role": "user", "content": t_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        n_tok = len(tok(t_prompt, add_special_tokens=False)["input_ids"])
        if n_tok > args.max_teacher_tokens:
            dropped += 1
            continue
        rows.append({
            "data_source": "opsd_openthoughts_math",
            "prompt": [{"role": "user", "content": student_user_message(problem)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": i,
                "split": "train",
                "teacher_prompt": t_prompt,          # fully templated, TM-on
                "teacher_prompt_tokens": n_tok,
            },
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    lens = [r["extra_info"]["teacher_prompt_tokens"] for r in rows]
    print(f"wrote {len(df)} rows -> {args.out}  (dropped {dropped} over {args.max_teacher_tokens} tok)")
    print(f"teacher prompt tokens: mean {sum(lens)/len(lens):.0f}  max {max(lens)}")
    print("\n--- student prompt (raw user message) ---\n" + rows[0]["prompt"][0]["content"][:400])
    print("\n--- teacher prompt (templated) ---\n" + rows[0]["extra_info"]["teacher_prompt"][:700])


if __name__ == "__main__":
    main()
