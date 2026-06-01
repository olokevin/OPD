#!/usr/bin/env python3
"""Build OpenThoughts3-1.2M-math.parquet from the full open-thoughts/OpenThoughts3-1.2M corpus.

Filters the 120-shard upstream dataset to ``domain == "math"`` (~850k rows) and writes a
single parquet in the verl *prompt* format that ``scripts/infer/vllm_rollout.py`` consumes
(it reads ``df["prompt"]`` and feeds each entry to ``tokenizer.apply_chat_template``).

Output schema (matches datasets/OpenThoughts3_opd.parquet, the released complement slice):
    prompt        list[{"role": "user", "content": str}]   <- the only column the rollout needs
    data_source   str
    ability       "math"
    reward_model  {"ground_truth": str, "style": "rule"}    <- best-effort \\boxed{} from the gpt turn
    extra_info    {"index": int, "split": "train", "source": str, "difficulty": float|None}

The human question is suffixed with the canonical
"Please reason step by step, and put your final answer within \\boxed{}." instruction so the
prompt is identical in spirit to the released complement slice.
"""
import argparse
import glob
import math
import os

import pandas as pd

BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(s: str):
    """Return the content of the last top-level \\boxed{...} in s, or None."""
    if not isinstance(s, str):
        return None
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return None


def human_and_gpt(conv):
    """Pull the (human_question, gpt_answer) pair out of a conversations list."""
    human, gpt = None, None
    for turn in conv:
        role = turn.get("from")
        if role in ("human", "user") and human is None:
            human = turn.get("value")
        elif role in ("gpt", "assistant") and gpt is None:
            gpt = turn.get("value")
    return human, gpt


def build_prompt(question: str) -> list:
    q = question.rstrip()
    if not q.endswith(BOXED_SUFFIX):
        q = q + " " + BOXED_SUFFIX
    return [{"role": "user", "content": q}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards-glob",
        default="/data/yequan/datasets/OpenThoughts3-1.2M/data/train-*.parquet",
        help="Glob for the downloaded upstream parquet shards.",
    )
    ap.add_argument(
        "--out",
        default="/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet",
        help="Output parquet path.",
    )
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    if not shards:
        raise SystemExit(f"No shards matched {args.shards_glob}")
    print(f"Found {len(shards)} shards.")

    rows = []
    n_math = 0
    n_boxed = 0
    for si, path in enumerate(shards):
        df = pd.read_parquet(path)
        mdf = df[df["domain"] == "math"]
        if len(mdf) == 0:
            print(f"[{si+1}/{len(shards)}] {os.path.basename(path)}: 0 math (skip)")
            continue
        for _, r in mdf.iterrows():
            human, gpt = human_and_gpt(r["conversations"])
            if not human:
                continue
            gt = last_boxed(gpt) if gpt else None
            if gt is not None:
                n_boxed += 1
            diff = r.get("difficulty")
            if isinstance(diff, float) and math.isnan(diff):
                diff = None
            rows.append(
                {
                    "prompt": build_prompt(human),
                    "data_source": "open-thoughts/OpenThoughts3-1.2M",
                    "ability": "math",
                    "reward_model": {
                        "ground_truth": gt if gt is not None else "",
                        "style": "rule",
                    },
                    "extra_info": {
                        "index": len(rows),
                        "split": "train",
                        "source": r.get("source"),
                        "difficulty": diff,
                    },
                }
            )
        n_math += len(mdf)
        print(
            f"[{si+1}/{len(shards)}] {os.path.basename(path)}: +{len(mdf)} math "
            f"(running total {len(rows)})"
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print("\n" + "=" * 50)
    print(f"Math rows seen:      {n_math}")
    print(f"Rows written:        {len(out_df)}")
    print(f"With boxed answer:   {n_boxed} ({100*n_boxed/max(len(out_df),1):.1f}%)")
    print(f"Wrote:               {args.out}")
    print(f"Size:                {os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
