#!/usr/bin/env python3
"""Rebuild MATH train/eval parquets with SimpleRL-Zoo's ``qwen25-math-cot`` prompt
structure, for GRPO-via-``scripts/grpo/full.sh`` ONLY.

The shared OPD parquets (``datasets/train_data/math-lv3to5/train.parquet`` and
``datasets/test_data/MATH-500/test.parquet``) bake the instruction into the user
turn:

    user: "{question} Please reason step by step, and put your final answer within \\boxed{}."

so verl's chat template renders a "You are a helpful assistant." system prompt.
SimpleRL-Zoo instead evaluates/trains with the ``qwen25-math-cot`` template:

    system: "Please reason step by step, and put your final answer within \\boxed{}."
    user:   "{question}"                      # raw question, no baked instruction

With the prompt split into (system=instruction, user=raw_question), the stock
Qwen2.5 chat template reproduces ``qwen25-math-cot`` byte-for-byte (verified), so
no custom chat template is needed — only a data restructuring.

This writes NEW copies under datasets/{train_data,test_data}/*_simplerl/ and does
NOT modify the shared parquets, so OPD runs are unaffected. ``full.sh`` points at
these copies via TRAIN_DATASET / TEST_FILE.
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# The exact suffix the shared parquets append to every question (verified uniform).
SUFFIX = " " + INSTRUCTION

# (source parquet, destination parquet) pairs. Destinations live in *_simplerl dirs.
JOBS = [
    (
        REPO_ROOT / "datasets/train_data/math-lv3to5/train.parquet",
        REPO_ROOT / "datasets/train_data/math-lv3to5_simplerl/train.parquet",
    ),
    (
        REPO_ROOT / "datasets/test_data/MATH-500/test.parquet",
        REPO_ROOT / "datasets/test_data/MATH-500_simplerl/test.parquet",
    ),
]


def restructure_prompt(prompt):
    """[{user: '{q} <suffix>'}] -> [{system: instruction}, {user: '{q}'}]."""
    msgs = list(prompt)
    if len(msgs) != 1 or msgs[0].get("role") != "user":
        raise ValueError(f"unexpected prompt shape: {msgs!r}")
    content = msgs[0]["content"]
    if not content.endswith(SUFFIX):
        raise ValueError(f"prompt does not end with the expected boxed suffix: {content!r}")
    raw_question = content[: -len(SUFFIX)]
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": raw_question},
    ]


def convert(src: Path, dst: Path) -> None:
    df = pd.read_parquet(src)
    df = df.copy()
    df["prompt"] = df["prompt"].apply(restructure_prompt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    # Spot-check row 0.
    p0 = df.iloc[0]["prompt"]
    print(f"[ok] {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    print(f"      system: {p0[0]['content']!r}")
    print(f"      user:   {p0[1]['content'][:80]!r}...")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    for src, dst in JOBS:
        if not src.exists():
            raise FileNotFoundError(src)
        convert(src, dst)


if __name__ == "__main__":
    main()
