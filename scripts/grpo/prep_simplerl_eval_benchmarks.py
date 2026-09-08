#!/usr/bin/env python3
"""Restructure the table-1 eval benchmarks to SimpleRL-Zoo ``qwen25-math-cot`` prompt
structure (instruction in SYSTEM turn, raw question in user), for offline eval of the
SimpleRL-aligned GRPO checkpoints — the same prompt the models trained on.

Mirrors scripts/grpo/prep_simplerl_prompts.py but for the eval suite. Reads the shared
verl-format parquets under datasets/test_data/<bench>/test.parquet (single [user] prompt
with the boxed instruction baked into the user turn) and writes NEW copies under
datasets/test_data/<bench>_simplerl/test.parquet. Does NOT modify the shared parquets.

Benchmarks = SimpleRL-Zoo Table 1 set present in this repo (GSM8K absent):
  MATH-500, AIME24, AMC23, Minerva, Olympiad-Bench.
(MATH-500_simplerl may already exist from prep_simplerl_prompts.py; it is rewritten
 identically here so the whole suite is produced by one command.)
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SUFFIX = " " + INSTRUCTION
BENCHES = ["MATH-500", "AIME24", "AMC23", "Minerva", "Olympiad-Bench"]


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


def main():
    for b in BENCHES:
        src = REPO_ROOT / f"datasets/test_data/{b}/test.parquet"
        dst = REPO_ROOT / f"datasets/test_data/{b}_simplerl/test.parquet"
        if not src.exists():
            print(f"[skip] {b}: source missing ({src})")
            continue
        df = pd.read_parquet(src).copy()
        df["prompt"] = df["prompt"].apply(restructure_prompt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst)
        p0 = df.iloc[0]["prompt"]
        print(f"[ok] {b}: {len(df)} rows -> {dst.relative_to(REPO_ROOT)}")
        print(f"     system: {p0[0]['content']!r}")
        print(f"     user:   {p0[1]['content'][:70]!r}...")


if __name__ == "__main__":
    main()
