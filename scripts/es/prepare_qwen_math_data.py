"""Build MATH train / MATH-500 eval parquet files that use the *Qwen-Math* template.

Paper reference (ES at Scale, Appendix A.6, Table 7):

    <|im_start|>system\nPlease reason step by step, and put your final
    answer within \boxed{}.<|im_end|>\n<|im_start|>user\n{question}
    <|im_end|>\n<|im_start|>assistant\n

The verl parquet files already shipped in this repo append the "Please reason
step by step ..." instruction to the *user* turn.  The ES trainer feeds the
`prompt` message list straight through `tokenizer.apply_chat_template`, so to
reproduce Table 7 exactly we re-emit the rows with the instruction moved into a
`system` message and the bare question left in the `user` message.

Outputs (parquet, verl schema: prompt / reward_model / extra_info):
    datasets/es_math/math_lv3to5_qwenmath_train.parquet   (MATH lvl 3-5, 8890 rows)
    datasets/es_math/math500_qwenmath_test.parquet        (MATH-500, 500 rows)
"""

import argparse
import os

import pandas as pd

SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_MSG = "Please reason step by step, and put your final answer within \\boxed{}."


def strip_suffix(text: str) -> str:
    return text[: -len(SUFFIX)] if text.endswith(SUFFIX) else text


def convert(src: str, dst: str, shuffle_seed: int | None = None) -> None:
    df = pd.read_parquet(src)
    if shuffle_seed is not None:
        # The ES trainer takes the FIRST `train_max_samples` rows as its fixed
        # training batch; the source parquet is ordered by subject/level, so
        # shuffle once (deterministically) to make any prefix representative.
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        question = strip_suffix(r["prompt"][0]["content"])
        extra = dict(r["extra_info"]) if r.get("extra_info") is not None else {}
        if "level" in df.columns:
            extra["level"] = int(r["level"])
        rows.append(
            {
                "data_source": r.get("data_source", "MATH"),
                "prompt": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(r["reward_model"]["ground_truth"]),
                },
                "extra_info": extra,
            }
        )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst)
    print(f"{src} -> {dst}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.expanduser("~/Project/compression/OPD"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = args.repo_root
    out = args.out_dir or os.path.join(root, "datasets", "es_math")

    convert(
        os.path.join(root, "datasets/train_data/math-lv3to5/train.parquet"),
        os.path.join(out, "math_lv3to5_qwenmath_train.parquet"),
        shuffle_seed=0,
    )
    convert(
        os.path.join(root, "datasets/test_data/MATH-500/test.parquet"),
        os.path.join(out, "math500_qwenmath_test.parquet"),
    )


if __name__ == "__main__":
    main()
