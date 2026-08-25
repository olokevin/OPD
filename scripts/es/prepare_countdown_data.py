"""Countdown-3to4 train/val parquets for the ES trainer (`data.task_type=countdown`).

The paper this section tests (arXiv:2601.20861) measures forgetting on
Countdown -> HellaSwag, so reproducing its setting needs the same new task.  The ES
countdown prompt processor reads the simple `{"nums": [...], "target": ...}` format
directly, so all this does is slice and split the HF dataset deterministically.

    python scripts/es/prepare_countdown_data.py --n-train 200 --n-val 500
"""

import argparse
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = ("/data/yequan/huggingface/hub/datasets--Jiayi-Pan--Countdown-Tasks-3to4/"
       "snapshots/408f70d177020686d34a56bba5952feb45aaaee4/data/train-00000-of-00001.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--n-train", type=int, default=200)   # paper: 200 examples
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "datasets/es_countdown"))
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train = df.iloc[: args.n_train].copy()
    val = df.iloc[args.n_train : args.n_train + args.n_val].copy()
    for d in (train, val):
        d["nums"] = d["nums"].apply(lambda a: [int(x) for x in a])
        d["target"] = d["target"].astype(int)

    os.makedirs(args.out_dir, exist_ok=True)
    tp = os.path.join(args.out_dir, f"countdown_train{args.n_train}.parquet")
    vp = os.path.join(args.out_dir, f"countdown_val{args.n_val}.parquet")
    train.to_parquet(tp, index=False)
    val.to_parquet(vp, index=False)
    print(f"train {len(train)} -> {tp}\nval   {len(val)} -> {vp}")
    print(train.head(2).to_dict("records"))


if __name__ == "__main__":
    main()
