"""Build datasets/DeepMath-103K.parquet (verl format) from the HF dataset zwhe99/DeepMath-103K.

  HF_HOME=/data/yequan/huggingface python3 scripts/iso/prep_deepmath.py
"""
import glob

import pandas as pd
from huggingface_hub import snapshot_download

p = snapshot_download("zwhe99/DeepMath-103K", repo_type="dataset", allow_patterns=["data/*"])
d = pd.concat([pd.read_parquet(f, columns=["question", "final_answer"])
               for f in sorted(glob.glob(f"{p}/data/train-*.parquet"))], ignore_index=True)
sfx = " Please reason step by step, and put your final answer within \\boxed{}."
pd.DataFrame({
    "data_source": "DeepMath-103K",
    "prompt": [[{"content": q + sfx, "role": "user"}] for q in d.question],
    "ability": "math",
    "reward_model": [{"ground_truth": str(a), "style": "rule"} for a in d.final_answer],
    "extra_info": [{"index": i, "question": q} for i, q in enumerate(d.question)],
}).to_parquet("datasets/DeepMath-103K.parquet")
print(len(d), "rows -> datasets/DeepMath-103K.parquet")
