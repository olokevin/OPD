"""eval_decomp.py -- decompose the capped-accuracy evals of the ds15b arms into
(mean length, fraction hitting the cap, accuracy among uncapped answers), per step.

  python scripts/zo_opd/ds15b/eval_decomp.py [--cap 7168]
"""
import argparse, glob, json, os
from transformers import AutoTokenizer

ARMS = {
    "BP":   "ds15b_bp_opd_k0_r7168_lr1e-6",
    "ES-A": "ds15b_es_opd_k0_N32_sig1e-3_a5e-4",
    "ES-B": "ds15b_es_opd_k0_N128_sig1e-3_a1e-3",
    "ES-C": "ds15b_es_opd_k0_N32_sig1e-3_a1.25e-3",
    "ES-D": "ds15b_es_opd_k0_N8_sig1e-3_a1.25e-3",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=7168); a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"{'arm':5s} {'step':>4s} {'n':>5s} {'len':>6s} {'capped':>7s} {'acc':>6s} {'acc|unc':>8s}")
    for arm, exp in ARMS.items():
        for f in sorted(glob.glob(f"validation_log/{exp}/*.jsonl"), key=lambda x: int(os.path.basename(x)[:-6])):
            step = int(os.path.basename(f)[:-6])
            n = ln = cap = acc = acc_u = n_u = 0
            for line in open(f):
                r = json.loads(line)
                txt = r.get("output", "")
                L = len(tok(txt, add_special_tokens=False).input_ids)
                sc = float(r.get("score", 0))
                n += 1; ln += L; acc += sc
                if L >= a.cap - 8: cap += 1
                else: n_u += 1; acc_u += sc
            print(f"{arm:5s} {step:4d} {n:5d} {ln/n:6.0f} {cap/n:7.2f} {acc/n:6.3f} {acc_u/max(n_u,1):8.3f}")


if __name__ == "__main__":
    main()
