"""Memory test: does the COMBINED (fwd+bwd) calibration fit at bs=1 + cap 8192?

Runs the exact production combined-calibration path on Qwen3-4B — full sequences,
DROP (not truncate) traces > 8192 tokens, bs=1, 128 seqs, sequence-reweighted —
and reports peak GPU allocation. The peak is governed by the longest kept sequence
(bs=1) + the fixed-size covariance accumulators, so it is ~independent of num_seqs.

Run in the sft env on one 80GB GPU (PYTHONPATH must include src/ for `compress`).
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from torch.utils.data import DataLoader  # noqa: E402
from compress.loaders import (  # noqa: E402
    build_fullseq_calib_loader, _VarLenSeqDataset, _pad_collate,
)
from compress.calibration import (  # noqa: E402
    collect_both_covariances_from_loader,
    collect_nystrom_combined_statistics,
)

JSONL = "/pscratch/sd/y/yequan/opd/datasets/OpenThought3-Qwen3-4B/data/train.jsonl"
MODEL = "Qwen/Qwen3-4B"
N = int(os.environ.get("N", "128"))
CAP = int(os.environ.get("CAP", "8192"))
# SYNTH_LEN>0 bypasses real data and feeds N synthetic sequences of exactly that many
# tokens (worst-case memory probe at the cap ceiling, regardless of real trace lengths).
SYNTH_LEN = int(os.environ.get("SYNTH_LEN", "0"))
# LONGEST_FIRST=1 sorts the kept (<=CAP) traces by length descending before taking N,
# so even a small N stresses the memory ceiling (the longest real backward).
LONGEST_FIRST = os.environ.get("LONGEST_FIRST", "0") == "1"

tok = AutoTokenizer.from_pretrained(MODEL)


def render(path, n):
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line).get("messages")
            except Exception:
                continue
            if not msgs:
                continue
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
            out.append(t)
    return out


if SYNTH_LEN > 0:
    import torch as _t
    seqs = [_t.randint(1, tok.vocab_size, (SYNTH_LEN,), dtype=_t.long) for _ in range(N)]
    loader = DataLoader(_VarLenSeqDataset(seqs), batch_size=1, shuffle=False,
                        collate_fn=lambda b: _pad_collate(b, pad_id=0))
    print(f"[data] SYNTHETIC: {N} sequences of exactly {SYNTH_LEN} tokens (cap ceiling probe)",
          flush=True)
else:
    texts = render(JSONL, N * 20)
    # Tokenize once; DROP (do not truncate) traces > CAP; report the distribution.
    pairs = [(len(tok(t, add_special_tokens=False)["input_ids"]), t) for t in texts]
    kept_pairs = [(n, t) for (n, t) in pairs if n <= CAP]
    lens = sorted(n for (n, _t) in pairs)
    print(f"[data] candidates={len(pairs)} kept(<= {CAP})={len(kept_pairs)} "
          f"dropped(> {CAP})={len(pairs) - len(kept_pairs)} "
          f"| longest kept={max((n for n, _ in kept_pairs), default=0)} "
          f"| overall longest={lens[-1]} | longest_first={LONGEST_FIRST} N={N}", flush=True)
    if LONGEST_FIRST:
        kept_pairs.sort(key=lambda p: p[0], reverse=True)
    sel_texts = [t for (_n, t) in kept_pairs[:N]]
    loader = build_fullseq_calib_loader(tok, sel_texts, num_seqs=N, length_filter="full",
                                        max_seq_len=CAP, batch_size=1)

print(f"[load] {MODEL} bf16 sdpa on cuda ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
model.eval()

import gc  # noqa: E402
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
print("[calib] collect_both_covariances_from_loader (fwd+bwd) ...", flush=True)
fwd, bwd = collect_both_covariances_from_loader(
    model, loader, device="cuda", skip_layers=("lm_head",), reweight="sequence")
n_attn = len(fwd)
print(f"[calib] both-cov done @ {time.time() - t0:.0f}s | "
      f"peak so far {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
# Mirror compress_setup.py: free the full cov dicts before the Nystrom-stats pass.
del fwd, bwd
gc.collect()
torch.cuda.empty_cache()
print("[calib] collect_nystrom_combined_statistics ...", flush=True)
mlp = collect_nystrom_combined_statistics(
    model, loader, device="cuda", skip_layers=("lm_head",), reweight="sequence")
torch.cuda.synchronize()

peak_a = torch.cuda.max_memory_allocated() / 1e9
peak_r = torch.cuda.max_memory_reserved() / 1e9
print(f"[RESULT] COMBINED CALIB OK in {time.time() - t0:.0f}s | "
      f"peak alloc={peak_a:.1f} GB  peak reserved={peak_r:.1f} GB  / 80 GB | "
      f"attn_cov={n_attn} mlp_stats={len(mlp)}", flush=True)
