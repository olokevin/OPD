"""Shared helpers for the A/B/D mechanism-fix method drivers.

Single source of truth so `bi_whitened_svd.py` (D), `lr_sparse_residual.py` (A)
and `sequential_src.py` (B) share the same operating point, eval contract, and
last-decoder-layer-skip logic spelled out in
`docs/aris/reason_aware_compress/EXPERIMENT_PLAN.md` § "Operating point & protocol".

Operating point (whole A/B/D plan):
  - retain 0.8 first, then sweep down (0.7/0.6/0.5/0.36);
  - the last decoder layer's linears are NEVER compressed (left dense);
  - every cell reports BOTH MATH-500 (greedy, ttrl_math grader) and C4 PPL.

Last-layer skip: `skip_layers` in the compress core is leaf-name-only
(`name.split('.')[-1]`), so passing "layers.35" does nothing. The working,
repo-established pattern (see tail_rescue.py) is to DROP the last layer's keys
from the collected statistics dict before compression — a layer with no
covariance/statistics entry is silently left dense by every compress path
(svd_llm_v2_compress_model: svd_llm_v2.py:256-258; nystrom_compress_model:
nystrom.py:394-396). `protected_layer_set` + `drop_protected_stats` below.

Reuses (does NOT duplicate) the standing eval + calibration helpers from
`layer_sensitivity.py`: build_openthought3_loader, attn_linear_names,
mlp_down_names, eval_math500.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo plumbing (mirrors every sibling script: src for compress, verl for grader)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "verl"))
# So `from layer_sensitivity import ...` resolves regardless of CWD.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------- #
# model loading / param accounting
# --------------------------------------------------------------------------- #
def load_model(model_name: str, dtype: torch.dtype, device: str):
    # sdpa attention is O(seq) memory; the transformers default can be eager
    # (O(seq^2)), which OOMs the combined fwd+bwd calibration on long traces.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="sdpa")
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device)


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def count_params(model) -> tuple[int, int]:
    """(total_params, nonzero_params). Nonzero matters for the sparse-residual
    (Block A) cells where the +Patch contributes structurally-dense-but-mostly-
    zero weights."""
    total = nonzero = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
    return total, nonzero


# --------------------------------------------------------------------------- #
# last-decoder-layer skip (the operating-point protocol)
# --------------------------------------------------------------------------- #
def num_hidden_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def layer_index_of(name: str) -> Optional[int]:
    """Decoder-layer index parsed from a parameter/module name, or None."""
    try:
        return int(name.split(".layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def protected_layer_set(model, skip_last_layers: int) -> Set[int]:
    """The top-N decoder layer indices to leave DENSE (never compressed)."""
    n = num_hidden_layers(model)
    if skip_last_layers <= 0:
        return set()
    return set(range(n - skip_last_layers, n))


def drop_protected_stats(stats: Dict[str, torch.Tensor], protect: Set[int]
                         ) -> Dict[str, torch.Tensor]:
    """Remove statistics whose decoder-layer index is protected, so those layers
    are left dense by the compress core. Returns a new dict (does not mutate)."""
    if not protect:
        return dict(stats)
    return {n: s for n, s in stats.items() if layer_index_of(n) not in protect}


def attn_names_excluding_protected(model, protect: Set[int]) -> list[str]:
    """All self_attn projection linear names except those in protected layers."""
    from layer_sensitivity import attn_linear_names
    by_layer = attn_linear_names(model)
    return [n for li, names in by_layer.items() if li not in protect for n in names]


# --------------------------------------------------------------------------- #
# evaluation (MATH-500 + C4 PPL) — the standing eval contract for every cell
# --------------------------------------------------------------------------- #
def eval_c4_ppl(model, tokenizer, *, seqlen: int = 2048, seed: int = 0,
                device: str = "cuda") -> float:
    return evaluate_model_ppl(
        model, tokenizer, seqlen=seqlen, seed=seed,
        datasets=("c4",), device=device,
    )["c4"]


def eval_math(model, tokenizer, *, device: str, limit: int,
              max_new_tokens: int = 2048, batch_size: int = 16) -> float:
    """MATH-500 greedy accuracy via the standing eval_math500 (ttrl_math grader,
    dataset ground truth from datasets/test_data/MATH-500/test.parquet)."""
    from layer_sensitivity import eval_math500
    return eval_math500(
        model, tokenizer, device=device, limit=limit,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )


def _first_correct_boxed_char(text: str, gold: str):
    """Char index just past the first \\boxed{...} whose prefix grades correct,
    or None. Reused for the 'relaxed' (contains-correct-answer) metric."""
    import re
    from verl.utils.reward_score.ttrl_math import compute_score
    i = 0
    while True:
        m = re.search(r"\\boxed\s*{", text[i:])
        if not m:
            return None
        j = i + m.end()
        depth = 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if compute_score(text[:j], str(gold)).get("acc", False):
            return j
        i = j


@torch.no_grad()
def eval_math_capture(model, tokenizer, *, device: str, limit: int,
                      max_new_tokens: int = 2048, batch_size: int = 16,
                      save_path: Optional[str] = None) -> dict:
    """MATH-500 eval that CAPTURES every response and reports four metrics in one
    generation pass:
      - strict_acc        : ttrl_math on the full response (final-answer graded)
      - relaxed_acc       : 'contains a correct answer' — ANY \\boxed{} grades
                            correct anywhere in the response (pardons looping-past-answer)
      - mean_gen_tokens   : mean #generated tokens (looping/blow-up signal)
      - mean_tok_to_correct : mean #generated tokens up to the first correct \\boxed{}
                            (over responses that reach it; isolates 'how far to the answer')
    Same prompt/decoding as eval_math500 (chat template, enable_thinking=False,
    greedy, dataset gold)."""
    import pandas as pd
    from verl.utils.reward_score.ttrl_math import compute_score

    df = pd.read_parquet(REPO_ROOT / "datasets" / "test_data" / "MATH-500" / "test.parquet")
    if limit > 0:
        df = df.iloc[:limit]
    prompts, gts = [], []
    for _, row in df.iterrows():
        messages = list(row["prompt"])
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        gts.append(str(row["reward_model"]["ground_truth"]))

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    records = []
    n = len(prompts)
    for i in range(0, n, batch_size):
        bp, bg = prompts[i:i + batch_size], gts[i:i + batch_size]
        enc = tokenizer(bp, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for k in range(gen.shape[0]):
            gen_ids = gen[k]
            # strip right-padding for an honest generated-token count
            keep = (gen_ids != tokenizer.pad_token_id).sum().item()
            n_gen = int(keep)
            txt = tokenizer.decode(gen_ids[:keep], skip_special_tokens=True)
            strict = bool(compute_score(txt, bg[k]).get("acc", False))
            fc_char = _first_correct_boxed_char(txt, bg[k])
            relaxed = fc_char is not None
            # tokens-to-first-correct: re-tokenize the correct prefix
            tok_to_correct = None
            if relaxed:
                tok_to_correct = len(tokenizer(txt[:fc_char], add_special_tokens=False)["input_ids"])
            records.append({"gold": bg[k], "n_gen_tokens": n_gen, "strict": strict,
                            "relaxed": relaxed, "tok_to_correct": tok_to_correct,
                            "response": txt})

    strict_acc = sum(r["strict"] for r in records) / len(records)
    relaxed_acc = sum(r["relaxed"] for r in records) / len(records)
    mean_gen = sum(r["n_gen_tokens"] for r in records) / len(records)
    reached = [r["tok_to_correct"] for r in records if r["tok_to_correct"] is not None]
    mean_ttc = (sum(reached) / len(reached)) if reached else None
    if save_path:
        save_json({"strict_acc": strict_acc, "relaxed_acc": relaxed_acc,
                   "mean_gen_tokens": mean_gen, "mean_tok_to_correct": mean_ttc,
                   "n": len(records), "n_reached": len(reached),
                   "records": records}, save_path)
    return {"strict_acc": strict_acc, "relaxed_acc": relaxed_acc,
            "mean_gen_tokens": mean_gen, "mean_tok_to_correct": mean_ttc,
            "n_reached": len(reached), "n": len(records)}


def eval_cell(model, tokenizer, *, device: str, math_limit: int,
              math_max_new_tokens: int = 2048, math_batch_size: int = 16,
              ppl_seqlen: int = 2048, skip_math: bool = False,
              skip_ppl: bool = False) -> dict:
    """Both standing metrics for one compressed model, plus param accounting.

    skip_ppl=True skips C4 PPL (the only metric needing internet/C4 — useful on
    offline compute nodes where only the local MATH-500 parquet is available)."""
    total, nonzero = count_params(model)
    c4 = None if skip_ppl else eval_c4_ppl(model, tokenizer, seqlen=ppl_seqlen, device=device)
    math_acc = None
    if not skip_math:
        math_acc = eval_math(
            model, tokenizer, device=device, limit=math_limit,
            max_new_tokens=math_max_new_tokens, batch_size=math_batch_size,
        )
    return {
        "params_total_B": total / 1e9,
        "params_nonzero_B": nonzero / 1e9,
        "c4_ppl": c4,
        "math500_acc": math_acc,
    }


# --------------------------------------------------------------------------- #
# calibration loader (OpenThought3 math traces — the held-fixed calibration)
# --------------------------------------------------------------------------- #
def build_calib_loader(calib: str, tokenizer, *, num_seqs: int, max_length: int,
                       batch_size: int, seed: int = 3, length: str = "full",
                       max_seq_len: int | None = None):
    """OpenThought3 (default, the held-fixed calibration) or C4.

    length:
      - "full"   : DEFAULT (2026-06-04) — full un-windowed sequences (one per
                   conversation), pad+mask collated, paired with the new
                   sequence-reweighted covariance collection. Beats the window
                   scheme on reasoning (FULLSEQ_CALIB_RESULTS.md). max_seq_len=None
                   (default) keeps every trace at full length; set an int to DROP
                   (never truncate) traces longer than it.
      - "lt2048" : full sequences but only conversations < 2048 tokens (best
                   strict-MATH variant in the tune).
      - "window2048" : legacy 2048-token windows (the pre-2026-06-04 behavior;
                   pass this + reweight="token" to reproduce old baselines).
    For OpenThought3 only; C4 stays windowed (it is the prior-art PPL harness).
    """
    if calib == "openthought3":
        if length in ("full", "lt2048"):
            from compress.loaders import build_fullseq_calib_loader
            from layer_sensitivity import _openthought3_texts
            path = REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
            texts = _openthought3_texts(tokenizer, path, n=num_seqs * 20)
            # bs=1 for "full": a full-length backward over a 4B model is the
            # memory ceiling for the combined/bwd path; batching would OOM. With
            # max_seq_len=None (never truncate) very long traces can OOM even at
            # bs=1 — lower num_seqs or pass a max_seq_len cap if so. lt2048 is
            # short enough to batch.
            bs = 4 if length == "lt2048" else 1
            return build_fullseq_calib_loader(
                tokenizer, texts, num_seqs=num_seqs, length_filter=length,
                max_seq_len=max_seq_len, batch_size=bs)
        # legacy windowed escape hatch
        from layer_sensitivity import build_openthought3_loader
        return build_openthought3_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size,
        )
    if calib == "c4":
        from compress.loaders import build_c4_calib_loader
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size, seed=seed,
        )
    raise ValueError(f"Unknown calib dataset {calib!r}")


# --------------------------------------------------------------------------- #
# results IO
# --------------------------------------------------------------------------- #
def save_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def add_cli_common(ap):
    """Operating-point CLI flags shared by every A/B/D driver."""
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="retain ratio (operating point: 0.8 first, then sweep down)")
    ap.add_argument("--skip-last-layers", type=int, default=1,
                    help="leave the top-N decoder layers DENSE (protocol: 1)")
    ap.add_argument("--calib", default="openthought3",
                    choices=["openthought3", "c4"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=100,
                    help="MATH-500 problems to grade (pilot=100, headline=200)")
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--out", required=True)
    return ap
