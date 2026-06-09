"""Standardized calibration for the moe_compress recovery atlas.

The headline comparison requires every compression method to see the SAME
calibration data and the SAME total token budget, so a family-level inversion
cannot be blamed on calibration differences (RESEARCH_REVIEW fix #2). We use a
fixed 256 x 2048 = 524,288-token window set drawn from OpenThoughts3 (the same
distribution as the recovery data) by default; C4 is available as an alternative
corpus for the appendix sensitivity table.

OLMoE routes top-8 of 64 experts, so a given expert sees only ~1/8 of tokens.
256 sequences x 2048 tokens ~= 524k tokens => ~65k tokens/expert/layer in
expectation, which is enough to estimate a 2048-dim input covariance per expert.
``calib_coverage`` reports the min tokens any expert was routed, so a caller can
assert every expert was hit before trusting per-expert statistics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from compress.loaders import build_c4_calib_loader, build_fullseq_calib_loader  # noqa: E402

# OpenThoughts3 traces in chat-format JSONL ({messages:[user,assistant]}).
# DEFAULT is the OLMoE-NATIVE traces (OpenThoughts3 prompts -> OLMoE-Instruct
# completions, regen 2026-06-08) for on-distribution calibration. The old
# Qwen3-4B-trace file is the fallback if native traces are absent. Override via
# the MOE_CALIB_JSONL env var.
_NATIVE = Path("/data/yequan/moe_compress/calib_src/ot3_olmoe_native.jsonl")
_QWEN = REPO / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
import os as _os  # noqa: E402
OPENTHOUGHTS_JSONL = Path(_os.environ.get("MOE_CALIB_JSONL")
                          or (_NATIVE if _NATIVE.exists() else _QWEN))

DEFAULT_NUM_SEQS = 256
DEFAULT_MAX_LEN = 2048


def _render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:  # noqa: BLE001 - base tokenizers may lack a template
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


# Concrete source files for the split-source nystrom_combined recipes.
# OT3_JSONL = the OpenThoughts3 "training response" source. DEFAULT is now the
# ORIGINAL OpenThoughts3 traces (QwQ-distilled, the dataset's own reasoning) if
# present, else the Qwen3-4B re-rollout. Override via MOE_OT3_JSONL.
_ORIG_OT3 = Path("/data/yequan/moe_compress/calib_src/ot3_original_math.jsonl")
NATIVE_JSONL = Path(_os.environ.get("MOE_NATIVE_JSONL") or _NATIVE)  # prompt + OLMoE self-gen
OT3_JSONL = Path(_os.environ.get("MOE_OT3_JSONL")
                 or (_ORIG_OT3 if _ORIG_OT3.exists() else _QWEN))   # prompt + OT3 training trace


def _texts_from(tokenizer, jsonl_path, n: int) -> list[str]:
    texts: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages")
            if msgs:
                texts.append(_render(tokenizer, msgs))
            if len(texts) >= n:
                break
    return texts


def _openthoughts_texts(tokenizer, n: int) -> list[str]:
    return _texts_from(tokenizer, OPENTHOUGHTS_JSONL, n)


def build_calib_loader_from(tokenizer, jsonl_path, *, num_seqs=DEFAULT_NUM_SEQS,
                            max_len=DEFAULT_MAX_LEN, batch_size=4):
    """Calib loader from a SPECIFIC jsonl (for split-source recipes)."""
    texts = _texts_from(tokenizer, jsonl_path, num_seqs * 4)
    logger.info(f"calib from {Path(jsonl_path).name}: {num_seqs}x{max_len} "
                f"(loaded {len(texts)} traces)")
    return build_fullseq_calib_loader(
        tokenizer, texts, num_seqs=num_seqs, length_filter="full",
        max_seq_len=max_len, batch_size=batch_size,
    )


def build_standard_calib_loader(
    tokenizer,
    *,
    corpus: str = "openthoughts",
    num_seqs: int = DEFAULT_NUM_SEQS,
    max_len: int = DEFAULT_MAX_LEN,
    batch_size: int = 4,
):
    """The ONE calibration loader used by every method in the main comparison.

    corpus: "openthoughts" (default, same distribution as recovery) | "c4"
            (appendix sensitivity). Token budget is fixed at num_seqs * max_len.
    """
    if corpus == "c4":
        logger.info(f"standard calib: C4 {num_seqs}x{max_len} (appendix corpus)")
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_len, batch_size=batch_size,
        )
    if corpus != "openthoughts":
        raise ValueError(f"corpus must be openthoughts|c4, got {corpus!r}")
    # Oversample texts (each trace is one variable-length sequence; truncate to
    # max_len so the per-method token budget is exactly num_seqs * max_len).
    texts = _openthoughts_texts(tokenizer, num_seqs * 4)
    logger.info(f"standard calib: OpenThoughts3 {num_seqs}x{max_len} "
                f"(loaded {len(texts)} traces)")
    return build_fullseq_calib_loader(
        tokenizer, texts, num_seqs=num_seqs, length_filter="full",
        max_seq_len=max_len, batch_size=batch_size,
    )


@torch.no_grad()
def calib_coverage(model, loader, device: str = "cuda") -> dict:
    """Count, per MoE expert, how many calibration tokens were routed to it.

    Returns {"min": int, "n_dead": int, "per_layer_min": [...], "total_tokens": int}.
    Use to assert every expert is routed >= 1 before trusting per-expert stats
    (the compress_sft OLMoE config used a large calib for exactly this reason).
    """
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    handles = []

    def _make_hook(name):
        def _hook(mod, inputs, output):
            counts[name] += int(inputs[0].shape[0])  # tokens this expert saw
        return _hook

    for name, mod in model.named_modules():
        # hook each expert's gate_proj (fires once per routed-token batch)
        if name.endswith(".gate_proj") and ".experts." in name:
            handles.append(mod.register_forward_hook(_make_hook(name)))

    model.eval()
    total = 0
    try:
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch.get("attention_mask")
            am = am.to(device) if am is not None else None
            total += int(am.sum()) if am is not None else int(ids.numel())
            model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    vals = list(counts.values())
    n_experts = model.config.num_hidden_layers * model.config.num_experts
    n_dead = n_experts - len(vals)
    return {
        "min": min(vals) if vals else 0,
        "n_dead": n_dead,
        "n_experts": n_experts,
        "total_tokens": total,
    }
