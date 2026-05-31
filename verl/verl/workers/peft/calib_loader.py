"""Glue between verl's PEFTConfig and compress.integration.build_calib_loader.

compress.integration.build_calib_loader expects argparse-style args (calib_mode,
calib_source, ...); we adapt the PEFTConfig dataclass to that shape.

For ``calib.source=training_data`` the fsdp_worker has no easy handle on the
top-level RL train dataset (workers see only ``actor_rollout_ref``), so we
synthesise the calibration dataset from the parquet path exported by the OPD
launch scripts (env var ``TRAIN_DATASET``, optionally a comma-separated list).
The prompts are run through the same chat template the RL dataloader uses, so
the calibration text matches what the trainer will actually condition on.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_training_parquet_paths() -> List[str]:
    """Pick up the launcher's TRAIN_DATASET env var (single path or comma list)."""
    raw = os.environ.get("TRAIN_DATASET") or os.environ.get("TRAIN_FILES")
    if not raw:
        return []
    parts = [p.strip().strip('"').strip("'") for p in raw.split(",")]
    return [p for p in parts if p]


def _load_training_prompt_texts(
    parquet_paths: Sequence[str],
    tokenizer,
    *,
    num_seqs: int,
    prompt_key: str = "prompt",
    apply_chat_template_kwargs: Optional[dict] = None,
) -> List[str]:
    import pandas as pd

    apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
    texts: List[str] = []
    for path in parquet_paths:
        if len(texts) >= num_seqs:
            break
        df = pd.read_parquet(path)
        if prompt_key not in df.columns:
            continue
        for messages in df[prompt_key].tolist():
            if len(texts) >= num_seqs:
                break
            try:
                rendered = tokenizer.apply_chat_template(
                    list(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_chat_template_kwargs,
                )
            except Exception:
                # Fall back to raw concatenation if chat template fails.
                rendered = "\n".join(
                    m.get("content", "") for m in messages if isinstance(m, dict)
                )
            if rendered:
                texts.append(rendered)
    return texts


class _PerPromptDataset(Dataset):
    def __init__(self, examples: List[dict]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        return self._examples[idx]


def _build_per_prompt_calib_loader(
    tokenizer, texts: Sequence[str], *, max_length: int, batch_size: int, pad_id: int,
) -> DataLoader:
    """Per-prompt calibration loader (no concat / no full-window requirement).

    Each text becomes one calibration example, truncated to ``max_length``.
    Batches are padded right with ``pad_id`` and accompanied by an
    ``attention_mask`` so the compress backward-covariance path can ignore
    padding positions when ``calib_loss='opd'``.
    """
    examples: List[dict] = []
    for text in texts:
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=int(max_length),
        )["input_ids"].squeeze(0).to(dtype=torch.long)
        if ids.numel() == 0:
            continue
        examples.append({"input_ids": ids, "labels": ids.clone()})
    if not examples:
        raise ValueError(
            "per-prompt calibration loader: tokenization produced no usable "
            "examples (every text tokenised to zero tokens?)."
        )

    def _collate(batch):
        max_len = max(e["input_ids"].shape[0] for e in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), fill_value=-100, dtype=torch.long)
        for i, e in enumerate(batch):
            L = e["input_ids"].shape[0]
            input_ids[i, :L] = e["input_ids"]
            attn[i, :L] = 1
            labels[i, :L] = e["labels"]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return DataLoader(
        _PerPromptDataset(examples),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate,
        drop_last=False,
    )


def build_calib_loader_for_peft(peft_cfg, *, tokenizer, training_dataset=None,
                                training_collate_fn=None, rl_rollout_fn=None,
                                apply_chat_template_kwargs=None):
    """Returns a DataLoader or None when peft.calib.mode == 'none'."""
    if peft_cfg.calib.mode == "none":
        return None
    from compress.integration import build_calib_loader

    # calib.source=training_data + no in-process RL dataset: synthesise from
    # the parquet path the launch script already exports via TRAIN_DATASET.
    if (
        peft_cfg.calib.source == "training_data"
        and training_dataset is None
        and rl_rollout_fn is None
    ):
        parquet_paths = _resolve_training_parquet_paths()
        if not parquet_paths:
            raise RuntimeError(
                "peft.calib.source='training_data' but the fsdp_worker has no "
                "training_dataset and TRAIN_DATASET / TRAIN_FILES env var is "
                "unset. Either export TRAIN_DATASET=<parquet path> (the OPD "
                "launch scripts already do this) or switch peft.calib.source "
                "to 'c4' / 'traces'."
            )
        # Pick up enable_thinking from env so the calibration prompts match
        # what the trainer will see (the top-level data.apply_chat_template_
        # kwargs config is not visible inside the worker).
        if apply_chat_template_kwargs is None:
            apply_chat_template_kwargs = {}
            et_env = os.environ.get("ENABLE_THINKING")
            if et_env is not None:
                apply_chat_template_kwargs["enable_thinking"] = (
                    et_env.strip().lower() not in {"0", "false", "no"}
                )
        texts = _load_training_prompt_texts(
            parquet_paths,
            tokenizer,
            num_seqs=int(peft_cfg.calib.num_seqs),
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        if not texts:
            raise RuntimeError(
                "training_data calibration: failed to extract any prompts from "
                f"{parquet_paths!r} (expected a 'prompt' column)."
            )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
        return _build_per_prompt_calib_loader(
            tokenizer,
            texts,
            max_length=int(peft_cfg.calib.max_length),
            batch_size=int(peft_cfg.calib.batch_size),
            pad_id=int(pad_id),
        )

    ns = SimpleNamespace(
        calib_mode=peft_cfg.calib.mode,
        calib_source=peft_cfg.calib.source,
        calib_traces_path=peft_cfg.calib.traces_path,
        calib_num_seqs=peft_cfg.calib.num_seqs,
        calib_max_length=peft_cfg.calib.max_length,
        calib_seed=peft_cfg.calib.seed,
        calib_batch_size=peft_cfg.calib.batch_size,
    )
    return build_calib_loader(
        ns,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        training_collate_fn=training_collate_fn,
        rl_rollout_fn=rl_rollout_fn,
        hyphen_style=True,
    )
