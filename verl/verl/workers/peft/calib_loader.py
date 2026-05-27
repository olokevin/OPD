"""Glue between verl's PEFTConfig and compress.integration.build_calib_loader.

compress.integration.build_calib_loader expects argparse-style args (calib_mode,
calib_source, ...); we adapt the PEFTConfig dataclass to that shape.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional


def build_calib_loader_for_peft(peft_cfg, *, tokenizer, training_dataset=None,
                                training_collate_fn=None, rl_rollout_fn=None):
    """Returns a DataLoader or None when peft.calib.mode == 'none'."""
    if peft_cfg.calib.mode == "none":
        return None
    from compress.integration import build_calib_loader
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
