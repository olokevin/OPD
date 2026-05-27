"""GPU smoke: calib_mode=v2 with c4 source and num_seqs=4 installs BTT topology
and modifies at least one core from its random init.

These tests require a GPU and network access to download the C4 calibration
dataset on first run. Run with ``pytest -m gpu``.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_blocktt_v2_calibration_smoke(tiny_model, tiny_tokenizer):
    from omegaconf import OmegaConf
    from verl.workers.config.peft import PEFTConfig
    from verl.workers.peft import PEFTAdapter
    from verl.workers.peft.calib_loader import build_calib_loader_for_peft
    from compress.btt.btt_linear import BTTLinear

    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": 0.5,
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": False}},
        "calib": {"mode": "v2", "source": "c4", "num_seqs": 4,
                  "max_length": 64, "batch_size": 2, "seed": 0},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    loader = build_calib_loader_for_peft(cfg, tokenizer=tiny_tokenizer)
    assert loader is not None
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: loader)
    n_btt = sum(1 for m in out.modules() if isinstance(m, BTTLinear))
    assert n_btt > 0
