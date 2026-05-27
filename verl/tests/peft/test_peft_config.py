"""Unit tests for PEFTConfig parsing and validation."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig


def test_default_is_none_mode():
    cfg = PEFTConfig()
    assert cfg.mode == "none"


def test_lora_mode_from_omegaconf():
    raw = OmegaConf.create(
        {"mode": "lora", "target_modules": "all",
         "lora": {"rank": 16, "alpha": 32, "dropout": 0.05}}
    )
    cfg = PEFTConfig.from_omegaconf(raw)
    assert cfg.mode == "lora"
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 32
    assert cfg.lora.dropout == 0.05


def test_blocktt_mode_with_calib():
    raw = OmegaConf.create(
        {"mode": "blocktt",
         "blocktt": {"decomp_mode": "input_one_block", "train_position": "small",
                     "rank": "full", "qfura": {"enabled": False}},
         "calib": {"mode": "v2", "source": "c4", "num_seqs": 64}}
    )
    cfg = PEFTConfig.from_omegaconf(raw)
    assert cfg.mode == "blocktt"
    assert cfg.blocktt.train_position == "small"
    assert cfg.calib.mode == "v2"
    assert cfg.calib.num_seqs == 64


def test_invalid_mode_rejected():
    raw = OmegaConf.create({"mode": "bogus"})
    with pytest.raises(ValueError, match="mode must be one of"):
        PEFTConfig.from_omegaconf(raw)


def test_qlora_requires_lora_rank():
    raw = OmegaConf.create({"mode": "qlora", "lora": {"rank": 0}})
    with pytest.raises(ValueError, match="qlora requires peft.lora.rank > 0"):
        PEFTConfig.from_omegaconf(raw)


def test_blocktt_calib_with_int_rank_rejected():
    raw = OmegaConf.create(
        {"mode": "blocktt",
         "blocktt": {"rank": 4},
         "calib": {"mode": "v2"}}
    )
    with pytest.raises(ValueError, match="integer .*rank is only valid"):
        PEFTConfig.from_omegaconf(raw)


def test_legacy_lora_rank_shim():
    """Old-style actor_rollout_ref.model.lora_rank populates peft.lora.*."""
    model_cfg = OmegaConf.create(
        {"lora_rank": 16, "lora_alpha": 32, "target_modules": "all-linear"}
    )
    peft_cfg = OmegaConf.create({"mode": "none"})
    merged = PEFTConfig.legacy_shim(peft_cfg=peft_cfg, model_cfg=model_cfg)
    assert merged.mode == "lora"
    assert merged.lora.rank == 16
    assert merged.lora.alpha == 32
    assert merged.target_modules == "all-linear"
