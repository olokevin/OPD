"""Per-adapter apply / export / save tests."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def test_null_adapter_apply_is_identity(tiny_model, tiny_tokenizer):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({"mode": "none"}))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    assert adapter.mode == "none"
    assert adapter.needs_calibration() is False
    new_model = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert new_model is tiny_model
    assert adapter.export_for_vllm(new_model) is None
    assert adapter.vllm_engine_kwargs() == {}
    assert adapter.peft_config() is None
    assert adapter.topology_meta() == {"mode": "none", "target_modules": "all"}
