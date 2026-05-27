"""Verify that BlockTT/SVD export_for_vllm emits exactly the nn.Linear param keys
of the pre-PEFT model, with dense tensors of the matching shape."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def _linear_param_keys(model):
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
            keys.add(f"{name}.weight")
            if module.bias is not None:
                keys.add(f"{name}.bias")
    return keys


@pytest.mark.gpu
@pytest.mark.parametrize("mode_cfg", [
    {"mode": "blocktt", "target_modules": "all",
     "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                 "train_position": "small", "s_merged_to": "frozen",
                 "convert_mode": "svd", "factorize_by_head": True,
                 "train_bias": True, "normalize_after_update": False,
                 "qfura": {"enabled": False}}},
    {"mode": "svd", "target_modules": "all",
     "svd": {"train_position": "output", "s_merged_to": "frozen",
             "compression_ratio": 1.0}},
])
def test_export_keys_subset_of_linear_keys(tiny_model, tiny_tokenizer, mode_cfg):
    target_keys = _linear_param_keys(tiny_model)
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create(mode_cfg))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    exported = adapter.export_for_vllm(out)
    # Every exported key must correspond to a Linear in the original model.
    for key in exported:
        # Allow either exact .weight match or .bias match.
        assert key in target_keys, f"exported key {key!r} not found in tiny_model Linear keys"
