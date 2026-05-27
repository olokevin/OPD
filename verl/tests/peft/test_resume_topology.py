"""Resume path: write peft_meta.json + btt_topology.json, then on a fresh
model call adapter.rebuild_from_meta and verify the module topology matches
the original adapter.apply output."""
from __future__ import annotations

import json
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter
from verl.workers.peft.blocktt import BlockTTAdapter


@pytest.mark.gpu
def test_blocktt_rebuild_from_meta(tiny_model, tiny_tokenizer, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": False}},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    original = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                             calib_loader_builder=lambda: None)
    # Persist sidecar.
    meta = adapter.topology_meta()
    adapter.write_compress_sidecar(str(tmp_path))
    meta["_resolved_topology_path"] = str(tmp_path / "compress" / "btt_topology.json")

    # Fresh model — rebuild.
    fresh = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype=torch.float32,
    ).cuda()
    rebuilt = BlockTTAdapter.rebuild_from_meta(fresh, meta)

    def _module_class_map(model):
        return {n: type(m).__name__ for n, m in model.named_modules()
                if type(m).__name__ in {"BTTLinear", "QBTTLinear",
                                        "SVDCompressedLinear", "Linear"}}

    assert _module_class_map(rebuilt) == _module_class_map(original)
