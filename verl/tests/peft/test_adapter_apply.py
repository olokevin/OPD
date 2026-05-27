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


from peft import PeftModel


def _lora_cfg(rank=4, alpha=8):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "lora",
        "target_modules": "all",
        "lora": {"rank": rank, "alpha": alpha, "dropout": 0.0},
    }))


def test_lora_adapter_wraps_with_peft(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert isinstance(out, PeftModel)
    # only adapter params should require grad
    grad_param_names = [n for n, p in out.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in grad_param_names), grad_param_names
    assert any(".q_proj" in n or ".o_proj" in n for n in grad_param_names)


def test_lora_adapter_export_for_vllm_returns_none(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert adapter.export_for_vllm(out) is None


def test_lora_adapter_vllm_engine_kwargs_enables_lora(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg(rank=8)
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    kw = adapter.vllm_engine_kwargs()
    assert kw["enable_lora"] is True
    assert kw["max_loras"] == 1
    assert kw["max_lora_rank"] >= 8


def test_lora_adapter_save_pretrained_writes_adapter_dir(tiny_model, tiny_tokenizer, tmp_path):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out_model = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out_model, str(save_dir))
    assert (save_dir / "adapter_config.json").exists()
    assert (save_dir / "adapter_model.safetensors").exists()


import pytest


def _qlora_cfg(rank=4):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "qlora",
        "target_modules": "all",
        "lora": {"rank": rank, "alpha": rank * 2},
        "qlora": {"bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": "bfloat16",
                  "bnb_4bit_use_double_quant": True},
    }))


@pytest.mark.gpu
def test_qlora_adapter_reloads_base_in_4bit(tiny_model, tiny_tokenizer, tmp_path):
    # The adapter ignores the prebuilt tiny_model and reloads the model id in 4-bit,
    # so QLoRAAdapter must be told a model path via peft_cfg.model_config.
    pytest.importorskip("bitsandbytes")
    from transformers import AutoModelForCausalLM
    cfg = _qlora_cfg()
    class MockModelConfig:
        path = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        local_path = path
        trust_remote_code = False
    adapter = PEFTAdapter.from_config(cfg, model_config=MockModelConfig())
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    # at least one Linear should now be Linear4bit
    import bitsandbytes as bnb
    found_4bit = any(isinstance(m, bnb.nn.Linear4bit) for m in out.modules())
    assert found_4bit, "expected at least one Linear4bit after QLoRA apply"


def test_qlora_topology_meta_has_qlora_block(tiny_tokenizer):
    cfg = _qlora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    meta = adapter.topology_meta()
    assert meta["mode"] == "qlora"
    assert meta["qlora"]["bnb_4bit_quant_type"] == "nf4"
