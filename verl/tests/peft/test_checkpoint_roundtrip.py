"""Save → reload → forward-pass parity for every PEFT mode.

merged_hf/ must be loadable via stock from_pretrained (compress/none) or
PeftModel.from_pretrained (lora/qlora). Logits must match within 1e-4 on a
fixed input."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def _forward_logits(model, input_ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids.to(next(model.parameters()).device)).logits.detach().cpu()


def test_none_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({"mode": "none"}))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32)
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4), f"max diff {(pre - post).abs().max()}"


def test_lora_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    from peft import PeftModel
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "lora", "target_modules": "all",
        "lora": {"rank": 4, "alpha": 8},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    base = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, str(save_dir))
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("qfura", [False, True])
def test_blocktt_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path, qfura):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": qfura}},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32).cuda()
    post = _forward_logits(reloaded, fixed_inputs)
    # qfura has NF4 quantization error; loosen tolerance there.
    atol = 5e-2 if qfura else 1e-4
    assert torch.allclose(pre, post, atol=atol), f"max diff {(pre - post).abs().max()}"


@pytest.mark.gpu
def test_svd_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "svd", "target_modules": "all",
        "svd": {"train_position": "output", "s_merged_to": "frozen",
                "compression_ratio": 1.0},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32).cuda()
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4)
