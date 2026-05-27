"""Shared fixtures for verl/workers/peft tests."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture
def tiny_model():
    """Fresh tiny Llama on CPU. Cheap to recreate per test."""
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID, torch_dtype=torch.float32)


@pytest.fixture
def fixed_inputs(tiny_tokenizer):
    """Deterministic input ids for forward-parity checks."""
    text = "The quick brown fox jumps over the lazy dog."
    enc = tiny_tokenizer(text, return_tensors="pt")
    return enc["input_ids"]
