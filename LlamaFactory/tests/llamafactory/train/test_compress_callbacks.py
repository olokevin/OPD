"""Unit tests for CompressNormalizeCallback and CompressSaveCallback."""
import pathlib

import pytest
import torch
import torch.nn as nn


class _FakeFA:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_normalize_callback_disabled_when_flag_off():
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=False,
    ))
    assert cb.enabled is False


def test_normalize_callback_disabled_for_svd():
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="svd", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is False


def test_normalize_callback_calls_compress_helper(monkeypatch):
    """When enabled, on_step_end should call normalize_trainable_blocktt_cores_."""
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    calls = []
    monkeypatch.setattr(
        ci, "normalize_trainable_blocktt_cores_",
        lambda m: calls.append(m),
    )

    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is True

    fake_model = nn.Linear(4, 4)
    cb.on_step_end(args=None, state=None, control=None, model=fake_model)
    assert calls == [fake_model]


def test_normalize_callback_no_model_kwarg_is_noop(monkeypatch):
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    calls = []
    monkeypatch.setattr(
        ci, "normalize_trainable_blocktt_cores_",
        lambda m: calls.append(m),
    )
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    cb.on_step_end(args=None, state=None, control=None, model=None)
    assert calls == []
