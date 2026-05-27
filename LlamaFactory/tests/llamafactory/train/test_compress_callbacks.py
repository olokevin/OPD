"""Unit tests for CompressNormalizeCallback."""
import torch.nn as nn


def test_normalize_callback_disabled_when_flag_off(fake_finetuning_args):
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(fake_finetuning_args(
        finetuning_type="blocktt", blocktt_normalize_after_update=False,
    ))
    assert cb.enabled is False


def test_normalize_callback_disabled_for_svd(fake_finetuning_args):
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(fake_finetuning_args(
        finetuning_type="svd", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is False


def test_normalize_callback_calls_compress_helper(monkeypatch, fake_finetuning_args):
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
    cb = CompressNormalizeCallback(fake_finetuning_args(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is True

    fake_model = nn.Linear(4, 4)
    cb.on_step_end(args=None, state=None, control=None, model=fake_model)
    assert calls == [fake_model]


def test_normalize_callback_no_model_kwarg_is_noop(monkeypatch, fake_finetuning_args):
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    calls = []
    monkeypatch.setattr(
        ci, "normalize_trainable_blocktt_cores_",
        lambda m: calls.append(m),
    )
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(fake_finetuning_args(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    cb.on_step_end(args=None, state=None, control=None, model=None)
    assert calls == []
