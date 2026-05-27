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


def _build_btt_converted_tiny_model(fake_finetuning_args):
    """Build a tiny Qwen-shaped model and run plain BlockTT conversion on
    it so the model contains real BTTLinear modules whose materialization
    is exercised by _materialize_and_save. Requires CUDA (the underlying
    compress.svd_finetune raises on CPU tensors)."""
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("compress conversion requires CUDA")
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()

    # Reuse the tiny model factory from the compress_setup tests
    import sys, pathlib
    test_dir = pathlib.Path(__file__).resolve().parents[1] / "model"
    sys.path.insert(0, str(test_dir))
    try:
        from test_compress_setup import (
            _tiny_qwen_like_model, _default_fa_kwargs,
        )
    finally:
        sys.path.remove(str(test_dir))

    fa = fake_finetuning_args(**_default_fa_kwargs(finetuning_type="blocktt"))
    model = _tiny_qwen_like_model().cuda()
    return compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )


def test_save_callback_plain_writes_merged_dir(tmp_path, fake_finetuning_args):
    """Plain BTT model: _build_materialized_state_dict materializes BTT
    cores into dense nn.Linear-shaped weights, with no .btt_l/.btt_r keys."""
    from llamafactory.train.callbacks import _build_materialized_state_dict

    model = _build_btt_converted_tiny_model(fake_finetuning_args)
    sd = _build_materialized_state_dict(model)

    btt_keys = [k for k in sd if k.endswith(".btt_l") or k.endswith(".btt_r")]
    assert btt_keys == [], f"unexpected BTT keys: {btt_keys}"
    weight_keys = [k for k in sd if k.endswith(".weight")]
    assert len(weight_keys) > 0


def test_save_callback_rank_zero_guard(tmp_path, fake_finetuning_args):
    """Non-rank-0 callers must be no-ops — no checkpoint-N-merged dir."""
    from llamafactory.train.callbacks import CompressSaveCallback

    class _State:
        is_world_process_zero = False
        global_step = 5

    class _Args:
        output_dir = str(tmp_path)

    cb = CompressSaveCallback(
        fake_finetuning_args(finetuning_type="blocktt", calib_mode="none"),
    )
    cb.on_save(args=_Args(), state=_State(), control=None, model=object())
    assert not (tmp_path / "checkpoint-5-merged").exists()
