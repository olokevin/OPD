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


def test_build_materialized_state_dict_has_no_btt_keys(fake_finetuning_args):
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


def test_save_callback_does_not_mutate_live_model(tmp_path, fake_finetuning_args, monkeypatch):
    """Critical: CompressSaveCallback must not replace BTT modules with
    nn.Linear on the live training model, even for calibrated runs.
    Regression for the bug where save_calibrated_btt_hf_pretrained
    mutated the model in place via materialize_calibrated_btt_to_linear.

    The tiny test model has no HF ``.config``, so we stub the peer-model
    construction/save — the test's concern is whether the live model is
    mutated by ``_materialize_and_save``, not the downstream HF save path.
    """
    from llamafactory.train.callbacks import _materialize_and_save
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    from compress.integration import BTTLinear

    model = _build_btt_converted_tiny_model(fake_finetuning_args)
    btt_modules_before = [
        n for n, m in model.named_modules() if isinstance(m, BTTLinear)
    ]
    assert btt_modules_before, "test precondition: model has BTT modules"

    # Stub out the HF peer construction/save: the tiny fixture model
    # has no .config, and the test is only concerned with whether the
    # live model is mutated.
    class _StubPeer:
        def load_state_dict(self, sd, strict=False):  # noqa: ARG002
            return None, None
        def save_pretrained(self, path):  # noqa: ARG002
            return None

    import transformers
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_config",
        classmethod(lambda cls, *a, **kw: _StubPeer()),
    )
    # _materialize_and_save reads model.config; give the tiny model a
    # placeholder attribute so the lookup itself doesn't AttributeError.
    monkeypatch.setattr(model, "config", object(), raising=False)

    _materialize_and_save(model, str(tmp_path / "merged"))

    btt_modules_after = [
        n for n, m in model.named_modules() if isinstance(m, BTTLinear)
    ]
    assert btt_modules_after == btt_modules_before, (
        f"BTT modules were mutated by _materialize_and_save. "
        f"Before: {btt_modules_before!r}, after: {btt_modules_after!r}"
    )


def test_save_callback_writes_tokenizer_when_provided(tmp_path, fake_finetuning_args, monkeypatch):
    """CompressSaveCallback should save the tokenizer (passed via the HF
    processing_class kwarg) alongside the merged model weights."""
    from llamafactory.train.callbacks import CompressSaveCallback

    class _State:
        is_world_process_zero = True
        global_step = 7

    class _Args:
        output_dir = str(tmp_path)

    class _StubTokenizer:
        def __init__(self):
            self.saved_to = None
        def save_pretrained(self, path):
            self.saved_to = str(path)

    # Stub _materialize_and_save so the test doesn't need a real BTT model.
    def fake_materialize_and_save(model, ckpt_dir):
        return None
    monkeypatch.setattr(
        "llamafactory.train.callbacks._materialize_and_save",
        fake_materialize_and_save,
    )

    tok = _StubTokenizer()
    cb = CompressSaveCallback(fake_finetuning_args(finetuning_type="blocktt", calib_mode="none"))
    cb.on_save(
        args=_Args(), state=_State(), control=None, model=object(),
        processing_class=tok,
    )
    assert tok.saved_to == str(tmp_path / "checkpoint-7-merged")
