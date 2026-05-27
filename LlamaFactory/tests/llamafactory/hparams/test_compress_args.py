"""Unit tests for CompressArguments mixin on FinetuningArguments."""
import pytest

from llamafactory.hparams.finetuning_args import FinetuningArguments


def _make(**overrides):
    return FinetuningArguments(**overrides)


def test_default_compress_fields_present():
    fa = _make()
    # Shared defaults
    assert fa.trainable_type == "all"
    assert fa.train_position is None
    assert fa.s_merged_to is None
    # BTT-only defaults
    assert fa.decomp_mode == "input_one_block"
    assert fa.blocktt_rank == "full"
    assert fa.convert_mode == "svd"
    assert fa.train_bias is True
    assert fa.blocktt_normalize_after_update is False
    assert fa.blocktt_factorize_by_head is True
    # Calib defaults (mirroring add_calibrated_btt_args)
    assert fa.calib_mode == "none"
    assert fa.calib_source == "c4"
    assert fa.calib_num_seqs == 128
    assert fa.calib_max_length == 2048
    assert fa.calib_seed == 3
    assert fa.calib_batch_size == 8
    assert fa.calib_traces_path is None
    assert fa.compression_ratio == 1.0


def test_finetuning_type_blocktt_and_svd_accepted():
    assert _make(finetuning_type="blocktt").finetuning_type == "blocktt"
    assert _make(finetuning_type="svd").finetuning_type == "svd"


def test_finetuning_type_invalid_rejected():
    with pytest.raises(AssertionError, match="Invalid fine-tuning method"):
        _make(finetuning_type="bogus")


def test_blocktt_default_train_position_small():
    fa = _make(finetuning_type="blocktt")
    assert fa.train_position == "small"


def test_svd_default_train_position_output():
    fa = _make(finetuning_type="svd")
    assert fa.train_position == "output"


def test_compress_default_s_merged_to_frozen():
    assert _make(finetuning_type="blocktt").s_merged_to == "frozen"
    assert _make(finetuning_type="svd").s_merged_to == "frozen"


def test_blocktt_train_position_rejects_svd_values():
    with pytest.raises(ValueError, match="train_position"):
        _make(finetuning_type="blocktt", train_position="output")


def test_svd_train_position_rejects_blocktt_values():
    with pytest.raises(ValueError, match="train_position"):
        _make(finetuning_type="svd", train_position="small")


def test_blocktt_both_with_frozen_s_merged_rejected():
    with pytest.raises(ValueError, match="train_position.*both"):
        _make(finetuning_type="blocktt", train_position="both", s_merged_to="frozen")


def test_blocktt_qr_with_s_merged_to_warns():
    # The "llamafactory" library root logger sets propagate=False and writes
    # through its own StreamHandler bound to a captured sys.stdout (see
    # llamafactory.extras.logging._configure_library_root_logger). Both
    # caplog (which relies on propagation to the root logger) and capsys/
    # capfd (which require the handler to write to the *current* fd) fail
    # to observe the record cleanly. The least invasive approach is to
    # attach our own list-collecting handler to the emitting submodule
    # logger for the duration of the test, then assert against it. This
    # avoids any monkey-patching of propagation or library globals.
    import logging
    captured: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    target = logging.getLogger("llamafactory.hparams.finetuning_args")
    handler = _ListHandler(level=logging.WARNING)
    prev_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:
        fa = _make(finetuning_type="blocktt", convert_mode="qr", s_merged_to="output")
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)

    assert fa.s_merged_to is None
    assert any("convert_mode=qr" in r.getMessage() for r in captured)


def test_calib_mode_requires_compress_finetuning():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="full", calib_mode="v2")


def test_svd_calib_mode_rejects_blocktt_method():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="blocktt", calib_mode="svd_v2")


def test_btt_calib_mode_rejects_svd_method():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="svd", calib_mode="v2")


def test_blocktt_rank_must_parse():
    with pytest.raises(ValueError, match="blocktt_rank"):
        _make(finetuning_type="blocktt", blocktt_rank="notanumber")
    _make(finetuning_type="blocktt", blocktt_rank="full")  # OK
    _make(finetuning_type="blocktt", blocktt_rank="16")    # OK


def test_compress_rejects_galore_apollo_badam():
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="blocktt", use_galore=True)
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="svd", use_apollo=True)
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="blocktt", use_badam=True)


def test_blocktt_rank_accepts_float_in_calibrated_mode():
    # Float blocktt_rank is valid under calibrated BTT (0, 1] ratio.
    _make(finetuning_type="blocktt", blocktt_rank="0.5", calib_mode="v2")
    _make(finetuning_type="blocktt", blocktt_rank="1.0", calib_mode="v2")


def test_blocktt_rank_rejects_float_in_plain_mode():
    with pytest.raises(ValueError, match="float values are only valid"):
        _make(finetuning_type="blocktt", blocktt_rank="0.5", calib_mode="none")


def test_blocktt_rank_rejects_out_of_range_float():
    with pytest.raises(ValueError, match=r"float must be in \(0, 1\]"):
        _make(finetuning_type="blocktt", blocktt_rank="1.5", calib_mode="v2")
    with pytest.raises(ValueError, match=r"float must be in \(0, 1\]"):
        _make(finetuning_type="blocktt", blocktt_rank="0", calib_mode="v2")


def test_blocktt_rank_rejects_integer_in_calibrated_mode():
    with pytest.raises(ValueError, match="integer blocktt_rank is only valid"):
        _make(finetuning_type="blocktt", blocktt_rank="16", calib_mode="v2")
