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


def test_blocktt_qr_with_s_merged_to_warns(caplog):
    import logging
    # The "llamafactory" library root logger sets propagate=False (see
    # llamafactory.extras.logging), which prevents pytest's caplog from
    # capturing records from submodules. Temporarily re-enable propagation
    # so caplog (attached to the stdlib root) sees this warning.
    caplog.set_level(logging.WARNING, logger="llamafactory.hparams.finetuning_args")
    lf_root = logging.getLogger("llamafactory")
    prev_propagate = lf_root.propagate
    lf_root.propagate = True
    try:
        fa = _make(finetuning_type="blocktt", convert_mode="qr", s_merged_to="output")
    finally:
        lf_root.propagate = prev_propagate
    assert fa.s_merged_to is None
    assert any("convert_mode=qr" in r.message for r in caplog.records)


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
