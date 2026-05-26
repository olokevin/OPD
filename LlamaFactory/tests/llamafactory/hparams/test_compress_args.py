"""Unit tests for CompressArguments mixin on FinetuningArguments."""
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
    _make(finetuning_type="blocktt")
    _make(finetuning_type="svd")
