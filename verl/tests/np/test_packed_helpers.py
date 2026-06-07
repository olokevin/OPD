import pytest

from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _packed_row_blocks,
    _assign_rollout_ids,
    _select_bucket,
    _pad_waves_to_pack_width,
)


def test_packed_row_blocks_layout():
    # B_pack=3 prompts, N=2 perturbed rails -> each prompt owns (1+N)=3 rows.
    blocks = _packed_row_blocks(b_pack=3, n_sample=2)
    # prompt p: clean row = p*(1+N), perturbed rows = next N.
    assert blocks == [
        {"clean": 0, "perturbed": [1, 2]},
        {"clean": 3, "perturbed": [4, 5]},
        {"clean": 6, "perturbed": [7, 8]},
    ]
    # total rows R = B_pack*(1+N)
    assert blocks[-1]["perturbed"][-1] + 1 == 3 * (1 + 2)


def test_assign_rollout_ids_matches_serial_global_index():
    # Serial loop seeds prompt b with rollout_idx = step*batch_size + b (spec §4.6).
    # Wave-chunked packing must reproduce the SAME per-prompt rollout_idx so the
    # noise draw is identical -> parity-by-construction.
    ids = _assign_rollout_ids(step=2, batch_size=8, n_rollout=1)
    assert ids == [16, 17, 18, 19, 20, 21, 22, 23]  # 2*8 + b


def test_assign_rollout_ids_n_rollout_gt_1():
    # n_rollout>1: batch_size*n_rollout slots, each (prompt,rollout) a distinct id.
    ids = _assign_rollout_ids(step=0, batch_size=2, n_rollout=2)
    # (step*batch_size + b) * n_rollout + r:
    # (0+0)*2+0, (0+0)*2+1, (0+1)*2+0, (0+1)*2+1
    assert ids == [0, 1, 2, 3]


def test_select_bucket_smallest_ge_B():
    # E3 picks the smallest bucket >= B; leftover slots become PAD rows (C-4).
    buckets = [2, 4, 8, 16]
    assert _select_bucket(1, buckets) == 2
    assert _select_bucket(2, buckets) == 2
    assert _select_bucket(3, buckets) == 4
    assert _select_bucket(4, buckets) == 4
    assert _select_bucket(5, buckets) == 8
    assert _select_bucket(8, buckets) == 8
    assert _select_bucket(16, buckets) == 16


def test_select_bucket_exact_and_unsorted_input():
    # Input order must not matter (helper sorts internally); exact-fit returns B.
    assert _select_bucket(4, [16, 2, 8, 4]) == 4
    assert _select_bucket(3, [8, 2, 4]) == 4


def test_select_bucket_overflow_raises():
    # B beyond the largest bucket: the graphed driver can't wave one fixed-width
    # graph over more prompts than captured -> the caller must chunk. Raise.
    with pytest.raises(ValueError):
        _select_bucket(17, [2, 4, 8, 16])


def test_select_bucket_invalid_B():
    with pytest.raises(ValueError):
        _select_bucket(0, [2, 4, 8, 16])


def test_pad_waves_exact_multiple_no_padding():
    # batch_size == pack_width: ONE full wave, no padding, real_count == width.
    pids = [[1], [2], [3], [4]]
    rids = [10, 11, 12, 13]
    waves = _pad_waves_to_pack_width(pids, rids, pack_width=4)
    assert len(waves) == 1
    wave_pids, wave_rids, real_count = waves[0]
    assert wave_pids == [[1], [2], [3], [4]]
    assert wave_rids == [10, 11, 12, 13]
    assert real_count == 4


def test_pad_waves_short_final_wave_padded_to_pack_width():
    # 6 slots, pack_width=4 -> wave0 full (4 real), wave1 has 2 real + 2 pad.
    # The padded tail repeats slot 0 so _select_bucket(4) picks the SAME bucket
    # for BOTH waves -> one captured graph (no multi-bucket allocator assert).
    pids = [[1], [2], [3], [4], [5], [6]]
    rids = [10, 11, 12, 13, 14, 15]
    waves = _pad_waves_to_pack_width(pids, rids, pack_width=4)
    assert len(waves) == 2
    # Every wave is EXACTLY pack_width prompts (the invariant that pins one bucket).
    for wave_pids, wave_rids, _ in waves:
        assert len(wave_pids) == 4
        assert len(wave_rids) == 4
    # wave0: all real.
    assert waves[0] == ([[1], [2], [3], [4]], [10, 11, 12, 13], 4)
    # wave1: 2 real ([5],[6]); padded tail repeats slot 0 ([1], rid 10); real=2.
    w1_pids, w1_rids, w1_real = waves[1]
    assert w1_real == 2
    assert w1_pids[:2] == [[5], [6]]
    assert w1_pids[2:] == [[1], [1]]   # padded with slot 0
    assert w1_rids == [14, 15, 10, 10]


def test_pad_waves_single_short_wave_padded():
    # batch_size < pack_width: a single wave padded up; real_count tracks the truth.
    pids = [[7], [8]]
    rids = [3, 4]
    waves = _pad_waves_to_pack_width(pids, rids, pack_width=4)
    assert len(waves) == 1
    w_pids, w_rids, w_real = waves[0]
    assert len(w_pids) == 4 and w_real == 2
    assert w_pids == [[7], [8], [7], [7]]
    assert w_rids == [3, 4, 3, 3]


def test_pad_waves_pack_width_one():
    # pack_width=1 -> one wave per slot, each fully real.
    pids = [[1], [2], [3]]
    rids = [0, 1, 2]
    waves = _pad_waves_to_pack_width(pids, rids, pack_width=1)
    assert [w[2] for w in waves] == [1, 1, 1]
    assert [w[0] for w in waves] == [[[1]], [[2]], [[3]]]


def test_pad_waves_validation():
    with pytest.raises(ValueError):
        _pad_waves_to_pack_width([[1]], [1, 2], pack_width=2)   # len mismatch
    with pytest.raises(ValueError):
        _pad_waves_to_pack_width([], [], pack_width=2)          # empty
    with pytest.raises(ValueError):
        _pad_waves_to_pack_width([[1]], [1], pack_width=0)      # bad width
