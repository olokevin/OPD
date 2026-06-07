import pytest

from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _packed_row_blocks,
    _assign_rollout_ids,
    _select_bucket,
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
