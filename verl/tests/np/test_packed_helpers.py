import pytest
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _packed_row_blocks,
    _assign_rollout_ids,
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
    # 2 prompts x 2 rollouts = 4 slots; ids must be distinct and stable.
    assert len(ids) == 4
    assert len(set(ids)) == 4
