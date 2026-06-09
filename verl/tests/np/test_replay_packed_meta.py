"""CPU tests for the E2 packed-graph replay seams:

  - `_packed_replay_row_meta` (C-4 pad-row metadata): a FINISHED/PAD slot must
    carry the slot's LAST-VALID seq_len/position (NEVER 0/stale) and clean_slot
    forced to -1; an ACTIVE slot carries its real values. A zero seqused_k makes
    FlashAttention's KV iteration undefined, so the "valid, not zero" property is
    the load-bearing C-4 guardrail and is asserted directly here.
  - `_np_fill_u_buf_all_layers_packed` (C-6 single noise-refill site): every
    layer's packed u_buf is seeded per slot with the SAME noise_seed key the
    eager packed / serial paths use -> bit-identical noise (parity-by-construction).
"""
import torch

from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _packed_replay_row_meta,
    WorkerExtension,
)
from verl.trainer.np.seeding import noise_seed, draw_noise


def test_active_slot_gets_real_meta():
    slot_states = [{
        "active": True, "q_token": 7, "q_pos": 12, "clean_slot": 345,
        "last_q_token": 7, "last_q_pos": 12,
    }]
    meta = _packed_replay_row_meta(slot_states)
    assert meta == [{"q_token": 7, "q_pos": 12, "seq_len": 13, "clean_slot": 345}]


def test_finished_slot_gets_last_valid_not_zero():
    # C-4: a finished slot reuses its LAST VALID token/position; seq_len must be
    # q_pos+1 (> 0), NOT zero/stale; clean_slot must be forced to -1 (no KV write).
    slot_states = [{
        "active": False, "q_token": 999, "q_pos": 0, "clean_slot": 111,
        "last_q_token": 42, "last_q_pos": 8,
    }]
    meta = _packed_replay_row_meta(slot_states)[0]
    assert meta["clean_slot"] == -1, "finished slot must write NO KV (slot=-1)"
    assert meta["q_pos"] == 8 and meta["q_token"] == 42, "must reuse last-valid"
    assert meta["seq_len"] == 9, "seq_len = last_q_pos+1"
    assert meta["seq_len"] > 0, "C-4: seqused_k must be VALID (never zero)"


def test_pad_slot_seq_len_strictly_positive_for_min_position():
    # Even the earliest-possible finished slot (last_q_pos=0, a 1-token prompt
    # that emitted EOS at its first decode) yields seq_len=1 > 0, never 0.
    slot_states = [{
        "active": False, "q_token": 5, "q_pos": 0, "clean_slot": 9,
        "last_q_token": 5, "last_q_pos": 0,
    }]
    meta = _packed_replay_row_meta(slot_states)[0]
    assert meta["seq_len"] == 1 and meta["clean_slot"] == -1


def test_mixed_bucket_active_and_finished():
    # Staggered bucket: slot 0 active, slot 1 finished, slot 2 pad.
    slot_states = [
        {"active": True, "q_token": 1, "q_pos": 5, "clean_slot": 50,
         "last_q_token": 1, "last_q_pos": 5},
        {"active": False, "q_token": 2, "q_pos": 3, "clean_slot": 60,
         "last_q_token": 11, "last_q_pos": 7},
        {"active": False, "q_token": 0, "q_pos": 0, "clean_slot": 70,
         "last_q_token": 22, "last_q_pos": 4},
    ]
    meta = _packed_replay_row_meta(slot_states)
    # active slot: real
    assert meta[0] == {"q_token": 1, "q_pos": 5, "seq_len": 6, "clean_slot": 50}
    # finished / pad slots: last-valid + clean_slot=-1
    assert meta[1] == {"q_token": 11, "q_pos": 7, "seq_len": 8, "clean_slot": -1}
    assert meta[2] == {"q_token": 22, "q_pos": 4, "seq_len": 5, "clean_slot": -1}
    # NONE of them has a zero seq_len (the C-4 landmine).
    assert all(m["seq_len"] > 0 for m in meta)


def test_fill_u_buf_all_layers_packed_parity_with_eager_loop():
    # The packed refill must write the SAME bytes the eager packed driver writes:
    # slot p's rows [p*N : (p+1)*N], seeded by noise_seed(global_seed, step,
    # layer, slot_rollout_ids[p], q). Bit-identical -> parity-by-construction.
    we = WorkerExtension.__new__(WorkerExtension)
    bucket_b_pack, n_sample, d_out = 2, 3, 6
    u = {
        "L0": torch.zeros(bucket_b_pack * n_sample, d_out),
        "L1": torch.zeros(bucket_b_pack * n_sample, d_out),
    }
    cfg = dict(global_seed=42, sample_method="gaussian")
    slot_rollout_ids = [16, 23]  # arbitrary, distinct
    we._np_fill_u_buf_all_layers_packed(
        u, cfg, ["L0", "L1"], step=3, slot_rollout_ids=slot_rollout_ids,
        n_sample=n_sample)
    # Reconstruct the eager packed driver's exact draw for each (layer, slot, q).
    for ln in ("L0", "L1"):
        for p, rid in enumerate(slot_rollout_ids):
            for q in range(n_sample):
                exp = draw_noise(noise_seed(42, 3, ln, rid, q), (d_out,),
                                 torch.device("cpu"), torch.float32, "gaussian")
                got = u[ln][p * n_sample + q]
                assert torch.equal(got, exp), f"{ln} slot{p} q{q} not bit-identical"


def test_fill_u_buf_packed_distinct_per_slot_and_layer():
    # Different slots (different rollout id) and different layers -> different noise.
    we = WorkerExtension.__new__(WorkerExtension)
    bucket_b_pack, n_sample, d_out = 2, 2, 8
    u = {"L0": torch.zeros(bucket_b_pack * n_sample, d_out)}
    cfg = dict(global_seed=7, sample_method="gaussian")
    we._np_fill_u_buf_all_layers_packed(
        u, cfg, ["L0"], step=0, slot_rollout_ids=[0, 1], n_sample=n_sample)
    # slot 0 row 0 vs slot 1 row 0 differ (different rollout id).
    assert not torch.equal(u["L0"][0], u["L0"][n_sample])
