"""Seed/regeneration parity: the assembly must regenerate the EXACT bytes the
decode drew, from (global_seed, t, rollout_id) alone. CPU only."""
import torch

from verl.trainer.es_token.seeding import (
    build_noise_layout, draw_token_noise, es_token_seed)
from verl.trainer.np.seeding import noise_seed


def test_es_seed_namespace_disjoint_from_np():
    # Same (global_seed, step, rollout, q=0) must NOT collide with an NP layer key.
    s_es = es_token_seed(42, 3, 7)
    s_np = noise_seed(42, 3, "model.layers.0.mlp.down_proj", 7, 0)
    assert s_es != s_np


def test_draw_token_noise_regenerates_bit_identical():
    for method in ("bernoulli", "gaussian", "uniform"):
        a = draw_token_noise(42, 5, 13, 256, torch.device("cpu"),
                             torch.float32, method)
        b = draw_token_noise(42, 5, 13, 256, torch.device("cpu"),
                             torch.float32, method)
        assert torch.equal(a, b)
        c = draw_token_noise(42, 6, 13, 256, torch.device("cpu"),
                             torch.float32, method)
        assert not torch.equal(a, c)


def test_noise_layout_contiguous_disjoint():
    dims = [("qkv", 12, 8), ("o", 8, 8), ("down", 8, 24)]
    layout, d_total = build_noise_layout(dims)
    assert d_total == (12 + 8) + (8 + 8) + (8 + 24)
    spans = []
    for name, d_out, d_in in dims:
        off_u, do, off_v, di = layout[name]
        assert (do, di) == (d_out, d_in)
        spans += [(off_u, off_u + do), (off_v, off_v + di)]
    spans.sort()
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0, "slices must tile the flat vector with no gap/overlap"
    assert spans[0][0] == 0 and spans[-1][1] == d_total


def test_layout_slices_select_consistent_noise():
    dims = [("a", 16, 8), ("b", 8, 32)]
    layout, d_total = build_noise_layout(dims)
    flat = draw_token_noise(0, 0, 0, d_total, torch.device("cpu"),
                            torch.float32, "gaussian")
    off_u, do, off_v, di = layout["b"]
    u = flat[off_u:off_u + do]
    v = flat[off_v:off_v + di]
    assert u.shape == (8,) and v.shape == (32,)
    assert torch.equal(u, flat[24:32]) and torch.equal(v, flat[32:64])
