"""CPU unit tests for the V2 buffer-in-graph helpers (no GPU).

Covers the pure-Python contracts the GPU parity gate
(scripts/zo_opd/np_checks/check_graphed_parity.py) relies on:
  - perturb_graph mode adds sigma*u_buf to the perturbed-row block only,
    leaves the clean row untouched, and is RNG-free (reads u_buf).
  - perturb_graph writes x[0] into the persistent x_buf in place.
  - sigma=0 in perturb_graph is a no-op (the sigma=0 gate's premise).
  - _np_fill_u_buf writes BIT-IDENTICAL noise to what V1's in-forward draw
    produces for the same (global_seed, step, layer, rollout, q) key -- the
    parity-by-construction guarantee.
"""
import torch

from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    PerturbedLinear, WorkerExtension,
)
from verl.trainer.np.seeding import noise_seed, draw_noise


class FakeLinear(torch.nn.Module):
    """vLLM-linear stand-in: returns (output, bias) like ColumnParallelLinear."""

    def __init__(self, d_in, d_out, return_tuple=True):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d_out, d_in))
        self.return_tuple = return_tuple

    def forward(self, x):
        y = x @ self.w.t()
        return (y, None) if self.return_tuple else y


def _graph_state(d_out, d_in, n_sample, sigma, u_buf, x_buf):
    return {
        "mode": "perturb_graph", "layer": "L", "sigma": float(sigma),
        "n_clean_rows": 1, "u_buf": u_buf, "x_buf": x_buf,
    }


def test_perturb_graph_adds_sigma_u_buf_to_perturbed_rows_only():
    d = 4
    base = FakeLinear(d, d)
    u_buf = torch.arange(2 * d, dtype=torch.float32).reshape(2, d)  # [N=2, d_out]
    x_buf = torch.zeros(d)
    st = _graph_state(d, d, n_sample=2, sigma=3.0, u_buf=u_buf, x_buf=x_buf)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.zeros(3, d)              # 1 clean + 2 perturbed, identity weight
    out, _ = pl(x)
    # clean row (0) untouched
    assert torch.allclose(out[0], torch.zeros(d))
    # perturbed rows == 0 + sigma*u_buf
    assert torch.allclose(out[1], 3.0 * u_buf[0])
    assert torch.allclose(out[2], 3.0 * u_buf[1])


def test_perturb_graph_writes_x_buf_in_place():
    d = 3
    base = FakeLinear(d, d)
    u_buf = torch.zeros(2, d)
    x_buf = torch.zeros(d)
    st = _graph_state(d, d, n_sample=2, sigma=1.0, u_buf=u_buf, x_buf=x_buf)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])
    pl(x)
    # x_buf holds the CLEAN row's input, in place (same storage)
    assert torch.allclose(x_buf, torch.tensor([1.0, 2.0, 3.0]))


def test_perturb_graph_sigma_zero_is_noop():
    d = 4
    base = FakeLinear(d, d)
    u_buf = torch.randn(2, d)
    x_buf = torch.zeros(d)
    st = _graph_state(d, d, n_sample=2, sigma=0.0, u_buf=u_buf, x_buf=x_buf)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.randn(3, d)
    out, _ = pl(x)
    # sigma=0 -> 0*u_buf added -> exact passthrough on every row
    assert torch.allclose(out, x @ base.w.t())


def test_perturb_graph_only_acts_on_matched_layer():
    d = 3
    base = FakeLinear(d, d)
    u_buf = torch.ones(2, d)
    x_buf = torch.zeros(d)
    st = _graph_state(d, d, n_sample=2, sigma=5.0, u_buf=u_buf, x_buf=x_buf)
    st["layer"] = "OTHER"              # this module is "L", not the active layer
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.randn(3, d)
    out = pl(x)
    # wrong layer -> pure passthrough (tuple preserved, no perturbation)
    assert isinstance(out, tuple)
    assert torch.allclose(out[0], x @ base.w.t())


def test_fill_u_buf_is_bit_identical_to_v1_in_forward_draw():
    """_np_fill_u_buf must produce the SAME bytes V1 draws in-forward for the
    same key. This is the M1 parity-by-construction guarantee."""
    d_out, n_sample = 6, 3
    np_cfg = dict(global_seed=42, sample_method="bernoulli")
    layer, step, rollout = "model.layers.0.mlp.down_proj", 7, 2

    u_buf = torch.zeros(n_sample, d_out, dtype=torch.float32)
    ext = WorkerExtension.__new__(WorkerExtension)   # no __init__ needed
    ext._np_fill_u_buf(u_buf, np_cfg, layer, step, rollout, n_sample)

    # V1's in-forward draw (np_worker_extension perturb mode, line ~91):
    v1 = torch.stack([
        draw_noise(noise_seed(42, step, layer, rollout, q),
                   (d_out,), torch.device("cpu"), torch.float32, "bernoulli")
        for q in range(n_sample)
    ])
    assert torch.equal(u_buf, v1), "u_buf bytes differ from V1 in-forward draw"


def test_fill_u_buf_varies_with_step_and_rollout():
    d_out, n_sample = 5, 2
    np_cfg = dict(global_seed=42, sample_method="gaussian")
    ext = WorkerExtension.__new__(WorkerExtension)
    a = torch.zeros(n_sample, d_out)
    b = torch.zeros(n_sample, d_out)
    c = torch.zeros(n_sample, d_out)
    ext._np_fill_u_buf(a, np_cfg, "L", step=0, rollout=0, n_sample=n_sample)
    ext._np_fill_u_buf(b, np_cfg, "L", step=1, rollout=0, n_sample=n_sample)
    ext._np_fill_u_buf(c, np_cfg, "L", step=0, rollout=1, n_sample=n_sample)
    assert not torch.equal(a, b)      # different step -> different noise
    assert not torch.equal(a, c)      # different rollout -> different noise
