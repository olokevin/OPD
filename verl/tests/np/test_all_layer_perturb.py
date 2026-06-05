import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import PerturbedLinear


class _FakeLinear(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(d_out, d_in))  # identity-ish: y == x when square
    def forward(self, x):
        return x @ self.weight.t(), None


def _alllayer_state(sigma=1.0, n_sample=2):
    # all-layer mode: u_buf / x_buf are DICTS keyed by layer name.
    return {
        "mode": "perturb_all_layers", "sigma": sigma, "n_clean_rows": 1,
        "u_buf": {"L0": torch.zeros(n_sample, 4), "L1": torch.zeros(n_sample, 4)},
        "x_buf": {"L0": torch.zeros(4), "L1": torch.zeros(4)},
    }


def test_all_layer_applies_own_u_per_layer():
    st = _alllayer_state(sigma=2.0)
    st["u_buf"]["L0"] = torch.arange(8.0).reshape(2, 4)
    st["u_buf"]["L1"] = torch.arange(8.0, 16.0).reshape(2, 4)
    pl0 = PerturbedLinear(_FakeLinear(4, 4), "L0", lambda: st)
    pl1 = PerturbedLinear(_FakeLinear(4, 4), "L1", lambda: st)
    x = torch.arange(12.0).reshape(3, 4)  # 1 clean + 2 perturbed
    y0, _ = pl0(x)
    y1, _ = pl1(x)
    # clean row (0) unchanged at both layers
    assert torch.allclose(y0[0], x[0]) and torch.allclose(y1[0], x[0])
    # perturbed rows got THIS layer's u (y == x for identity weight, then += sigma*u)
    for q in range(2):
        assert torch.allclose(y0[1 + q], x[1 + q] + 2.0 * st["u_buf"]["L0"][q])
        assert torch.allclose(y1[1 + q], x[1 + q] + 2.0 * st["u_buf"]["L1"][q])
    # each layer captured its OWN clean-row input
    assert torch.allclose(st["x_buf"]["L0"], x[0]) and torch.allclose(st["x_buf"]["L1"], x[0])


def test_all_layer_sigma_zero_is_noop():
    st = _alllayer_state(sigma=0.0)
    st["u_buf"]["L0"] = torch.randn(2, 4)
    pl0 = PerturbedLinear(_FakeLinear(4, 4), "L0", lambda: st)
    x = torch.arange(12.0).reshape(3, 4)
    y0, _ = pl0(x)
    assert torch.allclose(y0, x)  # 0*u -> exact passthrough


def test_clean_and_perturbed_rows_disjoint_invariant():
    # estimator validity: clean row never receives perturbation
    st = _alllayer_state(sigma=5.0)
    st["u_buf"]["L0"] = torch.ones(2, 4)
    pl0 = PerturbedLinear(_FakeLinear(4, 4), "L0", lambda: st)
    x = torch.zeros(3, 4)
    y0, _ = pl0(x)
    assert torch.allclose(y0[0], torch.zeros(4))  # clean row untouched
    assert (y0[1:] != 0).any()                     # perturbed rows changed
