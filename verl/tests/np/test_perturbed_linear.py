import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import PerturbedLinear


class FakeLinear(torch.nn.Module):
    """Stand-in for a vLLM linear: returns (output, bias) like ColumnParallelLinear."""
    def __init__(self, d_in, d_out, return_tuple=True):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d_out, d_in))
        self.return_tuple = return_tuple

    def forward(self, x):
        y = x @ self.w.t()
        return (y, None) if self.return_tuple else y


def _state(mode, **kw):
    s = {"mode": mode, "layer": "L", "global_seed": 0, "step": 0, "rollout": 0,
         "sigma": 1.0, "n_sample": 2, "sample_method": "gaussian",
         "n_clean_rows": 1, "captured_x": {}, "captured_u": {}}
    s.update(kw)
    return s


def test_off_mode_is_passthrough_tuple_preserved():
    base = FakeLinear(3, 3)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: _state("off"))
    x = torch.randn(4, 3)
    out = pl(x)
    assert isinstance(out, tuple) and out[1] is None
    assert torch.allclose(out[0], x)   # identity weight -> passthrough


def test_perturb_mode_adds_noise_only_to_perturbed_rows():
    base = FakeLinear(3, 3)
    st = _state("perturb", n_clean_rows=1, n_sample=2, sigma=5.0)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.zeros(3, 3)          # 1 clean row + 2 perturbed rows, all zero input
    out, _ = pl(x)
    assert torch.allclose(out[0], torch.zeros(3))   # clean row untouched
    assert not torch.allclose(out[1], torch.zeros(3))  # perturbed
    assert not torch.allclose(out[2], torch.zeros(3))
    # the two perturbed rows differ (independent u_q)
    assert not torch.allclose(out[1], out[2])


def test_capture_mode_records_clean_row_input():
    base = FakeLinear(3, 3)
    st = _state("capture", n_clean_rows=1)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])  # row0 clean, row1 perturbed copy
    pl(x)
    assert "L" in st["captured_x"]
    assert torch.allclose(st["captured_x"]["L"], torch.tensor([1.0, 2.0, 3.0]))


def test_sigma_zero_is_noop_in_perturb_mode():
    base = FakeLinear(3, 3)
    st = _state("perturb", sigma=0.0, n_sample=2)
    pl = PerturbedLinear(base, name="L", np_state_ref=lambda: st)
    x = torch.randn(3, 3)
    out, _ = pl(x)
    assert torch.allclose(out, x @ base.w.t())   # sigma=0 -> exact passthrough
