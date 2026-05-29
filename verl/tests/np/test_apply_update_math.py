import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    PerturbedLinear,
    WorkerExtension,
    assemble_layer_delta,
)


def test_assemble_layer_delta_single_token():
    # 1 step, n_sample=2, d_out=3, d_in=2
    L_q_per_step = [torch.tensor([2.0, 4.0])]     # L_t^(q)
    L_clean_per_step = [1.0]
    u_per_step = [torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]])]  # [n_sample, d_out]
    x_per_step = [torch.tensor([1.0, 4.0])]       # [d_in]
    dw = assemble_layer_delta(
        L_q_per_step, L_clean_per_step, u_per_step, x_per_step,
        sigma=0.5, sample_mode="average", normalize=False, token_agg="sum",
    )
    # scales = (L_q - 1)/0.5 = [2, 6]; g = mean([2*u0, 6*u1]) ; dw = outer(g, x)
    scales = torch.tensor([(2.0 - 1.0) / 0.5, (4.0 - 1.0) / 0.5])
    g = (scales[:, None] * u_per_step[0]).mean(dim=0)
    assert torch.allclose(dw, torch.outer(g, x_per_step[0]), atol=1e-5)


def test_assemble_layer_delta_mean_token_agg_divides_by_T():
    L_q = [torch.tensor([1.0]), torch.tensor([1.0])]
    L_clean = [0.0, 0.0]
    u = [torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 1.0]])]   # d_out=2
    x = [torch.tensor([2.0]), torch.tensor([2.0])]                 # d_in=1
    dw_sum = assemble_layer_delta(L_q, L_clean, u, x, sigma=1.0, sample_mode="average",
                                  normalize=False, token_agg="sum")
    dw_mean = assemble_layer_delta(L_q, L_clean, u, x, sigma=1.0, sample_mode="average",
                                   normalize=False, token_agg="mean")
    assert torch.allclose(dw_mean, dw_sum / 2.0, atol=1e-6)


class _FakeLinear(torch.nn.Module):
    """Minimal vLLM-linear stand-in with a `weight` parameter."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d_out, d_in))

    def forward(self, x):
        return x @ self.weight.t(), None


def test_apply_node_update_does_gradient_descent_sign():
    """W must move OPPOSITE to delta_w (gradient DESCENT). See C1 in review."""
    base = _FakeLinear(2, 3)
    wrapped = PerturbedLinear(base, name="L", np_state_ref=lambda: {"mode": "off"})
    we = WorkerExtension.__new__(WorkerExtension)
    we.np_modules = {"L": wrapped}

    dw = torch.tensor([[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 2.0]])
    lr = 0.1
    norm_returned = we.apply_node_update("L", dw, lr=lr)

    # weight should equal -lr * dw (started at zero)
    assert torch.allclose(base.weight.data, -lr * dw, atol=1e-6), (
        f"sign error: weight={base.weight.data}, expected={-lr * dw}")
    assert norm_returned == float(dw.norm().item())
