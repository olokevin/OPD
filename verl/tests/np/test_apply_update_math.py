import pytest
import torch
from verl.trainer.np.grad_estimator import accumulate_delta_w, sample_scale
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    PerturbedLinear,
    WorkerExtension,
    assemble_layer_delta,
)


def _assemble_ref_cpu_loop(L_q, L_clean, u, x, sigma, sample_mode, normalize,
                           token_agg, eps=1e-6):
    """The original per-token CPU torch.outer accumulation -- the parity oracle
    the batched (GEMM) assemble_layer_delta must reproduce. Kept inline in the
    test so the production path can drop the slow loop entirely."""
    d_out = u[0].shape[1]
    d_in = x[0].shape[0]
    dw = torch.zeros(d_out, d_in, dtype=torch.float32)
    T = max(len(L_q), 1)
    for Lq, Lc, uu, xx in zip(L_q, L_clean, u, x):
        scales = sample_scale(Lq.float(), Lc, sigma, sample_mode)
        accumulate_delta_w(dw, scales=scales, u=uu.float(), x_t=xx.float(),
                           normalize=normalize, eps=eps)
    if token_agg == "mean":
        dw.div_(T)
    return dw


@pytest.mark.parametrize("sample_mode", ["average", "grpo"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("token_agg", ["sum", "mean"])
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_batched_assemble_matches_cpu_loop(sample_mode, normalize, token_agg, device):
    """The GEMM-batched assemble_layer_delta must match the per-token CPU loop
    across every (sample_mode, normalize, token_agg) combination, on CPU and GPU.
    This is the parity gate for moving the 5-min CPU-bound assemble onto the GPU."""
    torch.manual_seed(0)
    T, n_sample, d_out, d_in = 37, 8, 64, 48
    sigma = 0.01
    L_q = [torch.randn(n_sample) for _ in range(T)]
    L_clean = [float(torch.randn(())) for _ in range(T)]
    u = [torch.randn(n_sample, d_out) for _ in range(T)]
    x = [torch.randn(d_in) for _ in range(T)]

    ref = _assemble_ref_cpu_loop(L_q, L_clean, u, x, sigma, sample_mode,
                                 normalize, token_agg)
    got = assemble_layer_delta(L_q, L_clean, u, x, sigma=sigma,
                               sample_mode=sample_mode, normalize=normalize,
                               token_agg=token_agg, device=device)
    assert got.device.type == "cpu", "assemble must return a CPU tensor"
    # CPU sequential rank-1 adds vs the GPU GEMM reduce in a different float32
    # order, so parity is at fp32-epsilon, NOT bit-identity. The scale-free
    # metric is the relative Frobenius norm (~1.4e-7 measured across all modes);
    # absolute atol must track the (sum-agg) magnitude, hence atol=1e-3.
    rel_fro = (got - ref).norm() / (ref.norm() + 1e-12)
    assert rel_fro < 1e-5, (
        f"batched assemble diverged from CPU loop "
        f"(mode={sample_mode} norm={normalize} agg={token_agg} dev={device}): "
        f"rel_fro={rel_fro:.2e} max|d|={(got - ref).abs().max():.2e}")
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-4)


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
