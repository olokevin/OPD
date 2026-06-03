import math
import torch
from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w


def test_sample_scale_average_is_forward_difference():
    # average: (L_q - L_clean) / sigma
    L_q = torch.tensor([2.0, 4.0])
    s = sample_scale(L_q, L_clean=1.0, sigma=0.5, mode="average")
    assert torch.allclose(s, torch.tensor([(2.0 - 1.0) / 0.5, (4.0 - 1.0) / 0.5]))


def test_sample_scale_grpo_is_mean_centered_over_sigma():
    # grpo: (L_q - mean_q) / sigma -- mean-centered advantage on the finite-
    # difference scale. The 1/std z-scoring is DROPPED (it self-amplifies into
    # divergence on low-signal tokens; see docs/results/zo_opd.md sec 5-6).
    L_q = torch.tensor([1.0, 2.0, 3.0])
    sigma = 0.1
    s = sample_scale(L_q, L_clean=None, sigma=sigma, mode="grpo")
    assert torch.allclose(s, (L_q - L_q.mean()) / sigma, atol=1e-5)


def test_accumulate_delta_w_rank1_outer_product_shape_and_value():
    # One token, n_sample=2, d_out=3, d_in=2.
    # u stacked [n_sample, d_out]; x_t is [d_in]; g_t = (1/n) sum_q scale_q * u_q  -> [d_out]
    # delta_w += g_t (outer) x_t  -> [d_out, d_in]
    d_out, d_in = 3, 2
    u = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]])      # [n_sample=2, d_out=3]
    scales = torch.tensor([1.0, 0.5])                          # per-sample scalar
    x_t = torch.tensor([1.0, 4.0])                             # [d_in=2]
    g_t = (scales[:, None] * u).mean(dim=0)                    # [d_out]
    expected = torch.outer(g_t, x_t)                           # [d_out, d_in]

    dw = torch.zeros(d_out, d_in)
    accumulate_delta_w(dw, scales=scales, u=u, x_t=x_t)
    assert dw.shape == (d_out, d_in)
    assert torch.allclose(dw, expected, atol=1e-6)


def test_accumulate_delta_w_anp_normalizes_per_sample():
    # With normalize=True each u_q is divided by ||u_q||^2 (ANP), clamped at eps.
    d_out, d_in = 4, 1
    u = torch.tensor([[3.0, 4.0, 0.0, 0.0]])   # ||u||^2 = 25
    scales = torch.tensor([1.0])
    x_t = torch.tensor([2.0])
    dw = torch.zeros(d_out, d_in)
    accumulate_delta_w(dw, scales=scales, u=u, x_t=x_t, normalize=True, eps=1e-6)
    g_t = (scales[:, None] * (u / 25.0)).mean(dim=0)
    assert torch.allclose(dw, torch.outer(g_t, x_t), atol=1e-6)
