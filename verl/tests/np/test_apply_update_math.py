import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import assemble_layer_delta


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
