"""Estimator math gates: loss forms (incl. the teacher-cancellation pitfall),
chunked-GEMM == naive outer-product loop, descent sign, ES unbiasedness on a
quadratic. CPU only."""
import torch

from verl.trainer.es_token.grad_estimator import (
    assemble_chunk, assemble_naive, rail_scales, sampled_token_losses)
from verl.trainer.es_token.signs import sign_rows


def test_sampled_token_losses_shapes_and_clean():
    T, N = 5, 4
    payload = torch.randn(T, 1 + N) - 2.0
    logq = torch.randn(T) - 2.0
    losses, clean = sampled_token_losses(payload, logq, "student_iw")
    assert losses.shape == (T, N) and clean.shape == (T,)
    assert torch.allclose(clean, payload[:, 0] - logq)


def test_student_iw_is_unbiased_single_sample_kl():
    """E_{y~p0}[ (p_n(y)/p0(y)) * (log p_n(y) - log q(y)) ] == KL(p_n || q)."""
    torch.manual_seed(0)
    V = 7
    logits0 = torch.randn(V)
    logits_n = logits0 + 0.1 * torch.randn(V)
    logits_q = torch.randn(V)
    p0 = torch.log_softmax(logits0, -1).exp()
    logpn = torch.log_softmax(logits_n, -1)
    logq = torch.log_softmax(logits_q, -1)
    # Enumerate the expectation over y ~ p0 of the IW rail loss.
    est = 0.0
    for y in range(V):
        payload = torch.tensor([[torch.log(p0[y]).item(), logpn[y].item()]])
        losses, _ = sampled_token_losses(payload, logq[y:y + 1], "student_iw",
                                         iw_clamp=None)
        est += p0[y].item() * losses[0, 0].item()
    kl = (logpn.exp() * (logpn - logq)).sum().item()
    assert abs(est - kl) < 1e-5


def test_weight_mode_none_cancels_teacher_in_scales():
    """The documented degeneracy: with w=1, the mean-baselined scales are
    INDEPENDENT of the teacher logq -- the update direction carries no teacher
    signal. (Why student_iw is the default.)"""
    T, N = 4, 8
    payload = torch.randn(T, 1 + N)
    qa, qb = torch.randn(T), torch.randn(T)
    la, _ = sampled_token_losses(payload, qa, "none")
    lb, _ = sampled_token_losses(payload, qb, "none")
    sa = rail_scales(la, payload[:, 0] - qa, 0.01, "mean_baseline")
    sb = rail_scales(lb, payload[:, 0] - qb, 0.01, "mean_baseline")
    assert torch.allclose(sa, sb, atol=1e-5)
    # ...whereas student_iw scales DO depend on the teacher.
    la_iw, ca = sampled_token_losses(payload, qa, "student_iw")
    lb_iw, cb = sampled_token_losses(payload, qb, "student_iw")
    sa_iw = rail_scales(la_iw, ca, 0.01, "mean_baseline")
    sb_iw = rail_scales(lb_iw, cb, 0.01, "mean_baseline")
    assert not torch.allclose(sa_iw, sb_iw, atol=1e-5)


def test_rail_scales_modes():
    losses = torch.tensor([[1.0, 2.0, 3.0]])
    clean = torch.tensor([1.5])
    s_mean = rail_scales(losses, clean, 0.5, "mean_baseline")
    assert torch.allclose(s_mean, torch.tensor([[-2.0, 0.0, 2.0]]))
    s_clean = rail_scales(losses, clean, 0.5, "clean_baseline")
    assert torch.allclose(s_clean, torch.tensor([[-1.0, 1.0, 3.0]]))


def test_assemble_chunk_matches_naive():
    torch.manual_seed(1)
    M, N, d_out, d_in = 6, 4, 32, 16
    S = sign_rows(N, d_out, seed=1)
    R = sign_rows(N, d_in, seed=2)
    u = torch.randn(M, d_out)
    v = torch.randn(M, d_in)
    scales = torch.randn(M, N)
    acc = torch.zeros(d_out, d_in)
    assemble_chunk(scales, u, v, S, R, acc)
    ref = assemble_naive(scales, u, v, S, R)
    assert torch.allclose(acc.double(), ref, atol=1e-4), (
        f"max abs diff {(acc.double() - ref).abs().max().item()}")


def test_assemble_chunk_accumulates_across_chunks():
    torch.manual_seed(2)
    M, N, d_out, d_in = 8, 2, 8, 8
    S = sign_rows(N, d_out, seed=3)
    R = sign_rows(N, d_in, seed=4)
    u, v, sc = torch.randn(M, d_out), torch.randn(M, d_in), torch.randn(M, N)
    whole = torch.zeros(d_out, d_in)
    assemble_chunk(sc, u, v, S, R, whole)
    halves = torch.zeros(d_out, d_in)
    assemble_chunk(sc[:4], u[:4], v[:4], S, R, halves)
    assemble_chunk(sc[4:], u[4:], v[4:], S, R, halves)
    assert torch.allclose(whole, halves, atol=1e-5)


def test_es_estimator_descends_a_quadratic():
    """End-to-end sign check on f(W) = 0.5||W - W*||^2_F: the assembled dW from
    forward differences must align with +grad f, so W - lr*dW reduces f."""
    torch.manual_seed(3)
    N, d_out, d_in, T = 8, 16, 16, 64
    sigma = 1e-3
    W = torch.zeros(d_out, d_in)
    W_star = torch.randn(d_out, d_in)
    S = sign_rows(N, d_out, seed=5)
    R = sign_rows(N, d_in, seed=6)

    def f(Wm):
        return 0.5 * ((Wm - W_star) ** 2).sum()

    acc = torch.zeros(d_out, d_in)
    for t in range(T):
        u = (torch.randint(0, 2, (d_out,)).float() * 2 - 1)
        v = (torch.randint(0, 2, (d_in,)).float() * 2 - 1)
        losses = torch.empty(1, N)
        for n in range(N):
            dW = sigma * torch.outer(S[n] * u, R[n] * v)
            losses[0, n] = f(W + dW)
        clean = torch.tensor([f(W)])
        sc = rail_scales(losses, clean, sigma, "mean_baseline")
        assemble_chunk(sc, u[None, :], v[None, :], S, R, acc)
    dW_hat = acc / (N * T)
    g_true = W - W_star
    cos = torch.nn.functional.cosine_similarity(
        dW_hat.flatten(), g_true.flatten(), dim=0).item()
    assert cos > 0.5, f"cos(dW_hat, grad) = {cos}"
    assert f(W - 0.5 * dW_hat) < f(W)
