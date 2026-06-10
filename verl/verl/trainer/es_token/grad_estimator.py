"""es_token gradient estimation (pure math; no vLLM/GPU coupling).

Chain (see docs/plans/es_token_trainer.md §2):
  1. sampled_token_losses: per-(token, rail) loss l_{n,t} from the decode-stored
     student logprobs of the clean sampled token + the teacher's logprob of it.
  2. rail_scales: s~_{n,t} = (l_{n,t} - baseline_t) / sigma.
  3. assemble: delta_W = (1/N) sum_{t,n} s~_{n,t} (s_n (.) u_t)(r_n (.) v_t)^T,
     approximating +dL/dW (descent: W <- W - lr * delta_W).

Sampled-token rail loss -- the importance-weighted single-sample reverse-KL
estimate (the teacher term must stay rail-coupled or it cancels in the rail
baseline-difference; see the plan §2.3 pitfall):
    l_{n,t} = w_{n,t} * (log pi_n(y_t) - log q(y_t)),
    w = exp(logp_n - logp_0)  ("student_iw", default; E_y[l] = KL(pi_n || q))
      | exp(logp_n)           ("student_p", NP reverse_kl_topk k=1 analog)
      | 1                     ("none", DEGENERATE -- teacher cancels; ablation only)
"""
from typing import Tuple

import torch


def sampled_token_losses(
    payload: torch.Tensor,      # [T, 1+N] student logprobs of the clean sampled
                                #          token (col 0 = clean rail)
    teacher_logq: torch.Tensor, # [T] teacher logprob of the clean sampled token
    weight_mode: str = "student_iw",
    iw_clamp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (losses [T, N], clean_loss [T]). Minimization-oriented."""
    payload = payload.float()
    teacher_logq = teacher_logq.float()
    logp0 = payload[:, 0:1]                    # [T, 1]
    logpn = payload[:, 1:]                     # [T, N]
    diff_n = logpn - teacher_logq[:, None]     # [T, N] log pi_n - log q
    diff_0 = payload[:, 0] - teacher_logq      # [T]
    if weight_mode == "student_iw":
        w = (logpn - logp0).exp()
        if iw_clamp is not None:
            w = w.clamp(max=float(iw_clamp))
        losses = w * diff_n
        clean = diff_0                          # iw weight is exactly 1 at clean
    elif weight_mode == "student_p":
        losses = logpn.exp() * diff_n
        clean = payload[:, 0].exp() * diff_0
    elif weight_mode == "none":
        losses = diff_n
        clean = diff_0
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode!r}")
    return losses, clean


def rail_scales(
    losses: torch.Tensor,       # [T, N]
    clean_loss: torch.Tensor,   # [T]
    sigma: float,
    mode: str = "mean_baseline",
) -> torch.Tensor:
    """s~_{n,t} = (l_{n,t} - baseline_t) / sigma  ->  [T, N].

    mean_baseline (default): baseline = mean over the N rails (the NP `grpo`
    form; the 1/std z-scoring stays dropped -- it self-amplified to divergence,
    docs/results/zo_opd.md §5-6).
    clean_baseline: baseline = the clean rail's loss (one-sided FD vs rail 0).
    """
    losses = losses.float()
    if mode == "mean_baseline":
        base = losses.mean(dim=1, keepdim=True)
    elif mode == "clean_baseline":
        base = clean_loss.float()[:, None]
    else:
        raise ValueError(f"unknown grad_estimate_sample mode: {mode!r}")
    return (losses - base) / float(sigma)


def assemble_chunk(
    scales: torch.Tensor,   # [M, N] rail scales for M token-records
    u: torch.Tensor,        # [M, d_out] base noise (output side)
    v: torch.Tensor,        # [M, d_in]  base noise (input side)
    S: torch.Tensor,        # [N, d_out] fixed sign rails (output side)
    R: torch.Tensor,        # [N, d_in]  fixed sign rails (input side)
    out: torch.Tensor,      # [d_out, d_in] fp32 accumulator (in/out)
) -> None:
    """out += sum_n ((u * scales[:, n]) (.) S[n])^T @ (v (.) R[n]).

    N GEMMs with inner dim M; mathematically identical to summing the
    per-(record, rail) rank-1 outer products. GEMMs run in the inputs' dtype
    (bf16 on GPU -> fp32 tensor-core accumulation inside the GEMM); the chunk
    result is accumulated into the fp32 `out`.
    """
    n_rails = scales.shape[1]
    sc = scales.to(u.dtype)
    for n in range(n_rails):
        un = (u * sc[:, n:n + 1]) * S[n][None, :]   # [M, d_out]
        vn = v * R[n][None, :]                      # [M, d_in]
        out.add_((un.t() @ vn).float())


def assemble_naive(
    scales: torch.Tensor,   # [M, N]
    u: torch.Tensor,        # [M, d_out]
    v: torch.Tensor,        # [M, d_in]
    S: torch.Tensor,        # [N, d_out]
    R: torch.Tensor,        # [N, d_in]
) -> torch.Tensor:
    """Reference per-(record, rail) outer-product loop (tests only)."""
    M, N = scales.shape
    d_out, d_in = u.shape[1], v.shape[1]
    dw = torch.zeros(d_out, d_in, dtype=torch.float64)
    for j in range(M):
        for n in range(N):
            left = (S[n] * u[j]).double()
            right = (R[n] * v[j]).double()
            dw += float(scales[j, n]) * torch.outer(left, right)
    return dw
