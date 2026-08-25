"""ESTokenLinear math gate: the in-layer rank-1 rail op must equal a dense
forward through the materialized perturbed weight (W + dW_n) for each rail,
while the clean rows stay byte-identical to W x. CPU only."""
import torch

from verl.trainer.es_token.seeding import build_noise_layout
from verl.trainer.es_token.signs import sign_rows
from verl.workers.rollout.vllm_rollout.es_token_worker_extension import (
    ESTokenLinear)


def _build(bucket=2, n_rails=4, d_out=16, d_in=8, sigma=0.05):
    torch.manual_seed(0)
    lin = torch.nn.Linear(d_in, d_out, bias=False)
    layout, d_total = build_noise_layout([("L", d_out, d_in)])
    S = sign_rows(n_rails, d_out, seed=1)
    R = sign_rows(n_rails, d_in, seed=2)
    noise_buf = torch.randn(bucket, d_total)
    width = 1 + n_rails
    clean_row_idx = torch.tensor([p * width for p in range(bucket)])
    perturbed_row_idx = torch.tensor(
        [p * width + 1 + n for p in range(bucket) for n in range(n_rails)])
    rail_idx = torch.tensor(
        [n for _ in range(bucket) for n in range(n_rails)])
    prompt_idx = torch.tensor(
        [p for p in range(bucket) for _ in range(n_rails)])
    st = {
        "mode": "perturb_es",
        "es_noise_buf": noise_buf,
        "es_layout": layout,
        "es_signs": {"L": (S, R)},
        "es_sigma_buf": {"L": torch.tensor([sigma])},
        "perturbed_row_idx": perturbed_row_idx,
        "clean_row_idx": clean_row_idx,
        "es_rail_idx": rail_idx,
        "es_prompt_idx": prompt_idx,
    }
    wrapped = ESTokenLinear(lin, "L", lambda: st)
    return wrapped, lin, st, S, R, noise_buf, layout, sigma, width


def test_rank1_op_equals_dense_perturbed_weight():
    bucket, n_rails, d_out, d_in = 2, 4, 16, 8
    wrapped, lin, st, S, R, noise_buf, layout, sigma, width = _build(
        bucket, n_rails, d_out, d_in)
    R_rows = bucket * width
    x = torch.randn(R_rows, d_in)
    y = wrapped(x)
    y_clean_ref = lin(x)   # batch forward: clean rows must be BIT-identical

    off_u, _, off_v, _ = layout["L"]
    for p in range(bucket):
        u = noise_buf[p, off_u:off_u + d_out]
        v = noise_buf[p, off_v:off_v + d_in]
        # clean row: untouched (compare against the same batched-GEMM path)
        c = p * width
        assert torch.equal(y[c], y_clean_ref[c])
        for n in range(n_rails):
            row = p * width + 1 + n
            dW = sigma * torch.outer(S[n] * u, R[n] * v)
            ref = x[row] @ (lin.weight + dW).t()
            assert torch.allclose(y[row], ref, atol=1e-5), (
                f"rail {n} slot {p}: max diff "
                f"{(y[row] - ref).abs().max().item()}")


def test_sigma_zero_is_noop_and_off_mode_passthrough():
    wrapped, lin, st, *_ = _build(sigma=0.0)
    x = torch.randn(10, 8)
    assert torch.allclose(wrapped(x), lin(x), atol=0)
    st["mode"] = "off"
    st["es_sigma_buf"]["L"].fill_(0.7)
    assert torch.equal(wrapped(x), lin(x))


def test_tuple_output_repack():
    """vLLM linears may return (tensor, bias); the wrapper must repack."""
    class TupleLinear(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.weight = lin.weight
            self._lin = lin

        def forward(self, x):
            return self._lin(x), None

    wrapped, lin, st, *_ = _build()
    tlin = TupleLinear(lin)
    wrapped2 = ESTokenLinear(tlin, "L", lambda: st)
    x = torch.randn(2 * 5, 8)
    out = wrapped2(x)
    assert isinstance(out, tuple) and out[1] is None
    assert torch.allclose(out[0], wrapped(x), atol=1e-6)
