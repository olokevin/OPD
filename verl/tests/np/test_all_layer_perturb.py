import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import PerturbedLinear
from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
from verl.trainer.np.seeding import noise_seed, draw_noise


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


def test_alloc_layer_buffers_shapes():
    import torch
    from verl.workers.rollout.vllm_rollout.np_worker_extension import _alloc_layer_buffers
    class W:  # fake wrapped linear with weight [d_out,d_in]
        def __init__(s, o, i): s.wrapped = type("x", (), {"weight": torch.zeros(o, i)})()
    mods = {"L0": W(64, 48), "L1": W(32, 96)}
    u, x = _alloc_layer_buffers(mods, n_sample=8, device=torch.device("cpu"))
    assert u["L0"].shape == (8, 64) and x["L1"].shape == (96,)


def test_all_layer_packed_scatter_branch():
    """Packed (pri set) branch of perturb_all_layers on CPU: every layer reads
    its OWN per-layer u_buf/x_buf dict, perturbed rows are SCATTERED across prompt
    blocks and get +sigma*u, clean rows are untouched, and each layer's x_buf
    captures the clean rows' inputs. Mirrors the perturb_graph packed scatter test
    but with per-layer dicts (the all-layer packed graph contract, Stage E1).

    2 prompts, n_sample=2 -> R=6 rows (prompt-major: [c0,p0a,p0b,c1,p1a,p1b])."""
    d = 4
    b_pack, n_sample = 2, 2
    clean_row_idx = torch.tensor([0, 3], dtype=torch.long)
    perturbed_row_idx = torch.tensor([1, 2, 4, 5], dtype=torch.long)
    # Per-layer u_buf [b_pack*n_sample, d_out]; row i aligns with pri[i].
    u_buf = {
        "L0": torch.arange(1, 1 + 4 * d, dtype=torch.float32).reshape(4, d),
        "L1": torch.arange(100, 100 + 4 * d, dtype=torch.float32).reshape(4, d),
    }
    x_buf = {"L0": torch.zeros(b_pack, d), "L1": torch.zeros(b_pack, d)}
    st = {
        "mode": "perturb_all_layers", "sigma": 2.0, "n_clean_rows": 1,
        "u_buf": u_buf, "x_buf": x_buf,
        "clean_row_idx": clean_row_idx,
        "perturbed_row_idx": perturbed_row_idx,
    }
    pl0 = PerturbedLinear(_FakeLinear(d, d), "L0", lambda: st)  # identity -> y==x
    pl1 = PerturbedLinear(_FakeLinear(d, d), "L1", lambda: st)
    # distinct per-row inputs so clean-row survival is verifiable.
    x = torch.arange(6 * d, dtype=torch.float32).reshape(6, d)
    y0, _ = pl0(x)
    y1, _ = pl1(x)

    # (a) perturbed rows got + sigma*u for THIS layer (row pri[i] += u[i]).
    for i, r in enumerate(perturbed_row_idx.tolist()):
        assert torch.allclose(y0[r], x[r] + 2.0 * u_buf["L0"][i]), f"L0 row {r}"
        assert torch.allclose(y1[r], x[r] + 2.0 * u_buf["L1"][i]), f"L1 row {r}"
    # (b) clean rows are UNCHANGED by the perturbation at both layers.
    for r in clean_row_idx.tolist():
        assert torch.allclose(y0[r], x[r]), f"L0 clean row {r} perturbed"
        assert torch.allclose(y1[r], x[r]), f"L1 clean row {r} perturbed"
    # (c) each layer's x_buf holds the clean rows' inputs (x[clean_row_idx]).
    assert torch.allclose(x_buf["L0"], x[clean_row_idx])
    assert torch.allclose(x_buf["L1"], x[clean_row_idx])


def test_all_layer_contiguous_path_still_taken_when_no_pri():
    """Regression guard for C-2's sibling: when perturbed_row_idx is ABSENT (or
    None) the contiguous single-prompt path runs (the existing eager all-layer
    decode relies on st.get returning None -> slice form). Mirrors the production
    state which sets perturbed_row_idx=None defensively."""
    st = _alllayer_state(sigma=2.0)
    st["perturbed_row_idx"] = None      # explicit None -> contiguous path
    st["clean_row_idx"] = None
    st["u_buf"]["L0"] = torch.arange(8.0).reshape(2, 4)
    pl0 = PerturbedLinear(_FakeLinear(4, 4), "L0", lambda: st)
    x = torch.arange(12.0).reshape(3, 4)  # 1 clean + 2 perturbed (contiguous)
    y0, _ = pl0(x)
    assert torch.allclose(y0[0], x[0])    # clean row untouched
    for q in range(2):
        assert torch.allclose(y0[1 + q], x[1 + q] + 2.0 * st["u_buf"]["L0"][q])
    assert torch.allclose(st["x_buf"]["L0"], x[0])  # contiguous: x_buf is [d_in]


def test_fill_u_buf_all_layers_independent_per_layer():
    we = WorkerExtension.__new__(WorkerExtension)
    u = {"L0": torch.zeros(2, 6), "L1": torch.zeros(2, 6)}
    cfg = dict(global_seed=42, sample_method="gaussian")
    we._np_fill_u_buf_all_layers(u, cfg, ["L0", "L1"], step=0, rollout=0, n_sample=2)
    # bit-identical to the per-layer seed formula
    exp = draw_noise(noise_seed(42, 0, "L0", 0, 0), (6,), torch.device("cpu"),
                     torch.float32, "gaussian")
    assert torch.equal(u["L0"][0], exp)
    # different layer -> different seed -> different noise
    assert not torch.equal(u["L0"][0], u["L1"][0])
