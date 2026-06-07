# Fully-CUDA-Graphed All-Layer ZO-NP Trainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the NP (node-perturbation / zeroth-order OPD) trainer so it perturbs **all matched linear layers simultaneously** in **one fully-CUDA-graphed packed decode**, scores the teacher with **batched top-k** reverse-KL, and assembles **all layers' δW in one batched GPU pass** — then prove a single update step is **≤ BP-OPD wall-clock** at batch=64 / max_tokens=1024.

**Architecture:** Each decode token runs `1 + N` rows (1 clean + N perturbed). Every perturbed row q is perturbed at *every* matched linear via an independent per-(layer, q) buffer add `y += σ·u_buf[layer][q]`; each layer captures its own clean-row input `x[layer]` in the same forward. The combined loss `L_q` is attributed back per layer via `dW^layer = outer(mean_q[(L_q−L_clean)/σ · u^layer_q], x^layer_q)` (node-perturbation trick; unbiased to first order because noise is independent per (layer, q) and x is captured per layer). The `(1+N)·B_pack`-row step forward is captured into a CUDA graph at a few **fixed bucket widths** and replayed per token; prompts that hit EOS become **PAD rows** (slot=−1, masked, outputs ignored) padded up to the nearest captured bucket — exactly how vLLM captures its own decode graphs (`cudagraph_capture_sizes` + pad-up), so the graph shape never changes and there is no per-token recapture.

**Tech Stack:** vLLM 0.11.0 (V1 engine, FLASH_ATTN, `enforce_eager`), PyTorch CUDA graphs (`torch.cuda.CUDAGraph`), Ray single-controller, Hydra. Student Qwen3-1.7B + teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500. conda env `verl` (`/home/yequan/miniconda3/envs/verl/bin/python`).

---

## Context the implementer needs (read before starting)

**All paths from repo root `/home/yequan/Project/compression/OPD`. Run tests/commands from repo root. verl python: `/home/yequan/miniconda3/envs/verl/bin/python`. GPU gates: `cd verl && CUDA_VISIBLE_DEVICES=<free 4-7> NP_KEEP_CUDA_VISIBLE=1 <python> ...`. Honor: one job per GPU, GPUs 4–7 only.**

**Files:**
- `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` — the vLLM `WorkerExtension`: `PerturbedLinear` (46–129), `install_perturb_layers` (138–159), decode drivers `run_np_decode` (169–215) / `run_np_decode_graphed` (251–344) / `run_np_decode_packed` (346–459), eager forward `_np_run_forward` (897–912), single-prompt graph `_np_capture_step` (558–659) / `_np_replay_step` (661–703) / `_np_build_attn_metadata_persistent` (705+), noise refill `_np_fill_u_buf` (229–250), assemble `assemble_layer_delta`/`assemble_and_apply` (1158–1227). **Most new code lands here.**
- `verl/verl/trainer/np/ray_trainer.py` — `RayNPTrainer.fit()` loop (506–649); `decode_mode` validation (~491); `NPNcclLLM` forces `enforce_eager` (31–46).
- `verl/verl/trainer/np/grad_estimator.py` — `sample_scale` (13–43), `accumulate_delta_w` (46–60). **Do not change the math.**
- `verl/verl/trainer/np/teacher_scorer.py` — `TeacherScorer.score_rollout` (55–96), `_select_ids` (98–132), `_teacher_topk_logprobs` (134–155).
- `verl/verl/trainer/np/layer_resolve.py` — `resolve_modules` (returns forward-execution order, de-duped), `active_layers_for_step` (33–39).
- `verl/verl/trainer/np/seeding.py` — `noise_seed(global_seed, step, layer, rollout, q)` (blake2b key **includes layer** → independent noise per layer automatically), `draw_noise`. **Do not change.**
- `verl/verl/trainer/config/np_trainer.yaml` — Hydra config.
- `scripts/zo_opd/opd_math_np.sh` — env→Hydra launcher.
- `scripts/zo_opd/bench_np_vs_bp.sh` — the canonical one-step NP-vs-BP wall-clock harness (currently hard-codes `DECODE_MODE=packed`).
- `verl/tests/np/` — CPU pytest suite (no GPU). **All new unit tests go here** (NOT a top-level `tests/`). Mirror `test_perturb_graph.py` / `test_apply_update_math.py` style (`FakeLinear` identity + plain `np_state` dict).

**Locked design decisions (do not re-litigate):**
1. **Row layout = (1+N) shared.** Each perturbed row perturbed at *every* layer with independent per-(layer,q) noise; per-layer dW from that layer's own (u, x). N stays 8–16 (memory-bound "N-free" regime: +19% at N=8). NOT (1+N·L).
2. **Goal = beat/match BP one-step speed.** Teacher-score and assemble must be optimized too, not just decode.
3. **EOS under fixed-shape graph = vLLM's trick.** Capture at fixed bucket widths (`b_pack_buckets`, e.g. `[2,4,8,16]`); each token pad active prompts up to the nearest bucket; finished prompts' rows → PAD (slot=−1, masked, outputs ignored); never recapture within a bucket.

**Three correctness landmines the critique surfaced — guardrails baked into the tasks below:**
- **C-1:** Top-k GPU slicing must store the **union of student-top-k and teacher-top-k ids** (or a large-enough `k` window), or `union`/`intersection`/`teacher_p` strategies silently degrade to `only_stu`. Production default is `only_stu` (lossless), but the plan keeps the others correct by storing a wider window + a dedicated test with a teacher id outside the student top-k.
- **C-2/C-8:** `PerturbedLinear.forward` reads `st["u_buf"][self.name]`; the graph capture must pin that **same dict object** for the graph's life — forward and capture must agree on where the per-layer buffers live, or the captured graph perturbs nothing.
- **C-4:** Bucket-padded (finished) rows must carry **valid** `seq_lens`/`positions` (not stale/zero), and staggered-EOS bucket-crossing must be parity-tested against the eager oracle — not just the shallow "one prompt inactive" case.

**Estimator validity invariant (must hold in every all-layer task):** perturbation is added to layer **output `y`**, never input `x`; the clean row (index 0 per prompt) is never in the perturbed-row set, so `x[layer]` captured from the clean row is the genuine unperturbed input at that layer. Test: `clean_row_idx ∩ perturbed_row_idx = ∅`.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `np_worker_extension.py` | Modify | All-layer `PerturbedLinear` mode; `_np_fill_u_buf_all_layers`; per-layer captured u/x dicts; top-k logit slice; `assemble_all_layers_and_apply`; `_np_capture_step_packed`/`_np_replay_step_packed`/`run_np_decode_packed_graphed`; extend `run_np_decode*` to a layer-name list. |
| `teacher_scorer.py` | Modify | Vectorized top-k `score_rollout` over `[T,1+N,k]`; batched teacher prefill over B prompts; top-k id-window handling. |
| `ray_trainer.py` | Modify | All-layer `fit()` branch: one decode → one score → one all-layer assemble per step; `packed_graphed` decode_mode; allow-list. |
| `layer_resolve.py` | Modify | `active_layers_for_step` returns all matched in all-layer mode (keep forward order — never sort). |
| `np_trainer.yaml` | Modify | `decode_mode: packed_graphed`, `b_pack_buckets`, `teacher_batch_size`, `topk_store_k`. |
| `opd_math_np.sh` | Modify | Env plumbing for the new knobs + `EN_LAYERWISE=false`. |
| `bench_np_vs_bp.sh` | Modify | Re-point NP side at `packed_graphed` all-layer. |
| `verl/tests/np/test_all_layer_*.py`, `test_teacher_scorer.py`, `test_apply_update_math.py` | Create/Modify | CPU unit + parity tests. |
| `scripts/zo_opd/np_checks/check_alllayer_*.py` | Create | GPU gates (σ=0, parity, staggered-EOS, util). |
| `docs/wiki/zo_np_trainer.md`, `docs/index.md`, `docs/log.md` | Modify | Closeout ingest. |

---

# STAGE A — All-layer perturbation forward (foundation; everything depends on it)

### Task A1: CPU test — multi-layer perturb in one forward

**Files:** Create `verl/tests/np/test_all_layer_perturb.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_all_layer_perturb.py -q`
Expected: FAIL — `perturb_all_layers` mode not handled; `PerturbedLinear.forward` returns `out` unchanged (no `mode=="perturb_all_layers"` branch).

- [ ] **Step 3: Implement (Task A2 makes these pass).** Proceed to A2; rerun after.

- [ ] **Step 4: Commit** (after A2 passes)

```bash
git add verl/tests/np/test_all_layer_perturb.py
git commit -m "np: failing CPU tests for all-layer perturbation forward"
```

### Task A2: `PerturbedLinear.forward` all-layer mode

**Files:** Modify `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` (PerturbedLinear.forward, 61–129)

- [ ] **Step 1: Add the all-layer branch.** Insert BEFORE the existing `if mode == "perturb_graph" ...` block (line 76), a new branch that handles every matched layer with its own buffer dict. It must NOT gate on `self.name == st["layer"]`:

```python
        if mode == "perturb_all_layers":
            # Every matched layer perturbs with ITS OWN buffer slice and captures
            # ITS OWN clean-row input, all in this one forward. u_buf/x_buf are
            # dicts keyed by layer name (pinned by the caller / graph capture).
            # Perturbation is added to OUTPUT y, never input x, so the clean row
            # (index 0) stays the genuine unperturbed input at this layer.
            u_buf = st["u_buf"][self.name]          # [n_sample, d_out]
            x_buf = st["x_buf"][self.name]          # [d_in]
            sigma = st["sigma"]
            n_clean = st["n_clean_rows"]
            x_buf.copy_(x[0])
            y[n_clean:n_clean + u_buf.shape[0]] = (
                y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            return _repack(y, bias, was_tuple)
```

(Note: for the packed/graphed path the scatter form `y[pri] += σ·u_buf` and `x_buf.copy_(x[cri])` mirrors the existing `perturb_graph` packed branch — added in Stage E. For the single-prompt all-layer eager path the contiguous slice above is correct.)

- [ ] **Step 2: Run the A1 tests**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_all_layer_perturb.py -q`
Expected: PASS (3 passed).

- [ ] **Step 3: Confirm no regression**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/ -q`
Expected: all pass (existing single-layer `perturb`/`perturb_graph`/`capture` modes untouched — the new branch is additive).

- [ ] **Step 4: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_all_layer_perturb.py
git commit -m "np: PerturbedLinear perturb_all_layers mode (per-layer u_buf/x_buf dict, additive)"
```

### Task A3: `_np_fill_u_buf_all_layers` (one canonical copy)

**Files:** Modify `np_worker_extension.py` (add after `_np_fill_u_buf`, ~250); Create test in `verl/tests/np/test_all_layer_perturb.py`

- [ ] **Step 1: Write the failing test** (append to `test_all_layer_perturb.py`)

```python
from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
from verl.trainer.np.seeding import noise_seed, draw_noise


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
```

- [ ] **Step 2: Run to verify it fails** — `... -m pytest verl/tests/np/test_all_layer_perturb.py::test_fill_u_buf_all_layers_independent_per_layer -q` → FAIL (`AttributeError: _np_fill_u_buf_all_layers`).

- [ ] **Step 3: Implement**

```python
    def _np_fill_u_buf_all_layers(self, u_buf_dict, np_cfg, matched_layers,
                                  step, rollout, n_sample):
        """Refill every matched layer's u_buf with independent noise per (layer,q),
        seeded identically to V1's single-layer draw -> parity by construction."""
        for layer_name in matched_layers:
            buf = u_buf_dict[layer_name]
            d_out = buf.shape[1]
            for q in range(n_sample):
                seed = noise_seed(int(np_cfg["global_seed"]), int(step),
                                  layer_name, int(rollout), q)
                u = draw_noise(seed, (d_out,), buf.device, buf.dtype,
                               np_cfg["sample_method"])
                buf[q].copy_(u)
```

- [ ] **Step 4: Run to verify it passes** — same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_all_layer_perturb.py
git commit -m "np: _np_fill_u_buf_all_layers (independent per-(layer,q) noise, parity seeds)"
```

### Task A4: `active_layers_for_step` returns all matched in forward order (NO sort)

**Files:** Modify `verl/verl/trainer/np/layer_resolve.py`; test `verl/tests/np/test_layer_resolve.py`

- [ ] **Step 1: Write the test** (append to existing `test_layer_resolve.py`)

```python
from verl.trainer.np.layer_resolve import active_layers_for_step

def test_all_layer_mode_returns_all_in_forward_order():
    matched = ["model.layers.2.mlp.down_proj", "model.layers.10.mlp.down_proj"]
    # en_layerwise=False already returns all matched, in the SAME order (NOT sorted)
    out = active_layers_for_step(matched, step=0, en_layerwise=False)
    assert out == matched  # forward/resolve order preserved; layers.2 before layers.10
```

- [ ] **Step 2: Run** — `... -m pytest verl/tests/np/test_layer_resolve.py -q`. Expected: PASS (the existing `en_layerwise=False` path at `layer_resolve.py:39` already returns `list(matched)` in order). **No code change needed** — this task asserts the existing behavior is correct and forbids the "sort" anti-pattern. If it passes, just commit the test.

- [ ] **Step 3: Commit**

```bash
git add verl/tests/np/test_layer_resolve.py
git commit -m "np: assert all-layer mode preserves forward order (guard against sorting)"
```

### Task A5: CPU parity — all-layer signals == single-layer V1 per layer

**Files:** Create `verl/tests/np/test_all_layer_parity.py`

- [ ] **Step 1: Write the parity test.** Run a 3-layer toy through both: (a) the all-layer forward (all 3 perturbed in one pass), and (b) three separate single-layer `perturb_graph` forwards. For each layer, with the SAME seeds, the captured `u` must be bit-identical and (because perturbation is added to output, clean-row x is the same in both) `x` identical. Assert per-layer `u` equality via `torch.equal`.

```python
import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    PerturbedLinear, WorkerExtension)

def test_all_layer_u_matches_single_layer_seed():
    we = WorkerExtension.__new__(WorkerExtension)
    layers = ["L0", "L1", "L2"]
    cfg = dict(global_seed=7, sample_method="bernoulli")
    # all-layer fill
    u_all = {n: torch.zeros(4, 8) for n in layers}
    we._np_fill_u_buf_all_layers(u_all, cfg, layers, step=3, rollout=1, n_sample=4)
    # single-layer fill, one layer at a time (V1 seed path)
    for n in layers:
        u_one = torch.zeros(4, 8)
        we._np_fill_u_buf_all_layers(u_one_dict := {n: u_one}, cfg, [n],
                                     step=3, rollout=1, n_sample=4)
        assert torch.equal(u_all[n], u_one_dict[n]), f"{n} u not bit-identical"
```

- [ ] **Step 2: Run** → PASS (proves all-layer fill == per-layer fill; the noise is layer-keyed).
- [ ] **Step 3: Commit**

```bash
git add verl/tests/np/test_all_layer_parity.py
git commit -m "np: CPU parity gate -- all-layer u == single-layer u per layer (bit-identical)"
```

---

# STAGE B — All-layer batched assemble (pure CPU math; can run parallel to Stage C)

### Task B1: `assemble_all_layers_and_apply`

**Files:** Modify `np_worker_extension.py` (after `assemble_and_apply`, ~1190); test `verl/tests/np/test_apply_update_math.py`

- [ ] **Step 1: Write the failing test.** Assert the all-layer assemble produces, per layer, a dW identical (rel-Frobenius < 1e-5) to calling the existing single-layer `assemble_layer_delta` on that layer's signals. Pass `L_q`/`L_clean` ONCE (shared across layers) and `{layer: (u, x)}` separately (avoids the T×L loss-copy bug, C-5):

```python
import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    assemble_layer_delta, assemble_all_layers)

def test_all_layer_assemble_matches_per_layer():
    torch.manual_seed(0)
    T, n, layers = 12, 8, {"L0": (64, 48), "L1": (32, 96)}
    sigma = 0.01
    L_q = [torch.randn(n) for _ in range(T)]
    L_clean = [float(torch.randn(())) for _ in range(T)]
    sig = {ln: {"u": [torch.randn(n, d_out) for _ in range(T)],
                "x": [torch.randn(d_in) for _ in range(T)]}
           for ln, (d_out, d_in) in layers.items()}
    got = assemble_all_layers(L_q, L_clean, sig, sigma=sigma, sample_mode="grpo",
                              normalize=False, token_agg="mean", device="cpu")
    for ln in layers:
        ref = assemble_layer_delta(L_q, L_clean, sig[ln]["u"], sig[ln]["x"],
                                   sigma=sigma, sample_mode="grpo",
                                   normalize=False, token_agg="mean", device="cpu")
        rel = (got[ln] - ref).norm() / (ref.norm() + 1e-12)
        assert rel < 1e-5, f"{ln} rel_fro={rel:.2e}"
```

- [ ] **Step 2: Run to fail** — `AttributeError`/`ImportError: assemble_all_layers`.

- [ ] **Step 3: Implement** a thin wrapper that loops the existing per-layer GEMM (reuse, don't rewrite):

```python
def assemble_all_layers(L_q_per_step, L_clean_per_step, layer_signals,
                        sigma, sample_mode, normalize, token_agg, device=None):
    """Batched assemble for ALL layers. layer_signals: {name: {"u":[...], "x":[...]}}.
    L_q/L_clean are SHARED across layers (one combined loss per (token,sample)).
    Returns {name: dW [d_out,d_in] on CPU}. Reuses assemble_layer_delta per layer."""
    return {ln: assemble_layer_delta(L_q_per_step, L_clean_per_step,
                                     sig["u"], sig["x"], sigma=sigma,
                                     sample_mode=sample_mode, normalize=normalize,
                                     token_agg=token_agg, device=device)
            for ln, sig in layer_signals.items()}
```

And the RPC `assemble_all_layers_and_apply(self, layer_signals, L_q_steps, L_clean_steps, sigma, sample_mode, normalize, token_agg, lr, update_clip)` that calls `assemble_all_layers` then `self.apply_node_update(ln, dW, lr, update_clip)` per layer, returning `{ln: dw_norm}`.

- [ ] **Step 4: Run to pass.** **Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_apply_update_math.py
git commit -m "np: assemble_all_layers + assemble_all_layers_and_apply (reuse per-layer GEMM)"
```

---

# STAGE C — Teacher-scoring speed (RESOLVE the top-k semantics FIRST)

### Task C0: Decide + test the top-k id-window (guards C-1)

**Files:** Modify `teacher_scorer.py`; test `verl/tests/np/test_teacher_scorer.py`

- [ ] **Step 1: Write the failing test that a teacher id OUTSIDE the student top-k survives.** This is the test that would catch the silent `union`/`intersection` degradation:

```python
import torch
from verl.trainer.np.teacher_scorer import TeacherScorer

def test_topk_window_preserves_teacher_id_outside_student_topk():
    # student top-1 is id 0; teacher's top id is 5 (outside student top-1).
    # With strategy="union", the scored id set MUST include id 5.
    sc = TeacherScorer.__new__(TeacherScorer)
    sc.top_k = 1; sc.top_k_strategy = "union"; sc.weight_mode = "none"
    sc.teacher_temperature = 1.0
    s_clean_full = torch.tensor([10.0, -1, -1, -1, -1, -1])  # student argmax=0
    t_ids = torch.tensor([5, 0]); t_logp = torch.tensor([-0.1, -2.0])
    ids, t_aligned = sc._select_ids(s_clean_full, t_ids, t_logp, fallback=-50.0)
    assert 5 in ids.tolist() and 0 in ids.tolist()  # union keeps teacher id 5
```

(`_select_ids` already exists and computes union from FULL student logits — so this passes today. The point of C0 is the **storage decision**: the decode-side top-k slice (C1) must keep a window wide enough that `_select_ids` can still see teacher ids. **Decision:** store the student top-`topk_store_k` ids where `topk_store_k = max(log_prob_top_k, 512)` so the union/intersection window is preserved for the non-`only_stu` strategies; document that ids beyond this window are unreachable.)

- [ ] **Step 2: Run** → PASS (asserts the union semantics we must NOT break in C1).
- [ ] **Step 3: Commit**

```bash
git add verl/tests/np/test_teacher_scorer.py
git commit -m "np: pin top-k id-window semantics test (teacher id outside student top-k survives union)"
```

### Task C1: Top-k logit slice in decode drivers

**Files:** Modify `np_worker_extension.py` (logit storage at 198, 325, 438); test `verl/tests/np/test_teacher_scorer.py`

- [ ] **Step 1: Write the failing test** that top-k-sliced logits + ids reproduce the full-vocab reverse-KL (within fp tol) for `only_stu`:

```python
def test_topk_slice_matches_full_vocab_kl_only_stu():
    import torch
    from verl.trainer.np.teacher_scorer import reverse_kl_topk
    vocab, n, k = 200, 4, 32
    logits = torch.randn(1 + n, vocab)
    # full-vocab reference: take student top-k ids, score
    ids = torch.topk(logits[0], k).indices
    s_full = torch.log_softmax(logits.float(), -1)
    ref = s_full[:, ids]
    # sliced storage: gather all rows on the same ids, store [1+n, k] + ids
    sliced = logits[:, ids]
    s_sliced = torch.log_softmax(logits.float(), -1)[:, ids]  # log_softmax over full vocab, then slice
    assert torch.allclose(ref, s_sliced, atol=1e-5)
```

(The subtlety the test pins: `log_softmax` MUST be over the full vocab BEFORE slicing — so the decode side must store either the pre-softmax top-k logits AND the full-vocab log-sum-exp normalizer, OR store top-k log-probs directly. **Decision:** store top-k **log-probs** `log_softmax(logits)[:, ids]` computed on GPU before the D2H copy — exact, and the scorer consumes log-probs directly.)

- [ ] **Step 2: Run to fail** if you naively slice-then-softmax; PASS once the store-log-probs decision is implemented.

- [ ] **Step 3: Implement** a helper `_topk_store(logits, k)` → `(topk_logp[1+N,k], ids[k])` computing `lp = torch.log_softmax(logits.float(), -1); ids = torch.topk(lp[0], k).indices; return lp[:, ids].to("cpu"), ids.to("cpu")`. Call it at all three decode sites (198, 325, 438) instead of `logits.detach().to("cpu")`; store `(topk_logp, ids)` tuples in `candidate_logits`.

- [ ] **Step 4: Run to pass. Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_teacher_scorer.py
git commit -m "np: store top-k log-probs+ids from decode (GPU-side) instead of full vocab"
```

### Task C2: Vectorized `score_rollout` over `[T,1+N,k]`

**Files:** Modify `teacher_scorer.py` (score_rollout 55–96); test `verl/tests/np/test_teacher_scorer.py`

- [ ] **Step 1: Write the failing test** asserting the vectorized scorer matches the current per-token Python loop within rel-Frobenius < 1e-4 across `weight_mode ∈ {student_p, teacher_p, none}` (only_stu strategy, the lossless one), on random `(topk_logp, ids)` inputs. (Full code: build a list of T `(lp[1+N,k], ids)` tuples + a fake teacher logp map; compare `score_rollout_vectorized` vs the existing `score_rollout` per-token result.)

- [ ] **Step 2: Run to fail** (`score_rollout_vectorized` missing).

- [ ] **Step 3: Implement** `score_rollout` to accept the tuple format: stack to `[T,1+N,k]`, align teacher logps per token, compute reverse-KL in one broadcast reduce over `[T,N,k]` (reuse `reverse_kl_topk`'s math, vectorized). Keep a legacy branch detecting full-vocab tensor input (back-compat).

- [ ] **Step 4: Run to pass. Step 5: Commit**

```bash
git add verl/verl/trainer/np/teacher_scorer.py verl/tests/np/test_teacher_scorer.py
git commit -m "np: vectorized top-k score_rollout (one batched reverse-KL reduce; per-token-loop parity <1e-4)"
```

### Task C3: Batched teacher prefill over B prompts

**Files:** Modify `teacher_scorer.py` (_teacher_topk_logprobs 134–155)

- [ ] **Step 1: Write the failing test** `test_teacher_batched_logprobs_matches_serial` — a fake engine returning deterministic prompt_logprobs; assert batching B prompts in one `generate.remote` returns the same per-prompt logps as B serial calls.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implement** a `score_wave(list_of_prefixes, list_of_candidate_logits)` that issues ONE `engine.generate.remote(list_of_prompts, ...)` (vLLM accepts a list) and splits the result per prompt; cap at `teacher_batch_size` (config; default 16) to avoid teacher OOM under co-location; fall back to serial above the cap.
- [ ] **Step 4: Run to pass. Step 5: Commit**

```bash
git add verl/verl/trainer/np/teacher_scorer.py verl/tests/np/test_teacher_scorer.py
git commit -m "np: batched teacher prefill over B prompts (teacher_batch_size cap, serial fallback)"
```

### Task C4: CPU parity suite — all strategies × weight modes

**Files:** Modify `verl/tests/np/test_teacher_scorer.py`

- [ ] **Step 1: Write** a parametrized test over `top_k_strategy ∈ {only_stu, only_tch, intersection, union}` × `weight_mode ∈ {student_p, teacher_p, none}`, INCLUDING a case where a teacher id lies outside the student top-k (exercises C0/C1), asserting vectorized == per-token-loop within tol AND that union/intersection actually include the off-window teacher id (uses `topk_store_k≥512`).
- [ ] **Step 2: Run → PASS. Step 3: Commit**

```bash
git add verl/tests/np/test_teacher_scorer.py
git commit -m "np: parity suite -- vectorized scorer across all strategies/weight modes incl off-window teacher id"
```

---

# STAGE D — Trainer consolidation (one decode + one score + all-layer assemble per step)

### Task D1: Decode drivers accept a layer-name LIST; return per-layer u/x dicts

**Files:** Modify `np_worker_extension.py` (`run_np_decode` 169–215 + `run_np_decode_packed` 346–459)

- [ ] **Step 1: Write the failing GPU-free structural test** (mock model_runner is heavy — instead assert signature + that passing `layer_names=[...]` sets `st["mode"]="perturb_all_layers"` and allocates per-layer `u_buf`/`x_buf` dicts). If a true unit test needs the engine, mark it as the GPU gate in Task F and here just assert the Python plumbing (dict allocation helper) via a small function `_alloc_layer_buffers(np_modules, n_sample, device)` returning the two dicts.

```python
def test_alloc_layer_buffers_shapes():
    import torch
    from verl.workers.rollout.vllm_rollout.np_worker_extension import _alloc_layer_buffers
    class W:  # fake wrapped linear with weight [d_out,d_in]
        def __init__(s,o,i): s.wrapped=type("x",(),{"weight":torch.zeros(o,i)})()
    mods = {"L0": W(64,48), "L1": W(32,96)}
    u,x = _alloc_layer_buffers(mods, n_sample=8, device=torch.device("cpu"))
    assert u["L0"].shape==(8,64) and x["L1"].shape==(96,)
```

- [ ] **Step 2: Run to fail. Step 3: Implement** `_alloc_layer_buffers` + extend `run_np_decode`/`run_np_decode_packed` to accept `layer_names` (list); when set, use `perturb_all_layers` mode, fill all layers via `_np_fill_u_buf_all_layers`, and return `captured_u`/`captured_x` as `{layer: {t: tensor}}`. Keep the single `layer_name` (str) path for back-compat (parity oracle).
- [ ] **Step 4: Run to pass. Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_all_layer_perturb.py
git commit -m "np: decode drivers accept layer_names list -> per-layer captured u/x dicts (all-layer eager)"
```

### Task D2: `fit()` all-layer branch (one decode → one score → one assemble)

**Files:** Modify `ray_trainer.py` (fit loop 506–649)

- [ ] **Step 1:** Add a branch: when `len(active) > 1` (all-layer mode, `en_layerwise=False`), do ONE decode over all `active` layers, ONE `score_rollout` on the clean rollout (the loss is layer-agnostic — scored on the clean row, identical across layers; **replicate `L_clean`/`L_q` references for all layers** when building `layer_signals`), ONE `assemble_all_layers_and_apply`, then broadcast each layer. Keep the `len(active)==1` serial path verbatim (oracle).
- [ ] **Step 2: Smoke** (depends on Stage E for the graphed path; here test the EAGER all-layer path): run the launcher with `EN_LAYERWISE=false DECODE_MODE=packed BATCH_SIZE=4 MAX_RESP_LENGTH=32 N_SAMPLE=8 NUM_ITERATIONS=2` on a free GPU; expect 2 steps, `weight_sync_ok:1.0`, and `dW_norm` logged for EVERY matched layer (not just one). Paste the per-layer dW lines.
- [ ] **Step 3: Commit**

```bash
git add verl/verl/trainer/np/ray_trainer.py
git commit -m "np: fit() all-layer branch -- one decode/score/assemble per step over all matched layers"
```

### Task D3: Wire top-k tuples into the all-layer loop

**Files:** Modify `ray_trainer.py` (the score call site in the new all-layer branch)

- [ ] **Step 1:** Pass the top-k `(logp, ids)` tuples from D1's decode into the vectorized `score_rollout` (Stage C). **Step 2:** Re-run the D2 smoke; confirm identical `weight_sync_ok` + per-layer dW with the top-k path. **Step 3: Commit**

```bash
git add verl/verl/trainer/np/ray_trainer.py
git commit -m "np: feed top-k logp tuples into all-layer vectorized scoring in fit()"
```

---

# STAGE E — Fully CUDA-graphed packed all-layer decode

### Task E1: `_np_capture_step_packed` — capture fixed-bucket all-layer forward

**Files:** Modify `np_worker_extension.py` (new method near `_np_capture_step` 558)

- [ ] **Step 1:** Implement capture at a FIXED bucket width `R = bucket_b_pack * (1 + n_sample)`. Reuse the `_np_capture_step` pattern: persistent `input_ids_buf`/`positions_buf`, `_np_build_attn_metadata_persistent` with `max_seq_len_override=cap`, the graph-pool-release gotcha (lines 631–637). **Critical (C-2):** install the per-layer `u_buf`/`x_buf` DICTS onto `self.np_state` BEFORE capture and DO NOT re-bind them after — the captured `PerturbedLinear.forward` reads `st["u_buf"][self.name]`, so the graph pins those exact tensors. Set `st["mode"]="perturb_all_layers"`, `st["perturbed_row_idx"]`/`st["clean_row_idx"]` for the packed scatter (reuse the existing packed scatter branch, extended to all layers). Return `graph_state` holding graph + all persistent buffers + the layer dicts.
- [ ] **Step 2:** GPU smoke: capture once for `bucket_b_pack=2, n_sample=4` on Qwen3-1.7B, assert no exception and `graph_state["graph"]` is a `torch.cuda.CUDAGraph`. **Step 3: Commit.**

### Task E2: `_np_replay_step_packed` — per-token replay + bucket-pad EOS

**Files:** Modify `np_worker_extension.py`

- [ ] **Step 1:** Per token: refill per-prompt `clean_slot`/`seq_lens`/`positions`/`input_ids` in place for ACTIVE prompts; **pad finished prompts' rows with slot=−1 AND a VALID seq_len/position** (C-4 — never stale/zero; reuse the last valid position so attention is well-defined though its output is ignored); refill ALL layers' `u_buf` via `_np_fill_u_buf_all_layers` (ONE refill site, C-6); `graph.replay()`; per-token `torch.cuda.synchronize()`; eager `compute_logits` + top-k store on active rows. Pad active count UP to the nearest captured bucket; pick the graph for that bucket; NEVER recapture within a bucket.
- [ ] **Step 2:** Unit-ish GPU test: 3 prompts, staggered EOS, bucket set `[2,4]`; assert per-prompt clean tokens match the eager packed oracle bit-for-bit up to each prompt's EOS (the deep staggered-EOS test, not the shallow one). **Step 3: Commit.**

### Task E3: `run_np_decode_packed_graphed` orchestrator

**Files:** Modify `np_worker_extension.py`

- [ ] **Step 1:** Prefill all prompts (`_np_prefill_packed`), allocate per-layer buffers sized to the bucket, capture per needed bucket (cache by `bucket_b_pack`), loop tokens calling `_np_replay_step_packed`, collect per-prompt per-layer `captured_u`/`captured_x` + top-k logits, return the per-prompt lists. **Single noise-refill site** (inside replay). **Step 2:** GPU smoke (b_pack=4, n_sample=8, max_tokens=64) → returns clean tokens + per-layer signals, no exception. **Step 3: Commit.**

### Task E4: Config knobs + decode_mode allow-list

**Files:** Modify `verl/verl/trainer/config/np_trainer.yaml`; `ray_trainer.py` (~491 validation)

- [ ] **Step 1:** Add to yaml: `pack_width`/`b_pack_buckets: [2,4,8,16]`, `teacher_batch_size: 16`, `topk_store_k: 512`, and extend `decode_mode` comment to include `packed_graphed`. **Step 2:** Extend the `decode_mode` allow-list in `ray_trainer.py` to accept `"packed_graphed"` (else `ValueError` before the new branch — the critique's M-1 bug). **Step 3:** `python -c "from omegaconf import OmegaConf; print(OmegaConf.load('verl/verl/trainer/config/np_trainer.yaml').np.b_pack_buckets)"` → `[2, 4, 8, 16]`. **Step 4: Commit.**

### Task E5: `fit()` packed_graphed branch (all matched layers)

**Files:** Modify `ray_trainer.py`

- [ ] **Step 1:** Add the `decode_mode == "packed_graphed"` branch routing to `run_np_decode_packed_graphed` over ALL matched layers (fix the `matched[0]` single-layer bug — use the full list). Reuse the D2 score+assemble tail. **Step 2:** Smoke `DECODE_MODE=packed_graphed EN_LAYERWISE=false` 2 steps → per-layer dW, `weight_sync_ok:1.0`. **Step 3: Commit.**

### Task E6: Launcher plumbing

**Files:** Modify `scripts/zo_opd/opd_math_np.sh`

- [ ] **Step 1:** Add env passthrough: `DECODE_MODE` (allow `packed_graphed`), `EN_LAYERWISE` (default keep, but doc that all-layer needs `false`), `B_PACK_BUCKETS`, `TEACHER_BATCH_SIZE`, `TOPK_STORE_K` → corresponding `np.*` Hydra overrides. **Step 2:** `bash -n scripts/zo_opd/opd_math_np.sh` → ok. **Step 3: Commit.**

---

# STAGE F — GPU gates + the goal proof

### Task F1: GPU parity gates (σ=0, u bit-identity, logits tol, staggered-EOS)

**Files:** Create `scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py`

- [ ] **Step 1:** Gate (a) σ=0 packed_graphed all-layer clean tokens == stock greedy for every prompt; (b) packed_graphed vs eager all-layer: per-layer `u` bit-identical (`torch.equal`), logits within `rtol=1e-2`; (c) staggered-EOS bucket-crossing parity vs eager oracle. **Step 2:** Run on a free GPU (4–7); paste PASS lines. **Step 3: Commit.**

### Task F2: Non-optional e2e smoke on real Qwen3-1.7B

**Files:** Create `scripts/zo_opd/np_checks/check_alllayer_e2e.sh`

- [ ] **Step 1:** `DECODE_MODE=packed_graphed EN_LAYERWISE=false` on all `down_proj` layers, ~20 steps, batch=8, max_tokens=128. Assert: per-layer `dW_norm > 0` for EVERY matched layer, `weight_changed_frac > 0`, `weight_sync_ok = 1`, and `eval/heldout_kl` trends DOWN over the 20 steps (the honest learning signal at `ray_trainer.py:716-722`). **Step 2:** Run; paste the heldout_kl trajectory. **Step 3: Commit.**

### Task F3: FINAL — NP(packed_graphed all-layer) vs BP-OPD one-step

**Files:** Modify `scripts/zo_opd/bench_np_vs_bp.sh`

- [ ] **Step 1:** Re-point the NP side: `DECODE_MODE=packed_graphed EN_LAYERWISE=false PACK_WIDTH=8 BATCH_SIZE=64 MAX_RESP_LENGTH=1024`, all matched layers. **Step 2:** Run NP and BP on two separate free GPUs (one job per GPU, 4–7). Record one-step wall-clock (NP `step:0 ... step_time`; BP `perf/time_per_step`) + peak mem. **GATE:** report the ratio; the goal is NP ≤ BP. If NP > BP, record the decode/score/assemble breakdown so the residual is attributable (the forward-count floor may dominate). **Step 3: Commit** the bench change + a results file `scripts/zo_opd/results/np_vs_bp_alllayer_graphed.txt`.

### Task F4: Docs closeout

**Files:** Modify `docs/wiki/zo_np_trainer.md` (new §11), `docs/index.md`, `docs/log.md`

- [ ] **Step 1:** Document the all-layer graphed design + the F1/F2/F3 measured results (verbatim numbers). **Step 2:** Update index + append a log line. **Step 3: Commit.**

---

## Dependency order (critical path)

```
A1→A2→A3→A5  (all-layer forward + noise + parity)   ─┐
A4 (order guard)                                      │
B1 (assemble)  ───────────────────────────────────── ┤→ D1→D2→D3 (eager all-layer trainer works e2e)
C0→C1→C2→C3→C4 (top-k + vectorized teacher) ───────── ┘                    │
                                                                            ▼
                              E1→E2→E3→E4→E5→E6 (graphed packed all-layer)  │
                                                                            ▼
                                          F1 (parity) → F2 (e2e) → F3 (NP-vs-BP goal) → F4 (docs)
```

**Hard rules:** E (graphed) MUST NOT start until A lands and A5 passes (else the captured graph bakes in a single-layer perturb — unfixable post-capture, critique O-1/C-2). D2 (trainer all-layer loop) MUST precede D3 (top-k wiring) to avoid double-editing the same call sites (O-2). Resolve C0/C1 (top-k window) before C2 or `union`/`intersection`/`teacher_p` silently degrade (C-1).

---

## Test Plan (verification deliverables)

- **CPU unit/parity (every stage, must stay green):** `pytest verl/tests/np/ -q`. New: all-layer perturb (A1), fill parity (A3/A5), assemble all-layer (B1), top-k window + vectorized scorer all strategies (C0/C2/C4), buffer alloc (D1).
- **GPU correctness gates:** σ=0 routing, eager-vs-graphed all-layer `u` bit-identity + logits tol, **staggered-EOS bucket-crossing parity** (F1); non-optional e2e with heldout_kl-down (F2).
- **The goal proof:** F3 — NP packed_graphed all-layer vs BP-OPD one-step wall-clock + peak mem at batch=64/max_tokens=1024, with decode/score/assemble breakdown. This is the single number that decides success.

---

## Self-Review notes

- **Spec coverage:** all-layer perturb (A), batched assemble (B), top-k + vectorized + batched teacher (C), trainer consolidation (D), fully-graphed packed decode with bucket-pad EOS (E), GPU gates + NP-vs-BP proof + docs (F). The five gaps + all four critique "missing" items (config M-1, launcher M-2, NP-vs-BP M-3, non-optional e2e M-4, docs M-6) are tasks.
- **Three landmines as explicit guardrails:** C-1 (top-k window, C0+C4), C-2/C-8 (forward/capture agree on per-layer buffer dict, E1), C-4 (valid padded seq_lens + staggered-EOS test, E2/F1).
- **Type consistency:** decode drivers return per-layer `captured_u/x` as `{layer: {t: tensor}}` (D1) consumed by `assemble_all_layers` `{layer:{"u":[...],"x":[...]}}` — D2 reshapes the dict-of-dicts into per-layer lists before calling assemble; `L_q`/`L_clean` passed once (B1 signature), shared across layers. Top-k stored as `(topk_logp[1+N,k], ids[k])` tuples (C1) consumed by vectorized `score_rollout` (C2).
- **Reused machinery (not rebuilt):** `assemble_layer_delta` GEMM, `_np_build_attn_metadata_persistent`, the graph-pool-release gotcha, the (1+N) mask, `reverse_kl_topk` math, `noise_seed` layer-keying.
