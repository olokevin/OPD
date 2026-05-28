# Node-Perturbation (NP) Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone zeroth-order "node perturbation" trainer for the OPD verl fork: a single clean student trajectory whose every decode step is evaluated `1+n_sample`-wide with ephemeral per-row perturbations (no KV pollution), scored per-step by a teacher, committing only the clean token, and updating the perturbed layer's weights from the per-token gradient estimate.

**Architecture:** Self-contained in the verl fork (no edits to installed `vllm` or existing verl files). Mirrors the committed ES sibling trainer (`verl/trainer/es/`, `es_worker_extension.py`). The novel piece is a worker-side custom decode driver that reuses vLLM v1's multi-query-per-step machinery (the speculative-decode path: variable `query_start_loc`, `slot_mapping=-1` to suppress perturbed-row KV writes, shared-prefix decode) plus a `PerturbedLinear` shim installed on `perturb_rules`-matched modules. `enforce_eager=True` is mandatory (RNG noise can't live in a CUDA graph). Teacher is a second vLLM engine scoring candidate distributions per step into a per-token reverse-KL loss `L_t^(q)`.

**Tech Stack:** Python 3.12, conda env `verl`, vLLM 0.11.0 (v1 engine), Ray, PyTorch, Hydra/OmegaConf, pytest. Reference files: `verl/verl/trainer/es/ray_trainer.py`, `verl/verl/trainer/main_es.py`, `verl/verl/workers/rollout/vllm_rollout/es_worker_extension.py`, `verl/verl/trainer/config/es_trainer.yaml`, `opd_es.sh`. Design spec: `docs/superpowers/specs/2026-05-28-np-trainer-design.md`.

---

## Conventions for this plan

- **Env:** all `python`/`pytest` runs use the `verl` conda env. Prefix commands with `conda run -n verl` (e.g. `conda run -n verl pytest ...`) or activate it in your shell first.
- **Two test styles** (the repo has no repo-root test runner; `verl/tests/` is upstream pytest):
  - **Unit tests** (pure logic, no GPU/vLLM): real pytest under `verl/tests/np/`. These follow strict TDD (write failing test → run red → implement → run green → commit).
  - **GPU/vLLM verification scripts** (decode driver, perturbed layer, end-to-end): runnable assertion scripts under `scripts/zo_opd/np_checks/`. These are the spec's "verification ladder." They need a GPU and a small model; treat their assertions as the acceptance gate for GPU-coupled tasks. Where a GPU isn't available during authoring, the step says so and the assertion script is still committed so it can be run on the box.
- **Module names are vLLM-real (fused):** `self_attn.qkv_proj`, `self_attn.o_proj`, `mlp.gate_up_proj`, `mlp.down_proj`. The HF names `q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj` do NOT exist in vLLM `named_modules()`. See spec §4.
- **Never store `u_q`:** perturbations are always regenerated from a seed via `torch.Generator`. Only seeds cross RPC boundaries.
- **Sign:** `L_t^(q)` is minimization-oriented (lower = closer to teacher). `average` estimator is `(L_t^(q) − L_t)/σ` (one-sided, divide by σ, not 2σ).

---

## File structure

| Path | Responsibility | Created in |
|---|---|---|
| `verl/verl/trainer/np/__init__.py` | package marker + re-exports | Task 1 |
| `verl/verl/trainer/np/seeding.py` | `noise_seed(...)` + `draw_noise(...)` (gaussian/bernoulli/uniform) | Task 2 |
| `verl/verl/trainer/np/layer_resolve.py` | regex → matched module names; layer schedule | Task 3 |
| `verl/verl/trainer/np/grad_estimator.py` | `g_t` from `{L_t^(q)}`; accumulate `Σ_t g_t⊗x_t`; ANP-normalize `δW` | Task 4 |
| `verl/verl/trainer/np/teacher_scorer.py` | per-step teacher reverse-KL `L_t^(q)` over top-k set | Task 5 |
| `verl/verl/trainer/np/task_utils.py` | thin re-export of ES `get_task_components` | Task 6 |
| `verl/verl/trainer/config/np_trainer.yaml` | Hydra config (the `np.*` interface) | Task 7 |
| `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` | `PerturbedLinear` shim install + worker-side decode driver + `apply_node_update` + NCCL broadcast | Tasks 8–11 |
| `verl/verl/trainer/np/ray_trainer.py` | `RayNPTrainer` + `NPNcclLLM` + `fit()` | Task 12 |
| `verl/verl/trainer/main_np.py` | Hydra entry (mirror `main_es.py`) | Task 13 |
| `scripts/zo_opd/opd_np.sh` | top-level launcher (sibling of `opd_es.sh`) | Task 14 |
| `scripts/zo_opd/np_checks/*.py` | GPU verification scripts (σ=0 smoke, KV-grows-by-1, cosine-sim) | Tasks 9, 11, 15 |
| `verl/tests/np/test_*.py` | unit tests for pure-logic units | Tasks 2–5 |

Implementation order is dependency-first: pure-logic units (seeds, regex, estimator, teacher math) → config → worker extension (layer shim → decode driver → update) → trainer → entry → launcher → end-to-end verification.

---

## Task 1: Package skeleton

**Files:**
- Create: `verl/verl/trainer/np/__init__.py`
- Create: `verl/tests/np/__init__.py`

- [ ] **Step 1: Create the np package marker**

Create `verl/verl/trainer/np/__init__.py`:

```python
"""Node-Perturbation (NP) zeroth-order trainer.

See docs/superpowers/specs/2026-05-28-np-trainer-design.md for the algorithm.
Mirrors the ES sibling trainer (verl/verl/trainer/es/) but perturbs linear-layer
*outputs* during a custom n_sample-wide decode rather than perturbing weights.
"""
```

- [ ] **Step 2: Create the test package marker**

Create `verl/tests/np/__init__.py` (empty file).

- [ ] **Step 3: Verify the package imports**

Run: `conda run -n verl python -c "import verl.trainer.np; print('ok')"`
Expected: prints `ok` (run from `verl/` dir, or with `verl/` on PYTHONPATH — the verl env installs verl editable, so it resolves from anywhere).

- [ ] **Step 4: Commit**

```bash
git add verl/verl/trainer/np/__init__.py verl/tests/np/__init__.py
git commit -m "feat(np): package skeleton for node-perturbation trainer"
```

---

## Task 2: Seeding & noise draws (`seeding.py`)

Deterministic seed derivation and noise sampling. No `u_q` is ever stored — callers regenerate from `(seed, shape, device, dtype)`.

**Files:**
- Create: `verl/verl/trainer/np/seeding.py`
- Test: `verl/tests/np/test_seeding.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_seeding.py`:

```python
import torch
from verl.trainer.np.seeding import noise_seed, draw_noise


def test_noise_seed_is_deterministic_and_64bit():
    s1 = noise_seed(global_seed=42, step=3, layer="model.layers.0.mlp.down_proj", rollout=1, q=2)
    s2 = noise_seed(global_seed=42, step=3, layer="model.layers.0.mlp.down_proj", rollout=1, q=2)
    assert s1 == s2                      # deterministic
    assert 0 <= s1 < 2**63               # fits a signed 64-bit generator seed


def test_noise_seed_varies_with_each_field():
    base = dict(global_seed=42, step=3, layer="L", rollout=1, q=2)
    s = noise_seed(**base)
    assert noise_seed(**{**base, "step": 4}) != s
    assert noise_seed(**{**base, "layer": "M"}) != s
    assert noise_seed(**{**base, "rollout": 2}) != s
    assert noise_seed(**{**base, "q": 3}) != s


def test_draw_noise_reproducible_from_seed():
    seed = 123456789
    a = draw_noise(seed, (4, 8), torch.device("cpu"), torch.float32, method="gaussian")
    b = draw_noise(seed, (4, 8), torch.device("cpu"), torch.float32, method="gaussian")
    assert torch.equal(a, b)             # same seed -> identical noise
    assert a.shape == (4, 8)


def test_draw_noise_bernoulli_is_pm1():
    n = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="bernoulli")
    uniq = set(n.unique().tolist())
    assert uniq.issubset({-1.0, 1.0})    # Rademacher: only +/-1


def test_draw_noise_uniform_in_unit_range():
    n = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="uniform")
    assert n.min() >= -1.0 and n.max() <= 1.0


def test_draw_noise_methods_differ():
    g = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="gaussian")
    b = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="bernoulli")
    assert not torch.equal(g, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_seeding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verl.trainer.np.seeding'`.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/trainer/np/seeding.py`:

```python
"""Deterministic perturbation seeds and noise draws.

Invariant: no perturbation tensor is ever stored. Callers regenerate noise
on demand from a seed produced by noise_seed(...). Only integer seeds cross
RPC/Ray boundaries. See spec §2 "Never store u_q".
"""
import hashlib
from typing import Tuple

import torch

_MASK_63 = (1 << 63) - 1


def noise_seed(global_seed: int, step: int, layer: str, rollout: int, q: int) -> int:
    """Stable 63-bit seed for the perturbation of sample q, rollout, layer, step.

    Uses blake2b over the field tuple so the namespace can grow without
    collision and is stable across processes (Python's hash() is salted).
    """
    key = f"{int(global_seed)}|{int(step)}|{layer}|{int(rollout)}|{int(q)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK_63


def draw_noise(
    seed: int,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    method: str = "gaussian",
) -> torch.Tensor:
    """Regenerate the noise tensor for a seed. Deterministic per (seed, shape, method).

    method:
      - "gaussian":  N(0, 1)
      - "bernoulli": Rademacher, values in {-1, +1} (symmetric two-point)
      - "uniform":   U(-1, 1)
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if method == "gaussian":
        n = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    elif method == "bernoulli":
        bits = torch.randint(0, 2, shape, generator=gen, device=device, dtype=torch.int64)
        n = bits.to(torch.float32) * 2.0 - 1.0
    elif method == "uniform":
        n = torch.rand(shape, generator=gen, device=device, dtype=torch.float32) * 2.0 - 1.0
    else:
        raise ValueError(f"unknown sample_method: {method!r}")
    return n.to(dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_seeding.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/trainer/np/seeding.py verl/tests/np/test_seeding.py
git commit -m "feat(np): deterministic seeding + gaussian/bernoulli/uniform noise draws"
```

---

## Task 3: Layer resolution (`layer_resolve.py`)

Resolve `perturb_rules` regexes against module names and pick the active layer(s) per step.

**Files:**
- Create: `verl/verl/trainer/np/layer_resolve.py`
- Test: `verl/tests/np/test_layer_resolve.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_layer_resolve.py`:

```python
from verl.trainer.np.layer_resolve import resolve_modules, active_layers_for_step

# Representative vLLM-real module names (fused qkv_proj / gate_up_proj).
MODULES = [
    "model.embed_tokens",
    "model.layers.0.self_attn.qkv_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.1.mlp.down_proj",
    "model.norm",
    "lm_head",
]


def test_fullmatch_single_type():
    rules = [r"model\.layers\.\d+\.mlp\.down_proj"]
    assert resolve_modules(rules, MODULES) == [
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    ]


def test_fullmatch_specific_layer():
    rules = [r"model\.layers\.0\.self_attn\.qkv_proj"]
    assert resolve_modules(rules, MODULES) == ["model.layers.0.self_attn.qkv_proj"]


def test_partial_substring_does_not_match():
    # fullmatch semantics: a rule must match the WHOLE name
    rules = [r"qkv_proj"]
    assert resolve_modules(rules, MODULES) == []


def test_multiple_rules_union_dedup_in_module_order():
    rules = [r"model\.layers\.\d+\.mlp\.down_proj", r"model\.layers\.0\.self_attn\.o_proj"]
    assert resolve_modules(rules, MODULES) == [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    ]


def test_no_match_raises():
    # HF-style name that never exists in vLLM -> empty -> caller must fail loudly
    import pytest
    with pytest.raises(ValueError, match="matched no modules"):
        resolve_modules([r"model\.layers\.\d+\.mlp\.up_proj"], MODULES, error_if_empty=True)


def test_layer_schedule_layerwise_roundrobin():
    matched = ["A", "B", "C"]
    assert active_layers_for_step(matched, step=0, en_layerwise=True) == ["A"]
    assert active_layers_for_step(matched, step=1, en_layerwise=True) == ["B"]
    assert active_layers_for_step(matched, step=3, en_layerwise=True) == ["A"]  # wraps


def test_layer_schedule_all_at_once():
    matched = ["A", "B", "C"]
    assert active_layers_for_step(matched, step=5, en_layerwise=False) == ["A", "B", "C"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_layer_resolve.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/trainer/np/layer_resolve.py`:

```python
"""Resolve perturb_rules regexes to module names and pick active layers per step.

Resolution uses re.fullmatch: a rule must match the WHOLE module name. Output
preserves the order modules appear in (model.named_modules() order), de-duped.
Names are vLLM-real (fused qkv_proj / gate_up_proj); see spec §4.
"""
import re
from typing import List


def resolve_modules(
    rules: List[str],
    module_names: List[str],
    error_if_empty: bool = False,
) -> List[str]:
    compiled = [re.compile(r) for r in rules]
    out, seen = [], set()
    for name in module_names:
        if name in seen:
            continue
        if any(c.fullmatch(name) for c in compiled):
            out.append(name)
            seen.add(name)
    if error_if_empty and not out:
        raise ValueError(
            f"perturb_rules {rules!r} matched no modules. "
            f"Note vLLM fuses qkv/gate_up: use self_attn.qkv_proj / mlp.gate_up_proj, "
            f"not q_proj / up_proj."
        )
    return out


def active_layers_for_step(matched: List[str], step: int, en_layerwise: bool) -> List[str]:
    """en_layerwise=True -> one layer per step (round-robin); False -> all matched."""
    if not matched:
        return []
    if en_layerwise:
        return [matched[step % len(matched)]]
    return list(matched)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_layer_resolve.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/trainer/np/layer_resolve.py verl/tests/np/test_layer_resolve.py
git commit -m "feat(np): perturb_rules regex resolution + per-step layer schedule"
```

---

## Task 4: Gradient estimator (`grad_estimator.py`)

Pure math, no vLLM. Turns per-step `{L_t^(q)}` + regenerated `u_q` + captured `x_t` into a weight delta `δW` for one layer.

**Files:**
- Create: `verl/verl/trainer/np/grad_estimator.py`
- Test: `verl/tests/np/test_grad_estimator.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_grad_estimator.py`:

```python
import math
import torch
from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w


def test_sample_scale_average_is_forward_difference():
    # average: (L_q - L_clean) / sigma
    L_q = torch.tensor([2.0, 4.0])
    s = sample_scale(L_q, L_clean=1.0, sigma=0.5, mode="average")
    assert torch.allclose(s, torch.tensor([(2.0 - 1.0) / 0.5, (4.0 - 1.0) / 0.5]))


def test_sample_scale_grpo_is_zscore_over_samples():
    L_q = torch.tensor([1.0, 2.0, 3.0])
    s = sample_scale(L_q, L_clean=None, sigma=0.1, mode="grpo")
    mean = L_q.mean()
    std = L_q.std(unbiased=False) + 1e-8
    assert torch.allclose(s, (L_q - mean) / std, atol=1e-5)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_grad_estimator.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/trainer/np/grad_estimator.py`:

```python
"""Node-perturbation gradient estimation (pure math; no vLLM/GPU coupling).

Per step t, given per-sample losses L_t^(q), the clean baseline L_t, and the
regenerated perturbations u_q (rows of `u`), form the per-sample scalar scale,
then the per-token gradient g_t = (1/n) sum_q scale_q * u_q, and accumulate the
rank-1 outer product g_t (x) x_t into the layer's delta_W. See spec §1, §3.
"""
from typing import Optional

import torch


def sample_scale(
    L_q: torch.Tensor,            # [n_sample] per-sample loss at this token
    L_clean: Optional[float],     # baseline (clean) loss; required for "average"
    sigma: float,
    mode: str,                    # "average" | "grpo"
) -> torch.Tensor:
    """Per-sample scalar weighting of u_q. Lower L is better (minimization)."""
    if mode == "average":
        if L_clean is None:
            raise ValueError("average mode requires L_clean baseline")
        return (L_q - float(L_clean)) / sigma
    if mode == "grpo":
        mean = L_q.mean()
        std = L_q.std(unbiased=False) + 1e-8
        return (L_q - mean) / std
    raise ValueError(f"unknown grad_estimate_sample mode: {mode!r}")


def accumulate_delta_w(
    delta_w: torch.Tensor,        # [d_out, d_in] in/out accumulator
    scales: torch.Tensor,         # [n_sample]
    u: torch.Tensor,              # [n_sample, d_out] regenerated perturbations
    x_t: torch.Tensor,            # [d_in] captured clean input to the layer
    normalize: bool = True,
    eps: float = 1e-6,
) -> None:
    """delta_w += outer( (1/n) sum_q scales_q * (u_q [/ ||u_q||^2]), x_t )."""
    u_eff = u
    if normalize:
        sq = (u * u).sum(dim=-1, keepdim=True).clamp_min(eps)  # [n_sample,1]
        u_eff = u / sq
    g_t = (scales[:, None] * u_eff).mean(dim=0)                # [d_out]
    delta_w.add_(torch.outer(g_t.to(delta_w.dtype), x_t.to(delta_w.dtype)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_grad_estimator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/trainer/np/grad_estimator.py verl/tests/np/test_grad_estimator.py
git commit -m "feat(np): node-perturbation gradient estimator (sample_scale + rank-1 accumulate)"
```

---

## Task 5: Teacher reverse-KL math (`teacher_scorer.py`, pure-math core first)

Split into a pure-math kernel (testable now) and a vLLM-engine wrapper (wired in Task 12). This task does the kernel: given student & teacher log-probs over a top-k token set at one position, return the per-token reverse-KL `L_t` (minimization-oriented).

**Files:**
- Create: `verl/verl/trainer/np/teacher_scorer.py`
- Test: `verl/tests/np/test_teacher_scorer.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_teacher_scorer.py`:

```python
import torch
from verl.trainer.np.teacher_scorer import reverse_kl_topk


def test_reverse_kl_zero_when_identical():
    # student == teacher -> KL = 0
    logp = torch.log_softmax(torch.tensor([2.0, 1.0, 0.5, 0.0]), dim=-1)
    kl = reverse_kl_topk(student_logp=logp, teacher_logp=logp, weight_mode="none")
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_reverse_kl_positive_and_minimization_oriented():
    s = torch.log_softmax(torch.tensor([3.0, 0.0, 0.0]), dim=-1)   # peaked
    t = torch.log_softmax(torch.tensor([0.0, 0.0, 0.0]), dim=-1)   # uniform
    kl = reverse_kl_topk(student_logp=s, teacher_logp=t, weight_mode="none")
    assert kl.item() > 0.0   # reverse KL E_student[log p_s - log p_t] > 0 here


def test_reverse_kl_student_p_weighting_uses_student_probs():
    s = torch.log_softmax(torch.tensor([3.0, 0.0]), dim=-1)
    t = torch.log_softmax(torch.tensor([0.0, 0.0]), dim=-1)
    # student_p weighting weights each vocab term by softmax(student) over the k set
    kl_w = reverse_kl_topk(student_logp=s, teacher_logp=t, weight_mode="student_p")
    # equals sum_v softmax(s)_v * (s_v - t_v) == standard reverse KL
    p = s.exp()
    expected = (p * (s - t)).sum()
    assert torch.allclose(kl_w, expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_teacher_scorer.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/trainer/np/teacher_scorer.py`:

```python
"""Per-step teacher scoring -> per-token reverse-KL loss L_t (minimization-oriented).

This file holds two layers:
  1. reverse_kl_topk(...): pure-math kernel over a top-k token set (tested in
     test_teacher_scorer.py).
  2. TeacherScorer (added in Task 12): wraps a second vLLM engine, gathers the
     teacher's log-probs over the OPD top-k set, and calls the kernel per token.

Sign: L_t is positive reverse-KL, already minimization-oriented (lower = student
closer to teacher). If ever sourcing from dp_actor.compute_distillation_reward
(which returns -kl*w, maximization-oriented), negate before use. See spec §3.
"""
import torch


def reverse_kl_topk(
    student_logp: torch.Tensor,   # [k] student log-probs over the top-k token set
    teacher_logp: torch.Tensor,   # [k] teacher log-probs over the SAME k tokens
    weight_mode: str = "student_p",
) -> torch.Tensor:
    """Reverse KL: sum_v w_v * (log p_student - log p_teacher).

    weight_mode:
      - "student_p": w_v = softmax(student)_v  (standard reverse KL E_student[...])
      - "teacher_p": w_v = softmax(teacher)_v
      - "none":      w_v = 1 (unweighted sum of log-prob differences over the set)
    """
    diff = student_logp - teacher_logp
    if weight_mode == "student_p":
        w = student_logp.exp()
    elif weight_mode == "teacher_p":
        w = teacher_logp.exp()
    elif weight_mode == "none":
        w = torch.ones_like(diff)
    else:
        raise ValueError(f"unknown reward_weight_mode: {weight_mode!r}")
    return (w * diff).sum()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_teacher_scorer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/trainer/np/teacher_scorer.py verl/tests/np/test_teacher_scorer.py
git commit -m "feat(np): reverse-KL top-k kernel for per-token teacher scoring"
```

---

## Task 6: Task-utils re-export (`np/task_utils.py`)

NP reuses ES's `get_task_components` (which already has the `opd_math` branch). Re-export, do not copy.

**Files:**
- Create: `verl/verl/trainer/np/task_utils.py`

- [ ] **Step 1: Create the re-export**

Create `verl/verl/trainer/np/task_utils.py`:

```python
"""NP reuses the ES task components verbatim (incl. the opd_math branch).

Importing from the ES module keeps a single source of truth for prompt
processors and reward functions. See verl/verl/trainer/es/task_utils.py.
"""
from verl.trainer.es.task_utils import get_task_components  # noqa: F401
```

- [ ] **Step 2: Verify it imports and exposes opd_math**

Run:
```bash
conda run -n verl python -c "
from verl.trainer.np.task_utils import get_task_components
pp, rf = get_task_components('opd_math', {})
print('opd_math ok', callable(pp), callable(rf))
"
```
Expected: prints `opd_math ok True True`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/trainer/np/task_utils.py
git commit -m "feat(np): re-export ES get_task_components (opd_math reuse)"
```

---

## Task 7: Hydra config (`np_trainer.yaml`)

**Files:**
- Create: `verl/verl/trainer/config/np_trainer.yaml`

- [ ] **Step 1: Create the config**

Create `verl/verl/trainer/config/np_trainer.yaml` (mirrors `es_trainer.yaml` structure; `np.*` block from spec §4):

```yaml
# Node-Perturbation (NP) Trainer Configuration
# Mirrors es_trainer.yaml structure. See docs/superpowers/specs/2026-05-28-np-trainer-design.md
defaults:
  - _self_

np:
  # core NP knobs
  sigma: 0.01
  n_sample: 8                      # perturbed copies scored per decode step
  n_rollout: 8                     # rollouts for sequence-level grad estimation
  sample_method: bernoulli         # gaussian | bernoulli | uniform
  en_layerwise_perturbation: true  # false = perturb all matched layers at once
  perturb_method: forward          # forward (one-sided) | antithetic
  perturb_granularity: token       # token | rollout
  grad_estimate_sample: grpo       # average | grpo   (over n_sample, per-token)
  grad_estimate_sequence: grpo     # average | grpo   (over n_rollout)
  perturb_rules:
    - '^model\.layers\.\d+\.mlp\.down_proj$'   # vLLM-real names (see spec §4)
  lr: 1.0e-4
  token_agg: sum                   # sum | mean
  update_clip: null                # delta_W / ||u||^2 safety (null = eps-floor only)

  # teacher scorer (v1, core)
  teacher_model_path: null         # required for loss_type=opd
  loss_type: opd                   # opd (teacher reverse-KL) | grpo (rule reward)
  log_prob_top_k: 256
  top_k_strategy: only_stu         # only_stu | only_tch | intersection | union | union-intersection
  teacher_temperature: 1.0
  reward_weight_mode: student_p    # student_p | teacher_p | none

  # engine / eval (mirror es_trainer.yaml)
  num_engines: 4
  num_iterations: 800
  precision: bfloat16
  max_tokens: 1024
  temperature: 0.0
  eval_interval: 25
  eval_batch_size: 256
  gpu_memory_utilization: 0.7
  global_seed: 42
  verbose: false
  worker_extension_cls: "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"

model:
  path: model/Qwen3-1.7B
  trust_remote_code: false

data:
  task_type: opd_math
  train_files: datasets/dapo-math-17k.parquet
  val_files: datasets/test_data/AIME24/test.parquet
  train_max_samples: 200
  val_max_samples: -1
  reward_fn_path: null
  reward_fn_name: null
  prompt_processor_path: null
  prompt_processor_name: null

trainer:
  project_name: OPD-NP
  experiment_name: np-run
  logger:
    - console
    - wandb
  default_local_dir: /tmp/${oc.env:USER}/verl/np_checkpoints
  default_hdfs_dir: null
  device: cuda
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: null
  test_freq: null
  save_freq: 100
  npu_profile:
    enable: false

ray_kwargs:
  ray_init:
    runtime_env: {}
```

- [ ] **Step 2: Verify Hydra can compose it**

Run:
```bash
conda run -n verl python -c "
from hydra import compose, initialize_config_dir
import os
d = os.path.abspath('verl/verl/trainer/config')
with initialize_config_dir(version_base=None, config_dir=d):
    cfg = compose(config_name='np_trainer')
    print('n_sample', cfg.np.n_sample, '| rules', list(cfg.np.perturb_rules), '| loss', cfg.np.loss_type)
"
```
Expected: prints `n_sample 8 | rules ['^model\\.layers\\.\\d+\\.mlp\\.down_proj$'] | loss opd`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/trainer/config/np_trainer.yaml
git commit -m "feat(np): np_trainer.yaml Hydra config with np.* interface"
```

---

## Task 8: `PerturbedLinear` shim + hook install (`np_worker_extension.py` part 1)

The worker extension begins. This task adds the shim that perturbs a matched module's output for the perturbed rows and captures `x` for the clean row, driven by worker-local `self.np_state`. It also adds `install_perturb_layers`.

**Files:**
- Create: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`
- Test: `verl/tests/np/test_perturbed_linear.py` (CPU unit test of the shim logic, no vLLM engine)

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_perturbed_linear.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_perturbed_linear.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` with the shim and install method (decode driver + update come in later tasks; keep this file growing):

```python
"""Node-Perturbation worker extension (vLLM WorkerExtension).

Registered via np.worker_extension_cls. Runs on each per-GPU vLLM Worker (self
is the Worker; self.model_runner.model is the loaded model). Responsibilities,
added across plan Tasks 8-11:
  - PerturbedLinear shim + install_perturb_layers  (Task 8)
  - n_sample-wide custom decode driver             (Task 9)
  - apply_node_update (weights) + NCCL broadcast   (Task 10-11)

Perturbations are regenerated from seeds, never stored. enforce_eager=True is
mandatory (set by NPNcclLLM) so these eager-Python hooks actually run.
See docs/superpowers/specs/2026-05-28-np-trainer-design.md.
"""
import torch

from verl.trainer.np.seeding import noise_seed, draw_noise


def _unpack(output):
    """vLLM linears may return a bare tensor or (tensor, bias). Normalize."""
    if isinstance(output, tuple):
        return output[0], output[1], True
    return output, None, False


def _repack(tensor, bias, was_tuple):
    return (tensor, bias) if was_tuple else tensor


class PerturbedLinear(torch.nn.Module):
    """Wraps a matched vLLM linear. Behavior keyed by worker-local np_state.

    Row layout per decode step (see spec §2): the active sequence contributes
    n_clean_rows clean row(s) followed by n_sample perturbed rows, contiguous.
    np_state_ref() returns the live dict so the trainer can switch modes per RPC
    without reinstalling.
    """

    def __init__(self, wrapped: torch.nn.Module, name: str, np_state_ref):
        super().__init__()
        self.wrapped = wrapped
        self.name = name
        self._np_state_ref = np_state_ref

    def forward(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        st = self._np_state_ref()
        mode = st.get("mode", "off")
        if mode == "off":
            return out
        x = args[0]
        y, bias, was_tuple = _unpack(out)
        n_clean = st["n_clean_rows"]

        if mode == "capture":
            # record the clean row's input x_t (detached) for the rank-1 update
            st["captured_x"][self.name] = x[0].detach().clone()
            return out

        if mode == "perturb" and self.name == st["layer"]:
            sigma = float(st["sigma"])
            if sigma == 0.0:
                return out
            n_sample = st["n_sample"]
            d_out = y.shape[-1]
            # perturbed rows occupy [n_clean : n_clean + n_sample]
            u_rows = []
            for q in range(n_sample):
                seed = noise_seed(st["global_seed"], st["step"], self.name, st["rollout"], q)
                u = draw_noise(seed, (d_out,), y.device, y.dtype, st["sample_method"])
                u_rows.append(u)
                y[n_clean + q] = y[n_clean + q] + sigma * u
            # stash regenerated u (stacked) so the update step can reuse identical noise
            st["captured_u"][self.name] = torch.stack(u_rows, dim=0)
            return _repack(y, bias, was_tuple)

        return out


class WorkerExtension:
    def _ensure_np_state(self):
        if not hasattr(self, "np_state"):
            self.np_state = {"mode": "off"}
        return self.np_state

    def install_perturb_layers(self, perturb_rules):
        """Wrap every perturb_rules-matched module with PerturbedLinear. Idempotent."""
        from verl.trainer.np.layer_resolve import resolve_modules

        self._ensure_np_state()
        model = self.model_runner.model
        names = [n for n, _ in model.named_modules()]
        matched = resolve_modules(list(perturb_rules), names, error_if_empty=True)
        self.np_modules = {}
        for layer_name in matched:
            parent = model
            *path, leaf = layer_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            child = getattr(parent, leaf)
            if isinstance(child, PerturbedLinear):
                wrapped = child  # already installed
            else:
                wrapped = PerturbedLinear(child, layer_name, lambda: self.np_state)
                setattr(parent, leaf, wrapped)
            self.np_modules[layer_name] = wrapped
        return list(matched)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_perturbed_linear.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_perturbed_linear.py
git commit -m "feat(np): PerturbedLinear shim + install_perturb_layers"
```

---

## Task 9: Custom `n_sample`-wide decode driver (`np_worker_extension.py` part 2)

The heart. A worker-side decode loop that builds a `1+n_sample`-wide step over shared-prefix KV, suppresses perturbed-row KV writes (`slot_mapping=-1`), runs the model, harvests `1+n_sample` logits, and commits only the clean token. This is GPU/vLLM-coupled — verified with an assertion script, not a CPU unit test.

> **Implementation note for the engineer:** This is the highest-risk task. Mirror vLLM v1 speculative decoding (`vllm/v1/spec_decode/`, `vllm/v1/worker/gpu_model_runner.py:1413-1479`) which already packs `1+k` queries per sequence against shared KV and commits only accepted tokens. The decode-driver design (spec §2): model the `1+n_sample` rows as separate prefix-sharing sequences (each a vanilla single-query causal decode against the shared prefix — NO custom attention mask) and set the perturbed rows' `slot_mapping` entries to `-1` (`PAD_SLOT_ID`, `vllm/v1/attention/backends/utils.py:37`) so `reshape_and_cache` skips them (`vllm/attention/ops/triton_reshape_and_cache_flash.py:33-37`). Confirm the variable-query + shared-prefix path is exposed by the active attention backend under `enforce_eager=True` (FlashAttention v1 was verified; TORCH_SDPA was NOT — pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` if SDPA lacks it; this is the spec §5 open question and the σ=0 smoke is its gate).

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` (add `run_np_decode` + helpers)
- Create: `scripts/zo_opd/np_checks/check_decode_sigma0.py` (σ=0 smoke + KV-grows-by-1 + width assertions)

- [ ] **Step 1: Add the decode driver method**

Append to the `WorkerExtension` class in `np_worker_extension.py`:

```python
    def run_np_decode(self, prompt_token_ids, sampling_params, layer_name, np_cfg, rollout_idx):
        """Custom decode for ONE prompt. Returns dict with:
          - "clean_tokens": list[int]            committed (row-0) response tokens
          - "candidate_logits": list[Tensor]     per-step [1+n_sample, vocab] logits
          - "captured_x": dict[layer -> Tensor]  clean-row input per active layer (last step)
          - "captured_u": dict[step -> Tensor]   regenerated u per step  [n_sample, d_out]

        Mechanics (spec §2): prefill the prompt once (normal, writes KV). Then per
        step t, build a (1 + n_sample)-row batch of the SAME next position sharing
        the prompt+committed-token KV blocks; rows 1.. get slot_mapping=-1 so they
        never write KV; arm np_state mode="perturb"/"capture"; run model forward;
        read 1+n_sample logits; sample row 0; append; advance.
        """
        import torch
        from vllm import SamplingParams

        st = self._ensure_np_state()
        model = self.model_runner.model
        device = next(model.parameters()).device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])

        # --- prefill prompt (clean, normal KV write) ---
        # Reuse the worker's existing manual-forward pattern (cf. es_worker_extension
        # get_logits_for_prompt at es_worker_extension.py:295-298 for set_forward_context).
        # NOTE: the concrete construction of input_ids/positions/attn_metadata/slot_mapping
        # for the 1+n_sample-wide step reuses self.model_runner's _prepare_inputs helpers
        # and the spec-decode metadata layout; see the implementation note above.
        state = self._np_prefill(model, device, prompt_token_ids)  # helper below

        clean_tokens, candidate_logits, captured_u = [], [], {}
        for t in range(max_tokens):
            st.update({
                "mode": "perturb", "layer": layer_name,
                "global_seed": int(np_cfg["global_seed"]), "step": t,
                "rollout": int(rollout_idx), "sigma": float(np_cfg["sigma"]),
                "n_sample": n_sample, "sample_method": np_cfg["sample_method"],
                "n_clean_rows": 1, "captured_x": st.get("captured_x", {}),
                "captured_u": {},
            })
            logits = self._np_step_forward(model, device, state, n_sample)  # [1+n_sample, vocab]
            candidate_logits.append(logits.detach().to("cpu"))
            captured_u[t] = st["captured_u"].get(layer_name)
            next_tok = self._np_sample_clean(logits[0], sampling_params)
            clean_tokens.append(int(next_tok))
            if self._np_is_eos(next_tok, sampling_params):
                break
            self._np_commit_clean(state, next_tok)  # advance shared KV by 1 (clean only)

        st["mode"] = "off"
        return {
            "clean_tokens": clean_tokens,
            "candidate_logits": candidate_logits,
            "captured_x": st.get("captured_x", {}),
            "captured_u": captured_u,
        }
```

> The helpers `_np_prefill`, `_np_step_forward`, `_np_sample_clean`, `_np_is_eos`, `_np_commit_clean` encapsulate the vLLM-internal input/metadata construction. Their exact bodies are **vLLM-0.11.0-version-specific and must be written against the installed engine** (they reach into `self.model_runner` internals that the offline probe could not fully reproduce without a live engine). Implement against: `set_forward_context` (cf. `es_worker_extension.py:295-298`), the `_prepare_inputs` building blocks in `vllm/v1/worker/gpu_model_runner.py:923-1266`, `block_table.compute_slot_mapping` then override perturbed-row slots to `-1`, and the spec-decode metadata layout in `gpu_model_runner.py:1413-1479` as the multi-query template. Keep each helper small and `# vLLM internal`-commented. **The contract each helper must satisfy is exactly the assertion script in Step 2** — do not consider Task 9 done until `check_decode_sigma0.py` prints PASS. Required behaviors:
> - `_np_prefill(model, device, prompt_token_ids) -> state`: run a normal prefill over the prompt (writes KV); return an opaque state holding the engine's running-sequence handle + committed length.
> - `_np_step_forward(model, device, state, n_sample) -> Tensor[1+n_sample, vocab]`: build the `1+n_sample`-row step (prefix-sharing sequences, same next position), set perturbed rows' `slot_mapping=-1`, run one forward under `set_forward_context` with `self.np_state` armed, return the `1+n_sample` next-token logits in row order (row 0 = clean).
> - `_np_sample_clean(logits_row0, sampling_params) -> int`: sample/argmax the clean next token (greedy when `temperature==0`).
> - `_np_is_eos(token, sampling_params) -> bool`: EOS / stop check.
> - `_np_commit_clean(state, token) -> None`: append the clean token to the running sequence so its KV (and only its KV) grows by one; advance `state`'s committed length.

- [ ] **Step 2: Create the σ=0 verification script**

Create `scripts/zo_opd/np_checks/check_decode_sigma0.py`:

```python
"""GPU verification (spec Verification #1): with sigma=0 the NP custom decode
must reproduce a stock greedy generate() byte-for-byte, the perturbed forward
must widen to 1+n_sample rows, and the KV cache must grow by exactly 1 per step.

Usage (needs 1 GPU + a small model):
  conda run -n verl python scripts/zo_opd/np_checks/check_decode_sigma0.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 4
"""
import argparse

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--prompt", default="What is 2+2? Answer:")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)

    tok = llm.get_tokenizer()
    prompt_ids = tok(args.prompt)["input_ids"]

    # Stock greedy reference.
    ref = llm.generate({"prompt_token_ids": prompt_ids},
                       SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                       use_tqdm=False)
    ref_tokens = list(ref[0].outputs[0].token_ids)

    # Install perturb layers, then run NP decode with sigma=0.
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))
    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens, global_seed=42,
                  sigma=0.0, sample_method="gaussian")
    out = llm.collective_rpc("run_np_decode",
                             args=(prompt_ids, SamplingParams(temperature=0.0),
                                   args.layer, np_cfg, 0))[0]
    np_tokens = out["clean_tokens"]

    # Assertions.
    assert np_tokens[: len(ref_tokens)] == ref_tokens, (
        f"sigma=0 decode diverged from greedy generate():\n ref={ref_tokens}\n np ={np_tokens}")
    # width: each step's candidate logits must be [1+n_sample, vocab]
    for i, cl in enumerate(out["candidate_logits"]):
        assert cl.shape[0] == 1 + args.n_sample, f"step {i} width {cl.shape[0]} != {1+args.n_sample}"
    print(f"PASS: sigma=0 matches greedy ({len(ref_tokens)} tokens); "
          f"width=1+{args.n_sample} every step.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verification script (on a GPU box)**

Run:
```bash
conda run -n verl python scripts/zo_opd/np_checks/check_decode_sigma0.py \
    --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 4
```
Expected: prints `PASS: sigma=0 matches greedy ... width=1+4 every step.`
If it diverges or width != 1+n_sample, the decode driver / backend pin is wrong — fix before proceeding (this is the spec §5 backend gate).

- [ ] **Step 4: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py scripts/zo_opd/np_checks/check_decode_sigma0.py
git commit -m "feat(np): n_sample-wide custom decode driver + sigma=0 verification script"
```

---

## Task 10: `apply_node_update` (`np_worker_extension.py` part 3)

Turn the per-step `{candidate_logits, captured_u, captured_x}` + teacher loss into `δW` and apply it to the layer's weight. Reuses `grad_estimator`. The teacher loss `L_t^(q)` is computed by the trainer (Task 12) and passed in.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`
- Test: `verl/tests/np/test_apply_update_math.py` (CPU test of the update assembly, mocking the model param)

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_apply_update_math.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n verl pytest verl/tests/np/test_apply_update_math.py -v`
Expected: FAIL — `assemble_layer_delta` not defined.

- [ ] **Step 3: Add `assemble_layer_delta` + `apply_node_update`**

Append to `np_worker_extension.py` (module-level function + method). Add `from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w` to the imports at the top:

```python
def assemble_layer_delta(L_q_per_step, L_clean_per_step, u_per_step, x_per_step,
                         sigma, sample_mode, normalize, token_agg, eps=1e-6):
    """Build delta_W [d_out, d_in] from per-step signals. Pure math (CPU/GPU)."""
    from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w

    assert len(L_q_per_step) == len(u_per_step) == len(x_per_step)
    d_out = u_per_step[0].shape[1]
    d_in = x_per_step[0].shape[0]
    dw = torch.zeros(d_out, d_in, dtype=torch.float32)
    T = max(len(L_q_per_step), 1)
    for L_q, L_clean, u, x_t in zip(L_q_per_step, L_clean_per_step, u_per_step, x_per_step):
        scales = sample_scale(L_q.float(), L_clean, sigma, sample_mode)
        accumulate_delta_w(dw, scales=scales, u=u.float(), x_t=x_t.float(),
                           normalize=normalize, eps=eps)
    if token_agg == "mean":
        dw.div_(T)
    return dw
```

```python
    def apply_node_update(self, layer_name, delta_w_cpu, lr, update_clip=None):
        """W <- W + lr * delta_W for the wrapped layer's weight. delta_w_cpu: [d_out,d_in]."""
        import torch

        wrapped = self.np_modules[layer_name]
        weight = wrapped.wrapped.weight  # vLLM linear weight [d_out, d_in]
        dw = delta_w_cpu.to(weight.device, weight.dtype)
        if update_clip is not None:
            dw = dw.clamp_(-float(update_clip), float(update_clip))
        with torch.no_grad():
            weight.add_(float(lr) * dw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(dw.norm().item())   # return ||delta_W|| for the >0 assertion
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n verl pytest verl/tests/np/test_apply_update_math.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_apply_update_math.py
git commit -m "feat(np): assemble_layer_delta + apply_node_update (weight write, returns ||dW||)"
```

---

## Task 11: NCCL broadcast (`np_worker_extension.py` part 4)

Sync the updated layer's weights across engines, reusing the ES NCCL pattern but scoped to one layer.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`

- [ ] **Step 1: Add the inter-engine group init + per-layer broadcast**

Append to `np_worker_extension.py` (mirror `es_worker_extension.py:13-19,126-137`). Add the NCCL helper import block at the top of the file:

```python
def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)
```

Methods on `WorkerExtension`:

```python
    def get_worker_ip(self):
        from vllm.utils import get_ip
        return get_ip()

    def init_inter_engine_group(self, master_address, master_port, rank, world_size):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_layer_weights(self, layer_name, src_rank):
        """Broadcast only the updated layer's weight (+bias if present)."""
        import torch
        wrapped = self.np_modules[layer_name]
        params = [wrapped.wrapped.weight]
        if getattr(wrapped.wrapped, "bias", None) is not None:
            params.append(wrapped.wrapped.bias)
        for p in params:
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
```

- [ ] **Step 2: Verify the file imports cleanly (no GPU needed)**

Run:
```bash
conda run -n verl python -c "
import verl.workers.rollout.vllm_rollout.np_worker_extension as m
assert hasattr(m.WorkerExtension, 'install_perturb_layers')
assert hasattr(m.WorkerExtension, 'run_np_decode')
assert hasattr(m.WorkerExtension, 'apply_node_update')
assert hasattr(m.WorkerExtension, 'broadcast_layer_weights')
assert hasattr(m.WorkerExtension, 'init_inter_engine_group')
print('worker extension surface ok')
"
```
Expected: prints `worker extension surface ok`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "feat(np): per-layer NCCL broadcast + inter-engine group init"
```

---

## Task 12: `RayNPTrainer` + `NPNcclLLM` + teacher engine + `fit()` (`ray_trainer.py`)

The orchestrator. Reuses ES engine-launch/eval patterns; the material new code is teacher-engine launch, per-step teacher scoring (wrapping the Task 5 kernel), and the per-step fit loop.

**Files:**
- Create: `verl/verl/trainer/np/ray_trainer.py`
- Modify: `verl/verl/trainer/np/teacher_scorer.py` (add `TeacherScorer` engine wrapper)

- [ ] **Step 1: Add the `TeacherScorer` engine wrapper to `teacher_scorer.py`**

Append to `verl/verl/trainer/np/teacher_scorer.py`:

```python
class TeacherScorer:
    """Wraps a teacher vLLM engine; scores per-step candidate logits into L_t^(q).

    Given the student's prefix tokens (committed) and the per-step candidate
    next-token distributions, query the teacher for its log-probs over the OPD
    top-k set and call reverse_kl_topk per (token, sample). Returns L_t^(q).
    """

    def __init__(self, teacher_engine, top_k, top_k_strategy, teacher_temperature, weight_mode):
        self.engine = teacher_engine
        self.top_k = int(top_k)
        self.top_k_strategy = top_k_strategy
        self.teacher_temperature = float(teacher_temperature)
        self.weight_mode = weight_mode

    def score_rollout(self, prefix_token_ids, candidate_logits):
        """candidate_logits: list over steps of [1+n_sample, vocab] (CPU tensors).

        Returns (L_q_per_step, L_clean_per_step):
          L_q_per_step[t]:   [n_sample] reverse-KL of each perturbed candidate vs teacher
          L_clean_per_step[t]: float reverse-KL of the clean (row 0) candidate vs teacher
        Implementation: feed [prefix + committed token sequence] to the teacher with
        prompt_logprobs=top_k to get teacher top-k log-probs per position (vLLM
        SamplingParams.prompt_logprobs, sampling_params.py:154-164). Align teacher
        position p with student step p, gather student log-probs over the same top-k
        token ids, call reverse_kl_topk. See spec §3.
        """
        import torch

        from verl.trainer.np.teacher_scorer import reverse_kl_topk

        L_q_per_step, L_clean_per_step = [], []
        teacher_logp_by_pos, topk_ids_by_pos = self._teacher_topk_logprobs(prefix_token_ids,
                                                                           len(candidate_logits))
        for t, cl in enumerate(candidate_logits):
            ids = topk_ids_by_pos[t]                       # [k] teacher's top-k token ids
            t_logp = teacher_logp_by_pos[t]                # [k]
            s_full_logp = torch.log_softmax(cl.float(), dim=-1)  # [1+n_sample, vocab]
            s_logp = s_full_logp[:, ids]                   # [1+n_sample, k]
            L_clean_per_step.append(
                float(reverse_kl_topk(s_logp[0], t_logp, self.weight_mode)))
            L_q_per_step.append(torch.stack([
                reverse_kl_topk(s_logp[1 + q], t_logp, self.weight_mode)
                for q in range(s_logp.shape[0] - 1)
            ]))
        return L_q_per_step, L_clean_per_step

    def _teacher_topk_logprobs(self, prefix_token_ids, num_steps):
        """Query the teacher engine for per-position top-k log-probs over the response.

        Returns (logp_by_pos: list[Tensor[k]], ids_by_pos: list[LongTensor[k]]).
        """
        import torch
        from vllm import SamplingParams

        sp = SamplingParams(temperature=self.teacher_temperature, max_tokens=1,
                            prompt_logprobs=self.top_k)
        out = self.engine.generate.remote({"prompt_token_ids": prefix_token_ids}, sp,
                                          use_tqdm=False)
        import ray
        out = ray.get(out)[0]
        # prompt_logprobs: list[dict[token_id -> Logprob]] aligned to prompt positions.
        # The response region is the last num_steps positions of prefix_token_ids.
        plp = out.prompt_logprobs[-num_steps:]
        logp_by_pos, ids_by_pos = [], []
        for d in plp:
            ids = list(d.keys())
            logp_by_pos.append(torch.tensor([d[i].logprob for i in ids]))
            ids_by_pos.append(torch.tensor(ids, dtype=torch.long))
        return logp_by_pos, ids_by_pos
```

> **Engineer note:** vLLM's `prompt_logprobs` returns a sparse dict per position (the model's own top-k, not an arbitrary requested set). For `top_k_strategy != only_tch` you may need to also gather the student's top-k ids and union them; v1 ships `only_stu`/`only_tch` cleanly, the union strategies are a refinement. Keep the kernel call site stable.

- [ ] **Step 2: Create `ray_trainer.py`**

Create `verl/verl/trainer/np/ray_trainer.py`. Reuse ES patterns; only `fit()` and teacher launch are new. (Read `verl/verl/trainer/es/ray_trainer.py` for `_launch_engines`, `_init_inter_engine_group`, `_evaluate_model`, `_compute_metrics`, `Tracking` usage and copy them, adapting `es_config` → `np_config` and `ESNcclLLM` → `NPNcclLLM`.)

```python
"""Node-Perturbation trainer with Ray single-controller (mirrors RayESTrainer).

Reuses the ES engine-launch / NCCL / eval scaffolding verbatim; the material
differences are: NPNcclLLM forces enforce_eager + prefix caching; a teacher
engine is launched for per-step scoring; fit() drives the custom n_sample-wide
decode + per-token NP update instead of weight-perturbation ES.
See docs/superpowers/specs/2026-05-28-np-trainer-design.md.
"""
import gc
import os
import time
from datetime import datetime

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from verl.trainer.np.layer_resolve import active_layers_for_step, resolve_modules
from verl.trainer.np.teacher_scorer import TeacherScorer
from verl.utils.tracking import Tracking


class NPNcclLLM(LLM):
    """vLLM wrapper for NP. enforce_eager mandatory; prefix caching ON (shared-prefix decode)."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        kwargs["enforce_eager"] = True
        kwargs["enable_prefix_caching"] = True
        super().__init__(*args, **kwargs)


class RayNPTrainer:
    def __init__(self, config, tokenizer, reward_fn, val_reward_fn=None,
                 train_data=None, eval_data=None, prompt_processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn or reward_fn
        self.train_data = train_data or []
        self.eval_data = eval_data or []
        self.prompt_processor = prompt_processor
        self.np_config = config.np if hasattr(config, "np") else config
        self.engines = []
        self.teacher_engine = None
        self.placement_groups = []

    # --- engine launch: copy _launch_engines / _init_inter_engine_group / _cleanup
    #     from es/ray_trainer.py, swapping ESNcclLLM->NPNcclLLM and es_config->np_config.
    #     Then ADDITIONALLY launch one teacher engine (NPNcclLLM with the teacher path)
    #     on its own placement group when np.loss_type == "opd".

    def init_workers(self, model_path):
        self._launch_engines(model_path)                  # students (copied from ES)
        self._init_inter_engine_group()                   # copied from ES
        ray.get([e.collective_rpc.remote("install_perturb_layers",
                                         args=(list(self.np_config.perturb_rules),))
                 for e in self.engines])
        if self.np_config.get("loss_type", "opd") == "opd":
            self._launch_teacher_engine(self.np_config.teacher_model_path)
            self.scorer = TeacherScorer(
                self.teacher_engine,
                top_k=self.np_config.log_prob_top_k,
                top_k_strategy=self.np_config.top_k_strategy,
                teacher_temperature=self.np_config.teacher_temperature,
                weight_mode=self.np_config.reward_weight_mode,
            )

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.get("project_name", "OPD-NP"),
            experiment_name=self.config.trainer.get("experiment_name", "np-run"),
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        prompts = ([self.prompt_processor(d, self.tokenizer) for d in self.train_data]
                   if self.prompt_processor else
                   [d.get("prompt", d.get("context")) for d in self.train_data])

        cfg = self.np_config
        num_iterations = self.config.trainer.get("total_epochs") or cfg.num_iterations
        eval_interval = self.config.trainer.get("test_freq") or cfg.get("eval_interval", 25)
        sp = SamplingParams(temperature=cfg.get("temperature", 0.0))

        # module names for layer scheduling (query engine 0 once)
        matched = ray.get(self.engines[0].collective_rpc.remote(
            "install_perturb_layers", args=(list(cfg.perturb_rules),)))[0]

        np_cfg = dict(n_sample=int(cfg.n_sample), max_tokens=int(cfg.max_tokens),
                      global_seed=int(cfg.global_seed), sigma=float(cfg.sigma),
                      sample_method=cfg.sample_method)

        bar = tqdm(range(num_iterations), desc="NP Training")
        for step in bar:
            t0 = time.time()
            active = active_layers_for_step(matched, step, cfg.en_layerwise_perturbation)
            dw_norms = {}
            for layer_name in active:
                # shard prompts across engines; here: engine 0 drives n_rollout rollouts
                prompt = prompts[step % len(prompts)]
                pid = prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt
                L_q_steps, L_clean_steps, u_steps, x_steps = [], [], [], []
                for r in range(int(cfg.n_rollout)):
                    out = ray.get(self.engines[0].collective_rpc.remote(
                        "run_np_decode", args=(pid, sp, layer_name, np_cfg, r)))[0]
                    full = pid + out["clean_tokens"]
                    L_q, L_clean = self.scorer.score_rollout(full, out["candidate_logits"])
                    L_q_steps += L_q
                    L_clean_steps += L_clean
                    u_steps += [out["captured_u"][t] for t in range(len(out["candidate_logits"]))]
                    # capture x: re-run one capture pass (mode=capture) to get clean x per step
                    x_steps += self._capture_x(self.engines[0], pid, out["clean_tokens"], layer_name, np_cfg)
                # assemble + apply on engine 0, broadcast to all
                dw = ray.get(self.engines[0].collective_rpc.remote(
                    "assemble_and_apply", args=(layer_name, L_q_steps, L_clean_steps,
                                                u_steps, x_steps, float(cfg.sigma),
                                                cfg.grad_estimate_sample,
                                                True, cfg.token_agg, float(cfg.lr),
                                                cfg.update_clip)))[0]
                dw_norms[layer_name] = dw
                ray.get([e.collective_rpc.remote("broadcast_layer_weights",
                                                 args=(layer_name, 0)) for e in self.engines])
            logger.log({"train/step_time": time.time() - t0,
                        **{f"train/dW_norm/{k}": v for k, v in dw_norms.items()},
                        "training/global_step": step}, step=step)
            bar.set_postfix({"dW": f"{max(dw_norms.values()) if dw_norms else 0:.3e}"})
            if eval_interval and (step % eval_interval == 0 or step == num_iterations - 1):
                logger.log(self._evaluate_model(self.engines[0], self.eval_data, step, logger),
                           step=step)
            gc.collect(); torch.cuda.empty_cache()
        bar.close(); logger.finish(); self._cleanup()
```

> **Engineer notes:**
> - `assemble_and_apply(layer_name, ...)` is a worker-extension method (defined in Step 2b below) that calls `assemble_layer_delta(...)` then `apply_node_update(...)` in one RPC (avoids shipping the big `delta_w` tensor twice).
> - `self._capture_x(engine, prompt_ids, clean_tokens, layer_name, np_cfg) -> list[Tensor]` is a `RayNPTrainer` method (defined in Step 1b below) returning the clean input `x_t` to `layer_name` at each response step. It issues a `collective_rpc("run_capture_pass", ...)` that re-runs the fixed `prompt_ids + clean_tokens` as one teacher-forced prefill with `np_state["mode"]="capture"`, recording `x` per step. Per-step capture is correct; a last-step-only approximation is a documented fallback if perf-bound.
> - Copy `_launch_engines`, `_init_inter_engine_group`, `_cleanup`, `_evaluate_model`, `_evaluate_with_engine`, `_compute_metrics` from `es/ray_trainer.py` and adapt names (`es_config`→`np_config`, `ESNcclLLM`→`NPNcclLLM`). `_launch_teacher_engine(path)` is a one-placement-group variant of `_launch_engines` that stores the engine on `self.teacher_engine`.

- [ ] **Step 1b: Add `_capture_x` to `RayNPTrainer` and `run_capture_pass` to the worker extension**

Add to `RayNPTrainer`:

```python
    def _capture_x(self, engine, prompt_ids, clean_tokens, layer_name, np_cfg):
        """Re-run the fixed [prompt + clean response] once with mode=capture; return
        the clean input x_t to layer_name at each response step (list of CPU tensors)."""
        return ray.get(engine.collective_rpc.remote(
            "run_capture_pass", args=(prompt_ids, clean_tokens, layer_name, np_cfg)))[0]
```

Add to `WorkerExtension` in `np_worker_extension.py` (teacher-forced prefill over the fixed sequence; the `PerturbedLinear` in `capture` mode records row-0 input per step — for a single prefill it records once per position via repeated calls, so loop one position at a time, or capture the full input row-block and slice per response position):

```python
    def run_capture_pass(self, prompt_token_ids, clean_tokens, layer_name, np_cfg):
        """Teacher-forced re-run of the committed sequence to capture x_t per response
        step. Returns list[Tensor[d_in]] (CPU), one per token in clean_tokens."""
        import torch

        st = self._ensure_np_state()
        model = self.model_runner.model
        device = next(model.parameters()).device
        full = list(prompt_token_ids) + list(clean_tokens)
        n_prompt = len(prompt_token_ids)
        x_per_step = []
        # Re-decode the fixed sequence one response step at a time so the capture hook
        # sees a single clean row whose input is x_t. Reuse the same step machinery as
        # run_np_decode but with n_sample=0 and mode="capture".
        state = self._np_prefill(model, device, prompt_token_ids)
        for t, tok in enumerate(clean_tokens):
            st.update({"mode": "capture", "layer": layer_name, "n_clean_rows": 1,
                       "captured_x": {}})
            self._np_step_forward(model, device, state, n_sample=0)  # 1-row forward
            x_per_step.append(st["captured_x"][layer_name].detach().to("cpu"))
            self._np_commit_clean(state, tok)
        st["mode"] = "off"
        return x_per_step
```

- [ ] **Step 2b: Add `assemble_and_apply` to the worker extension**

Append to `WorkerExtension` in `np_worker_extension.py`:

```python
    def assemble_and_apply(self, layer_name, L_q_steps, L_clean_steps, u_steps, x_steps,
                           sigma, sample_mode, normalize, token_agg, lr, update_clip):
        dw = assemble_layer_delta(L_q_steps, L_clean_steps, u_steps, x_steps,
                                  sigma=sigma, sample_mode=sample_mode,
                                  normalize=normalize, token_agg=token_agg)
        return self.apply_node_update(layer_name, dw, lr, update_clip)
```

- [ ] **Step 3: Verify trainer imports (no GPU)**

Run:
```bash
conda run -n verl python -c "
from verl.trainer.np.ray_trainer import RayNPTrainer, NPNcclLLM
from verl.trainer.np.teacher_scorer import TeacherScorer
print('trainer surface ok')
"
```
Expected: prints `trainer surface ok`.

- [ ] **Step 4: Commit**

```bash
git add verl/verl/trainer/np/ray_trainer.py verl/verl/trainer/np/teacher_scorer.py verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "feat(np): RayNPTrainer + NPNcclLLM + teacher scorer + per-step fit loop"
```

---

## Task 13: Hydra entry point (`main_np.py`)

**Files:**
- Create: `verl/verl/trainer/main_np.py`

- [ ] **Step 1: Create the entry point**

Create `verl/verl/trainer/main_np.py` (mirror `main_es.py:101-235`; reuse its `load_data`, task dispatch, and Ray init):

```python
"""Hydra entry point for Node-Perturbation (NP) training. Mirrors main_es.py."""
import os
import socket
import tempfile
import time

import hydra
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.trainer.np.ray_trainer import RayNPTrainer
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="np_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_np(config)


def run_np(config) -> None:
    from pprint import pprint
    print(f"NP Training - hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    if not ray.is_initialized():
        for k in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(k, None)
        unique_dir = tempfile.mkdtemp(prefix=f"ray_np_session_{int(time.time())}_")
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True,
                 _temp_dir=unique_dir, dashboard_port=None)

    model_path = config.model.path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=config.model.get("trust_remote_code", False))

    from verl.trainer.main_es import load_data
    train_data, eval_data = [], []
    if config.data.get("train_files"):
        train_data = load_data(config.data.train_files)
        if config.data.get("train_max_samples", -1) > 0:
            train_data = train_data[: config.data.train_max_samples]
    if config.data.get("val_files"):
        eval_data = load_data(config.data.val_files)
        if config.data.get("val_max_samples", -1) > 0:
            eval_data = eval_data[: config.data.val_max_samples]

    task_type = config.data.get("task_type", "opd_math")
    if task_type in ["countdown", "gsm8k", "math", "math500", "olympiadbench",
                     "uspto50k", "common_gen", "mbpp", "rocstories", "opd_math"]:
        from verl.trainer.np.task_utils import get_task_components
        prompt_processor, reward_fn = get_task_components(
            task_type, OmegaConf.to_container(config.data, resolve=True))
    elif task_type == "custom":
        from verl.utils.import_utils import load_extern_object
        reward_fn = (load_extern_object(config.data.reward_fn_path, config.data.reward_fn_name)
                     if config.data.get("reward_fn_path") else None)
        prompt_processor = (load_extern_object(config.data.prompt_processor_path,
                                               config.data.prompt_processor_name)
                            if config.data.get("prompt_processor_path") else None)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if prompt_processor and eval_data:
        for d in eval_data:
            d["context"] = prompt_processor(d, tokenizer)

    trainer = RayNPTrainer(config=config, tokenizer=tokenizer, reward_fn=reward_fn,
                           val_reward_fn=reward_fn, train_data=train_data,
                           eval_data=eval_data, prompt_processor=prompt_processor)
    trainer.init_workers(model_path)
    trainer.fit()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify Hydra wiring (no GPU; config compose only)**

Run:
```bash
conda run -n verl python -c "
import importlib; m = importlib.import_module('verl.trainer.main_np')
assert hasattr(m, 'run_np') and hasattr(m, 'main'); print('main_np ok')
"
```
Expected: prints `main_np ok`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/trainer/main_np.py
git commit -m "feat(np): main_np.py Hydra entry point"
```

---

## Task 14: Launcher (`scripts/zo_opd/opd_np.sh`)

**Files:**
- Create: `scripts/zo_opd/opd_np.sh`

- [ ] **Step 1: Create the launcher**

Create `scripts/zo_opd/opd_np.sh` (mirror `opd_es.sh`; expose the `np.*` knobs):

```bash
#!/bin/bash
#SBATCH --job-name=opd_np
#SBATCH --output=logs/opd_np_output_%j.log
#SBATCH --error=logs/opd_np_error_%j.log
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}; mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/opd_np_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE  Start: $(date)"
fi

ray stop --force
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# FlashAttention v1 verified for the shared-prefix multi-query decode; SDPA was not.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- NP knobs ----
export SIGMA=${SIGMA:-0.01}
export N_SAMPLE=${N_SAMPLE:-8}
export N_ROLLOUT=${N_ROLLOUT:-8}
export SAMPLE_METHOD=${SAMPLE_METHOD:-bernoulli}
export PERTURB_GRANULARITY=${PERTURB_GRANULARITY:-token}
export GRAD_ESTIMATE_SAMPLE=${GRAD_ESTIMATE_SAMPLE:-grpo}
export GRAD_ESTIMATE_SEQUENCE=${GRAD_ESTIMATE_SEQUENCE:-grpo}
export EN_LAYERWISE=${EN_LAYERWISE:-true}
export LR=${LR:-1e-4}
export TOKEN_AGG=${TOKEN_AGG:-sum}
export LOSS_TYPE=${LOSS_TYPE:-opd}
# newline-separated regex list -> Hydra list; default = all decoder mlp.down_proj
export PERTURB_RULES=${PERTURB_RULES:-'^model\.layers\.\d+\.mlp\.down_proj$'}

# ---- teacher / OPD ----
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-model/Qwen3-4B-Non-Thinking-RL-Math}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-256}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}

# ---- hardware ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NUM_ENGINES=${NUM_ENGINES:-4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}

# ---- model & data ----
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export EVAL_DATASET=${EVAL_DATASET:-datasets/test_data/AIME24/test.parquet}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-200}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export NUM_ITERATIONS=${NUM_ITERATIONS:-200}

# ---- logging ----
export PROJECT_NAME=${PROJECT_NAME:-OPD-NP}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-np_${ACTOR_MODEL_NAME}_sigma_${SIGMA}_n_${N_SAMPLE}_$(date +%Y-%m-%d_%H-%M-%S)}
export SAVE_DIR=${SAVE_DIR:-/data/yequan/compress_train/OPD/checkpoint/${EXPERIMENT_NAME}}
export NP_LOGGER=${NP_LOGGER:-'["console","wandb"]'}
mkdir -p "$SAVE_DIR"

python3 -m verl.trainer.main_np --config-name np_trainer \
    np.sigma=${SIGMA} np.n_sample=${N_SAMPLE} np.n_rollout=${N_ROLLOUT} \
    np.sample_method=${SAMPLE_METHOD} np.perturb_granularity=${PERTURB_GRANULARITY} \
    np.grad_estimate_sample=${GRAD_ESTIMATE_SAMPLE} \
    np.grad_estimate_sequence=${GRAD_ESTIMATE_SEQUENCE} \
    np.en_layerwise_perturbation=${EN_LAYERWISE} np.lr=${LR} np.token_agg=${TOKEN_AGG} \
    np.loss_type=${LOSS_TYPE} "np.perturb_rules=[${PERTURB_RULES}]" \
    np.teacher_model_path=${TEACHER_MODEL_PATH} np.log_prob_top_k=${LOG_PROB_TOP_K} \
    np.top_k_strategy=${TOP_K_STRATEGY} np.teacher_temperature=${TEACHER_TEMPERATURE} \
    np.num_engines=${NUM_ENGINES} np.num_iterations=${NUM_ITERATIONS} \
    np.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    np.worker_extension_cls='verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension' \
    model.path=${ACTOR_MODEL_PATH} \
    data.task_type=opd_math data.train_files=${TRAIN_DATASET} data.val_files=${EVAL_DATASET} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} data.val_max_samples=${VAL_MAX_SAMPLES} \
    trainer.project_name=${PROJECT_NAME} trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=${NP_LOGGER} trainer.default_local_dir=${SAVE_DIR} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
```

- [ ] **Step 2: Make executable + shellcheck the bracket/quoting**

Run:
```bash
chmod +x scripts/zo_opd/opd_np.sh
bash -n scripts/zo_opd/opd_np.sh && echo "syntax ok"
```
Expected: prints `syntax ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/zo_opd/opd_np.sh
git commit -m "feat(np): opd_np.sh launcher (np.* env interface)"
```

---

## Task 15: End-to-end verification + gradient cosine-similarity check

The spec's Verification ladder #2 and #3. These need a GPU; commit the scripts so they're runnable on the box.

**Files:**
- Create: `scripts/zo_opd/np_checks/check_grad_cosine.py`
- Create: `datasets/dapo-math-17k-1percent.parquet` (tiny slice for smoke runs)

- [ ] **Step 1: Create the tiny training slice**

Run:
```bash
conda run -n verl python -c "
import pandas as pd
df = pd.read_parquet('datasets/dapo-math-17k.parquet')
df.head(max(8, len(df)//100)).to_parquet('datasets/dapo-math-17k-1percent.parquet')
print('wrote', 'datasets/dapo-math-17k-1percent.parquet', 'rows=', max(8, len(df)//100))
"
```
Expected: prints the row count (≥8).

- [ ] **Step 2: Create the gradient cosine-sim check**

Create `scripts/zo_opd/np_checks/check_grad_cosine.py`:

```python
"""GPU verification (spec Verification #2): NP's estimated delta_W should align
with the true autograd gradient on one layer/batch. Reports cosine similarity;
PASS if >= 0.05 (token-granularity expected >= 0.1 with enough samples).

This loads the model TWICE: once via HF (eager, autograd) for the reference
gradient, once via the NP worker path for the estimate. Run on 1 GPU.

  conda run -n verl python scripts/zo_opd/np_checks/check_grad_cosine.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' \
      --n-sample 64 --repeats 50
"""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.np.seeding import noise_seed, draw_noise
from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--prompt", default="Compute 7*8. Answer:")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
    model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

    # locate the HF analog of the layer (HF uses split names; map down_proj directly).
    # For this check we target a real HF Linear by attribute walk.
    hf_name = args.layer  # HF Qwen has model.layers.N.mlp.down_proj as a real module
    mod = model
    for p in hf_name.split("."):
        mod = getattr(mod, p)
    W = mod.weight  # [d_out, d_in]

    # --- reference gradient: dL/dW where L = next-token CE on the prompt's last token ---
    captured = {}
    h = mod.register_forward_hook(lambda m, i, o: captured.__setitem__("x", i[0].detach()))
    W.requires_grad_(True)
    out = model(ids)
    logits = out.logits[:, -1, :]
    target = logits.argmax(-1)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    g_true = W.grad.detach().clone()
    h.remove()
    W.requires_grad_(False)
    x_t = captured["x"].reshape(-1, captured["x"].shape[-1])[-1]  # last-token input [d_in]

    # --- NP estimate: perturb W's OUTPUT row-wise via u, measure loss delta ---
    d_out = W.shape[0]
    dw = torch.zeros_like(g_true, dtype=torch.float32)
    base = F.cross_entropy(model(ids).logits[:, -1, :], target).item()
    for rep in range(args.repeats):
        u = torch.stack([
            draw_noise(noise_seed(0, rep, args.layer, 0, q), (d_out,), dev, torch.float32, "gaussian")
            for q in range(args.n_sample)
        ])  # [n_sample, d_out]
        L_q = []
        for q in range(args.n_sample):
            hh = mod.register_forward_hook(
                lambda m, i, o, uq=u[q]: o + args.sigma * uq)
            L_q.append(F.cross_entropy(model(ids).logits[:, -1, :], target).item())
            hh.remove()
        scales = sample_scale(torch.tensor(L_q), L_clean=base, sigma=args.sigma, mode="average")
        accumulate_delta_w(dw, scales=scales, u=u, x_t=x_t, normalize=False)
    dw.div_(args.repeats)

    cos = F.cosine_similarity(dw.flatten(), g_true.flatten(), dim=0).item()
    print(f"cosine(NP_dW, true_grad) = {cos:.4f}  (n_sample={args.n_sample}, repeats={args.repeats})")
    assert cos > 0.05, f"FAIL: cosine {cos:.4f} <= 0.05 (sign/scale bug?)"
    print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the cosine check (on a GPU box)**

Run:
```bash
conda run -n verl python scripts/zo_opd/np_checks/check_grad_cosine.py \
    --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 64 --repeats 50
```
Expected: prints a positive cosine and `PASS`. If negative/near-zero, the sign convention or `δy⊗x` assembly is wrong — fix `grad_estimator`/`assemble_layer_delta` before the end-to-end run.

- [ ] **Step 4: Run the small end-to-end OPD smoke (on a GPU box)**

Run:
```bash
ACTOR_MODEL_PATH=model/Qwen3-1.7B \
TRAIN_DATASET=datasets/dapo-math-17k-1percent.parquet \
TRAIN_MAX_SAMPLES=4 VAL_MAX_SAMPLES=4 \
N_SAMPLE=4 N_ROLLOUT=2 NUM_ITERATIONS=5 NUM_ENGINES=1 \
PERTURB_RULES='^model\.layers\.0\.mlp\.down_proj$' \
TEACHER_MODEL_PATH=model/Qwen3-1.7B \
bash scripts/zo_opd/opd_np.sh
```
Expected: completes 5 iterations; logs show `train/dW_norm/...` > 0 for the active layer at each step (the spec's "‖δW‖ > 0" assertion). No crash in teacher scoring or NCCL broadcast.

- [ ] **Step 5: Commit**

```bash
git add scripts/zo_opd/np_checks/check_grad_cosine.py datasets/dapo-math-17k-1percent.parquet
git commit -m "test(np): gradient cosine-sim check + 1% training slice for smoke runs"
```

---

## Task 16: ES regression check (belt-and-suspenders)

Confirm the NP work touched nothing in the shared ES paths.

- [ ] **Step 1: Confirm no existing files were modified**

Run:
```bash
git diff --stat HEAD~15 -- verl/verl/trainer/es/ verl/verl/workers/rollout/vllm_rollout/es_worker_extension.py verl/verl/trainer/main_es.py verl/verl/workers/actor/dp_actor.py
```
Expected: **empty output** (NP lives entirely in new files; no ES/PPO file changed). If anything shows, revert that incidental edit.

- [ ] **Step 2: Run the ES launcher on the known-good config (on a GPU box, optional but recommended)**

Run:
```bash
bash scripts/zo_opd/es.sh   # or opd_es.sh with prior known-good env
```
Expected: produces the same early loss curve as before the NP work (NP changed no shared code).

- [ ] **Step 3: Run the full NP unit-test suite**

Run: `conda run -n verl pytest verl/tests/np/ -v`
Expected: all unit tests PASS (seeding, layer_resolve, grad_estimator, teacher_scorer, perturbed_linear, apply_update_math).

- [ ] **Step 4: Final commit (if any cleanup)**

```bash
git add -A && git commit -m "test(np): full unit suite green; ES paths untouched" || echo "nothing to commit"
```

---

## Self-review notes (for the implementer)

- **Highest risk = Task 9** (decode driver) and the **attention-backend pin** (spec §5). The σ=0 smoke (Task 9 Step 3) is the gate: if it fails, the multi-query/shared-prefix/`slot_mapping=-1` path isn't exposed by the active backend — pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` and recheck before building Tasks 10+.
- **Per-step `x` capture** (Task 12 `_capture_x`) is the subtlest correctness point: `δW = Σ_t g_t ⊗ x_t` needs the clean `x_t` per step. Prefer the per-step capture loop; the last-step approximation is a documented fallback only.
- **`prompt_logprobs` alignment** (Task 12 TeacherScorer) is the likely off-by-one: confirm teacher position `p` aligns to student step `p` before trusting `L_t`. The teacher_scorer unit test (Task 5) covers the kernel; the alignment is exercised by the end-to-end smoke (Task 15 Step 4).
- **v1 simplifications shipped deliberately:** `perturb_method=antithetic`, `perturb_granularity=rollout`, `grad_estimate_sequence` beyond per-rollout reward, and the `union*` top-k strategies are config-accepted but their richest forms are refinements — the core path is `forward` / `token` / `only_stu`/`only_tch`. Document any not-yet-wired branch with a clear `NotImplementedError` rather than a silent no-op.
