# NP-HF Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, single-GPU `src/np_hf/` package that runs Node-Perturbation On-Policy-Distillation on plain HuggingFace forwards — batched 1+N perturbed decode over a persistent clean KV cache — with zero verl/Ray/vLLM dependency.

**Architecture:** A student HF model rolls out greedily; at each decode step a single forward runs `1+N` rows (row 0 clean, rows 1..N perturbed by independent noise added to matched linear-layer outputs) sharing the committed-prefix KV cache via `DynamicCache.batch_repeat_interleave`. Only row 0's new KV is kept (`batch_select_indices`). After the full rollout, a teacher HF model prefills the committed sequence once and per-token reverse-KL over a top-k set drives a zeroth-order rank-1 weight update `W ← W − lr·δW`. All matched layers are perturbed simultaneously by default.

**Tech Stack:** Python 3.12, PyTorch 2.8, transformers 4.56.1 (`DynamicCache`), pytest. Runs in the `verl` conda env (`/home/yequan/miniconda3/envs/verl/bin/python`). No Ray, no vLLM.

**Spec:** `docs/superpowers/specs/2026-06-03-np-hf-opd-trainer-design.md`

---

## Conventions for the implementing engineer

- **Run python via the verl env:** `/home/yequan/miniconda3/envs/verl/bin/python`. For pytest: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest <path> -v`.
- **CPU-only tests** (math, seeding, kv-slice logic on tiny tensors, estimator) need no GPU and run fast. **GPU tests** (σ=0 equivalence, oracle equivalence, grad cosine, smoke, bench) require a free GPU — set `CUDA_VISIBLE_DEVICES=<id>` to a free device. Each GPU task notes which kind it is.
- **Model paths** (already on disk):
  - Student (small, for tests): `/data/yequan/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*` (use the snapshot dir). Use `Qwen3-0.6B` for fast GPU tests; `Qwen3-1.7B` for the smoke run.
  - Teacher: `/data/yequan/huggingface/hub/models--Keven16--Qwen3-4B-Non-Thinking-RL-Math-Step500/snapshots/*`.
  - Resolve a snapshot path with: `ls -d /data/yequan/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/`.
- **`src/` is tracked** (the `src/` line was removed from `.gitignore`). New files under `src/np_hf/` will be picked up by `git add`. Verify once at Task 0.
- **Commit after every green step.** Small commits. Use the message shown in each step.
- All commit messages end with a trailing blank line then:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

```
src/np_hf/
  __init__.py
  seeding.py          # COPIED verbatim from verl/verl/trainer/np/seeding.py
  grad_estimator.py   # COPIED verbatim from verl/verl/trainer/np/grad_estimator.py
  reverse_kl.py       # reverse_kl_topk kernel (lifted from teacher_scorer.py)
  layer_resolve.py    # resolve_modules + active_layers_for_step (HF names in error msg)
  perturb.py          # PerturbState + make_perturb_hook: adds σ·u to output rows 1..N, captures x_t
  rollout.py          # RolloutEngine (Approach A): batched 1+N decode, persistent clean KV, row-0 slice
  rollout_oracle.py   # RolloutEngineOracle (Approach B): full-reprefill reference, tests only
  teacher.py          # TeacherScorer: HF teacher prefill -> per-token top-k logp -> reverse-KL
  estimator.py        # assemble_layer_delta + apply_update (no Ray)
  config.py           # NpHfConfig dataclass mirroring np.* knobs
  trainer.py          # NpHfTrainer: per-iter loop
  main.py             # CLI entry
  bench.py            # ms/step vs N sweep harness
  tests/
    __init__.py
    test_math_reuse.py     # copied math byte-matches verl originals (drift guard)  [CPU]
    test_seeding.py        # seeding determinism                                     [CPU]
    test_reverse_kl.py     # reverse_kl_topk kernel                                  [CPU]
    test_layer_resolve.py  # regex resolution + round-robin                          [CPU]
    test_perturb.py        # hook adds noise to rows 1..N only, captures x_t         [CPU]
    test_kv_slice.py       # expand/select-row-0 cache logic on tiny tensors         [CPU]
    test_estimator.py      # assemble_layer_delta end-to-end on synthetic signals    [CPU]
    test_sigma0_equiv.py   # GATE 1: σ=0 == model.generate, token-for-token          [GPU]
    test_oracle_equiv.py   # GATE 2: A == B oracle logits, σ>0                        [GPU]
    test_grad_cosine.py    # GATE 3: δW cosine vs autograd, single + all-layers       [GPU]
```

---

## Task 0: Package skeleton + math copy + drift guard

**Files:**
- Create: `src/np_hf/__init__.py`, `src/np_hf/tests/__init__.py`
- Create: `src/np_hf/seeding.py`, `src/np_hf/grad_estimator.py` (copied)
- Create: `src/np_hf/tests/test_math_reuse.py`

- [ ] **Step 1: Confirm `src/` is tracked**

Run: `git check-ignore src/np_hf 2>&1 || echo "TRACKED"`
Expected: prints `TRACKED` (the `src/` gitignore rule was removed). If it prints `src/np_hf`, STOP — `.gitignore` still ignores `src/`; remove the `src/` line before continuing.

- [ ] **Step 2: Create package dirs + empty `__init__.py` files**

```bash
mkdir -p src/np_hf/tests
printf '"""Standalone HuggingFace-forward Node-Perturbation OPD trainer. See docs/superpowers/specs/2026-06-03-np-hf-opd-trainer-design.md."""\n' > src/np_hf/__init__.py
touch src/np_hf/tests/__init__.py
```

- [ ] **Step 3: Copy the two validated math files verbatim**

```bash
cp verl/verl/trainer/np/seeding.py src/np_hf/seeding.py
cp verl/verl/trainer/np/grad_estimator.py src/np_hf/grad_estimator.py
```

- [ ] **Step 4: Write the drift-guard test**

`src/np_hf/tests/test_math_reuse.py`:
```python
"""Drift guard: the copied math files must stay byte-identical to the verl
originals they were lifted from. If verl's math changes, this fails loudly so
we re-copy intentionally rather than silently diverge."""
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]  # .../OPD
PAIRS = [
    ("src/np_hf/seeding.py", "verl/verl/trainer/np/seeding.py"),
    ("src/np_hf/grad_estimator.py", "verl/verl/trainer/np/grad_estimator.py"),
]


def test_copied_math_matches_verl_originals():
    for copied, original in PAIRS:
        a = (REPO / copied).read_bytes()
        b = (REPO / original).read_bytes()
        assert a == b, f"{copied} drifted from {original}; re-copy intentionally."
```

- [ ] **Step 5: Run the drift guard**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_math_reuse.py -v`
Expected: PASS (both files byte-match).

- [ ] **Step 6: Sanity-check the copied math imports**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -c "import sys; sys.path.insert(0,'src'); from np_hf.grad_estimator import sample_scale, accumulate_delta_w; from np_hf.seeding import noise_seed, draw_noise; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/np_hf/__init__.py src/np_hf/tests/__init__.py src/np_hf/seeding.py src/np_hf/grad_estimator.py src/np_hf/tests/test_math_reuse.py
git commit -m "np_hf: package skeleton + copied NP math + drift guard"
```

---

## Task 1: Seeding determinism test (lock the reused contract)

**Files:**
- Test: `src/np_hf/tests/test_seeding.py`

- [ ] **Step 1: Write the test**

`src/np_hf/tests/test_seeding.py`:
```python
"""The seeding contract is load-bearing: the rollout adds σ·u and the estimator
must regenerate the SAME u. Lock determinism + the keying fields."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # .../src
import torch
from np_hf.seeding import noise_seed, draw_noise


def test_noise_seed_is_deterministic_and_field_sensitive():
    base = noise_seed(0, 3, "model.layers.0.mlp.down_proj", 1, 2)
    assert base == noise_seed(0, 3, "model.layers.0.mlp.down_proj", 1, 2)
    # changing ANY field changes the seed
    assert base != noise_seed(1, 3, "model.layers.0.mlp.down_proj", 1, 2)
    assert base != noise_seed(0, 4, "model.layers.0.mlp.down_proj", 1, 2)
    assert base != noise_seed(0, 3, "model.layers.1.mlp.down_proj", 1, 2)
    assert base != noise_seed(0, 3, "model.layers.0.mlp.down_proj", 9, 2)
    assert base != noise_seed(0, 3, "model.layers.0.mlp.down_proj", 1, 7)


def test_draw_noise_regenerates_identically():
    seed = noise_seed(0, 0, "L", 0, 0)
    a = draw_noise(seed, (16,), torch.device("cpu"), torch.float32, "bernoulli")
    b = draw_noise(seed, (16,), torch.device("cpu"), torch.float32, "bernoulli")
    assert torch.equal(a, b)
    assert set(a.unique().tolist()) <= {-1.0, 1.0}
```

- [ ] **Step 2: Run it**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_seeding.py -v`
Expected: PASS (seeding.py already implements this).

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/tests/test_seeding.py
git commit -m "np_hf: seeding determinism test (lock reused contract)"
```

---

## Task 2: `reverse_kl.py` — the reverse-KL kernel

**Files:**
- Create: `src/np_hf/reverse_kl.py`
- Test: `src/np_hf/tests/test_reverse_kl.py`

- [ ] **Step 1: Write the failing test**

`src/np_hf/tests/test_reverse_kl.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from np_hf.reverse_kl import reverse_kl_topk


def test_reverse_kl_zero_when_student_equals_teacher():
    lp = torch.log_softmax(torch.tensor([1.0, 2.0, 0.5]), dim=-1)
    assert abs(float(reverse_kl_topk(lp, lp, "student_p"))) < 1e-6
    assert abs(float(reverse_kl_topk(lp, lp, "teacher_p"))) < 1e-6
    assert abs(float(reverse_kl_topk(lp, lp, "none"))) < 1e-6


def test_reverse_kl_weight_modes():
    s = torch.log_softmax(torch.tensor([2.0, 0.0, 0.0]), dim=-1)
    t = torch.log_softmax(torch.tensor([0.0, 0.0, 2.0]), dim=-1)
    diff = s - t
    assert torch.allclose(reverse_kl_topk(s, t, "student_p"), (s.exp() * diff).sum())
    assert torch.allclose(reverse_kl_topk(s, t, "teacher_p"), (t.exp() * diff).sum())
    assert torch.allclose(reverse_kl_topk(s, t, "none"), diff.sum())
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_reverse_kl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'np_hf.reverse_kl'`.

- [ ] **Step 3: Write `reverse_kl.py`** (lifted from `teacher_scorer.py`, pure kernel only)

`src/np_hf/reverse_kl.py`:
```python
"""Reverse-KL kernel over a top-k token set (minimization-oriented: lower =
student closer to teacher). Lifted verbatim from
verl/verl/trainer/np/teacher_scorer.py::reverse_kl_topk."""
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

- [ ] **Step 4: Run to verify it passes**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_reverse_kl.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/reverse_kl.py src/np_hf/tests/test_reverse_kl.py
git commit -m "np_hf: reverse_kl_topk kernel + tests"
```

---

## Task 3: `layer_resolve.py` — module matching + round-robin

**Files:**
- Create: `src/np_hf/layer_resolve.py`
- Test: `src/np_hf/tests/test_layer_resolve.py`

- [ ] **Step 1: Write the failing test**

`src/np_hf/tests/test_layer_resolve.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pytest
from np_hf.layer_resolve import resolve_modules, active_layers_for_step

NAMES = [
    "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj",
    "model.layers.1.mlp.down_proj", "model.layers.1.self_attn.q_proj",
]


def test_resolve_fullmatch_and_order():
    out = resolve_modules([r"^model\.layers\.\d+\.mlp\.down_proj$"], NAMES)
    assert out == ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]


def test_resolve_empty_raises_with_hf_hint():
    with pytest.raises(ValueError) as e:
        resolve_modules([r"^nomatch$"], NAMES, error_if_empty=True)
    # HF (unfused) names in the hint, NOT vLLM's qkv_proj/gate_up_proj
    assert "down_proj" in str(e.value)
    assert "qkv_proj" not in str(e.value)


def test_active_layers_round_robin_vs_all():
    matched = ["A", "B", "C"]
    assert active_layers_for_step(matched, 0, en_layerwise=True) == ["A"]
    assert active_layers_for_step(matched, 4, en_layerwise=True) == ["B"]  # 4 % 3 = 1
    assert active_layers_for_step(matched, 0, en_layerwise=False) == ["A", "B", "C"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_layer_resolve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'np_hf.layer_resolve'`.

- [ ] **Step 3: Write `layer_resolve.py`** (copy of verl's, with HF-unfused names in the error hint)

`src/np_hf/layer_resolve.py`:
```python
"""Resolve perturb_rules regexes to module names and pick active layers per step.

Resolution uses re.fullmatch: a rule must match the WHOLE module name. Output
preserves model.named_modules() order, de-duped. Names are plain-HuggingFace
(UNFUSED: q_proj / k_proj / v_proj / up_proj / gate_proj / down_proj), unlike the
vLLM trainer which sees fused qkv_proj / gate_up_proj."""
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
            f"Plain-HF linears are unfused: use self_attn.q_proj / mlp.down_proj "
            f"(not the vLLM-fused qkv_proj / gate_up_proj)."
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

- [ ] **Step 4: Run to verify it passes**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_layer_resolve.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/layer_resolve.py src/np_hf/tests/test_layer_resolve.py
git commit -m "np_hf: layer_resolve (HF-unfused module matching) + tests"
```

---

## Task 4: `perturb.py` — perturbation hook + state

**Files:**
- Create: `src/np_hf/perturb.py`
- Test: `src/np_hf/tests/test_perturb.py`

The hook is the core new mechanism. On a batched `1+N`-row forward, a matched
`nn.Linear`'s output has batch dim `1+N` (row 0 clean). The hook adds
`σ·u_{layer,q}` to rows `1..N` of the output, regenerating `u` from the seed, and
captures the clean row-0 input `x_t` for the rank-1 update. Behaviour is keyed by
a shared mutable `PerturbState` so the same model serves clean and perturbed
forwards without re-registering hooks.

**Design contract (used by later tasks):**
- `PerturbState` fields: `mode` (`"off"|"perturb"`), `step:int`, `global_seed:int`, `rollout:int`, `sigma:float`, `n_sample:int`, `sample_method:str`, `active_layers:set[str]`, and two output dicts `captured_x:{layer->Tensor[d_in]}`, `captured_u:{layer->Tensor[n_sample,d_out]}`.
- `make_perturb_hook(name, state)` returns a **forward hook** `hook(module, inputs, output)` registered via `module.register_forward_hook`. `inputs[0]` is `x` shape `[B, S, d_in]`; `output` is `[B, S, d_out]` (or a tuple whose first element is that tensor).
- Row convention: batch dim is `1+n_sample`; during decode `S==1`. Row 0 = clean. Capture `x_t = inputs[0][0, -1, :]` (clean row, last position). Perturb `output[1+q, -1, :] += sigma * u_q`.

- [ ] **Step 1: Write the failing test**

`src/np_hf/tests/test_perturb.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from np_hf.perturb import PerturbState, make_perturb_hook
from np_hf.seeding import noise_seed, draw_noise


class TinyLinear(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.lin(x)


def _run(state, d_in=4, d_out=5, n_sample=3):
    m = TinyLinear(d_in, d_out)
    h = m.lin.register_forward_hook(make_perturb_hook("L", state))
    B = 1 + n_sample
    x = torch.randn(B, 1, d_in)  # decode step: S=1
    # make all rows share row 0's input so we can isolate the additive noise
    x[:] = x[0:1]
    y = m(x)
    h.remove()
    return x, y


def test_off_mode_is_identity():
    st = PerturbState(mode="off")
    x, y = _run(st)
    # recompute clean
    assert not st.captured_x and not st.captured_u


def test_perturb_adds_noise_to_rows_1_to_n_only_and_captures():
    n_sample, d_in, d_out, sigma = 3, 4, 5, 0.1
    st = PerturbState(mode="perturb", step=2, global_seed=7, rollout=1,
                      sigma=sigma, n_sample=n_sample, sample_method="bernoulli",
                      active_layers={"L"})
    x, y = _run(st, d_in, d_out, n_sample)
    # row 0 must be the unperturbed clean output
    clean = y[0, -1]
    # rows 1..n must equal clean + sigma*u_q with the SAME u the estimator will regen
    for q in range(n_sample):
        seed = noise_seed(7, 2, "L", 1, q)
        u = draw_noise(seed, (d_out,), y.device, y.dtype, "bernoulli")
        assert torch.allclose(y[1 + q, -1], clean + sigma * u, atol=1e-5)
    # captured x_t = clean row last-position input
    assert torch.allclose(st.captured_x["L"], x[0, -1], atol=1e-6)
    # captured u stacked [n_sample, d_out], matches regenerated noise
    assert st.captured_u["L"].shape == (n_sample, d_out)


def test_layer_not_active_is_skipped():
    st = PerturbState(mode="perturb", step=0, global_seed=0, rollout=0,
                      sigma=0.1, n_sample=2, sample_method="bernoulli",
                      active_layers=set())  # "L" NOT active
    x, y = _run(st, n_sample=2)
    assert torch.allclose(y[1, -1], y[0, -1])  # no perturbation
    assert "L" not in st.captured_u
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_perturb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'np_hf.perturb'`.

- [ ] **Step 3: Write `perturb.py`**

`src/np_hf/perturb.py`:
```python
"""Perturbation forward-hook + shared mutable state for the batched 1+N decode.

On a matched nn.Linear, during a (1+n_sample)-row forward (row 0 = clean), the
hook adds sigma*u_q to output rows 1..n_sample and captures the clean row's
input x_t. Noise is regenerated from a seed (never stored beyond the step) so
the estimator reproduces identical u. Mirrors the vLLM PerturbedLinear shim but
as a hook over plain HF linears. See spec section 5a."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch

from .seeding import noise_seed, draw_noise


@dataclass
class PerturbState:
    mode: str = "off"                       # "off" | "perturb"
    step: int = 0
    global_seed: int = 0
    rollout: int = 0
    sigma: float = 0.0
    n_sample: int = 0
    sample_method: str = "bernoulli"
    active_layers: Set[str] = field(default_factory=set)
    captured_x: Dict[str, torch.Tensor] = field(default_factory=dict)
    captured_u: Dict[str, torch.Tensor] = field(default_factory=dict)


def _as_tensor_and_repack(output):
    """HF linears return a bare tensor. Be tolerant of (tensor, ...) tuples."""
    if isinstance(output, tuple):
        return output[0], lambda t: (t,) + output[1:]
    return output, lambda t: t


def make_perturb_hook(name: str, state: PerturbState):
    """Return a forward hook for module `name`, behavior keyed by `state`."""

    def hook(module, inputs, output):
        if state.mode != "perturb" or name not in state.active_layers:
            return output
        x = inputs[0]                      # [B, S, d_in]
        y, repack = _as_tensor_and_repack(output)
        # capture clean row 0's last-position input for the rank-1 update
        state.captured_x[name] = x[0, -1, :].detach().clone()
        sigma = float(state.sigma)
        if sigma == 0.0:
            # still record u (zeros contribution) so estimator shapes line up
            d_out = y.shape[-1]
            us = []
            for q in range(state.n_sample):
                seed = noise_seed(state.global_seed, state.step, name, state.rollout, q)
                us.append(draw_noise(seed, (d_out,), y.device, y.dtype, state.sample_method))
            state.captured_u[name] = torch.stack(us, dim=0)
            return output
        d_out = y.shape[-1]
        us = []
        for q in range(state.n_sample):
            seed = noise_seed(state.global_seed, state.step, name, state.rollout, q)
            u = draw_noise(seed, (d_out,), y.device, y.dtype, state.sample_method)
            us.append(u)
            y[1 + q, -1, :] = y[1 + q, -1, :] + sigma * u
        state.captured_u[name] = torch.stack(us, dim=0)
        return repack(y)

    return hook
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_perturb.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/perturb.py src/np_hf/tests/test_perturb.py
git commit -m "np_hf: perturbation forward-hook + PerturbState + tests"
```

---

## Task 5: `estimator.py` — δW assembly + apply

**Files:**
- Create: `src/np_hf/estimator.py`
- Test: `src/np_hf/tests/test_estimator.py`

Ports `assemble_layer_delta` + `apply_node_update` from the vLLM worker
extension, minus Ray/vLLM. Pure tensors in, weight mutated in place.

**Contract:**
- `assemble_layer_delta(L_q_per_step, L_clean_per_step, u_per_step, x_per_step, sigma, sample_mode, normalize, token_agg) -> Tensor[d_out, d_in]` — identical math to the vLLM version (reuses `sample_scale` + `accumulate_delta_w`).
- `apply_update(weight: nn.Parameter, delta_w, lr, update_clip=None) -> float` — `W ← W − lr·δW` in place, returns `‖δW‖`.

- [ ] **Step 1: Write the failing test**

`src/np_hf/tests/test_estimator.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from np_hf.estimator import assemble_layer_delta, apply_update


def test_assemble_shape_and_descent_direction():
    # 2 steps, n_sample=3, d_out=4, d_in=5
    T, n, d_out, d_in = 2, 3, 4, 5
    L_q = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 1.0, 2.0])]
    L_clean = [1.5, 1.0]
    u = [torch.randn(n, d_out) for _ in range(T)]
    x = [torch.randn(d_in) for _ in range(T)]
    dw = assemble_layer_delta(L_q, L_clean, u, x, sigma=0.1,
                              sample_mode="average", normalize=False, token_agg="sum")
    assert dw.shape == (d_out, d_in)
    assert torch.isfinite(dw).all()


def test_apply_update_is_gradient_descent():
    w = torch.nn.Parameter(torch.zeros(3, 2))
    dw = torch.ones(3, 2)
    norm = apply_update(w, dw, lr=0.5, update_clip=None)
    # W <- W - lr*dW = -0.5
    assert torch.allclose(w.data, torch.full((3, 2), -0.5))
    assert abs(norm - dw.norm().item()) < 1e-5


def test_apply_update_clip():
    w = torch.nn.Parameter(torch.zeros(2, 2))
    dw = torch.tensor([[10.0, -10.0], [0.1, -0.1]])
    apply_update(w, dw, lr=1.0, update_clip=1.0)
    # dw clamped to +/-1 before applying
    assert torch.allclose(w.data, torch.tensor([[-1.0, 1.0], [-0.1, 0.1]]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_estimator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'np_hf.estimator'`.

- [ ] **Step 3: Write `estimator.py`**

`src/np_hf/estimator.py`:
```python
"""Assemble per-layer delta_W from per-step NP signals and apply it in place.

Pure port of assemble_layer_delta + apply_node_update from the vLLM worker
extension, with no Ray/vLLM. delta_W ~= +dL/dW (cosine-validated), so the
update is gradient DESCENT: W <- W - lr*delta_W. See spec section 4 + 5."""
import torch

from .grad_estimator import sample_scale, accumulate_delta_w


def assemble_layer_delta(L_q_per_step, L_clean_per_step, u_per_step, x_per_step,
                         sigma, sample_mode, normalize, token_agg, eps=1e-6):
    """Build delta_W [d_out, d_in] from per-step signals (CPU/GPU agnostic)."""
    assert len(L_q_per_step) == len(u_per_step) == len(x_per_step)
    d_out = u_per_step[0].shape[1]
    d_in = x_per_step[0].shape[0]
    dw = torch.zeros(d_out, d_in, dtype=torch.float32)
    T = max(len(L_q_per_step), 1)
    for L_q, L_clean, u, x_t in zip(L_q_per_step, L_clean_per_step, u_per_step, x_per_step):
        scales = sample_scale(L_q.float(), L_clean, sigma, sample_mode)
        accumulate_delta_w(dw, scales=scales, u=u.float().cpu(), x_t=x_t.float().cpu(),
                           normalize=normalize, eps=eps)
    if token_agg == "mean":
        dw.div_(T)
    return dw


def apply_update(weight, delta_w, lr, update_clip=None):
    """W <- W - lr*delta_W in place. Returns ||delta_W|| (post-clip) as float."""
    dw = delta_w.to(weight.device, weight.dtype)
    if update_clip is not None:
        dw = dw.clamp(-float(update_clip), float(update_clip))
    with torch.no_grad():
        weight.add_(dw, alpha=-float(lr))
    return float(dw.norm().item())
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_estimator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/estimator.py src/np_hf/tests/test_estimator.py
git commit -m "np_hf: estimator (assemble_layer_delta + apply_update) + tests"
```

---

## Task 6: KV-slice helpers — expand + keep-row-0 (CPU logic test)

**Files:**
- Create: `src/np_hf/kv_utils.py`
- Test: `src/np_hf/tests/test_kv_slice.py`

The one-forward-per-step mechanic (spec 5b) needs two cache operations:
1. **expand** the batch-1 clean cache to `1+N` rows → `DynamicCache.batch_repeat_interleave(1+N)` (on a COPY, never the persistent cache).
2. after the forward, **keep only row 0** → `DynamicCache.batch_select_indices([0])`.

This task validates the expand→forward→select-row-0 round-trips correctly with a
fake cache on tiny tensors (no model), so the real rollout in Task 7 can trust it.
`copy_cache` deep-copies layer K/V so the persistent cache is untouched by the
throwaway forward.

- [ ] **Step 1: Write the failing test**

`src/np_hf/tests/test_kv_slice.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from transformers import DynamicCache
from np_hf.kv_utils import copy_cache, expand_to_batch, keep_row0


def _seed_cache(n_layers=2, H=2, L=3, D=4):
    """A batch-1 DynamicCache with L positions already written, n_layers deep."""
    c = DynamicCache()
    for i in range(n_layers):
        k = torch.randn(1, H, L, D)
        v = torch.randn(1, H, L, D)
        c.update(k, v, i)
    return c


def test_expand_repeats_batch_dim_without_touching_source():
    src = _seed_cache()
    src_k0 = src.layers[0].keys.clone()
    work = copy_cache(src)
    expand_to_batch(work, 1 + 3)  # 1 clean + 3 perturbed
    assert work.layers[0].keys.shape[0] == 4
    # source cache unchanged
    assert torch.equal(src.layers[0].keys, src_k0)


def test_keep_row0_selects_clean_row():
    src = _seed_cache()
    work = copy_cache(src)
    expand_to_batch(work, 4)
    # simulate a decode step writing 1 new position for all 4 rows
    for i in range(len(work.layers)):
        H, _, D = work.layers[i].keys.shape[1], None, work.layers[i].keys.shape[3]
        newk = torch.randn(4, H, 1, D)
        newv = torch.randn(4, H, 1, D)
        # row 0 distinct so we can assert it survives
        newk[0] = 111.0
        work.update(newk, newv, i)
    keep_row0(work)
    assert work.layers[0].keys.shape[0] == 1
    # the kept row is row 0 (last appended position == 111.0)
    assert torch.allclose(work.layers[0].keys[0, :, -1, :], torch.full_like(
        work.layers[0].keys[0, :, -1, :], 111.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_kv_slice.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'np_hf.kv_utils'`.

- [ ] **Step 3: Write `kv_utils.py`**

`src/np_hf/kv_utils.py`:
```python
"""KV-cache helpers for the one-forward-per-step batched 1+N decode (spec 5b).

Persistent clean cache stays batch-1. Each step we COPY it, expand to 1+N rows,
run the throwaway forward, then collapse back to row 0 only and adopt that as
the new persistent clean cache. Uses transformers 4.56 DynamicCache batch ops."""
import copy

import torch
from transformers import DynamicCache


def copy_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-copy a DynamicCache's K/V so the original is never mutated by the
    throwaway 1+N forward."""
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


def expand_to_batch(cache: DynamicCache, batch: int) -> None:
    """In place: repeat batch dim 1 -> `batch` (DynamicCache.batch_repeat_interleave)."""
    assert cache.layers[0].keys.shape[0] == 1, "expand expects a batch-1 cache"
    cache.batch_repeat_interleave(batch)


def keep_row0(cache: DynamicCache) -> None:
    """In place: collapse batch dim to just row 0 (the clean row)."""
    idx = torch.tensor([0], device=cache.layers[0].keys.device)
    cache.batch_select_indices(idx)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_kv_slice.py -v`
Expected: PASS (2 tests). If `batch_repeat_interleave`/`batch_select_indices` differ in this transformers build, inspect with `python -c "from transformers import DynamicCache; help(DynamicCache.batch_select_indices)"` and adjust; the test pins the required behavior.

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/kv_utils.py src/np_hf/tests/test_kv_slice.py
git commit -m "np_hf: KV expand/keep-row0 helpers + CPU logic tests"
```

---

## Task 7: `rollout.py` — RolloutEngine (Approach A, production decode)

**Files:**
- Create: `src/np_hf/rollout.py`

This is the production decode driver. No new unit test here — its correctness is
established by GATE 1 (Task 9, σ=0 vs `model.generate`) and GATE 2 (Task 10, vs
the oracle). This task only needs to import cleanly and expose the contract the
gates call.

**Contract:**
```python
class RolloutEngine:
    def __init__(self, model, perturb_rules, en_layerwise=False, device=None): ...
    def rollout(self, prompt_token_ids, *, max_tokens, n_sample, sigma,
                sample_method, global_seed, rollout_idx, temperature=0.0,
                eos_token_ids=None) -> dict
        # returns {
        #   "clean_tokens": list[int],
        #   "candidate_logits": list[Tensor[1+n_sample, vocab]],  # CPU
        #   "captured_x": list[{layer: Tensor[d_in]}],            # per step
        #   "captured_u": list[{layer: Tensor[n_sample, d_out]}], # per step
        # }
```

- [ ] **Step 1: Write `rollout.py`**

`src/np_hf/rollout.py`:
```python
"""Approach A: batched 1+N perturbed decode over a persistent clean KV cache.

Per step: copy the batch-1 clean cache, expand to 1+n_sample, run ONE forward
(hooks add sigma*u to rows 1..N of matched layers), read [1+N, vocab] logits,
sample row 0, collapse the throwaway cache to row 0 and adopt it as the new
clean cache. One forward per decode step (the free-lunch path). See spec 5b."""
from typing import Dict, List, Optional

import torch

from .layer_resolve import resolve_modules, active_layers_for_step
from .perturb import PerturbState, make_perturb_hook
from .kv_utils import copy_cache, expand_to_batch, keep_row0


class RolloutEngine:
    def __init__(self, model, perturb_rules, en_layerwise=False, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        names = [n for n, _ in model.named_modules()]
        self.matched = resolve_modules(list(perturb_rules), names, error_if_empty=True)
        self.en_layerwise = en_layerwise
        self.state = PerturbState()
        self._handles = []
        name_to_mod = dict(model.named_modules())
        for layer_name in self.matched:
            h = name_to_mod[layer_name].register_forward_hook(
                make_perturb_hook(layer_name, self.state))
            self._handles.append(h)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def _prefill(self, prompt_token_ids):
        """Prefill all but the last prompt token into a fresh batch-1 cache; the
        last token becomes the first decode query."""
        from transformers import DynamicCache
        cache = DynamicCache()
        ids = torch.tensor([prompt_token_ids[:-1]], dtype=torch.long, device=self.device)
        pos = torch.arange(len(prompt_token_ids) - 1, device=self.device).unsqueeze(0)
        self.state.mode = "off"
        self.model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
        return cache, prompt_token_ids[-1], len(prompt_token_ids) - 1

    @torch.no_grad()
    def rollout(self, prompt_token_ids, *, max_tokens, n_sample, sigma,
                sample_method, global_seed, rollout_idx, temperature=0.0,
                eos_token_ids=None):
        eos = set(eos_token_ids or [])
        clean_cache, next_token, cursor = self._prefill(list(prompt_token_ids))
        B = 1 + n_sample
        clean_tokens, candidate_logits, cap_x, cap_u = [], [], [], []

        for t in range(max_tokens):
            active = set(active_layers_for_step(self.matched, t, self.en_layerwise))
            self.state.captured_x, self.state.captured_u = {}, {}
            self.state.mode = "perturb"
            self.state.step = t
            self.state.global_seed = int(global_seed)
            self.state.rollout = int(rollout_idx)
            self.state.sigma = float(sigma)
            self.state.n_sample = int(n_sample)
            self.state.sample_method = sample_method
            self.state.active_layers = active

            work = copy_cache(clean_cache)
            expand_to_batch(work, B)
            ids = torch.full((B, 1), int(next_token), dtype=torch.long, device=self.device)
            pos = torch.full((B, 1), int(cursor), dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, position_ids=pos,
                             past_key_values=work, use_cache=True)
            logits = out.logits[:, -1, :]                    # [B, vocab]
            candidate_logits.append(logits.detach().to("cpu"))
            cap_x.append({k: v for k, v in self.state.captured_x.items()})
            cap_u.append({k: v for k, v in self.state.captured_u.items()})

            tok = self._sample(logits[0], temperature)
            clean_tokens.append(int(tok))
            keep_row0(work)                                  # collapse to clean row
            clean_cache = work
            next_token, cursor = int(tok), cursor + 1
            if int(tok) in eos:
                break

        self.state.mode = "off"
        return {"clean_tokens": clean_tokens, "candidate_logits": candidate_logits,
                "captured_x": cap_x, "captured_u": cap_u}

    @staticmethod
    def _sample(logits_row0, temperature):
        if not temperature or temperature == 0.0:
            return int(torch.argmax(logits_row0).item())
        probs = torch.softmax(logits_row0.float() / float(temperature), dim=-1)
        return int(torch.multinomial(probs, 1).item())
```

- [ ] **Step 2: Verify it imports**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -c "import sys; sys.path.insert(0,'src'); from np_hf.rollout import RolloutEngine; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/rollout.py
git commit -m "np_hf: RolloutEngine (Approach A, one-forward 1+N decode)"
```

---

## Task 8: `rollout_oracle.py` — RolloutEngineOracle (Approach B, reference)

**Files:**
- Create: `src/np_hf/rollout_oracle.py`

A deliberately simple, obviously-correct decode used only by GATE 2 to validate
Approach A. No KV cache at all: each step re-runs the FULL `[prompt + committed]`
prefix for all `1+N` rows (O(T²), fine for short test rollouts). Same hook, same
perturb state, same return contract as `RolloutEngine`.

- [ ] **Step 1: Write `rollout_oracle.py`**

`src/np_hf/rollout_oracle.py`:
```python
"""Approach B (oracle): full-reprefill 1+N decode, no KV cache. Obviously
correct, O(T^2), tests-only. Validates Approach A in GATE 2. Same return
contract as RolloutEngine."""
import torch

from .layer_resolve import resolve_modules, active_layers_for_step
from .perturb import PerturbState, make_perturb_hook


class RolloutEngineOracle:
    def __init__(self, model, perturb_rules, en_layerwise=False, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        names = [n for n, _ in model.named_modules()]
        self.matched = resolve_modules(list(perturb_rules), names, error_if_empty=True)
        self.en_layerwise = en_layerwise
        self.state = PerturbState()
        self._handles = [dict(model.named_modules())[n].register_forward_hook(
            make_perturb_hook(n, self.state)) for n in self.matched]

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def rollout(self, prompt_token_ids, *, max_tokens, n_sample, sigma,
                sample_method, global_seed, rollout_idx, temperature=0.0,
                eos_token_ids=None):
        eos = set(eos_token_ids or [])
        seq = list(prompt_token_ids)
        B = 1 + n_sample
        clean_tokens, candidate_logits, cap_x, cap_u = [], [], [], []
        for t in range(max_tokens):
            active = set(active_layers_for_step(self.matched, t, self.en_layerwise))
            self.state.captured_x, self.state.captured_u = {}, {}
            self.state.mode = "perturb"
            self.state.step, self.state.global_seed = t, int(global_seed)
            self.state.rollout, self.state.sigma = int(rollout_idx), float(sigma)
            self.state.n_sample, self.state.sample_method = int(n_sample), sample_method
            self.state.active_layers = active
            ids = torch.tensor([seq] * B, dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, use_cache=False)
            logits = out.logits[:, -1, :]
            candidate_logits.append(logits.detach().to("cpu"))
            cap_x.append({k: v for k, v in self.state.captured_x.items()})
            cap_u.append({k: v for k, v in self.state.captured_u.items()})
            tok = int(torch.argmax(logits[0]).item()) if not temperature else int(
                torch.multinomial(torch.softmax(logits[0].float() / temperature, -1), 1).item())
            clean_tokens.append(tok)
            seq.append(tok)
            if tok in eos:
                break
        self.state.mode = "off"
        return {"clean_tokens": clean_tokens, "candidate_logits": candidate_logits,
                "captured_x": cap_x, "captured_u": cap_u}
```

- [ ] **Step 2: Verify it imports**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -c "import sys; sys.path.insert(0,'src'); from np_hf.rollout_oracle import RolloutEngineOracle; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/rollout_oracle.py
git commit -m "np_hf: RolloutEngineOracle (Approach B, full-reprefill reference)"
```

---

## Task 9: GATE 1 — σ=0 byte-equivalence vs `model.generate` [GPU]

**Files:**
- Test: `src/np_hf/tests/test_sigma0_equiv.py`

The load-bearing gate: with σ=0, Approach A must reproduce stock greedy decode
token-for-token, and every perturbed logit row must equal row 0. This validates
the expand + row-0-slice + hook plumbing end to end on a real model.

- [ ] **Step 1: Write the test**

`src/np_hf/tests/test_sigma0_equiv.py`:
```python
"""GATE 1: with sigma=0, RolloutEngine == stock greedy model.generate,
token-for-token, and all 1+N logit rows are identical. Validates the KV
expand/slice + hook machinery. [GPU]"""
import glob, os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

MODEL = glob.glob("/data/yequan/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")[0]


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    return model, tok


def _eos_list(model, tok):
    """Use the model's full EOS id set (Qwen3 config eos is a LIST), so the
    rollout and model.generate stop on identical tokens."""
    eos = model.config.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return list(eos or ([tok.eos_token_id] if tok.eos_token_id is not None else []))


def test_sigma0_matches_generate(model_and_tok):
    from np_hf.rollout import RolloutEngine
    from transformers import GenerationConfig
    model, tok = model_and_tok
    prompt = tok("Compute 7*8. Answer:", return_tensors="pt").input_ids[0].tolist()
    max_new = 16
    eos = _eos_list(model, tok)

    eng = RolloutEngine(model, [r"^model\.layers\.\d+\.mlp\.down_proj$"])
    out = eng.rollout(prompt, max_tokens=max_new, n_sample=4, sigma=0.0,
                      sample_method="bernoulli", global_seed=0, rollout_idx=0,
                      temperature=0.0, eos_token_ids=eos)
    eng.remove_hooks()

    # stock greedy reference. Pin a clean greedy GenerationConfig so the model's
    # do_sample=True / temperature=0.6 defaults don't leak in.
    gcfg = GenerationConfig(do_sample=False, num_beams=1, eos_token_id=eos,
                            pad_token_id=tok.pad_token_id or eos[0], max_new_tokens=max_new)
    gen = model.generate(torch.tensor([prompt]).cuda(), generation_config=gcfg)
    ref = gen[0, len(prompt):].tolist()
    ref = ref[:len(out["clean_tokens"])]  # generate may pad past EOS

    assert out["clean_tokens"] == ref, (out["clean_tokens"], ref)
    # all perturbed rows identical to clean at sigma=0
    for step in out["candidate_logits"]:
        assert torch.allclose(step[0], step[1], atol=1e-4)
        assert step.shape[0] == 1 + 4
```

- [ ] **Step 2: Run the gate**

Run: `CUDA_VISIBLE_DEVICES=<free_gpu> /home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_sigma0_equiv.py -v`
Expected: PASS. If it FAILS on token mismatch, the KV expand/slice is wrong — debug `kv_utils`/`rollout` before proceeding (this gate blocks trust in the whole rollout).

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/tests/test_sigma0_equiv.py
git commit -m "np_hf: GATE 1 sigma=0 byte-equivalence vs model.generate"
```

---

## Task 10: GATE 2 — Approach A == Approach B oracle [GPU]

**Files:**
- Test: `src/np_hf/tests/test_oracle_equiv.py`

With a fixed seed and σ>0, the one-forward KV-slice path (A) must produce the
same per-step `[1+N, vocab]` logits as the full-reprefill oracle (B).

- [ ] **Step 1: Write the test**

`src/np_hf/tests/test_oracle_equiv.py`:
```python
"""GATE 2: Approach A (KV-slice) == Approach B (full reprefill) on per-step
logits, fixed seed, sigma>0. [GPU]"""
import glob, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
MODEL = glob.glob("/data/yequan/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")[0]


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    return model, tok


def test_A_matches_oracle(model_and_tok):
    from np_hf.rollout import RolloutEngine
    from np_hf.rollout_oracle import RolloutEngineOracle
    model, tok = model_and_tok
    prompt = tok("Compute 7*8. Answer:", return_tensors="pt").input_ids[0].tolist()
    kw = dict(max_tokens=8, n_sample=4, sigma=1e-2, sample_method="gaussian",
              global_seed=0, rollout_idx=0, temperature=0.0,
              eos_token_ids=[tok.eos_token_id])
    rules = [r"^model\.layers\.\d+\.mlp\.down_proj$"]

    a = RolloutEngine(model, rules); oa = a.rollout(prompt, **kw); a.remove_hooks()
    b = RolloutEngineOracle(model, rules); ob = b.rollout(prompt, **kw); b.remove_hooks()

    assert oa["clean_tokens"] == ob["clean_tokens"]
    assert len(oa["candidate_logits"]) == len(ob["candidate_logits"])
    for la, lb in zip(oa["candidate_logits"], ob["candidate_logits"]):
        assert torch.allclose(la, lb, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run the gate**

Run: `CUDA_VISIBLE_DEVICES=<free_gpu> /home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_oracle_equiv.py -v`
Expected: PASS. A mismatch means A's KV-slice diverges from re-prefill — debug before proceeding.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/tests/test_oracle_equiv.py
git commit -m "np_hf: GATE 2 Approach A == oracle B logits"
```

---

## Task 11: `teacher.py` — TeacherScorer (HF prefill → reverse-KL)

**Files:**
- Create: `src/np_hf/teacher.py`

Ports `TeacherScorer` from the vLLM `teacher_scorer.py`, replacing the vLLM
`prompt_logprobs` query with a single HF teacher forward over the committed
sequence (one prefill, all positions). Reuses the `_select_ids` strategy logic
and the min-fallback for missing teacher ids.

**Contract:**
```python
class TeacherScorer:
    def __init__(self, teacher_model, top_k, top_k_strategy,
                 teacher_temperature, weight_mode): ...
    def score_rollout(self, prefix_token_ids, candidate_logits, n_prompt_tokens)
        # candidate_logits: list[Tensor[1+n_sample, vocab]] (CPU), one per response step
        # returns (L_q_per_step: list[Tensor[n_sample]], L_clean_per_step: list[float])
```

- [ ] **Step 1: Write `teacher.py`**

`src/np_hf/teacher.py`:
```python
"""Teacher scoring via a single HF prefill. For the committed student sequence,
one teacher forward gives per-position logits; we take top-k logprobs at each
RESPONSE position and score every student row (clean + perturbed) with
reverse-KL over the chosen top-k set. Port of verl TeacherScorer, HF backend.
See spec section 4 + teacher_scorer.py for the strategy/fallback semantics."""
import torch

from .reverse_kl import reverse_kl_topk


class TeacherScorer:
    def __init__(self, teacher_model, top_k, top_k_strategy,
                 teacher_temperature, weight_mode):
        self.model = teacher_model
        self.device = next(teacher_model.parameters()).device
        self.top_k = int(top_k)
        self.top_k_strategy = top_k_strategy
        self.teacher_temperature = float(teacher_temperature)
        self.weight_mode = weight_mode

    @torch.no_grad()
    def _teacher_topk(self, prefix_token_ids, n_response):
        """Top-k teacher logprobs at each response position. The logits at
        position p predict token p+1, so for response tokens at absolute
        positions [P .. P+n_response-1] we read logits at [P-1 .. P+n_response-2]."""
        ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=False)
        logits = out.logits[0]                                  # [seqlen, vocab]
        total = len(prefix_token_ids)
        start = total - n_response - 1                          # predicts first response tok
        logp_by_pos, ids_by_pos = [], []
        for j in range(n_response):
            row = logits[start + j].float() / self.teacher_temperature
            lp = torch.log_softmax(row, dim=-1)
            top = torch.topk(lp, self.top_k)
            logp_by_pos.append(top.values.cpu())
            ids_by_pos.append(top.indices.cpu())
        return logp_by_pos, ids_by_pos

    def _select_ids(self, s_clean_full_logp, t_ids, t_logp, fallback):
        strat = self.top_k_strategy
        if strat == "only_tch":
            return t_ids, t_logp
        s_top = torch.topk(s_clean_full_logp, self.top_k).indices
        s_set = set(s_top.tolist()); t_set = set(t_ids.tolist())
        t_map = {int(i): float(lp) for i, lp in zip(t_ids.tolist(), t_logp.tolist())}
        if strat == "only_stu":
            ids_list = sorted(s_set)
        elif strat == "intersection":
            ids_list = sorted(s_set & t_set)
        elif strat in ("union", "union-intersection"):
            ids_list = sorted(s_set | t_set)
        else:
            raise ValueError(f"unknown top_k_strategy: {strat!r}")
        if not ids_list:
            return t_ids, t_logp
        ids = torch.tensor(ids_list, dtype=torch.long)
        t_aligned = torch.tensor([t_map.get(int(i), fallback) for i in ids_list],
                                 dtype=t_logp.dtype)
        return ids, t_aligned

    def score_rollout(self, prefix_token_ids, candidate_logits, n_prompt_tokens):
        n_response = len(candidate_logits)
        t_logp_pos, t_ids_pos = self._teacher_topk(prefix_token_ids, n_response)
        L_q_per_step, L_clean_per_step = [], []
        for t, cl in enumerate(candidate_logits):
            s_full = torch.log_softmax(cl.float(), dim=-1)      # [1+n_sample, vocab]
            t_ids, t_logp = t_ids_pos[t], t_logp_pos[t]
            fallback = float(t_logp.min().item()) if t_logp.numel() else -50.0
            ids, t_aligned = self._select_ids(s_full[0], t_ids, t_logp, fallback)
            s_logp = s_full[:, ids]                             # [1+n_sample, k']
            L_clean_per_step.append(
                float(reverse_kl_topk(s_logp[0], t_aligned, self.weight_mode)))
            L_q_per_step.append(torch.stack([
                reverse_kl_topk(s_logp[1 + q], t_aligned, self.weight_mode)
                for q in range(s_logp.shape[0] - 1)]))
        return L_q_per_step, L_clean_per_step
```

- [ ] **Step 2: Write a CPU smoke test that the scorer shapes are right**

`src/np_hf/tests/test_teacher_shapes.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from np_hf.teacher import TeacherScorer


class FakeTeacher(torch.nn.Module):
    """Returns deterministic logits [1, seqlen, vocab] so we can test scoring
    plumbing without a real model."""
    def __init__(self, vocab=32):
        super().__init__()
        self.vocab = vocab
        self._p = torch.nn.Parameter(torch.zeros(1))  # gives .parameters() a device

    def forward(self, input_ids, use_cache=False):
        seqlen = input_ids.shape[1]
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, seqlen, self.vocab, generator=g)
        return type("O", (), {"logits": logits})()


def test_score_rollout_shapes():
    teacher = FakeTeacher(vocab=32)
    scorer = TeacherScorer(teacher, top_k=8, top_k_strategy="only_stu",
                           teacher_temperature=1.0, weight_mode="student_p")
    n_prompt, n_resp, n_sample, vocab = 3, 4, 2, 32
    prefix = list(range(n_prompt + n_resp))
    cand = [torch.randn(1 + n_sample, vocab) for _ in range(n_resp)]
    Lq, Lc = scorer.score_rollout(prefix, cand, n_prompt)
    assert len(Lq) == n_resp and len(Lc) == n_resp
    assert all(t.shape == (n_sample,) for t in Lq)
    assert all(isinstance(x, float) for x in Lc)
```

- [ ] **Step 3: Run the smoke test**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_teacher_shapes.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/np_hf/teacher.py src/np_hf/tests/test_teacher_shapes.py
git commit -m "np_hf: TeacherScorer (HF prefill -> top-k reverse-KL) + shape test"
```

---

## Task 12: `config.py` — NpHfConfig dataclass

**Files:**
- Create: `src/np_hf/config.py`

- [ ] **Step 1: Write `config.py`**

`src/np_hf/config.py`:
```python
"""Config for the standalone HF NP-OPD trainer. Mirrors the vLLM np.* knobs,
dropping vLLM/Ray-only ones. NOTE the INVERTED default: en_layerwise_perturbation
defaults to False here (= perturb ALL matched layers per step), vs the vLLM
trainer's single-layer round-robin. See spec section 2 + 8."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NpHfConfig:
    student_model_path: str = ""
    teacher_model_path: str = ""
    # perturbation
    sigma: float = 0.01
    n_sample: int = 8
    n_rollout: int = 8
    sample_method: str = "bernoulli"          # gaussian | bernoulli | uniform
    grad_estimate_sample: str = "grpo"        # average | grpo
    token_agg: str = "sum"                     # sum | mean
    lr: float = 1e-4
    update_clip: Optional[float] = None
    perturb_rules: List[str] = field(
        default_factory=lambda: [r"^model\.layers\.\d+\.mlp\.down_proj$"])
    en_layerwise_perturbation: bool = False    # False => ALL matched layers (new default)
    # teacher / OPD reward
    log_prob_top_k: int = 256
    top_k_strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    teacher_temperature: float = 1.0
    teacher_offload: bool = False
    # rollout / loop
    max_tokens: int = 512
    temperature: float = 0.0
    num_iterations: int = 100
    global_seed: int = 0
```

- [ ] **Step 2: Verify it imports and defaults are right**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -c "import sys; sys.path.insert(0,'src'); from np_hf.config import NpHfConfig; c=NpHfConfig(); assert c.en_layerwise_perturbation is False and c.grad_estimate_sample=='grpo'; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/config.py
git commit -m "np_hf: NpHfConfig (inverted all-layers default)"
```

---

## Task 13: `trainer.py` — NpHfTrainer per-iteration loop

**Files:**
- Create: `src/np_hf/trainer.py`

Ties the pieces together: per iteration, run `n_rollout` rollouts, score each
against the teacher, accumulate per-layer signals across all rollouts/tokens,
assemble one δW per layer, apply, repeat. Optional teacher offload around the
single prefill. Asserts `‖δW‖>0` per applied layer.

**Contract:**
```python
class NpHfTrainer:
    def __init__(self, cfg: NpHfConfig, student, teacher, tokenizer): ...
    def train_iteration(self, prompts_token_ids: list[list[int]]) -> dict
        # runs one NP update over the given prompts; returns metrics dict
        # {"L_clean_mean": float, "dw_norm": {layer: float}, "n_tokens": int}
```

- [ ] **Step 1: Write `trainer.py`**

`src/np_hf/trainer.py`:
```python
"""NpHfTrainer: one NP-OPD update per iteration over a batch of prompts.

Per iteration:
  for each of n_rollout prompts: RolloutEngine.rollout -> candidate_logits, u, x
    -> TeacherScorer.score_rollout -> per-step (L_q, L_clean)
  per matched layer: gather (L_q, L_clean, u_layer, x_layer) across ALL rollouts
    and ALL response tokens -> assemble_layer_delta -> apply_update (W -= lr*dW).
See spec section 4."""
from typing import Dict, List

import torch

from .config import NpHfConfig
from .rollout import RolloutEngine
from .teacher import TeacherScorer
from .estimator import assemble_layer_delta, apply_update


class NpHfTrainer:
    def __init__(self, cfg: NpHfConfig, student, teacher, tokenizer):
        self.cfg = cfg
        self.student = student
        self.teacher = teacher
        self.tok = tokenizer
        self.engine = RolloutEngine(
            student, cfg.perturb_rules, en_layerwise=cfg.en_layerwise_perturbation)
        self.matched = self.engine.matched
        self.scorer = TeacherScorer(
            teacher, cfg.log_prob_top_k, cfg.top_k_strategy,
            cfg.teacher_temperature, cfg.reward_weight_mode)
        self._name_to_weight = {n: m.weight for n, m in student.named_modules()
                                if n in set(self.matched)}
        eos = tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])

    def train_iteration(self, prompts_token_ids: List[List[int]]) -> Dict:
        cfg = self.cfg
        # per-layer accumulators of per-token signals across all rollouts
        acc = {n: {"Lq": [], "Lc": [], "u": [], "x": []} for n in self.matched}
        L_clean_all = []

        for r, prompt in enumerate(prompts_token_ids[:cfg.n_rollout]):
            ro = self.engine.rollout(
                prompt, max_tokens=cfg.max_tokens, n_sample=cfg.n_sample,
                sigma=cfg.sigma, sample_method=cfg.sample_method,
                global_seed=cfg.global_seed, rollout_idx=r,
                temperature=cfg.temperature, eos_token_ids=self.eos_ids)
            if not ro["clean_tokens"]:
                continue
            prefix = list(prompt) + ro["clean_tokens"]
            Lq, Lc = self._score(prefix, ro["candidate_logits"], len(prompt))
            L_clean_all.extend(Lc)
            # scatter per-step signals into per-layer accumulators
            for t in range(len(ro["candidate_logits"])):
                for n in self.matched:
                    u = ro["captured_u"][t].get(n)
                    x = ro["captured_x"][t].get(n)
                    if u is None or x is None:
                        continue   # layer not active this step (round-robin)
                    acc[n]["Lq"].append(Lq[t])
                    acc[n]["Lc"].append(Lc[t])
                    acc[n]["u"].append(u)
                    acc[n]["x"].append(x)

        dw_norms = {}
        for n in self.matched:
            a = acc[n]
            if not a["u"]:
                continue
            dw = assemble_layer_delta(
                a["Lq"], a["Lc"], a["u"], a["x"], sigma=cfg.sigma,
                sample_mode=cfg.grad_estimate_sample,
                normalize=False, token_agg=cfg.token_agg)
            norm = apply_update(self._name_to_weight[n], dw, cfg.lr, cfg.update_clip)
            assert norm > 0.0, f"layer {n}: ||dW||==0, update did not land"
            dw_norms[n] = norm

        return {
            "L_clean_mean": float(sum(L_clean_all) / max(len(L_clean_all), 1)),
            "dw_norm": dw_norms,
            "n_tokens": len(L_clean_all),
        }

    def _score(self, prefix, candidate_logits, n_prompt):
        cfg = self.cfg
        if cfg.teacher_offload:
            self.teacher.cuda()
        try:
            Lq, Lc = self.scorer.score_rollout(prefix, candidate_logits, n_prompt)
        finally:
            if cfg.teacher_offload:
                self.teacher.cpu()
                torch.cuda.empty_cache()
        return Lq, Lc
```

- [ ] **Step 2: Verify it imports**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -c "import sys; sys.path.insert(0,'src'); from np_hf.trainer import NpHfTrainer; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/trainer.py
git commit -m "np_hf: NpHfTrainer per-iteration loop"
```

---

## Task 14: GATE 3 — gradient cosine vs autograd, single + all-layers [GPU]

**Files:**
- Test: `src/np_hf/tests/test_grad_cosine.py`

Validates the NP estimate points the same direction as the true autograd
gradient. Runs in BOTH single-layer and all-layers mode; the all-layers run
proves the new default's variance is survivable. Uses a self-consistent CE loss
(argmax target) so no ground truth is needed, mirroring the vLLM cosine gate.

- [ ] **Step 1: Write the test**

`src/np_hf/tests/test_grad_cosine.py`:
```python
"""GATE 3: cos(NP dW, autograd dL/dW) > 0.05 on one layer. Run single-layer AND
all-layers-active. All-layers being above threshold validates the new default's
variance budget. [GPU, slow]"""
import glob, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
MODEL = glob.glob("/data/yequan/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")[0]
LAYER = "model.layers.0.mlp.down_proj"


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    return model, tok


def _true_grad(model, tok, prompt_ids):
    """Autograd dL/dW at LAYER on a 1-step CE loss against the last-token argmax."""
    cap = {}
    mod = dict(model.named_modules())[LAYER]
    h = mod.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0][0, -1].detach()))
    mod.weight.requires_grad_(True)
    ids = torch.tensor([prompt_ids]).cuda()
    out = model(input_ids=ids)
    logits = out.logits[0, -1]
    target = logits.argmax().detach()
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    model.zero_grad(); loss.backward()
    g = mod.weight.grad.detach().clone()
    h.remove(); mod.weight.requires_grad_(False)
    return g


def _np_dw(model, tok, prompt_ids, en_layerwise, n_sample=64, repeats=20, sigma=1e-3):
    """NP estimate of dL/dW at LAYER via forward differences (same CE loss)."""
    from np_hf.perturb import PerturbState, make_perturb_hook
    from np_hf.estimator import assemble_layer_delta
    from np_hf.layer_resolve import resolve_modules
    names = [n for n, _ in model.named_modules()]
    rules = [r"^model\.layers\.\d+\.mlp\.down_proj$"] if not en_layerwise else [f"^{LAYER}$"]
    matched = resolve_modules(rules, names)
    state = PerturbState()
    handles = [dict(model.named_modules())[n].register_forward_hook(
        make_perturb_hook(n, state)) for n in matched]

    def loss_for(target, B):
        ids = torch.tensor([prompt_ids] * B).cuda()
        out = model(input_ids=ids, use_cache=False)
        lg = out.logits[:, -1, :]
        tgt = target.expand(B)
        return torch.nn.functional.cross_entropy(lg, tgt, reduction="none"), lg

    # clean target from row 0
    state.mode = "off"
    ids = torch.tensor([prompt_ids]).cuda()
    target = model(input_ids=ids, use_cache=False).logits[0, -1].argmax().detach()

    dw_acc = None
    for rep in range(repeats):
        state.mode = "perturb"; state.step = rep; state.global_seed = 0
        state.rollout = 0; state.sigma = sigma; state.n_sample = n_sample
        state.sample_method = "gaussian"; state.active_layers = {LAYER}
        state.captured_x, state.captured_u = {}, {}
        B = 1 + n_sample
        losses, _ = loss_for(target, B)
        L_clean = float(losses[0]); L_q = losses[1:].detach().cpu()
        u = state.captured_u[LAYER]; x = state.captured_x[LAYER]
        dw = assemble_layer_delta([L_q], [L_clean], [u], [x], sigma=sigma,
                                  sample_mode="average", normalize=False, token_agg="sum")
        dw_acc = dw if dw_acc is None else dw_acc + dw
    for h in handles:
        h.remove()
    state.mode = "off"
    return dw_acc / repeats


def _cosine(model, tok, en_layerwise):
    prompt = tok("Compute 7*8. Answer:", return_tensors="pt").input_ids[0].tolist()
    g = _true_grad(model, tok, prompt).float().flatten().cpu()
    dw = _np_dw(model, tok, prompt, en_layerwise).float().flatten().cpu()
    return float(torch.nn.functional.cosine_similarity(dw, g, dim=0))


def test_cosine_single_layer(model_and_tok):
    model, tok = model_and_tok
    cos = _cosine(model, tok, en_layerwise=True)
    print(f"single-layer cosine = {cos:.4f}")
    assert cos > 0.05, cos


def test_cosine_all_layers(model_and_tok):
    model, tok = model_and_tok
    cos = _cosine(model, tok, en_layerwise=False)
    print(f"all-layers cosine = {cos:.4f}")
    assert cos > 0.05, cos
```

- [ ] **Step 2: Run the gate**

Run: `CUDA_VISIBLE_DEVICES=<free_gpu> /home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_grad_cosine.py -v -s`
Expected: both PASS, printed cosines > 0.05 (single-layer typically higher than all-layers). If all-layers is below threshold, that is the reported finding from spec gate 3 — raise `n_sample`/`repeats` in the test to confirm direction, and record the result; do not silently weaken the assert.

- [ ] **Step 3: Commit**

```bash
git add src/np_hf/tests/test_grad_cosine.py
git commit -m "np_hf: GATE 3 gradient cosine vs autograd (single + all-layers)"
```

---

## Task 15: `main.py` + end-to-end smoke [GPU]

**Files:**
- Create: `src/np_hf/main.py`
- Create: `scripts/zo_opd/np_hf_smoke.sh`

- [ ] **Step 1: Write `main.py`**

`src/np_hf/main.py`:
```python
"""CLI entry for the standalone HF NP-OPD trainer. Loads student + teacher,
reads prompts from a parquet (verl format: a 'prompt' or 'question' column, or
pre-tokenized 'input_ids'), and runs NpHfTrainer for cfg.num_iterations."""
import argparse, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # .../src
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from np_hf.config import NpHfConfig
from np_hf.trainer import NpHfTrainer


def load_prompts(parquet_path, tokenizer, limit):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    col = "prompt" if "prompt" in df.columns else (
        "question" if "question" in df.columns else None)
    out = []
    for _, row in df.head(limit).iterrows():
        if col is None and "input_ids" in df.columns:
            out.append(list(row["input_ids"]))
        else:
            text = row[col]
            if isinstance(text, list):  # chat format
                text = tokenizer.apply_chat_template(text, tokenize=False,
                                                     add_generation_prompt=True)
            out.append(tokenizer(text).input_ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prompts", required=True, help="parquet of prompts")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--n-rollout", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--teacher-offload", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32, attn_implementation="eager").eval()
    if not args.teacher_offload:
        teacher = teacher.cuda()

    cfg = NpHfConfig(student_model_path=args.student, teacher_model_path=args.teacher,
                     n_rollout=args.n_rollout, max_tokens=args.max_tokens,
                     sigma=args.sigma, lr=args.lr, num_iterations=args.iters,
                     teacher_offload=args.teacher_offload)
    prompts = load_prompts(args.prompts, tok, args.n_rollout)
    trainer = NpHfTrainer(cfg, student, teacher, tok)
    for it in range(args.iters):
        m = trainer.train_iteration(prompts)
        nz = sum(1 for v in m["dw_norm"].values() if v > 0)
        print(f"[iter {it}] L_clean={m['L_clean_mean']:.4f} "
              f"tokens={m['n_tokens']} layers_updated={nz}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke launcher**

`scripts/zo_opd/np_hf_smoke.sh`:
```bash
#!/usr/bin/env bash
# 5-iter end-to-end smoke for the standalone HF NP-OPD trainer.
set -euo pipefail
GPU="${1:-0}"
PY=/home/yequan/miniconda3/envs/verl/bin/python
STUDENT=$(ls -d /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/ | head -1)
TEACHER=$(ls -d /data/yequan/huggingface/hub/models--Keven16--Qwen3-4B-Non-Thinking-RL-Math-Step500/snapshots/*/ | head -1)
PROMPTS="${PROMPTS:-datasets/test_data/math-500/test.parquet}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/np_hf/main.py \
    --student "$STUDENT" --teacher "$TEACHER" --prompts "$PROMPTS" \
    --iters 5 --n-rollout 4 --max-tokens 64 --teacher-offload
```

- [ ] **Step 3: Confirm a prompts parquet exists; pick one**

Run: `ls datasets/test_data/*/test.parquet 2>/dev/null | head; ls datasets/*.parquet 2>/dev/null | head`
Expected: at least one parquet path. Set `PROMPTS=` to a real one if the default is missing.

- [ ] **Step 4: Run the smoke (5 iters)**

Run: `bash scripts/zo_opd/np_hf_smoke.sh <free_gpu>`
Expected: 5 lines `[iter k] L_clean=... tokens=... layers_updated=N` with `layers_updated > 0` every iter and no crash/NaN. `L_clean` need not monotonically fall in 5 iters; the gate is "runs, updates land, finite."

- [ ] **Step 5: Commit**

```bash
git add src/np_hf/main.py scripts/zo_opd/np_hf_smoke.sh
git commit -m "np_hf: CLI entry + 5-iter end-to-end smoke launcher"
```

---

## Task 16: `bench.py` — free-lunch N-sweep benchmark [GPU]

**Files:**
- Create: `src/np_hf/bench.py`

Measures ms/decode-step vs N ∈ {0,1,4,8,16,32} at fixed seq len, reports the
overhead ratio t(N)/t(0), the knee, and GPU memory vs N. This is the test of the
free-lunch hypothesis (spec section 7).

- [ ] **Step 1: Write `bench.py`**

`src/np_hf/bench.py`:
```python
"""Free-lunch benchmark: ms/decode-step vs N (number of phantom perturbed rows).
N=0 is the clean baseline (no hooks). Reports t(N)/t(0) and the knee. See spec 7."""
import argparse, glob, sys, pathlib, statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from np_hf.rollout import RolloutEngine


def time_rollout(model, prompt_ids, n_sample, max_tokens, sigma):
    rules = [r"^model\.layers\.\d+\.mlp\.down_proj$"]
    if n_sample == 0:
        # baseline: plain greedy generate of the same length
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor([prompt_ids]).cuda()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / max_tokens
        return ms, torch.cuda.max_memory_allocated() / 1e9
    eng = RolloutEngine(model, rules)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    out = eng.rollout(prompt_ids, max_tokens=max_tokens, n_sample=n_sample,
                      sigma=sigma, sample_method="bernoulli", global_seed=0,
                      rollout_idx=0, temperature=0.0, eos_token_ids=[])
    t1.record(); torch.cuda.synchronize()
    eng.remove_hooks()
    steps = len(out["clean_tokens"])
    ms = t0.elapsed_time(t1) / max(steps, 1)
    return ms, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=glob.glob(
        "/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    prompt = tok("Compute 7*8 and explain.", return_tensors="pt").input_ids[0].tolist()

    Ns = [0, 1, 4, 8, 16, 32]
    results = {}
    for N in Ns:
        for _ in range(args.warmup):
            time_rollout(model, prompt, N, args.max_tokens, 0.01)
        samples = [time_rollout(model, prompt, N, args.max_tokens, 0.01)
                   for _ in range(args.reps)]
        ms = statistics.median(s[0] for s in samples)
        mem = max(s[1] for s in samples)
        results[N] = (ms, mem)

    base = results[0][0]
    print(f"{'N':>4} {'ms/step':>10} {'t(N)/t(0)':>10} {'peakGB':>8}")
    knee = None
    for N in Ns:
        ms, mem = results[N]
        ratio = ms / base
        if knee is None and N > 0 and ratio > 1.10:
            knee = N
        print(f"{N:>4} {ms:>10.3f} {ratio:>10.3f} {mem:>8.2f}")
    print(f"\nknee (first N with >10% overhead): {knee}")
    print(f"free-lunch up to N=8: {'CONFIRMED' if results[8][0]/base <= 1.2 else 'FALSIFIED'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `CUDA_VISIBLE_DEVICES=<free_gpu> /home/yequan/miniconda3/envs/verl/bin/python src/np_hf/bench.py`
Expected: a table of N / ms-per-step / ratio / peak-GB, a knee value, and a CONFIRMED/FALSIFIED verdict. Either verdict is a valid result — record the printed table.

- [ ] **Step 3: Record the result in the results wiki**

Append a dated block to `docs/results/zo_opd.md` with the benchmark table and the CONFIRMED/FALSIFIED verdict (create a `## [2026-06-XX] np_hf free-lunch benchmark` section). Add a one-line entry to `docs/log.md` and a row to `docs/index.md` if not already catalogued.

- [ ] **Step 4: Commit**

```bash
git add src/np_hf/bench.py docs/results/zo_opd.md docs/log.md docs/index.md
git commit -m "np_hf: free-lunch N-sweep benchmark + recorded result"
```

---

## Task 17: Full suite green + import-purity gate (spec success criterion 6)

**Files:** none (verification only)

- [ ] **Step 1: Run all CPU tests**

Run: `/home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/ -v -k "not equiv and not cosine"`
Expected: all CPU tests PASS.

- [ ] **Step 2: Run all GPU gates**

Run: `CUDA_VISIBLE_DEVICES=<free_gpu> /home/yequan/miniconda3/envs/verl/bin/python -m pytest src/np_hf/tests/test_sigma0_equiv.py src/np_hf/tests/test_oracle_equiv.py src/np_hf/tests/test_grad_cosine.py -v -s`
Expected: all GATE tests PASS.

- [ ] **Step 3: Verify zero verl/Ray/vLLM imports (spec success criterion 6)**

Run: `! grep -rn -E "import (ray|vllm)|from (ray|vllm|verl)" src/np_hf/ --include="*.py"`
Expected: prints nothing and exits 0 (no matches). If any match, the package leaked a dependency — remove it.

- [ ] **Step 4: Update the wiki design doc**

Add a new section to `docs/wiki/zo_np_trainer.md` (or a sibling `docs/wiki/zo_np_hf_trainer.md`) summarizing the HF backend: module map, the one-forward-KV-slice mechanic, the inverted all-layers default, and the four gate results. Catalog it in `docs/index.md`; append to `docs/log.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "np_hf: wiki + index + log for the HF NP-OPD trainer"
```

---

## Done criteria (maps to spec section 9)

| # | Spec criterion | Task |
|---|---|---|
| 1 | σ=0 reproduces greedy decode | Task 9 (GATE 1) |
| 2 | One-forward KV-slice == oracle | Task 10 (GATE 2) |
| 3 | δW cosine vs autograd (single + all-layers) | Task 14 (GATE 3) |
| 4 | End-to-end OPD iter runs, ‖δW‖>0, no crash | Task 13 + Task 15 |
| 5 | Free-lunch curve measured & reported | Task 16 |
| 6 | Zero verl/Ray/vLLM imports in src/np_hf/ | Task 17 step 3 |
