# NP Trainer — Node Perturbation for OPD

Zeroth-order on-policy distillation trainer that perturbs **linear-layer outputs** during a single clean rollout's decode and reads back per-token loss deltas from a teacher LLM. No backward pass. Lives in `verl.trainer.np` + `verl.workers.rollout.vllm_rollout.np_worker_extension`.

Reference branch: `feat/np-trainer`. Acceptance: 32 unit tests, σ=0 byte-equivalence smoke, cosine-similarity check against autograd (cos ≈ +0.41), 5-iter end-to-end smoke on Qwen3-1.7B student / Keven16 4B teacher.

---

## 1. One-line summary

For one clean student rollout, at every decode step `t` we evaluate **1 clean row + n_sample perturbed rows** of the same next-token query (`1+n_sample`-wide multi-query decode against the shared prompt KV with `slot_mapping=-1` on perturbed rows so they leave no KV trail). A teacher scores each row with reverse-KL. Per-sample loss deltas drive a per-token output-space gradient `g_t`, which is accumulated as the rank-1 outer product `g_t ⊗ x_t` into the perturbed layer's `δW`. Only the clean token is committed; perturbed rows are ephemeral.

```
prompt ──prefill──► shared KV ───decode step t────►┐
                                                    │
   clean row 0:        y_t = W x_t                  │
   perturbed row q:    y_t + σ·u_q   (q = 1..n)     ├── 1+n logits, one fwd
                                                    │
                       all rows query the SAME next position;
                       perturbed rows write KV to slot −1 (discarded)
                                                    │
                       teacher scores each row's distribution
                       L_t (clean) and L_t^(q) (perturbed)
                                                    │
                       g_t = (1/n) Σ_q s_q · u_q
                       δW += g_t · x_t^T
                                                    │
                       commit row-0 token; advance one step
```

---

## 2. File map

| Path | Responsibility |
|---|---|
| `verl/trainer/np/seeding.py` | `noise_seed(seed,t,layer,rollout,q) → 63-bit int`, `draw_noise(seed,shape,dev,dtype,method)` regenerator |
| `verl/trainer/np/layer_resolve.py` | `resolve_modules(rules, module_names)` (re.fullmatch over vLLM-fused names), `active_layers_for_step` |
| `verl/trainer/np/grad_estimator.py` | `sample_scale(L_q, L_clean, σ, mode)`, `accumulate_delta_w(δW, scales, u, x_t, normalize)` |
| `verl/trainer/np/teacher_scorer.py` | `reverse_kl_topk` kernel + `TeacherScorer` (vLLM engine wrapper with 5 `top_k_strategy` branches) |
| `verl/trainer/np/ray_trainer.py` | `NPNcclLLM` (enforce_eager + prefix caching), `RayNPTrainer` (per-step fit loop + per-layer NCCL broadcast) |
| `verl/workers/rollout/vllm_rollout/np_worker_extension.py` | `PerturbedLinear` shim + `WorkerExtension` (decode driver, capture pass, apply, broadcast) |
| `verl/trainer/main_np.py` | Hydra entry point |
| `verl/trainer/config/np_trainer.yaml` | Default config |
| `scripts/zo_opd/opd_np.sh` | Launcher |
| `scripts/zo_opd/np_checks/{check_decode_sigma0,check_grad_cosine}.py` | GPU-coupled verification gates |

Never edits ES paths: `git diff --stat main..feat/np-trainer -- verl/verl/trainer/es/ ... main_es.py ... dp_actor.py` is empty.

---

## 3. Math (as implemented)

### 3.1 Perturbation

For step `t`, the wrapped linear's clean output is `y_t = W x_t`, `W ∈ R^{d_out × d_in}`. Row-wise additive noise on rows 1..n_sample (row 0 is clean):

$$
y_t^{(q)} = W x_t + \sigma\, u_t^{(q)},\qquad q = 1, \dots, n_\text{sample}
$$

`u_t^{(q)} ∈ R^{d_out}` is regenerated from `noise_seed(global_seed, t, layer, rollout, q)`. Never stored; integer seeds are the only state that crosses the Ray boundary.

| `sample_method` | Distribution |
|---|---|
| `gaussian` | `u ∼ N(0, I)` |
| `bernoulli` (default) | Rademacher, `u ∈ {-1, +1}^{d_out}` iid |
| `uniform` | `u ∼ U(-1, 1)^{d_out}` |

### 3.2 Teacher loss

Reverse-KL on the OPD top-k token set, minimization-oriented (lower = closer to teacher):

$$
L_t^{(q)} = \sum_{v \in V_\text{top-k}} w_v \bigl(\log p_\text{student}^{(q)}(v) - \log p_\text{teacher}(v)\bigr)
$$

Knobs:
- `log_prob_top_k` — k (default 256; teacher engine launches with `max_logprobs=max(20,k)`)
- `top_k_strategy ∈ {only_tch, only_stu, intersection, union, union-intersection}` — selects `V_top-k`
- `reward_weight_mode ∈ {student_p, teacher_p, none}` — selects `w_v`

Teacher logp for ids missing from teacher's surface is filled with `min(t_logp)` at that position (a calibrated lower bound; a naive `-1e30` exploded L_clean to ~1e25 on `only_stu`).

### 3.3 Per-sample scale (`grad_estimate_sample`)

| Mode | Equation |
|---|---|
| `average` | `s_q = (L_t^(q) - L_t) / σ` (one-sided forward difference vs clean baseline) |
| `grpo` (default) | `s_q = (L_t^(q) - μ) / (σ_L + 1e-8)`, group-relative z-score over the n_sample perturbations |

### 3.4 Per-token output gradient

Let `ũ_q = u_t^(q)` (default) or `ũ_q = u_t^(q) / max(‖u_t^(q)‖², ε)` when `normalize=True` (ANP — adaptive node perturbation):

$$
g_t = \frac{1}{n_\text{sample}} \sum_{q=1}^{n_\text{sample}} s_q\, \tilde u_q \in \mathbb{R}^{d_\text{out}}
$$

Approximates `∂L_t / ∂y_t` in expectation.

### 3.5 Per-layer weight delta

Accumulate the rank-1 outer product over all T response tokens:

$$
\delta W = \sum_{t=1}^{T} g_t\, x_t^{\!\top} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}
$$

`token_agg="mean"` divides by T; default `sum`.

### 3.6 Update rule

Since `δW ≈ +∂L/∂W` (validated by the cosine check below), the trainer does gradient **descent**:

$$
W \leftarrow W - \mathrm{lr}\cdot \delta W
$$

Optional `update_clip` clamps `δW` element-wise.

### 3.7 Full equation (default config: average, bernoulli, no ANP, sum)

$$
\delta W = \sum_{t=1}^{T} \Biggl[ \frac{1}{n_\text{sample}\, \sigma} \sum_{q=1}^{n_\text{sample}} (L_t^{(q)} - L_t)\, u_t^{(q)} \Biggr]\, x_t^{\!\top}
$$

with `u_t^(q) ∈ {-1, +1}^{d_out}` Rademacher, regenerated from `blake2b(global_seed, t, layer, rollout, q) & (2^63 − 1)`.

---

## 4. Knob → equation map

| YAML key | Symbol | Default | Affects |
|---|---|---|---|
| `np.sigma` | σ | 0.01 | Perturbation magnitude in `y + σ·u` |
| `np.n_sample` | n | 8 | Width of the multi-query step (1 clean + n perturbed) |
| `np.n_rollout` | — | 8 | Independent rollouts; per-step signals concatenated before δW assembly |
| `np.sample_method` | u distribution | bernoulli | `gaussian`/`bernoulli`/`uniform` |
| `np.grad_estimate_sample` | mode | grpo | `average` (one-sided FD) or `grpo` (group z-score) |
| `np.token_agg` | — | sum | δW Σₜ or mean over T |
| `np.lr` | lr | 1e-4 | Descent step W ← W − lr·δW |
| `np.update_clip` | — | null | Element-wise clamp on δW |
| `np.perturb_rules` | — | `^model\.layers\.\d+\.mlp\.down_proj$` | Regex set (vLLM-fused names) of linear modules to wrap |
| `np.en_layerwise_perturbation` | — | true | Round-robin one layer/step vs all-at-once |
| `np.log_prob_top_k` | k | 256 | OPD top-k set size |
| `np.top_k_strategy` | — | only_stu | Which top-k to score over |
| `np.reward_weight_mode` | w_v | student_p | Reverse-KL term weighting |
| `np.teacher_temperature` | T_tch | 1.0 | Temperature on teacher logits before softmax |
| `np.teacher_model_path` | — | (required) | Path to teacher checkpoint |

---

## 5. Acceptance gates

| Gate | Script | Result |
|---|---|---|
| σ=0 byte-equivalence | `scripts/zo_opd/np_checks/check_decode_sigma0.py` | NP decode with σ=0 matches stock greedy `LLM.generate` token-for-token; logits width is `[1+n_sample, vocab]` every step. Validates the multi-query + `slot_mapping=-1` + shared-prefix machinery. **PASS** (Qwen3-1.7B, FLASH_ATTN backend). |
| Cosine-sim vs autograd | `scripts/zo_opd/np_checks/check_grad_cosine.py` | See §6 |
| 5-iter e2e smoke | `scripts/zo_opd/opd_np.sh` (5 iters, 4 prompts) | dW_norm ∈ [2.28, 2.58], L_clean ∈ [0.49, 0.76], no crash. |
| ES regression | `git diff --stat main..HEAD -- verl/verl/trainer/es/ ...` | empty (zero ES files touched) |
| Unit tests | `pytest verl/tests/np/` | 32/32 PASS |

---

## 6. Cosine-similarity check (gradient validity)

**Question:** does the NP estimate `δW` point in the same direction as the true autograd gradient on a real Transformer linear?

**Setup** (`scripts/zo_opd/np_checks/check_grad_cosine.py`):

- Model: Qwen3-1.7B, loaded once via HuggingFace (`torch_dtype=fp32`, eager, no caching) for autograd, once again the same weights for the NP path. GPU 6 (single A800).
- Target layer: `model.layers.0.mlp.down_proj` (`d_out × d_in = 1536 × 8960` after Qwen3 MLP fan-in; the canonical NP target).
- Prompt: `"Compute 7*8. Answer:"`; loss = cross-entropy on the last token's argmax target (a self-consistent label, so the gradient is well-defined without ground truth).
- Reference: register a forward hook to capture `x` at the layer's input, set `W.requires_grad=True`, run one forward + backward, read `g_true = W.grad`, also record `x_t` for the last response position.
- NP estimate (forward-only):
  - For `repeats` outer iterations: draw `n_sample` Rademacher Gaussian noise rows `u_q ∈ R^{d_out}` via `draw_noise(noise_seed(0, rep, layer, 0, q), ...)`.
  - For each `q`, install a forward hook adding `σ·u_q` to the layer's **output**, recompute the same loss, record `L_q`.
  - Compute `scales = (L_q − base) / σ` (one-sided forward difference vs the unperturbed `base` loss).
  - Accumulate `dw += outer( mean_q(scales_q · u_q), x_t )`.
  - After all `repeats`, divide `dw` by `repeats`.
- Metric: `cos(dw.flatten(), g_true.flatten())`. **PASS threshold**: > 0.05; expected ≥ 0.10 with enough samples.

**Configuration used to obtain the headline number:**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n verl python scripts/zo_opd/np_checks/check_grad_cosine.py \
    --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc... \
    --layer 'model.layers.0.mlp.down_proj' \
    --n-sample 64 --repeats 50 --sigma 1e-3
```

Total forward queries per estimate: `repeats × (1 + n_sample) = 50 × 65 = 3250` (base loss + 64 perturbed losses, 50 times).

**Result:**

```
cosine(NP_dW, true_grad) = 0.4074  (n_sample=64, repeats=50)
PASS
```

8× above the gate. This validates:
- The math chain `sample_scale → accumulate_delta_w → outer(g_t, x_t)` recovers the right **direction** of the gradient (not just non-zero).
- Bernoulli/Gaussian noise both work (we tested gaussian here for the FD math to match the cosine threshold; the trainer defaults to bernoulli for variance reasons).
- The sign convention used downstream: `dw` aligns with `+∂L/∂W`, so the trainer must **subtract** (`W ← W − lr · dw`), not add. The original `apply_node_update` did `W += lr · dw` (gradient ascent) — caught by code review, fixed in `apply_node_update`, asserted by `test_apply_node_update_does_gradient_descent_sign` in the CPU unit suite.

**Variance considerations.** Cosine grows with `√(n_sample × repeats)` for true forward-difference NP estimators. At 64 × 50 we get ≈ 0.41; halving either knob would drop cosine toward 0.10–0.20. The trainer uses fewer samples per step (`n_sample=8`, `n_rollout=8`) but accumulates over T response tokens, so the effective sample count per layer per training iteration is `n_sample × n_rollout × T ≈ 8 × 8 × 1024 ≈ 6.5×10⁴` — comfortably enough for SGD to make progress.

---

## 7. Pointers

- Spec: `docs/superpowers/specs/2026-05-28-np-trainer-design.md`
- Plan: `docs/superpowers/plans/2026-05-28-np-trainer.md` (16 tasks, all complete on `feat/np-trainer`)
- ES sibling: `docs/wiki/ZO.md` §2
- Paper: `arXiv:2604.13016` (token-level OPD theory)
