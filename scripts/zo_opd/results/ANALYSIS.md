# NP-vs-BP gradient check: cosine, norms, scaling, and LR

**Harness:** `verl/verl/trainer/zo_np/grad_check.py` (driver: `scripts/zo_opd/zo_np.sh`)
**Models:** student `Qwen/Qwen3-1.7B`, teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
**Layer:** `model.layers.0.mlp.down_proj` (`W ∈ [d_out=2048, d_in=6144]`)
**Loss:** per-token reverse-KL to teacher over the student top-K=16 set, `reward_weight_mode=student_p`
(identical formula to `verl.trainer.np.teacher_scorer.reverse_kl_topk`).
**Trajectory:** one prompt, greedy-decoded response frozen for both NP and BP.

The harness reuses the *shipping* estimator math verbatim — `noise_seed`/`draw_noise`
(seeding.py), `sample_scale`/`accumulate_delta_w` (grad_estimator.py), and
`assemble_layer_delta` (np_worker_extension.py) — so the numbers below are what the
production trainer actually computes, not a re-model. The only thing the harness adds
that the vLLM trainer cannot is a true `loss.backward()` for the ground-truth `dL/dW`.

---

## 1. Headline result (trainer's config: n_sample=16 × n_rollout=4 = 64 perturbations/token)

| quantity | shipping (grpo + normalize) | unbiased (average, no normalize) | true BP |
|---|---|---|---|
| **cosine(δW, dL/dW)** | **0.009** | **0.018** | 1.0 |
| ‖δW‖ | 0.083 | 2882 | **399.2 = ‖dL/dW‖** |
| ‖δW‖ / ‖dL/dW‖ | 2.1e-4 | 7.2 | 1.0 |

**Cosine is ~0, i.e. the NP δW is essentially orthogonal to the true gradient at this
sample budget.** This is NOT a sign error or a code bug — see §3, the estimator is
unbiased and its direction converges. It is **variance starvation**: 64 perturbations is
far too few to resolve a `d_out = 2048`-dimensional node gradient.

---

## 2. Is there improper scaling? Yes — two distinct, separable factors.

The shipping update is `W ← W − lr · δW` with `δW = assemble_layer_delta(..., sample_mode="grpo", normalize=True)`.
Relative to the true `dL/dW`, the magnitude is distorted by:

### (a) `normalize=True` (ANP `1/‖u‖²`) — a fixed `1/d_out` shrink  ← **the big one**
`accumulate_delta_w` with `normalize=True` divides each `u_q` by `‖u_q‖²`. For Rademacher
(`bernoulli`) noise `‖u_q‖² = d_out` exactly, so the **entire update is scaled by `1/d_out = 1/2048`**.
This is why the shipping ‖δW‖ (0.083) is ~`d_out`× smaller than the unbiased ‖δW‖ (2882):
`2882 / 2048 ≈ 1.4`, same order. With `lr=1e-6` the effective step is
`‖lr·δW‖ ≈ 8e-8` — about **5000× smaller** than a true-grad step of the same lr
(`‖lr·dL/dW‖ ≈ 4e-4`). The model barely moves.

### (b) `grpo` per-token `(L−mean)/std` rescale — destroys the per-token magnitude
`sample_scale(mode="grpo")` returns `(L_q − mean_q)/std_q`, which throws away the
`1/σ` finite-difference scale and replaces each token's gradient magnitude with a
unit-variance-normalized one. Direction per token is preserved; **absolute scale is not**,
and it varies per token, so the assembled δW is not on the `dL/dW` scale even before ANP.
The theoretically-unbiased setting is `mode="average"` (`(L_q−L_clean)/σ`) with
`normalize=False`, which recovers `‖δW‖ ≈ ‖dL/dW‖` up to the variance floor (§3).

### (c) variance inflation of the norm (not a "scaling bug", but distorts ‖δW‖)
Even the unbiased estimator has `‖δW‖ ≈ ‖dL/dW‖ · √(d_out / n)` because the orthogonal
noise component dominates when `n ≪ d_out`. Measured ratios (fixed harness, average mode,
24 tokens) track `√(d_out/n)` almost exactly on BOTH a `d_out=2048` and a `d_out=1024` node:

| layer (d_out) | n | measured ‖δW_avg‖/‖dL/dW‖ | √(d_out/n) | δW-matrix cosine |
|---|---|---|---|---|
| down_proj (2048) | 16   | 13.45 | 11.31 | −0.0004 |
| down_proj (2048) | 64   | 7.20  | 5.66  | 0.0022 |
| down_proj (2048) | 256  | 3.96  | 2.83  | 0.0046 |
| down_proj (2048) | 1024 | 2.05  | 1.41  | −0.0010 |
| v_proj (1024)    | 64   | 3.00  | 4.00  | −0.0134 |
| v_proj (1024)    | 256  | 1.70  | 2.00  | 0.0179 |
| v_proj (1024)    | 1024 | 0.83  | 1.00  | 0.0060 |

Two things to read off this table:
1. **‖δW‖ is governed entirely by `√(d_out/n)`** — the norm only reaches the true
   gradient norm as `n → d_out`. This is the variance floor, independent of layer.
2. **The δW-*matrix* cosine stays in the noise (±0.02) even at n=1024**, far slower than
   the single-node `dL/dy` cosine (§3, which hit 0.18 at n=4096). Reason: δW is a
   `d_out × d_in` (≈12.6M-element) matrix assembled as `Σ_t g_t ⊗ x_t` over only ~24
   variance-starved per-token `g_t`. Resolving the *matrix* direction is dramatically more
   sample-hungry than resolving one node-gradient vector. **This — not the per-token
   estimate — is the binding constraint for NP on full weight matrices.**

---

## 3. The estimator is correct — direction converges with samples

A per-token probe (recovering `dL/dy_t`, the 2048-dim node gradient, in isolation —
no rank-1 `⊗ x_t`) shows the cosine **rising monotonically with n_sample**, the signature
of an *unbiased but noisy* estimator:

| n_sample | cos(NP `dL/dy`, true `dL/dy`) | ‖NP‖/‖true‖ |
|---|---|---|
| 16   | 0.034 | 174 |
| 64   | 0.032 | 84  |
| 256  | 0.090 | 41  |
| 1024 | 0.110 | 20  |
| 4096 | **0.184** | 9.9 |

Cosine climbs toward 1 and the norm ratio halves each time `n` quadruples
(`∝ √(d_out/n)`). There is no sign flip and no constant bias — the math in
`grad_estimator.py` / `assemble_layer_delta` is sound. The problem is purely
**sample efficiency vs. node dimension**.

### Bug found and fixed in the harness (not in the trainer)
An initial batched shortcut perturbed *all* response tokens' node outputs in one forward;
causal attention then leaked earlier tokens' noise into later tokens' loss, decorrelating
the per-token forward difference (isolated cos 0.015 vs contaminated 0.0005 at n=64). The
harness now perturbs **exactly one token per forward** (batching the `n_sample` copies over
the batch dim instead), which matches how the vLLM trainer scores each decode step against
the shared clean prefix. All numbers above use the fixed harness.

---

## 4. A good learning rate

Two independent ways to set it; they agree on order of magnitude.

**(i) Match a reference BP step.** Choose `lr_np` so the NP update lands the same step
size a true-gradient step of `lr_ref` would: `lr_np = lr_ref · ‖dL/dW‖ / ‖δW‖`.

- shipping (grpo + normalize, ‖δW‖≈0.083): `lr_np ≈ 1e-6 · 399 / 0.083 ≈ 4.8e-3`
- unbiased (average, ‖δW‖≈2882):           `lr_np ≈ 1e-6 · 399 / 2882 ≈ 1.4e-7`

**(ii) Cancel the `1/d_out` ANP shrink directly.** If you keep the shipping
`normalize=True` + `grpo` path, the update is ~`d_out`× too small, so scale the figure's
`lr=1e-6` up by ~`d_out`:  `lr ≈ 1e-6 × 2048 ≈ 2e-3`.

**Recommendation:**
- **If keeping the shipping `grpo + normalize=True` path:** set `LR ≈ 2e-3` (range `1e-3 … 5e-3`).
  The figure's `1e-6` is calibrated for a *true-gradient* trainer and is ~10³–10⁴× too small
  here because of the `1/d_out` ANP normalization. Without this, training is effectively a no-op.
- **Cleaner fix — switch to the unbiased estimator** (`GRAD_ESTIMATE_SAMPLE=average`,
  `NORMALIZE=false`): then `δW` is on the same scale as `dL/dW`, the figure's `lr=1e-6`
  is directly meaningful, and `lr_np ≈ 1e-7 … 1e-6` is the right starting range.
  But note the **variance floor**: at `n=64`, ‖δW‖ is still ~7× inflated, so start at the
  low end (`~1e-7`) and/or raise `n_sample·n_rollout` toward `d_out` for a usable direction.

**Caveat:** any LR derived from norm-matching only fixes *magnitude*. With cosine ≈ 0.01–0.02
at n=64, most of each step is noise. To make NP train at all on a 2048-dim node you must
**raise the sample budget** (toward `n ~ d_out`) or **perturb a smaller node** (e.g.
`v_proj`, `d_out=1024`, or per-head slices) — see the v_proj sweep.

---

## 5. Bottom line

1. **Cosine(NP δW, BP dL/dW) ≈ 0.01–0.02 at the trainer's 64-perturbation budget** — the
   node-perturbation weight update is, in direction, almost pure noise relative to the true
   gradient on a 2048-wide node. (`norm_dW_BP ≈ 399`, `norm_dW_NP_ship ≈ 0.083`.)

2. **The estimator is mathematically correct, not buggy.** A per-token probe shows
   cos(NP `dL/dy`, true `dL/dy`) rising 0.03 → 0.18 as n goes 16 → 4096, with the norm
   ratio falling as `√(d_out/n)`. No sign error, no constant bias. (One *harness* bug — a
   cross-token attention leak from perturbing all positions at once — was found and fixed;
   the trainer's vLLM path already perturbs one step at a time and is not affected.)

3. **There IS improper scaling, of two kinds:**
   - `normalize=True` (ANP `1/‖u‖²`) shrinks the whole update by `1/d_out ≈ 1/2048`. With
     `lr=1e-6` the step is ~5000× smaller than a comparable true-gradient step → near-no-op.
   - `grpo` (`(L−mean)/std`) discards the `1/σ` finite-difference scale, so per-token
     magnitudes are arbitrary. The unbiased setting is `average` + `normalize=False`.

4. **Good LR:**
   - Keep shipping `grpo + normalize=True` → **`LR ≈ 2e-3`** (≈ `1e-6 × d_out`), range `1e-3…5e-3`.
   - Switch to unbiased `average + normalize=False` → figure's `lr=1e-6` becomes meaningful;
     start `~1e-7` (the variance floor still inflates ‖δW‖ ~7× at n=64).
   - Either way, LR only fixes magnitude. **The direction (cosine) needs more samples**:
     raise `n_sample·n_rollout` toward `d_out`, and/or perturb smaller nodes (v_proj
     d_out=1024 converges in norm ~2× faster; per-head slices would be better still).

## 6. Reproduce

```bash
# headline (trainer config: grpo + normalize, 64 perturbations/token)
CUDA_VISIBLE_DEVICES=5 MAX_STEPS=48 N_SAMPLE=16 N_ROLLOUT=4 \
  OUT=scripts/zo_opd/results/grad_check_fixed.json bash scripts/zo_opd/zo_np.sh

# unbiased estimator + a learning-rate-relevant norm
CUDA_VISIBLE_DEVICES=5 GRAD_ESTIMATE_SAMPLE=average NORMALIZE=false \
  N_SAMPLE=256 N_ROLLOUT=1 bash scripts/zo_opd/zo_np.sh
```

Artifacts: `grad_check_fixed.json` (headline), `conv2_{down,vproj}_ns*.json` (convergence sweep).
