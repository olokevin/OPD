# reweighted_compress: KL-importance token reweighting for calibration covariances

> **Verdict (2026-06-08): NEGATIVE — this does not work.** The idea: within a
> calibration sequence, not every token is equally damaged by compression, so
> compress once, measure where the compressed *student* diverges from the
> uncompressed *teacher* per token (forward KL), upweight those tokens'
> activations in the covariance, and recompress. The derivation (§2–3) is clean
> and the implementation is verified (β=0 reproduces the §5 baseline), **but it
> costs −5pp MATH at retain 0.7 (67→62%)** — the KL chases *teacher-uncertain*
> tokens, not *task-leverage* tokens, pulling the reconstruction budget away from
> what reasoning needs (§6). **Uniform sequence-reweight (§5) remains the
> recipe.** Kept here as a documented dead end + reusable weighted-covariance
> machinery for a future, better `w_t`.

Companion to the calibration-format lever. Pairs with
[reasoning_aware_compress_calib](reasoning_aware_compress_calib.md) (operating
point, eval contract, the sequence-vs-token reweight axis) and reuses its
infra (`compress_common.py`, `collect_covariances_reweighted`,
`build_fullseq_calib_loader`).

---

## 1. Why token-uniform calibration leaves accuracy on the table

Every activation-aware structured method in this repo distills the data into a
single second-moment matrix per layer and compresses against it:

- **SVD-LLM-V2 (attn)** whitens with `Φ Φᵀ ≈ C_x` and truncates the SVD of
  `W Φ`, which minimizes `E_x ‖(W − Ŵ)x‖²` — the *expected output error over the
  calibration activations* (`svd/svd_llm_v2.py`, `utils/whitening.py`).
- **Nystrom/MoDeGPT (MLP)** selects neurons and refits `W_down` from
  `C_σ = E[z zᵀ]` of the gated hidden `z` (`structured/nystrom.py`).

In both, the covariance `C` is the **entire** contribution of the data. Write it
as a sum of per-token outer products:

$$
C \;=\; \frac{1}{\sum_t 1}\sum_{t} v_t v_t^\top ,
\qquad v_t \in \{x_t\ (\text{attn}),\ z_t\ (\text{MLP})\}.
$$

Each token enters with **weight 1**. So the reconstruction is tuned to preserve
the model's behaviour *uniformly* across calibration tokens — including the
"easy" tokens a low-rank/neuron-subsampled model already reproduces almost
perfectly. Budget spent matching an already-matched token is wasted. The
sequence-reweight lever (§5 of the calib page) fixed *cross-sequence* imbalance
(a 10k-token trace no longer drowns a 500-token one) but is still **uniform
within a sequence**. This page attacks the *within-sequence* axis.

The reasoning-trace failure mode makes this concrete: compression first breaks
**termination** — the model reaches the answer but loops past it
([reasoning_aware_compress_calib](reasoning_aware_compress_calib.md) §4). The
tokens that decide *"emit `\boxed{}` / stop"* are a tiny, high-leverage subset.
Token-uniform calibration gives them no special standing; a damage-aware weight
does.

## 2. The reconstruction objective is linear in a per-token weight

Replace the uniform sum with a non-negative weighted one:

$$
\boxed{\,C_w \;=\; \frac{\sum_t w_t\, v_t v_t^\top}{\sum_t w_t}\,},
\qquad w_t \ge 0 .
$$

Feeding `C_w` (instead of `C`) into the *same* whitening / Nystrom kernels
changes the minimized loss from uniform to weighted, **with no change to the
compression math** — only the statistic it consumes:

$$
\hat W \;=\; \arg\min_{\hat W}\ \sum_t w_t\,\big\|(W-\hat W)\,v_t\big\|^2
\;=\; \arg\min_{\hat W}\ \operatorname{tr}\!\big[(W-\hat W)\,C_w\,(W-\hat W)^\top\big].
$$

So the *only* design question is the choice of `w_t`. `w_t ≡ 1` recovers the
current recipe exactly — the reweighting is a strict generalization and gives a
clean β=0 ablation.

## 3. The weights: forward KL of compressed-student vs uncompressed-teacher

We want `w_t` large where compression **damaged the model's prediction** at
token `t`, small where it left it intact. The uncompressed model is the natural
reference. Concretely, a **two-pass** scheme:

1. **Pass 0 (probe compress).** Compress once with the standing uniform
   recipe (sequence-reweight, full-length) → compressed **student** `S`. Keep
   the uncompressed model as **teacher** `T`.
2. **Damage measurement.** Run *both* `T` and `S` over each calibration
   conversation `(prompt + reasoning response)` — the *same* sequences and chat
   template used for covariance collection. At every token position `t` both
   emit a next-token distribution over the vocabulary,
   `p^T_t = \mathrm{softmax}(z^T_t)` and `p^S_t = \mathrm{softmax}(z^S_t)`.
   The per-token **forward KL** quantifies the damage:

$$
\delta_t \;=\; D_{\mathrm{KL}}\!\big(p^T_t \,\big\|\, p^S_t\big)
\;=\; \sum_{v} p^T_t(v)\,\log\frac{p^T_t(v)}{p^S_t(v)}
\;=\; \mathbb{E}_{v\sim p^T_t}\!\Big[\log p^T_t(v) - \log p^S_t(v)\Big]\ \ge 0 .
$$

This is exactly the user's "ratio of the logits": KL is the *teacher-expected
log-ratio* of the two distributions. `δ_t = 0` iff compression left the
distribution at `t` untouched; `δ_t` grows as the compressed model becomes more
uncertain / wrong relative to the uncompressed one. (Forward KL `D(T‖S)`, not
reverse, because the teacher is the trusted reference and forward KL is
mass-covering — it penalizes the student dropping probability where the teacher
puts it, i.e. exactly the prediction the compressed model lost.)

3. **Pass 1 (reweighted compress).** Collect `C_w` with `w_t = g(δ_t)` (below)
   and compress the *original* uncompressed weights again. This is the model we
   evaluate. (We recompress from scratch, not on top of `S` — `S` is only a
   damage probe.)

### 3.1 KL → weight: bounded, scale-free, β-controlled

Raw `δ_t` spans orders of magnitude with a heavy right tail (a few catastrophic
tokens). Three requirements: (a) **scale-free** — invariant to the absolute KL
scale, which drifts with ratio/layer; (b) **no token zeroed** — every token
still contributes some second-moment mass (the easy tokens still define the bulk
subspace); (c) **bounded tail** — one exploded token must not hijack the whole
covariance. An exponential tilt of the per-sequence-normalized KL satisfies all
three:

$$
\tilde\delta_t \;=\; \frac{\delta_t}{\bar\delta_{\,\mathrm{seq}(t)} + \varepsilon},
\qquad
w_t \;=\; \min\!\big(\exp(\beta\,\tilde\delta_t),\; w_{\max}\big),
$$

where `\bar\delta_{seq(t)}` is the **mean KL over the (unpadded) tokens of token
t's own sequence**. Properties:

- **β = 0 → w_t ≡ 1**: exact recovery of the uniform recipe (the ablation
  anchor).
- **Scale-free**: `\tilde\delta_t` is a *ratio* to the sequence's own mean, so a
  sequence whose KL is uniformly 10× another's gets the *same* weight profile —
  this composes cleanly with sequence-level reweighting (cross-sequence balance
  stays intact; we only redistribute weight *within* a sequence).
- **Mean-1 tilt**: by construction `mean_t \tilde\delta_t ≈ 1`, so `β` is the
  only sharpness knob; the cap `w_max` (default 5) bounds the tail.
- **Monotone**: higher damage → higher weight, strictly, until the cap.

The final covariance, **mask-aware** (pad tokens excluded from both numerator and
the per-sequence mean) and composed with sequence reweighting, is:

$$
C_w \;=\; \operatorname*{mean}_{\,\mathrm{seq}\,}
\left[\frac{\sum_{t\in\mathrm{seq}} w_t\, v_t v_t^\top}{\sum_{t\in\mathrm{seq}} w_t}\right].
$$

The inner term is the *within-sequence weighted* covariance; the outer mean is
the established sequence reweighting. So this is **strictly orthogonal** to and
**stacks on** the §5 lever: set all `w_t = 1` and it is identical to
`reweight="sequence"`.

### 3.2 Why this is second-order and should help where the cliff is

The first pass is calibrated to the *uncompressed* activation statistics, which
is a fixed-point mismatch: the model it produces has different activations than
the one it was calibrated for (mechanism **M3**, accumulation, in the calib
page). The KL probe measures that realized mismatch directly and folds it back
into the covariance — a single Gauss–Newton-style reweighting step toward
"calibrate for the damage you actually caused." It is cheapest exactly where it
should matter most: as the retain ratio drops past the cliff, damage becomes
**concentrated** (few tokens, large `δ_t`), so the weight profile sharpens and
diverges most from uniform — predicting larger gains at lower ratios (to be
tested).

## 4. Cost & relation to OPD

- **One extra compress + two extra forward passes** over the 128-sequence calib
  set (teacher once, student once; both `no_grad`, no backward). Forward-only,
  so ~the cost of one §5 cell plus a cheap logit diff. No teacher *training*,
  no rollout.
- **Teacher == uncompressed self**, so unlike the deferred OPD-gradient cell
  (D3, which needs a *distinct* teacher or the KL is degenerate), here the KL is
  **non-trivial by construction** — `T` and `S` are *different models* (full vs
  compressed). This is the legitimate, non-degenerate way to get a
  teacher-signal into one-shot compression.
- Distinct from `collect_both_covariances_from_loader_opd` (calibration_opd_loss):
  that drives the *backward* covariance with an OPD gradient and needs an
  external teacher. Here we reweight the *forward* covariance with a self-KL
  damage signal — no gradients, no external model.

## 5. Design & implementation plan

### 5.1 Code (minimal, reuses §5 infra)

| Piece | Where | What |
|---|---|---|
| per-token weight in the accumulator | `src/compress/calibration.py` `_accumulate_cov` | accept optional `weights` (B,T); fold `w_t` into the `vᵀv` sum and the denom. `weights=None` → identical to today. |
| weighted forward collector | `src/compress/calibration.py` | `collect_covariances_weighted(model, loader, token_weights, reweight, ...)` — like `collect_covariances_reweighted` but pulls a per-batch weight tensor from a callback/dict keyed by batch index. |
| KL weight computation | `src/compress/kl_reweight.py` (new) | `compute_kl_token_weights(teacher, student, loader, *, beta, w_max, eps)` → per-sequence list of `(T,)` weight tensors. Forward-only, mask-aware, per-sequence mean-normalized exponential tilt. |
| driver | `scripts/reasoning_aware_compress/kl_reweight_compress.py` (new) | two-pass: §5 uniform compress → KL weights → reweighted compress → `eval_math_capture` + C4 PPL. `--beta`, `--w-max`, `--ratio`. |

Invariants kept: last-decoder-layer dense (`drop_protected_stats`); forward-only
SVD-V2 attn + Nystrom MLP; OpenThought3 calib (128 seqs, seed 3); bf16, 1×H100;
eval contract (MATH-500 greedy, ttrl_math grader, C4 sliding-window PPL).

The probe student `S` is built with the *uniform sequence-reweight, full-length*
recipe at the **same retain ratio** as the final model, so the damage measured is
the damage at the operating point.

### 5.2 Experiment — same cell as the existing benchmark

Operating point fixed to the §5 table's apples-to-apples cell: **Qwen3-4B
non-thinking, sequence-reweight, full length, retain 0.7, last layer dense,
MATH-500 (100 probes) + C4 PPL**.

| cell | reweight | within-seq weight | expected role |
|---|---|---|---|
| **B (baseline)** | sequence | `w_t ≡ 1` (β=0) | reproduces §5 `sequence:full @0.7` = **69% / PPL 98.7** |
| **K-mid** | sequence | KL tilt, β=1, w_max=5 | headline |
| **K-sharp** | sequence | KL tilt, β=2, w_max=8 | sharper; risk of tail over-focus |

Success criterion (pre-registered): **K beats B on strict MATH** at fixed budget,
with C4 PPL not regressing materially. B must reproduce the standing 69% (sanity
that the two-pass harness ≡ §5 at β=0). If K-mid wins, sweep ratio 0.6/0.5 to
test the "gain grows past the cliff" prediction (§3.2); if it ties, the
within-sequence axis is exhausted and uniform sequence-reweight is the recipe.

Secondary read-outs from `eval_math_capture`: `relaxed_acc`, `mean_gen_tokens`
(does damage-aware calib further bound the looping?), `mean_tok_to_correct`.

### 5.3 Risks / falsifiers

- **Heavy-tail capture**: if a few exploded-KL tokens dominate, the covariance
  collapses to their subspace and *general* PPL regresses while MATH may or may
  not move. `w_max` caps this; K-sharp probes the boundary. If even K-mid
  regresses PPL hard, the tilt is too aggressive.
- **Probe mismatch**: `S` at ratio ρ measures damage at ρ; if the final model is
  at a *different* ρ the weights are stale. We hold ρ fixed across probe & final.
- **β=0 must equal §5**: if cell B ≠ 69% the harness has a bug, not the method.
- **Reviewer framing**: keep it "a calibration reweighting" (orthogonal to M1
  rank-floor and to §5 sequence-reweight), ablate β=0, and report PPL alongside
  MATH so the claim is "spends budget on damaged tokens," not "overfits a few
  tokens."

## 6. Results — NEGATIVE (2026-06-08): KL reweighting hurts MATH

Ran the §5.2 cell exactly: Qwen3-4B non-thinking, sequence-reweight, full length,
retain 0.7, last layer dense, MATH-500 (100 probes) + C4 PPL. 128 calib seqs,
seed-fixed. `kl_reweight_compress.py`, results
`scripts/reasoning_aware_compress/results/reweight/kl_r0.7.json`.

| cell | β | w_max | strict MATH | relaxed | C4 PPL | gen_len | tok→correct |
| ---- | - | ----- | ----------- | ------- | ------ | ------- | ----------- |
| **B (anchor)** | 0 | — | **67.0%** | 67.0% | 96.9 | 861 | 551 |
| K-mid | 1 | 5 | **62.0%** | 62.0% | 95.6 | 841 | 479 |
| K-sharp | 2 | 8 | **62.0%** | 62.0% | 100.6 | 894 | 604 |

**The method does not work at this operating point.** KL-damage token reweighting
costs **−5pp strict MATH** (67→62%) and never recovers; sharpening the tilt (β=2)
gives no further movement on MATH and starts *hurting* PPL (96.9→100.6). The
hypothesis of §3.2 is **falsified** for retain 0.7.

What the read-outs rule out:

- **Harness is sound.** Cell B (β=0) hits 67.0%, reproducing the §5
  `sequence:full @0.7` baseline (69% there, within run noise — the two-pass code
  at β=0 *is* `collect_covariances_reweighted`, verified bit-identical in a unit
  test). So the drop is the *method*, not a bug.
- **It's a reasoning loss, not a looping artifact.** `relaxed == strict` and
  `n_reached` tracks strict exactly (67/62/62) at every β — the upweighted tokens
  made the model *reach the correct answer on fewer problems*, the opposite of
  the §5 lever (which fixed looping). Termination was never the issue here.
- **PPL is flat-to-worse while MATH drops.** β=1 even *lowers* C4 PPL (95.6) while
  losing MATH — the KL-upweighted subspace is not "generally better", it is
  reallocated *away* from what reasoning needs. This is the §5.3 "heavy-tail
  capture" risk realized: spending budget on the high-KL tokens (mean KL only
  0.138, but heavy-tailed) trades off the tokens that carry the math.

**Why it likely fails (interpretation).** The first-pass damage is *largest*
exactly on tokens whose distribution is intrinsically high-entropy / hard to
predict (a 4B model is already unsure there), not on the load-bearing reasoning
steps. Forward KL `D(T‖S)` upweights wherever the compressed student lost teacher
mass — which correlates with *teacher* uncertainty, not with *task* leverage. So
the weight chases prediction-hard tokens, and the rank/neuron budget gets pulled
toward reproducing their (high-variance, less structured) activations at the
expense of the cleaner directions that the uniform second moment already favored.
The uniform sequence-reweighted covariance is, empirically, the better target.

**Conclusion.** The **within-sequence** reweighting axis (attacked via
teacher–student KL) is exhausted as a *gain* — **uniform sequence-reweight (§5)
remains the recipe**. The §5 sequence-vs-token lever was already capturing the
reweighting that helps; pushing below the token granularity with a damage signal
does not pay off and mildly hurts.

### Not pursued (the prediction it would have tested)
The §3.2 "gain grows past the cliff" sweep (0.6/0.5) was **not run** — it is only
worth doing if a cell *wins* at 0.7, and none did. A damage signal weighted
toward *task* leverage rather than *teacher* uncertainty (e.g. gradient/saliency
of the final answer, or KL only on response tokens past the first `\boxed{}`)
could in principle still help, but that is a different weight, not this one.

→ **That "different weight" is derived in
[reweighted_compress_v2](reweighted_compress_v2.md):** the correct lever is the
**output**-side KL-Fisher curvature `G_ℓ` (task-leverage), not this input-side
teacher-uncertainty reweight, applied inside an iterative refinement loop. v2
explains *why* v1 failed (wrong space + wrong signal) and gives the corrected
formulation.

### Status
**DONE — negative.** Code (`calibration.py` weighted path, `kl_reweight.py`,
`kl_reweight_compress.py`) kept and unit-tested; reusable if a better `w_t` is
proposed. β=0 path is a free re-derivation of the §5 baseline. No change to the
production default.
