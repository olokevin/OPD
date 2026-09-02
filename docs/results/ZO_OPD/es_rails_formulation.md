# ES rails for OPD — is the formulation right, and can it ever match BP?

> Audit of the per-token weight-perturbation rails (`es_token`) as a training signal for
> OPD, written 2026-08-31 before the DeepSeek/JustRL runs finish. Short answers:
> the estimator is **unbiased but mis-targeted** (three fixable variance leaks); the OPD-KL
> landscape turns out to be extremely low-rank (`r_eff = tr(H)/κ_g ≈ 160`), so ES is not
> curvature-limited — it is limited by the random-walk displacement budget, which puts full-parameter
> forward-only OPD at ~1/40 of a BP step per step (2 orders of magnitude, not 5). For a per-token *differentiable* objective on a fixed
> rollout, decode-time rails are dominated by plain teacher-forced prefills of the perturbed
> model, so the "free generation rail" premise does not hold for OPD.
> Code: branch `feat/es-token-trainer`, `scripts/zo_opd/ds15b/`, `verl/trainer/ppo/es_update.py`.
> Companion pages: [zo_opd_short.md](zo_opd_short.md) (history), [zo_opd.md](zo_opd.md) §12–13.

## 1. Verdict table

| Question | Answer | Where |
|---|---|---|
| Is `es_token` an unbiased estimator of *something*? | Yes — of the history-detached gradient of the sampled-token reverse-KL, `E_ŷ[∇log π(ŷ)(log π − log q + 1)] = ∇KL(π‖q)` | §2.1 |
| Is it the gradient BP-OPD uses? | Same expectation, but (i) single-sample with an extra zero-mean `+1` term, (ii) exponential IW weight evaluated at finite σ, (iii) detached history (rails read clean KV) | §2.2 |
| Does the rank-1 weight probe waste information? | Yes: `ΔW = δℓ·u vᵀ/σ` re-estimates the layer input `x_t` from a random projection although `x_t` is known. Using `x_t` (node-perturbation readout) cuts noise energy by `d_in` and curvature-weighted damage by ≈ `r_x/3` | §2.3 |
| Why does it not learn at *any* step size? | Not curvature: `r_eff = tr(H)/κ_g ≈ 160` is *small*. The isotropic part of every ES step is a random walk that must stay inside the quadratic basin, and within that budget the coherent motion is `√(2δ_max C/tr(H))` — for all 1.78 B params, ~1/40 of a BP step per step. Token count cancels; total rail count `C` is the only lever | §3, §5 |
| Are decode rails the right rail? | No, for a fixed-trajectory objective: ~3× the cost per token-evaluation of a prefill under the same perturbation, plus the detached-history error. Prefill rails are exact and use stock machinery | §4 |
| What would make forward-only OPD work? | Nothing measured here: ~10⁶ prefill rails per run (≈ 17 GPU-days) for BP-like motion on all parameters; the best aligned subspace buys 1.6×. Empirically (§7): every ES arm reaches BP − 3 pp at a 7168-token cap by shortening, and ≤ +1.8 pp (noise) once the cap is lifted, vs BP's +6 / +23 pp | §5, §7.1 |

## 1.5 Run naming (adopted 2026-09-01 — use these tags for all future runs)

| tag | perturbation | evaluated via | status here |
|---|---|---|---|
| **es-prefill** | one per rail, held for the whole trajectory | extra teacher-forced prefill of the perturbed model on the fixed rollout | **what §7's ES arms A/B/C/D′ ran** (`algorithm.es_update`) |
| **es-token-prefill** | fresh per token, within a prefill | same prefill; the rank-1 rail op as a per-position output adjustment (perturbation propagates to later positions → full-history unbiased) | analysed (§4 addendum), not built — adds no probes over es-prefill |
| **es-decode** | one per rail, held while decoding | parallel rail rows riding the clean rollout's KV | offline control only (`es_seq_audit.py`, zo_opd §12.5c) |
| **es-token-decode** | fresh per token, at decode | the original `es_token` trainer (rails on clean KV, CUDA-graphed) | the ruled-out design (zo_opd §12); its kernels now accelerated (`es_profile_results.md`) |

wandb: all ds15b runs live in
[`yequan_zhao…/nersc_opd_qwen4b_1p7b`](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/nersc_opd_qwen4b_1p7b) —
BP `s7x9sidi`; es-prefill A `vgukgnn6` (+ resumed `x9xamqew`), B `3cvrauyy`, C `9vttn2zu`, D′ `j85h7lyh`;
run names `ds15b_*` (future runs: `ds15b_es-prefill_*` etc.).

## 2. What the current formulation computes

### 2.1 Unbiasedness — correct

Per token `t`, rail `n`, with `ŷ_t ~ π_0(·|s_t)` the clean sample, the rail loss is
`l_{n,t} = (π_n(ŷ)/π_0(ŷ)) · (log π_n(ŷ) − log q(ŷ))`. Its expectation over `ŷ` is
`Σ_y π_n(y)(log π_n(y) − log q(y)) = KL(π_n ‖ q)` — the docs' claim holds. Its gradient in `W` at
`W_0` is `∇log π_0(ŷ) · (log π_0(ŷ) − log q(ŷ) + 1)`, whose expectation is `∇KL(π_0‖q)` because
`Σ_y π_0 ∇log π_0 = 0`. The rank-1 Rademacher contraction `E[(uᵀGv) u vᵀ] = G` is exact. So the
estimator is unbiased for the **fixed-state, history-detached** reverse-KL gradient.

### 2.2 …but it is a noisier estimate of the *same* gradient BP-OPD uses

BP-OPD (`token_reward_direct`, `LOG_PROB_TOP_K=0`, thunlp's `k1` + policy gradient) computes
`Σ_t A_t ∇log π(y_t)` with `A_t = log q(y_t) − log π(y_t)` frozen — the single-sample estimate of
`−∇ Σ_t KL(π(·|s_t)‖q(·|s_t))` on fixed states. Three places where the rail loss adds variance
for no information:

| leak | what it costs | fix |
|---|---|---|
| the `+1` score-function term | zero-mean noise proportional to `∇log π(ŷ)` | use BP's frozen advantage: rail fitness `Σ_t A_t (log π_n(y_t) − log π_0(y_t))` — the objective BP maximises, no `+1` |
| exponential IW weight `π_n/π_0` at σ = 1e-2 (mean |Δlogp| 0.32, max 4.9, clamped at 10) | heavy-tailed rail differences | same fix — the frozen-advantage form has no IW weight; or use the exact top-K KL |
| detached history (rails perturb only the current position, read clean KV) | `cos(g_direct, g_full)` = +0.07 … +1.0 per layer (§12.4) | teacher-forced prefill of the perturbed model — the whole sequence sees the perturbation |

### 2.3 The rank-1 weight probe throws away the known input

For a linear layer `y = W x`, the rail's perturbation `σ u vᵀ` changes the output by
`σ (vᵀx_t) u` — it *is* a node perturbation with a token-dependent scalar. The loss change is
`δℓ = σ (vᵀx_t)(uᵀg_t)` with `g_t = ∂ℓ/∂y_t`. The true weight gradient is `g_t x_tᵀ`.

| readout | estimate of `g_t x_tᵀ` | noise energy (∝) |
|---|---|---|
| `es_token` (shipping) | `(vᵀx_t)(uᵀg_t) · u vᵀ` | `‖g‖²‖x‖² · d_out · d_in` |
| node-perturbation readout | `(vᵀx_t)²(uᵀg_t)/‖x_t‖² · u x_tᵀ` | `‖g‖²‖x‖² · d_out · κ₄`, κ₄ ∈ [1, 3] |

Both are unbiased; the second uses the captured `x_t` instead of re-estimating it as `(vᵀx_t) v`,
so the noise energy drops by `d_in/κ₄` (≈ 500–3000×). The *damage* it does is not `d_in` smaller,
because the second estimator's noise sits entirely in the activation subspace where the curvature
lives: the curvature-weighted damage per unit signal drops by ≈ `r_x/3`, with `r_x` the
participation ratio of the layer-input covariance (measurable; likely 3–30 for LLM residual
inputs). Real, but secondary — it does not change §3. It also needs `x_t` captured per layer per
token, which `es_token` does not do (NP-V3 did).

## 3. The right unit: curvature effective rank, not cosine

Every unbiased ES step is `Δ = −η ĝ` with `ĝ = (1/N) Σ_n ⟨g, ε_n⟩ ε_n` (Gaussian `ε`). Expanding the
loss to second order and taking expectations:

```
E[ΔL] = −η‖g‖² + ½ η² ( gᵀHg + ‖g‖² tr(H) / N )
```

At the optimal `η`, the per-step decrease is `½‖g‖² / (κ_g + tr(H)/N)` with `κ_g = gᵀHg/‖g‖²`,
against gradient descent's `½‖g‖²/κ_g`. Hence

```
progress_ES / progress_GD = 1 / (1 + r_eff / N),        r_eff = tr(H) / κ_g
```

Three consequences, all consistent with what §12 of [zo_opd.md](zo_opd.md) measured:

1. **The token count cancels.** With independent per-token probes the noise term becomes
   `Σ_t‖g_t‖² tr(H)/N` instead of `‖g‖² tr(H)/N` — a factor `1/ρ ≈ 1.3`, not `T`. Per-token rails
   and one perturbation per rollout are the *same estimator* in this bound (§12.5c saw this).
2. **`cos ≈ √(N/D)` is a symptom, not the cause.** A tiny cosine is fine if `tr(H)/κ_g` is small —
   the noise lives in flat directions and costs nothing. This is why dense ES with `cos ≈ 6e-5`
   gains +20 pp on MATH with an *accuracy* fitness ([es_results_short.md](../ES/es_results_short.md)):
   that landscape is effectively low-rank. Whether the OPD-KL landscape is, is an empirical
   question — §5 measures it.
3. **N is the only lever**, and each rail is one full forward. So the rail *cost model* decides
   everything (§4).

**But the per-step-optimal analysis is a trap once `r_eff` is small.** The optimal `η` above makes
the *random* part of every step large (`η‖g‖√(D/N)`), and those random displacements accumulate as a
random walk that leaves the quadratic basin within a handful of steps (this is exactly the
"random-walk damage" of [zo_opd.md](zo_opd.md) §12.8). In practice the binding constraint is a
**displacement budget**: the total isotropic noise must stay where the KL-rise curve is still
quadratic. With a total noise-damage budget `δ_max` (KL units) and `C = S·N` rail forwards in the
run, the coherent motion along the gradient is

```
coherent motion  =  sqrt( 2 δ_max C / tr(H) )                       (weight-norm units)
per-step ES / BP  ≈  (ρ_ES / ρ_BP) · sqrt(N / D)                    (ρ = relative step size)
```

so progress goes as `√C` regardless of how rails are split between N and S, and the only ways to
buy more are a larger budget (a flatter model), more rails, or a subspace with a better
`‖g_S‖² / tr(H_S)` (§8). Curvature effective rank drops out entirely.

## 4. Decode rails vs prefill rails

Per *token-evaluation under a perturbed weight* (Qwen3-1.7B / H100 numbers from
[zo_opd.md](zo_opd.md) §6–§10):

| rail type | cost | what it measures | machinery |
|---|---|---|---|
| decode rail (`es_token`, `pack_width` 64, N = 8) | 15.0 ms / token-step for 576 rows → **≈ 0.021 ms per row-step** | history-detached loss of the *current* token | custom CUDA-graphed packed decoder, per-token noise, chunked assembly |
| teacher-forced prefill of `W + σε` | 1.5 B model at ~130 k tok/s → **≈ 0.008 ms per token** | exact loss of the whole sequence under the perturbation | stock HF / vLLM forward; `perturb → forward → restore` |

The rail is "free" only in the latency-bound regime (`pack_width` 4: +0.1 ms per rail on a 3 ms
floor) — which is exactly the regime where generation itself is 4× under-utilising the GPU. At the
batch sizes an efficient trainer runs, rails cost ∝ N and are ~3× a prefill *per evaluation* while
measuring a worse quantity. The one thing only a decode rail can do — commit its own token and
branch — is not needed by a fixed-trajectory objective, and the branching objective was shown to
be length-hackable (§13.5–13.6).

**So the "parallel eval rail" for OPD is N prefills**, and the question reduces to §3.

**Per-token perturbations inside a prefill.** The rank-1 rail op is a per-position output
adjustment (`y_t += σ(v_tᵀx_t)u_t`), so a teacher-forced prefill *can* carry a fresh perturbation
per token, exactly like the decode rails — and unlike them the perturbation at position `t` then
propagates through the hidden states of positions ≥ t, so the per-rail scalar is unbiased for the
**full** (history-included) gradient, killing the detached-history error for free. But it adds no
statistical power: what carries information is the number of *independent fitness scalars* per
step, and the token-cancellation law (§3.1 / R2, measured in zo_opd.md §12.5) says the T×N
per-token readouts inform T different per-token gradients whose estimate noises do not cancel in
the sum — the aggregate update is still an N-probe estimate. Fresh noise per token multiplies
targets, not probes; more rails is the only way to add probes.

## 5. Measurement: `r_eff` of the OPD objective (DeepSeek-R1-Distill-1.5B ← JustRL-1.5B)

`scripts/zo_opd/ds15b/opd_curvature.py` — one on-policy batch (16 prompts, ≤4096 tokens, T = 1.0),
teacher `log q` over the full vocab, objective `f(W) = mean_t KL(π_W(·|s_t) ‖ q(·|s_t))` on the fixed
rollout (the zero-variance form of the k1 loss). `g` by autograd; `κ_g` by symmetric finite
differences along `g/‖g‖`; `tr(H)` by antithetic isotropic probes (Hutchinson); both checked
against the autograd directional derivative.

| quantity | value | note |
|---|---:|---|
| D (all params, incl. embed + lm_head) | 1.777 B | RMS(W) = 0.0536, ‖W‖ = 2259 |
| base KL/token `f0` | 0.2378 | resp_len mean 3361 (cap 4096); 53,781 scored tokens |
| ‖g‖ | 2.078 | FD directional derivative 2.076 — autograd validated |
| κ_g = gᵀHg/‖g‖² | **34.2** | identical at steps 3e-3 / 1e-2 / 3e-2 → clean quadratic regime |
| tr(H) | **≈ 5.5e3** | Hutchinson, 2 probes × 4 σ (5.19e3 – 5.97e3); FD ⟨g,ε⟩ = autograd to 1 % |
| λ_avg = tr(H)/D | 3.1e-6 | the curvature isotropic noise sees |
| **r_eff = tr(H)/κ_g** | **≈ 160** | *not* 10⁴–10⁶: the OPD-KL landscape is extremely low-rank |
| cos(−g, W_T − W_S) | +0.010 | the BP direction is essentially orthogonal to the teacher's weight delta (‖W_T − W_S‖ = 11.2, 0.5 % of ‖W‖) |
| KL rise, isotropic σ = 1e-3 / 2e-3 / 3e-3 / 5e-3 (1.9 / 3.7 / 5.6 / 9.3 % of RMS(W)) | ×1.011 / ×1.048 / ×1.115 / ×1.364 | quadratic prediction ×1.012 / ×1.046 / ×1.104 / ×1.29 — only ~25 % super-quadratic at 9 %: **no cliff up to ~10 %**, unlike Qwen3-1.7B (×17 at 9.8 %, zo_opd §13.2). The displacement budget here is ≈ 10 %, not 3 % |
| FD directional derivative ⟨g,ε⟩ vs autograd, σ = 1e-3 / 2e-3 / 3e-3 / 5e-3 | 3 % / 12 % / 28 % / 66 % off | the *rail readout* is linear only up to σ ≈ 1e-3 — use that for training rails |

What the two regimes predict (batch objective, first order):

| | per-step KL decrease | vs GD |
|---|---:|---:|
| GD, optimal step (`½‖g‖²/κ_g`) | 0.063 (26 % of f0) | 1 |
| ES optimal step, N = 32 / 128 / 512 (`½‖g‖²/(κ_g + tr(H)/N)`) | 0.010 / 0.028 / 0.048 | 1/6 · 1/2.3 · 0.76 |
| …but its random displacement per step | 3.3 % of ‖W‖ (N = 32) | leaves the basin in ~3 steps |
| ES, displacement-limited, N = 32, 300 steps, damage budget 10 % of f0 (4 % displacement) | coherent motion 0.29 total ≈ **7 aligned BP steps** (BP step ‖Δ‖ = lr·√D ≈ 0.042) | per step ≈ 1/44 of BP |
| same, N = 128 | 0.58 ≈ 14 BP steps | 1/22 |
| same, N = 32 but budget 36 % of f0 (the measured ×1.36 at 9.3 % displacement) | 0.55 ≈ 13 BP steps | 1/23 |
| same, 10⁶ rails (~17 GPU-days at 1.5 s/rail), 10 % budget | 2.9 ≈ 70 BP steps | — |

**Per parameter group** (`--groups`, isotropic probes restricted to one group at a time; the
displacement-limited figure of merit is `‖g_G‖²/tr(H_G)` — KL gain per rail budget goes as its
square root):

| group | D | share of ‖g‖² | share of tr(H) | `‖g_G‖²/tr(H_G)` |
|---|---:|---:|---:|---:|
| embed + lm_head | 467 M | 41 % | 54 % | 5.9e-4 |
| attention | 154 M | 16 % | 13 % | 9.4e-4 |
| MLP | 1156 M | 42 % | 35 % | 9.3e-4 |
| norms | 0.1 M | 0.3 % | 0.1 % | 2.8e-3 |
| **all** | 1777 M | | | **7.4e-4** |

Gradient energy is spread almost exactly in proportion to curvature: dropping embeddings buys
1.27×, the norms alone have a 3.7× better ratio but hold 0.3 % of the gradient. **No coarse subspace
is a lever.**

**Calibrated activation subspace** (`--subspace zoact`, the ES-math study's best low-D arm: per
decoder linear `ΔW_l = A_l V_lᵀ`, `V_l` = top-r eigenvectors of the layer-input second moment on
this batch; 8 seqs × ≤4096 tok, full-space ratio on the same batch 6.8e-4):

| rank r | D_S | share of ‖g‖² | tr(H_S) | `‖g_S‖²/tr(H_S)` | vs full space |
|---|---:|---:|---:|---:|---:|
| 1 | 0.65 M | 34 % | 704 | 1.84e-3 | **2.7×** |
| 4 | 2.6 M | 42 % | 888 | 1.82e-3 | 2.7× |
| 16 | 10.3 M | 49 % | 1185 | 1.56e-3 | 2.3× |
| 64 | 41.3 M | 52 % | 1510 | 1.32e-3 | 1.9× |

The aligned subspace is genuinely better — a 2700× smaller space holds a third of the gradient
energy at an eighth of the curvature — but the figure of merit enters the KL gain as a square root,
so the best case is **1.6× the full-space gain per rail budget**. Not the order of magnitude that
would change the verdict. (`docs/results/ZO_OPD/data/ds15b_curvature_*.json` hold the raw numbers.)

**Reading.** The cosine/`√(N/D)` story (§12 of zo_opd.md) was right about the *direction* and wrong
about *why it matters*: `r_eff ≈ 160` says a 128-rail ES step is worth half a gradient step on this
landscape *if it could take the optimal step* — it cannot, because the step's isotropic part is 3 %
of the weight norm. Under the displacement budget, forward-only OPD on all 1.78 B parameters is
**~20–50× slower than BP per step, i.e. 2 orders of magnitude, not 5** — a 300-step, 32-rail run
should look like the first ~7 BP steps. That is a testable prediction for the runs in §7.

## 6. What was built (branch `feat/es-token-trainer`)

| piece | file | note |
|---|---|---|
| thunlp-mirrored BP-OPD on 1 GPU | `scripts/zo_opd/ds15b/bp_opd.sh` | batch 64 × n=4, `ppo_mini_batch` 16 (4 updates of 64 seqs per step, same data-per-update as their 4-GPU run), 1024/7168 tokens, lr 1e-6, k1 + PG (`LOG_PROB_TOP_K=0`), MATH-500 + AIME24 n=2 @ T=0.6 every 20 steps |
| `PPO_MINI_BATCH_SIZE` knob | `on_policy_distillation.sh` | decouples optimizer mini-batch from `data.train_batch_size` (was one variable) |
| forward-only ES update inside the PPO trainer | `verl/trainer/ppo/es_update.py`, `fsdp_workers.py` (`es_perturb_weights`, `es_apply_update`), `algorithm.es_*` config | replaces `update_actor`; N antithetic prefill rails of the FSDP actor on the same rollout, fitness = BP's surrogate `Σ A_t Δlog π`, OpenAI-ES step with z-scored coefficients; seeds only cross Ray; works for FSDP1 flat params and FSDP2 DTensors (per-rank shard noise) |
| ES-OPD launcher | `scripts/zo_opd/ds15b/es_opd.sh` | same pipeline as `bp_opd.sh` with `algorithm.es_update=True`; 16 × 4 seqs/step, N = 32 |
| curvature / `r_eff` probe | `scripts/zo_opd/ds15b/opd_curvature.py` | §5 |

## 7. Results

Launched 2026-08-31 18:03–18:19, wandb project `nersc_opd_qwen4b_1p7b`. All arms share the
pipeline, data order, teacher and eval; step-0 eval is bit-identical across arms
(MATH-500 mean@2 **0.751**, AIME24 mean@2 **0.150** at the 7168-token cap).

| arm | GPU | seqs/step | update | σ | α | per-step footprint | predicted after 300 steps |
|---|---|---|---|---|---|---|---|
| BP | 4 | 64 × 4 = 256 | Adam lr 1e-6, 4 mini-batches | — | — | ≈ 1.9e-5 (lr·√D/‖W‖) | the reference (280 steps = 1 epoch) |
| ES-A | 1 | 16 × 4 = 64 | 32 rails (16 antithetic pairs), z-scored | 1e-3 | 5e-4 | 2.3e-3 | ≈ 7 aligned BP steps of coherent motion; cum. displacement 4 % |
| ES-B | 0 | 64 | 128 rails | 1e-3 | 1e-3 | 2.3e-3 | ≈ 14 BP steps; 4 % |
| ES-C | 5 | 64 | 32 rails | 1e-3 | 1.25e-3 | 5.8e-3 | ≈ 13 BP steps; cum. displacement 10 % (KL damage ≈ ×1.36) |
| ES-D (2026-09-01 00:50–02:00, **discarded**: a normalisation bug — dividing the antithetic differences by their std instead of their RMS — inflated single steps by up to 4× at 4 pairs; cum. displacement 9 % by step 20, train KL 0.27 → 0.38. Fixed in `es_update.py`; ≤ 8 % effect at 16 pairs, so B/C stand) | 1 | 64 | **8 rails** (4 pairs) | 1e-3 | 1.25e-3 | 1.2e-2 |
| ES-D′ (from 2026-09-01 ~02:00, the fixed rerun) | 1 | 64 | **8 rails** (4 pairs) | 1e-3 | 1.25e-3 | 1.2e-2 | the N control: if the benchmark gain is a 1-D "shorten" signal, 8 rails should find it about as well as 32 at ~1/3 the step time |

Per-step diagnostics logged by `es_update.py`: `es/d_std` (signal spread across pairs ≈ σ‖g‖),
`es/post_update_gain` (surrogate gain on the *same* batch after the step — the direct test that
the estimator descends), `es/cum_footprint` (random-walk displacement / RMS(W)).

**BP step 1** (2026-08-31 18:21): 433 s/step — gen 161 s (256 seqs, mean response 6447 tokens,
70 % hit the 7168 cap: R1-Distill thinks long at T = 1.0), teacher 61 s, 4 mini-batch updates 169 s;
peak 78.3 GB allocated. One epoch (280 steps) ≈ 34 h + evals. `critic/rewards/mean` = −1792 per
sequence (≈ 0.28 nats/token, consistent with the 0.238 KL/token measured by the curvature probe on
shorter rollouts).

**ES-A step 1** (18:24): 378 s/step for 64 seqs — gen 52 s, teacher 37 s, **32 rails 279 s (8.7 s
per rail = 420 k tokens, ≈ 145 TFLOP/s)**, apply 0.2 s. Per sequence that is 3.5× BP's cost.
Signal check: `es/d_std` = 1.75e-3 ≈ σ‖g‖ (the FD readout is the gradient projection, as designed);
**`es/post_update_gain` = +1.55e-3** — the surrogate on the *same* batch improved after the step by
almost exactly the first-order prediction (1.4–1.8e-3 for N = 32 at this footprint, §5). The
estimator descends; the question the run answers is whether 1/40 of a BP step per step compounds
into anything visible on the eval.

**First 10 steps** (train KL/token = −`critic/rewards/mean` / mean length; batch noise ≈ ±0.01):

| step | BP | ES-A (N32, α 5e-4) | ES-C (N32, α 1.25e-3) | ES-B (N128, α 1e-3) |
|---|---:|---:|---:|---:|
| 1 | 0.278 | 0.274 | 0.274 | 0.274 |
| 4 | 0.259 | 0.275 | 0.276 | — |
| 8 | **0.223** | 0.256 | 0.248 | — |
| 10 | — | 0.269 | — | — |
| mean `es/post_update_gain` (same-batch, per step) | — | 1.41e-3 ± 0.06e-3 | 2.82e-3 ± 0.17e-3 | 2.5e-3 (2 steps) |
| s / step | 433 | 360 | 355 | 1313 |

BP's train KL falls 20 % in 8 steps. Every ES arm descends its *own* batch by a small, remarkably
stable amount (the gain scales with α as z-scored ES predicts: coherent motion per step = α, N
only shrinks the noise), and shows **no cross-batch trend inside the noise** — which is exactly
what "≈ 1/40 of a BP step per step" looks like after 10 steps (expected cumulative ≈ 0.002).

**Step-20 evals** (MATH-500 + AIME24, n = 2 @ T = 0.6, 7168-token cap; step 0 = 0.751 / 0.150):

| arm | train KL/tok @20 | MATH-500 | AIME24 | eval mean length | % hitting cap | acc among *uncapped* |
|---|---:|---:|---:|---:|---:|---:|
| step 0 (all) | 0.278 | 0.751 | 0.150 | 3755 | 25 % | 0.948 |
| **BP** | **0.083** | **0.823** (+7.2) | **0.317** (+16.7) | 3577 | 18 % | 0.941 |
| ES-A (N32, α 5e-4) | 0.231 | 0.781 (+3.0) | 0.183 (+3.3) | 3593 | 21 % | 0.935 |
| ES-C (N32, α 1.25e-3) | 0.218 | 0.803 (+5.2) | 0.267 (+11.7) | 3286 | 16 % | **0.915** (−3.3) |

Two different stories in one table:

- **On the objective itself (KL to the teacher) the analysis holds.** BP cuts the train KL by 70 %
  in 20 steps; ES-A/C by 15–20 % — the predicted order of magnitude of gap, and ES-C's gain is
  exactly 2.5× ES-A's per step.
- **On capped accuracy ES gets 40–70 % of BP's gain**, because that gain is *not* the KL: the
  decomposition shows accuracy among completed answers is flat (BP) or down (ES-C), and the whole
  improvement is **fewer responses hitting the 7168 cap** (25 % → 16–18 %). "Think shorter" is a
  ~1-dimensional behaviour, and a 1-D direction is exactly what a 32-rail ES finds cheaply — the
  same low-rank mechanism the ES-math study saw (every subspace, same gain). ES-C buys its extra
  brevity with a 3 pp loss on completed answers, i.e. it is already trading quality for length.

So "comparable with BP" depends on the ruler: on a truncation-sensitive benchmark, forward-only
prefill rails reproduce most of the early gain at 1/4 the sequences per step; on the distillation
objective they do not. The step-40/60 evals will show whether ES plateaus (ES-math arms did by
step ~40) while BP keeps converting KL into quality.

**Step-40 evals** (same protocol; AIME24 has 60 samples, σ ≈ 6 pp — read MATH-500):

| arm | train KL/tok @40 | MATH-500 | AIME24 | eval mean length | % capped | acc among uncapped |
|---|---:|---:|---:|---:|---:|---:|
| **BP** | **0.033** | **0.843** (+9.2) | 0.267 | 3745 | 16 % | 0.937 (−1.1) |
| ES-A | 0.233 | 0.802 (+5.1) | 0.217 | 3422 | 18 % | 0.928 (−2.0) |
| ES-C | 0.206 | 0.815 (+6.4) | 0.283 | 3042 | **15 %** | **0.909 (−3.9)** |

The mechanism separates cleanly by step 40. ES-C now truncates *less* than BP (15 % vs 16 %) yet
scores 2.8 pp lower on MATH-500, because its completed answers have lost 3.9 pp while BP's are
flat: **BP shortens by distilling the teacher's reasoning (KL 0.278 → 0.033); ES shortens by
shortening**. The ES arms' capped accuracy still creeps up (+2 pp per 20 steps, A and C alike) but
each ES step spends its budget on the one direction it can resolve, and pays for it in quality.

**Step-60 evals** (2026-09-01 00:45):

| arm | train KL/tok @60 | MATH-500 | AIME24 | eval mean length | % capped | acc among uncapped |
|---|---:|---:|---:|---:|---:|---:|
| **BP** | **0.017** | **0.846** (+9.5) | 0.267 | 3829 | 17 % | **0.948** (±0) |
| ES-A | 0.220 | 0.815 (+6.4) | 0.200 | 3245 | 16 % | 0.919 (−2.9) |
| ES-C | 0.193 | 0.829 (+7.8) | 0.283 | 3012 | 15 % | 0.929 (−1.9) |

MATH-500 trajectory (step 0 / 20 / 40 / 60): BP 0.751 → 0.823 → 0.843 → 0.846 (plateauing);
ES-C 0.751 → 0.803 → 0.815 → 0.829; ES-A → 0.781 → 0.802 → 0.815 (both still +1.3 pp per 20 steps).

**Where this leaves the goal.** On the benchmark as the reference setting measures it (MATH-500 at a
7168-token cap), 32 prefill rails on 64 sequences per step — no backward pass — sit **1.7 pp below
BP at step 60** and are still climbing, at about the same wall-clock per step (300 s vs 340 s). On
the distillation objective they are nowhere near (KL −30 % vs −94 %), and the decomposition says
why the benchmark does not care: in this pair *both* methods' MATH-500 gains are truncation
reductions (BP's accuracy among completed answers is also flat at 0.948), so the benchmark rewards
"be concise", a low-rank behaviour ES finds; BP additionally keeps quality (ES −2 to −3 pp).

**The displacement budget, measured the hard way** — the discarded ES-D (8 rails, normalisation
bug, 9 % cumulative random-walk displacement by step 20): train KL 0.27 → 0.38, MATH-500 0.761,
capped 18 % (it *did* shorten), **accuracy among completed answers 0.887 (−6 pp)**. A 9 % isotropic
random walk costs six points of reasoning quality — the σ-probe's ×1.36 KL at 9.3 % (§5), seen on
the benchmark.

**Step-80 (BP, ES-C) and ES-B's step 20** (2026-09-01 02:30):

| arm | step | train KL/tok | MATH-500 | % capped | acc among uncapped |
|---|---:|---:|---:|---:|---:|
| **BP** | 80 | **0.013** | **0.848** | 16 % | **0.950** |
| ES-C (32 rails, α 1.25e-3) | 80 | 0.198 | **0.817** (↓ from 0.829 @60) | 16 % | 0.911 |
| ES-B (128 rails, α 1e-3 = A's footprint) | 20 | 0.222 | 0.786 | 19 % | 0.924 |
| ES-A (32 rails, α 5e-4) | 20 | 0.231 | 0.781 | 21 % | 0.935 |

MATH-500 trajectories: BP 0.751 → 0.823 → 0.843 → 0.846 → **0.848**; ES-C 0.751 → 0.803 → 0.815 →
0.829 → **0.817**. Two things settle here:

1. **ES-C has stopped improving.** Its 0.829 @ 60 (1.7 pp under BP) is the top of a plateau: 0.817 /
   0.817 / 0.825 at 80 / 100 / 120 (mean 0.822; the ruler's own noise is ≈ 1.1 pp per eval, see
   below), while BP keeps rising with quality intact. Gains from the continuing coherent motion are
   being cancelled by the accumulating random walk (6.6 % of RMS(W) at step 120) — the √S
   accumulation the displacement-budget analysis predicts (§3); the discarded ES-D and ES-D′ show
   the same thing as an outright collapse at 2× the per-step footprint.
2. **Rails do not buy benchmark progress; step size does.** ES-B (4× the rails of A at A's
   per-step footprint, hence 2× A's coherent motion) evaluates like A (0.786 vs 0.781). Under
   z-scored ES the coherent motion per step is α; N only shrinks the noise, and on this ruler the
   noise was not yet what limited A/B at step 20. What separates the arms is α — and α is capped
   by the random walk.

**ES-D′ (8 rails, α 1.25e-3, fixed normalisation) at step 20**: MATH-500 **0.800** (C with 32 rails:
0.803), capped 16 %, completed-answer accuracy 0.910 (C: 0.915), at **135 s/step** — 40 % of BP's
wall-clock, a quarter of its sequences, no backward pass, 2.3 pp under BP at the same step. The
cheap-side confirmation of point 2: on this ruler 8 rails ≈ 32 rails ≈ 128 rails at equal α. Its
per-step random displacement is 2× C's (5.2 % at step 20), so it should turn over correspondingly
sooner.

**Step 100 (BP, ES-C) and ES-D′ at 40** (2026-09-01 04:35):

| arm | step | MATH-500 | % capped | acc among uncapped | cum. displacement |
|---|---:|---:|---:|---:|---:|
| BP | 100 | 0.844 (plateau ≈ 0.845 since step 40) | 17 % | 0.947 | — |
| ES-C | 100 | 0.817 (flat since the drop at 80) | 17 % | 0.922 | 6.1 % |
| ES-D′ | 40 | **0.762** (↓ from 0.800 @20) | **22 %** | 0.924 | 7.4 % |

**The turnover obeys the budget law.** Each ES arm peaks when its cumulative random-walk
displacement `ρ√S` reaches `B ≈ 5 %` of RMS(W), i.e. at `S* = (B/ρ)²` steps, and then loses
ground (the completed-answer accuracy and, for D′, even the truncation rate go the wrong way):

| arm | per-step footprint ρ | predicted S* = (0.05/ρ)² | observed peak |
|---|---:|---:|---:|
| ES-D′ (4 pairs, α 1.25e-3) | 1.17 % | 18 | 0.800 @ 20, collapsed by 40 |
| ES-C (16 pairs, α 1.25e-3) | 0.58 % | 74 | 0.829 @ 60, down at 80 |
| ES-A (16 pairs, α 5e-4) | 0.25 % | 400 | still rising at 60 (stopped) |
| ES-B (64 pairs, α 1e-3) | 0.23 % | 470 | still rising at 20 |

The coherent motion accumulated by the peak scales as `α·S* ∝ n_pairs/α`, so the arms that would
peak *highest* are the slow ones (A, B) — but A needs ~400 steps ≈ 33 h to get there, and the
quality loss at the peak (≈ 2–3 pp on completed answers, set by B) does not depend on α. The
ceiling for this design on this ruler is therefore a couple of points under BP, reached ~7× slower.

**Step 120 (BP, ES-C) and the resumed ES-A at 80** (2026-09-01 06:45):

| arm | step | MATH-500 | % capped | acc among uncapped | cum. displacement |
|---|---:|---:|---:|---:|---:|
| **BP** | 120 | **0.859** | 16 % | **0.958** | — |
| ES-C | 120 | 0.825 | 17 % | 0.937 | 6.6 % |
| ES-A (resumed) | 80 | 0.813 | 17 % | 0.933 | 1.9 % + 1.0 % |

**Ruler resolution.** The resumed ES-A re-evaluated the *identical* step-60 weights: 0.799 vs the
original 0.815 (completed-answer accuracy 0.935 vs 0.919). So one MATH-500 n = 2 @ T = 0.6 eval has
σ ≈ 1.1 pp, and the "±0.01" batch-noise caveat applies to every single-eval comparison on this
page; trends over ≥ 3 evals are what to read. On that basis: BP 0.843–0.859 from step 40 on; ES-C
0.817–0.829 from step 60 on; ES-A still climbing slowly (0.781 → 0.802 → 0.807 ± → 0.813).

**Round at 08:45** (BP/ES-C 140, ES-A 100, ES-B 40):

| arm | step | MATH-500 | % capped | acc among uncapped | cum. displacement |
|---|---:|---:|---:|---:|---:|
| BP | 140 | 0.850 | 17 % | 0.957 | — |
| ES-C | 140 | **0.795** (↓, now beyond noise) | 19 % | 0.927 | 7.1 % |
| ES-A | 100 | 0.804 (flat since 60: 0.815/0.799/0.813/0.804) | 17 % | 0.926 | 3.4 % |
| ES-B | 40 | 0.814 (A @40: 0.802, C @40: 0.815) | 16 % | 0.921 | 1.5 % |

**The ES ceiling on this ruler is ≈ 0.82, for every arm.** Each ES arm climbs to the same
truncation floor (16–17 % capped, the same floor BP reaches) and stops there: A at ~0.805 from step
60, C at ~0.82 from step 60 (now declining at 7 % displacement), B on the same track. The slow arm
does *not* go on to a higher, later peak as the coherent-motion scaling suggested — because the
benchmark gain is a saturating one-dimensional effect (stop truncating), not the surrogate.
What separates BP is what happens *after* the floor: its completed-answer accuracy keeps improving
(0.948 → 0.958) while every ES arm sits 2–3 pp below its starting quality. Net: **ES ≈ BP − 3 pp,
reached by step 40–60, then flat or worse.**

### 7.1 The ruler without the cap — 16k-token re-evaluation (decisive)

Same checkpoints, MATH-500 + AIME24, n = 2 @ T = 0.6, **cap 16384** instead of 7168
(`scripts/zo_opd/paper_align/eval_math.py`; FSDP shards merged with `legacy_model_merger.py`;
raw JSON in `data/eval16k/`). σ per MATH-500 number ≈ 1.4 pp; AIME24 (30 problems) ≈ 7 pp.

| checkpoint | MATH-500 @16k | Δ vs base | % capped @16k | AIME24 @16k | Δ | (MATH-500 @7k, for reference) |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.837 | — | 7.3 % | 0.300 | — | 0.751 |
| **BP @140** | **0.897** | **+6.0** | 2.2 % | **0.533** | **+23** | 0.850 |
| ES-A @100 | 0.855 | +1.8 | 4.1 % | 0.350 | +5 | 0.804 |
| ES-B @40 | 0.848 | +1.1 | 3.9 % | 0.267 | −3 | 0.814 |
| ES-C @140 | 0.839 | +0.2 | 4.8 % | 0.267 | −3 | 0.795 |

Remove the truncation and the picture is unambiguous: **BP's gain survives and grows** (+6 pp on
MATH-500, +23 pp on AIME24 — the student has actually acquired the teacher's shorter, correct
reasoning), while **every ES arm's gain collapses to ≤ 1.8 pp, inside noise**. The +5–8 pp the ES
arms showed at the 7168 cap were the cap. This is the KL story (§7's first table) told on the
benchmark: ES moved the student 15–30 % of the way to the teacher in KL and learned to stop
earlier; BP moved it 95 % of the way and learned what the teacher knows.

**Final answer to the goal.** On the reference OPD setting there is no configuration of
forward-only rails — decode or prefill, 8 to 128 rails, any step size inside the displacement
budget — that is comparable to BP-OPD on what the benchmark actually measures once truncation is
removed. The rails do exactly what the analysis says a `√(2δ_max C/tr(H))` budget buys: a
low-dimensional behavioural shift (be shorter), at 2–3 pp of quality, in ~40–60 steps, and nothing
more. Where forward-only ES *is* the right tool remains the case the ES-math thread studied — a
non-differentiable objective with no BP alternative.

**Last round at the 7168 cap** (10:30): BP @160 0.851 (completed-answer accuracy **0.967**, still
rising); ES-A @120 0.798; ES-C @160 0.807 — both ES arms flat at ≈ 0.80 ± 0.01 for 60+ steps. ES-A
and ES-C were stopped here (2026-09-01 10:35); nothing further to learn from them. BP runs to its
280-step reference.

**BP endpoint — stopped at step 239 on the user's call (2026-09-01), plateaued.** Evals every 20
steps, MATH-500 @7168: 0.751 → 0.823 → 0.843 → then flat in 0.844–0.861 from step 40 through 220
(best 0.861 @180; AIME24 best 0.400 @180); train KL/tok 0.278 → 0.004. The 16k-cap numbers in §7.1
(BP @140: 0.897 / +23 pp AIME24) stand as its headline. Latest checkpoint: `global_step_220`.

*(B stopped at 40 — its role (N ablation) is done. **ES-A resumed from its step-60 checkpoint** (2026-09-01
04:40, GPU 1, verl `resume_mode=auto`) to test the law's prediction that the slow arm keeps rising
toward a higher, later peak (~step 400); its steps ≥ 61 use the RMS-normalised update, which
differs from the first 60 by ≤ 8 % in step size. Note `es/cum_footprint` restarts from 0 on
resume — add the 1.9 % accumulated before. Eval decompositions: `scripts/zo_opd/ds15b/eval_decomp.py`.)*

## 7.2 Follow-up runs (launched 2026-09-01, wandb `es_opd_JustRL_1p5b`)

| run | GPU | config | calibration notes |
|---|---|---|---|
| **es-token-decode** | 7 | `es_token` trainer with the rail-aware kernels (`attn_impl=shared`, `lm_head_impl=stream`), 64 prompts × n=1, pack_width 8, N=32, σ=1e-3 (probe 1.9 %), lr **4e-3** | lr set by `train/update_footprint`: 3e-4 gave 2.1e-4/step (15× under the es-prefill band) → restarted at 4e-3 → footprint 2.5–2.8e-3/step ✓. **Warm step 773 s = decode 498 + teacher 13 + assemble 262** (`dW_cos_prev` 2e-4 ≈ 0, as the old diagnosis predicts) |
| **es-rl** (reward-only baseline, no teacher) | 6 | `es` trainer, dense, N=10, greedy rollouts, batch 64 resampled, 7168 tok, α=2.89e-4 | on DAPO the binary reward is nearly floored for this student (train acc 6–8 %, matching BP's `true_reward` ≈ 3 %); σ=1e-3 gave `reward_std` 0.019–0.026 — under the ES-math working band (0.035–0.055) — → restarted at **σ=2e-3**. Base greedy MATH-500 @7168 = 66.6 % (the greedy ruler reads lower than n=2 @T=0.6's 75.1 %) |

**Training-efficiency comparison at matched settings** (64-seq objective, N = 32 where applicable,
7168-token responses, same GPU class):

| method | s/step (median over the run) | seqs/step | **s per sequence-evaluation** | note |
|---|---:|---:|---:|---|
| BP | 329 | 256 | **1.29** | gen 126 / teacher 79 / log-prob 34 / update 124 |
| es-prefill N=8 | 143 | 64 | 2.2 | rails 59 s |
| es-prefill N=32 | 322 | 64 | **5.0** | rails 239 s (7.5 s/rail) |
| es-prefill N=128 | 1215 | 64 | 19.0 | rails 1125 s |
| es-token-decode N=32 (shared-KV + streaming-head kernels) | 746 | 64 | **11.7** | decode 496 / assembly 238 |

Full N-scaling tables (64-seq and 256-seq batches) and the algorithm boxes:
[zo_opd_short.md](zo_opd_short.md). At BP's own 256-seq batch the profiled law is
`step(N) ≈ 260 + 32·N s`, so time-parity with BP allows only **N ≈ 2** — the equal-cost ES gets
one or two probes per step (run `ds15b_es-prefill_b256_N2_*`, non-antithetic, launched
2026-09-02).

Even with the rail-aware kernels, the per-token decode machinery costs **2.2× es-prefill per
sequence** for the same N and the same (actually weaker: detached-history) information — the §4
cost ordering, now measured end-to-end on identical settings.

**Standard-ruler check (n = 2 @ T = 0.6, 7168 tokens), es-token-decode step 60:** MATH-500
**0.803** (base 0.751; es-prefill @60: A 0.815 / C 0.829; BP @60: 0.846), AIME24 0.167, mean length
3333, 15 % capped. The per-token decode estimator lands at the bottom of the es-prefill plateau at
the same step count — the same "shorten" effect, nothing more — while costing 2.2× per sequence.
With that, **es-token-decode is dominated on both axes measured here: information (≤ es-prefill)
and cost (2.2×)**, kernels included.

**…and ES-RL best@80 on the same ruler:** MATH-500 **0.769** (+1.8 pp over base, ≈ 1σ), AIME24
0.183, length 3454. The +7.8 pp greedy climb collapses on the sampled ruler — most of it was
greedy-repetition unsticking, which sampling at T = 0.6 already provides. **Consolidated
standard-ruler ranking at comparable step counts** (MATH-500, n = 2 @ T = 0.6, 7168 tokens):

| | base | es-rl best@80 | es-token-decode @60 | es-rl **best@150** | es-prefill C @60 | BP @60 |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 | 0.751 | 0.769 | 0.803 | **0.808** | 0.829 | **0.846** |
| AIME24 | 0.150 | 0.183 | 0.167 | 0.200 | — | 0.267 (@60) |

Every forward-only arm sits strictly between base and BP, ordered by signal quality — with one
correction to the first reading: **the reward-only ES kept converting in its second half.** Its
greedy curve ran 66.6 → 74.4 @80 → 76.6 @150 (still rising at the end), and the sampled ruler
followed: best@80 0.769 → best@150 **0.808** (+5.7 pp over base) at 150 iterations ≈ 15 GPU-h,
teacher-free. Mean response length 3145 (the shortest of all arms) and 14 % capped, so the length
channel is again a large part of it — but at 150 iterations the reward-only arm has caught the
per-token decode probe and sits 2 pp under es-prefill. Consistent with the ES-math thread: on the
*accuracy* landscape, reward-fitness ES works and was not yet done at 150 iterations; on the *KL*
landscape nothing forward-only approaches BP.

**Interim curves (greedy MATH-500 @7168, base = 66.6 on this ruler; 2026-09-02 03:30):**

| run | evals | train side |
|---|---|---|
| es-token-decode (N=32, lr 4e-3, kernels on) | @0 70.6 → @20 71.0 → @40 **72.6** | L_clean 0.26 → 0.22; `dW_cos_prev` ≈ 0 throughout; 12.1 s/seq |
| es-rl (dense N=10, σ 2e-3, no teacher) | @10 69.0 → @30 67.6 → @50 69.2 → @60 72.6 → @80 **74.4** | train accuracy 7 % → 26 %, reward_std 0.03–0.07; ~6 min/iter, eval time shrinking (115 → 99 s → shortening) |

The **reward-fitness ES is climbing for real** — the same low-rank accuracy landscape the ES-math
study exploits — while the KL-probing es-token-decode shows the familiar one-jump-then-creep. The
greedy ruler is generous to any perturbation on this model (+4 pp after a single random-walk
update), so the standing plan is the standard-ruler offline eval (n = 2 @ T = 0.6 + the
length/truncation decomposition) on the final checkpoints; `es_coef_best.pt` (full bf16 weights,
saved on each new best) guarantees the es-rl side is evaluable.

## 8. Next steps / where rails could still pay

1. **Read `r_eff` first.** If `r_eff/N ≫ 1` at any affordable N, forward-only full-parameter OPD is
   closed and the ES run is a confirmation, not a search.
2. **Subspaces.** Measured (§5): coarse groups ≈ 1×, calibrated activation subspace ≤ 2.7× on
   `‖g_S‖²/tr(H_S)` → ≤ 1.6× on the gain. A learned/BP-aligned subspace could do better, but then
   BP is in the loop anyway.
3. **Hybrid use of the rail machinery** (now cheap: `es_update.py` can evaluate *any* direction):
   line-search / trust-region on the BP step, or ES only inside the span of recent BP updates. Low
   expected gain on a ±2 pp ruler; not started.
4. **Not worth doing:** more decode-rail engineering for a fixed-trajectory objective; per-token fresh
   noise; the sampled-token IW fitness.

## 9. The regime the method is actually aimed at: seeds-only distributed training

The motivation for weight perturbation (2026-09-01, user): at very large scale, WP-ES needs only
**(seed, fitness) scalars** on the wire — every worker regenerates ε from seeds and applies the
identical update — plus inference-grade memory (no grads, no Adam, no activations). BP is not
avoided for its FLOPs but for its **communication and memory**. Two consequences of the measured
constants for that regime:

1. **NP is out on exactly these grounds.** A node-perturbation update is `Σ_t g_t x_tᵀ` — it needs
   the data-dependent activation tape, so workers can no longer reconstruct the update from seeds;
   the scalars-only property is a WP exclusive. The per-probe statistical gap is already
   benchmarked: single-layer probe cosine NP 0.205 vs WP 0.0056 at K = 400
   ([wiki/es_token_trainer.md](../../wiki/es_token_trainer.md) §3) — exactly the `√d_in` factor of
   §2.3. WP pays that factor for the seeds-only wire format.
2. **The budget law is compute-friendly at scale.** Coherent motion = `√(2δ·C/tr(H))` with C = total
   rail forwards, and both data and rails shard perfectly (per-shard fitness → all-reduce of N
   scalars). With the measured tr(H) = 5.5e3, RMS-motion equal to BP's full 280-step run
   (280 × lr√D ≈ 11.8, or ~3.5 allowing Adam's imperfect alignment) needs:

   | C (rail forwards, 64-seq × ~6k-tok batch each) | motion | ≈ aligned BP steps | FLOPs vs the whole BP run | wall-clock on 1024 inference workers |
   |---:|---:|---:|---:|---:|
   | 1e5 | 0.93 | 22 | ~20× | ~40 min |
   | 1e6 | 2.9 | 70 | ~200× | ~7 h |
   | 1e7 | 9.3 | 222 | ~2000× | ~3 days |

   and the damage budget stops binding: at large C one can run δ = 2 % of the KL and still out-move
   BP. So in the communication-bound regime the trade is explicit: **~2–3 orders of magnitude more
   FLOPs, near-zero gradient traffic, inference-only workers.**

**The open question that our runs flag for this extrapolation:** the ES arms' *KL* progress tracked
accumulated coherent motion as the law predicts (train KL 0.28 → 0.17–0.19 ≈ 10–20 BP-step
equivalents), but the *benchmark* stopped following the KL at ES's motion scale (§7.1) — the small-C
arms only ever cashed in the 1-D "shorten" component. Whether BP-level motion bought with huge C
converts to BP-level quality is the hypothesis a mid-scale run (C ≈ 1e5–1e6, e.g. 8 GPUs × 1–2
days at small α) would test before any large-scale build-out.

## Reference

### R1. Derivation of the progress ratio
`ĝ = (1/N)Σ_n ⟨g,ε_n⟩ε_n` is unbiased. `E[ĝᵀHĝ] = gᵀHg (1 + 2/N) + ‖g‖² tr(H)/N` for Gaussian `ε`;
dropping the `2/N` term gives the text. Optimal `η* = ‖g‖²/(κ_g‖g‖² + ‖g‖² tr(H)/N)`; substitute.
Antithetic pairs remove even-order terms but not this one. Z-scored ES (`W += (α/N)Σ z_n ε_n`)
is the same update with `η` set by `α/(σ·std(F))`; the ratio is unchanged.

### R2. Why the token count cancels
Per-token gradients are near-orthogonal (`ρ = ‖Σ_t G_t‖²/Σ_t‖G_t‖² = 0.77`, §12.5). Signal
`‖G‖² = ρ Σ‖G_t‖²`; independent per-token probes give noise `Σ_t‖G_t‖² tr(H)/N`. Both scale with
`T`; the ratio is `ρ N/r_eff`.

### R3. Cost-model numbers used in §4
`pack_width` 64, N = 8: 15.036 ms per token-step, 576 rows (zo_opd wiki §9). Clean-only at the same
width ≈ 4 ms. Prefill: 1.5 B params × 2 FLOPs/param/token = 3 GFLOP/token; ~400 TFLOP/s sustained
→ ~130 k tok/s. Both are H100 numbers; the ratio, not the absolute, is the point.

### R4. Exact-objective forms available to a prefill rail
- k1 / PG surrogate: `F(W) = Σ_t m_t A_t (log π_W(y_t) − log π_0(y_t)) / Σ m_t` (what `es_update.py` uses; gradient at `W_0` = BP's).
- top-K (`LOG_PROB_TOP_K=16`): `Σ_{t,k} A_{t,k} Δlog π_W(k|s_t)` via `compute_log_probs_for_ids` on the frozen student top-K set (also wired in `es_update.py`).
- full-vocab reverse KL: needs teacher logits per position; used by `opd_curvature.py`, not by the trainer.
