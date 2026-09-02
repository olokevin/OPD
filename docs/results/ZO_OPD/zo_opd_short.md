# ZO-OPD — short version

> One-page overview of the zeroth-order on-policy-distillation thread. Full record with
> methods, raw numbers and falsified hypotheses: [zo_opd.md](zo_opd.md).
> Student `Qwen/Qwen3-1.7B` (non-thinking) ← teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`,
> DAPO-Math-17k, MATH-500 greedy n=1 as the ruler (**base = 73.60 ± 1.97**).

## 2026-09-01 — the DeepSeek/JustRL setting: forward-only rails vs BP, closed out

> New reference pair (thunlp/OPD's): student `DeepSeek-R1-Distill-Qwen-1.5B` ← teacher
> `JustRL-DeepSeek-1.5B`, DAPO-Math-17k, 1 GPU, k1 objective. Full analysis, theory and raw data:
> [es_rails_formulation.md](es_rails_formulation.md). wandb `es_opd_JustRL_1p5b`.
> Run naming from here on: **es-prefill / es-token-prefill / es-decode / es-token-decode**
> (perturbation granularity × where it is evaluated; table in es_rails_formulation.md §1.5).

**What ran.** BP-OPD (256 seqs/step, Adam lr 1e-6) vs **es-prefill** (`algorithm.es_update`: same
rollout/teacher/advantages, but the update comes from N antithetic seeded weight perturbations each
scored by one teacher-forced prefill; 64 seqs/step, no backward). Arms: N=32 α=5e-4 (A), N=128
α=1e-3 (B), N=32 α=1.25e-3 (C), N=8 α=1.25e-3 (D′).

**Results (MATH-500, n=2 @ T=0.6).** At the training cap (7168 tok), from base 0.751:

| | best | when | then |
|---|---:|---|---|
| BP | 0.859–0.861 | steps 120–180 | flat, completed-answer accuracy rises 0.948 → 0.967 |
| es-prefill C | 0.829 | step 60 | plateau 0.80–0.83, quality −2 to −3 pp |
| es-prefill A / B / D′ | 0.804–0.815 | 40–100 | same plateau; D′ (N=8) collapses by 40 |

**At a 16k cap (truncation removed): BP +6.0 pp MATH-500 / +23 pp AIME24 over base; every ES arm
≤ +1.8 pp (inside noise).** Decomposition: at 7168 both methods' gains are truncation reductions —
"think shorter" — but BP gets it by actually distilling (train KL 0.278 → 0.004) while ES only
shortens (KL → ~0.19) and pays quality for it.

**Key takeaways**

1. **The old `cos ≈ √(N/D)` story was the wrong unit.** Measured on this pair: `κ_g = 34`,
   `tr(H) = 5.5e3` → `r_eff ≈ 160` — the OPD-KL landscape is very low-rank, ES is **not**
   curvature-limited. What binds is the **random-walk displacement budget**: within it, coherent
   motion = `√(2δ·C/tr(H))` for C total rail forwards ≈ 1/40 of a BP step per step at N=32.
   Verified in-run: every arm's per-batch descent matched the first-order prediction, and each arm
   peaked at `S* = (5 %/per-step footprint)²` steps then stalled (C @ ~60, D′ @ ~20).
2. **Rail count buys nothing on the benchmark; step size does, and step size is capped by the
   budget.** N=8 ≈ N=32 ≈ N=128 at equal α (z-scored ES moves α per step; N only shrinks noise).
3. **For a fixed-trajectory differentiable objective, decode rails are dominated by prefill
   rails** (~3× cost per token-evaluation, detached-history error, custom machinery) — and fresh
   per-token noise multiplies targets, not probes (token count cancels; zo_opd §12.5).
4. **Subspaces don't rescue it**: coarse groups ≈ 1×; the calibrated activation subspace (zoact,
   0.04 % of params holding 34 % of ‖g‖²) buys only 2.7× on `‖g_S‖²/tr(H_S)` → 1.6× on the gain.
5. **Where WP-ES still makes sense**: the seeds-only distributed regime (only (seed, fitness)
   scalars on the wire, inference-only workers — NP can't do this). Matching a full BP run's
   coherent motion costs ~10⁶–10⁷ rail-forwards ≈ 200–2000× the FLOPs at near-zero gradient
   traffic; whether that motion converts to BP-level *quality* is the open mid-scale experiment
   (es_rails_formulation.md §9).
6. Cost at parity settings: BP 433 s/step (256 seqs) vs es-prefill N=32 ~360 s/step (64 seqs);
   one prefill rail = 8.7 s per 420 k tokens (~145 TFLOP/s).

**Now running (2026-09-01):** es-token-decode with the new rail-aware kernels
([es_profile_results.md](es_profile_results.md)) on the same setting (GPU 7) for the
efficiency/learning comparison, and a reward-only **ES-RL baseline** (dense, no teacher, GPU 6).

---

## What we set out to do

Replace OPD's backprop with a **forward-only** (zeroth-order) update, and find out whether it can
match BP-OPD, which on this setting gains +2.8 pp on MATH-500 (0.7250 → 0.7532 at n=8).

Two estimators were built and measured:

| | how it perturbs | status |
|---|---|---|
| **es_token** | rank-1 weight perturbation, **fresh per token**; rails read the clean row's KV | **dead end — diagnosed, see below** |
| **ES-OPD** (sequence-level) | one Gaussian perturbation of **every** parameter, held for the whole rollout | **answered on the DeepSeek/JustRL setting as es-prefill — see the 2026-09-01 section above** |

## Current results

**BP-OPD works** (the reference): +2.8 pp MATH-500, up on 4/4 benchmarks, tracks the 8-GPU NERSC
run exactly to step 60.

**es_token does not learn at any step size.** MATH-500 greedy vs the 73.60 base:

| LR | slope | endpoint |
|---|---:|---:|
| 1e-5 | flat within noise | — |
| 1e-4 | **−1.28 pp / 100 steps** (t = −3.2) | 70.8 |
| 1e-3 | **−6.52 pp / 100 steps** (t = −11.4) | 57.4 |

Monotone damage, no learning anywhere in the bracket.

**ES-OPD**: not yet answered. Three fitness designs were tried; the first two were gamed (below)
and the third now passes its gate, awaiting GPUs.

## The problems found, and what fixed them

**1. es_token's premise was wrong — and it is not fixable by tuning.**
The design assumed a training step collects `K = B·T·N` probes, predicting per-layer cosine ≈ 0.20.
Measured cosine is **7.0e-4** (one layer) and ≲2e-4 (all layers). The real law is
**`cos ≈ sqrt(N/D)`** — each token's perturbation probes *that token's own* gradient, and per-token
gradients are near-orthogonal, so the token count **cancels**. The original 0.86–0.99×-of-bound gate
never applied to training: it perturbed one layer and re-probed one loss.
Three independent measurements agree the update is **≥92 % incoherent**: the offline audit, the
in-run `dW_cos_prev` bound (cos ≲ 0.01), and the trained checkpoint's displacement (1.05e-2 measured
vs 9.2e-3–1.4e-2 for a random walk, against 0.13–0.20 if it were coherent).
*Ruled out by measurement, not assumption:* implementation bugs (the ES checkpoint moved **further**
from base than BP's, median 1.86×), the learning rate (ES already stepped 2–4× larger than BP), σ
(cosine flat across 1e-5…3e-2), and clean-KV myopia (cos(g_direct, g_full) = +0.28…+1.00).
**Conclusion: at cos ≈ 0 there is no good step size** — small steps do nothing, large steps are
random-walk damage, and the window between is empty. Fix the *direction*, not the step.

**2. ES-OPD fitness was length-hackable — twice, in opposite directions.**
Under ES the fitness *compares rollouts the policy produced*, so length is a decision variable:
- per-token **`mean`** → rewards padding. Length +20.8 tok/iter (t = +14.9), corr(len, KL) = −0.88,
  MATH-500 74.8 → 68.2.
- per-sequence **`sum`** → rewards truncation, because per-token KL is positive (≈0.29) so total KL
  tracks length. Length **−60.1** tok/iter, 893 → **197** tokens (−78 %).
- **Fixed** by removing length from the comparison: score every rail on **one clean rollout per
  prompt**, teacher-forced (`opd_fixed_traj`). Verified — `resp_len` is now *identical* at every σ.

**3. The fixed-trajectory fitness was initially inverted.**
With `y ~ π₀` instead of `y ~ π_n`, `E[Σ(log π_n − log q)] = KL(π₀‖q) − KL(π₀‖π_n)`, so minimising it
**maximises** distance from the current policy — a bigger perturbation always won. Replaced by the
teacher-probability-weighted likelihood `Σ_t q(y_t)·log π_n(y_t)` (the repo's `teacher_p` mode), which
is well defined on another policy's samples. **Gate now passes**: fitness decreases monotonically
with σ (−60.9 → −97.4 → −299.2 → −4767 for σ = 0 → 8e-3).

**4. Smaller but real.** σ must be calibrated at the *production* sequence length (a fixed
perturbation compounds along the trajectory: σ=3e-3 is 2.26× the reference KL at 512 tokens but
3.25× at 1536). The ES training decode used no top-p while BP and every eval used 0.95. `_np_is_eos`
missed token `151643`. vLLM's `ray` executor cannot co-locate a teacher (needs `uni`), and
`ESNcclLLM` silently sent engines to physical GPU 0.

## What is still open

1. **Does sequence-level ES-OPD learn at all?** The central question. Fitness is now sane and
   calibrated (σ ≈ 2e-3); the run needs GPUs.
2. **Full-parameter ES may be the wrong target regardless.** `cos ≈ sqrt(N/D)` says the lever is
   shrinking `D`, which is what the ES study's LoRA/structured arms do. A low-rank ES-OPD is the
   natural follow-up if the dense run stalls.
3. **KL is not accuracy.** Every ES fitness here optimises teacher agreement; whether that transfers
   to MATH-500 is a separate question the in-run eval must answer.
4. **es_token's remaining cost.** Even fixed, it buys no gradient information over sequence-level ES
   at the same rail count while costing far more per step — its per-token machinery has no payoff.
   *(The cost side is now measured: with the 2026-08-31 rail-aware kernels —
   [es_profile_results.md](es_profile_results.md) — decode rails are near-free up to `B(1+N) ≈ 130`
   rows, e.g. 48 rails within 10 % at B=1. That confirms the systems claim only; the information
   argument above is unchanged.)*
5. **`train/kl_mean` is not a learning curve** (the batch is resampled each iteration); only the
   fixed-set `eval/accuracy` is. BP now logs the *same* MATH-500 greedy metric to the same key, so
   BP / es_token / ES-OPD curves overlay directly.
