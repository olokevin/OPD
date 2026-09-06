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

**Follow-ups (2026-09-02, wandb `es_opd_JustRL_1p5b`):** es-token-decode with the rail-aware
kernels and a reward-only ES-RL baseline, both on the same setting. Standard-ruler ranking — base
0.751 · es-token-decode@60 0.803 · **es-rl@150 0.808 (teacher-free!)** · es-prefill@60 0.829 ·
**BP@60 0.846**. es-token-decode is dominated on both axes (≤ es-prefill information at 2.2× the
cost per sequence, 12.1 vs 5.6 s/seq, kernels on). The reward-only ES kept converting to the end
(greedy 66.6 → 76.6; sampled 0.769 @80 → 0.808 @150, still rising) — the accuracy landscape is
ES-friendly even where the KL landscape is not; its responses are also the shortest (3145), so the
length channel carries much of it. Details: [es_rails_formulation.md](es_rails_formulation.md) §7.2.

## The four rail algorithms (naming of es_rails_formulation.md §1.5)

Common to all: rollout `y ~ π_W` on DAPO prompts; teacher log-probs `log q`; frozen per-token
advantage `A_t = log q(y_t) − log π_W(y_t)` (k1); Gaussian noise `ε(s)` regenerated from integer
seeds — only `(seed, fitness)` scalars ever cross workers.

**es-prefill** (implemented: `algorithm.es_update` in the PPO trainer — the recommended rail):

```
1  y ~ π_W                                  # ONE clean rollout (stock vLLM)
2  A_t = log q(y_t) − log π_W(y_t)          # one teacher forward, frozen
3  for i = 1..N/2:                          # antithetic pairs
4      for sign in {+, −}:
5          W ± σ·ε(s_i) in place (fp32 master)
6          F± = Σ_t m_t A_t Δlog π(y_t) / Σ m_t     # ONE teacher-forced prefill
7          restore W
8      d_i = (F₊ − F₋)/2                    # = σ⟨g, ε_i⟩ + O(σ³)
9  W += (α / (N/2)) Σ_i (d_i / RMS(d)) · ε(s_i)     # z-scored ES step, motion α/√(N/2)
```

**es-token-decode** (implemented: `es_token` trainer + rail-aware kernels; ruled out):

```
per decode token t, rail n = 1..N (rails ride the CLEAN rollout's KV, never commit):
    at every linear:  y ← y + σ((r_n⊙v_t)ᵀx)·(s_n⊙u_t)      # fresh rank-1 ΔW per token
    l_{n,t} = (π_n(ŷ_t)/π_0(ŷ_t)) · (log π_n(ŷ_t) − log q(ŷ_t))   # sampled-token IW-KL
update:  δW = 1/(Nσ) Σ_{t,n} (l_{n,t} − mean_m l_{m,t}) · (s_n⊙u_t)(r_n⊙v_t)ᵀ ;  W −= lr·δW
```

Detached-history estimate; fresh-per-token noise multiplies *targets* not probes, so the
information is N scalars per step, same as es-prefill (zo_opd.md §12.5).

**es-decode** (implemented 2026-09-05: `es_token` trainer with `rail_mode=seq`, kernels in
`rail_seq_kernels.py`; arms running on GPU 7, wandb `ds15b_es-decode_{full,r1}_N32_sig1e-3_a1.25e-3`):
line 5's held `ε(s_n)` per rail, but evaluated by decode rails on the clean KV instead of a prefill —
the same estimator as es-prefill, measured with the detached-history error.

```
per step: seeds s_1..s_{N/2};  rail 2i = +ε(s_i), rail 2i+1 = −ε(s_i)   (held for EVERY token)
    ε full-rank:  {±1}^{m×n} per linear, stored as packed bits (1 bit/elt: 32 rails = 4.9 GB)
    ε rank-r:     (1/√r) Σ_k a_k b_kᵀ, a,b ∈ {±1}          (unit per-element RMS either way)
decode rails ride the clean rollout's KV: at every linear  y_n += σ·ε_n x_n
    (full: packed-bit tensor-core GEMV, 1 launch/linear; rank ≤ 8: inside the fused norm/silu/rope)
F_n = Σ_t A_t (log π_n(y_t) − log π_0(y_t)) / Σ_t 1,   A_t = log q(y_t) − log π_0(y_t)   (k1, as es-prefill)
d_i = (F_{2i} − F_{2i+1})/2 ;  W += (α / (N/2)) Σ_i (d_i / RMS(d)) ε(s_i)                  (es_update.py rule)
```
The update reuses the noise still resident on the GPU (no regeneration); fp32 master on the host.
Decode cost at B=1 (R1-Distill-1.5B, es_profile_results.md §15): held rank-1/4 ≈ es-token-decode's
rails; full-rank adds the bit traffic + unpack, ≈ N × 0.16 GB per token.

**es-token-prefill** (not implemented; analysed in es_rails_formulation.md §4): fresh per-position
`ΔW_t` *inside a prefill* via the rank-1 rail op as a per-position output adjustment
(`y_t += σ(v_tᵀx_t)u_t`). Unlike decode rails the perturbation at `t` propagates to positions > t,
so the per-rail scalar is unbiased for the **full** gradient — but it is still N scalars per step:
no probe gain over es-prefill, only the machinery. Build only if the detached-history error of
es-decode ever matters.

## Efficiency: step time vs N, and where it goes

Medians over the full ds15b runs (warm steps; 64 seqs/step at 64 × ~6 k tokens for ES, 256 for BP;
one H100 NVL, student+teacher co-located). es-prefill scales as
**`step(N) ≈ 85 s + N × 7.4–8.8 s`** — gen/teacher/log-prob are flat, one rail = one prefill of the
batch (≈ 400 k tokens at ~145 TFLOP/s), the ES apply is O(0.1 s):

| arm | gen | teacher | log-prob | rails / update | **step** | **s per sequence** |
|---|---:|---:|---:|---:|---:|---:|
| es-prefill N=8 | 53 | 22 | 9 | 59 (7.4 s/rail) + 0.1 | **143** | 2.2 |
| es-prefill N=32 | 54 | 22 | 9 | 239 (7.5) + 0.2 | **322** | 5.0 |
| es-prefill N=128 | 55 | 23 | 10 | 1125 (8.8) + 0.9 | **1215** | 19.0 |
| BP (256 seqs) | 126 | 79 | 34 | 124 (fwd+bwd+Adam) | **329** | **1.29** |
| es-token-decode N=32 (kernels on) | 496 (decode) | 12 | — | 238 (assembly) | **746** | 11.7 |
| es-decode full-rank N=32 (step 0, 2026-09-05) | 1317 (decode, 32 packed-bit rails) | 15 | — | 10 (apply) | **1343** | 21.0 |

**256-seq batch (BP's own batch), non-antithetic, profiled 2026-09-02** (`profile_es_prefill_N.sh`,
2 warm steps per point): `step(N) ≈ 260 s + N × 32 s` — the rail cost is exactly proportional to
batch tokens (32 s vs 7.5 s at 64 seqs), the fixed part is BP's own gen+teacher+log-prob:

| N (256 seqs) | 2 | 4 | 8 | 16 | BP |
|---|---:|---:|---:|---:|---:|
| step (s) | **325** | 383 | 513 | 759 | **329** |

So at BP's batch, the equal-wall-clock rail budget is **N ≈ 2** — the whole ES information budget
at time parity with BP is one or two probes per step. (Training run at this point:
`ds15b_es-prefill_b256_N2_sig1e-3_a4.4e-4`, plain sampling, mean baseline.)

Reads: (1) at equal wall-clock per step (N=32 ≈ BP), BP processes 4× the sequences — es-prefill
is ~4× BP per sequence-evaluation and rails are the whole marginal cost; (2) N=8 is near-free
(rails ≈ gen); (3) es-token-decode is 2.3× es-prefill at the same N *with* the 2026-08-31
rail-aware kernels — its decode still pays ~0.02 ms/rail-row-token and its per-token assembly
238 s/step. Kernel-level profile ([es_profile_results.md](es_profile_results.md)): the shared-KV
attention cuts rail KV traffic from 5–25× to 1.1–1.7× of clean decode and the streaming LM head
never materialises `[rows, V]` (1.4–1.5×, fp32-accurate); free-rail frontier N_free(10 %) = 4–8 at
B=8 up to the cuBLAS ridge `B(1+N) ≈ 130` — decode rails are cheap only below that ridge, i.e. at
batch sizes where generation itself under-uses the GPU.

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
