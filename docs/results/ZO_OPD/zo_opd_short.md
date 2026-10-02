# ZO-OPD — short version

> One-page overview of the zeroth-order on-policy-distillation thread. Full record with
> methods, raw numbers and falsified hypotheses: [zo_opd.md](zo_opd.md).
> Student `Qwen/Qwen3-1.7B` (non-thinking) ← teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`,
> DAPO-Math-17k, MATH-500 greedy n=1 as the ruler (**base = 73.60 ± 1.97**).

## 2026-09-07 (evening) — aligned pair running: es-prefill vs es-decode (rank-1), standing ES setting

> The setting audit ([zo_opd.md § "2026-09-07 BP es-prefill and es-decode setting audit"](zo_opd.md))
> found the earlier es-prefill-vs-es-decode comparison confounded (all-params vs linears-only probe,
> 16×4 vs 64×1 prompts, top-p 1 vs 0.95). This pair removes every difference except *where the rail is
> evaluated*. Full hyper-parameter table and running results:
> [zo_opd.md § "2026-09-07 aligned pair"](zo_opd.md). Launcher `scripts/zo_opd/ds15b/aligned_es_pair.sh`.

**Standing ES setting (all future ES runs, CLAUDE.md):** 64 prompts / step (one rollout each), **N=32
perturbations, non-antithetic**, σ=1e-3, α=1.25e-3 z-scored, 7168 tokens, **T=0 / top-p 1.0 for the
training rollout and the eval (greedy n=1)**, decoder blocks only (es-prefill `es_perturb_set=layers`,
1.31 B elements = es_token's linears), 280 steps (1 epoch), in-run MATH-500 + AIME24 greedy pass@1 every
20 steps, wandb `es_opd_JustRL_1p5b`. Runs: `ds15b_es-prefill_layers_T0_b64n1_N32rand_sig1e-3_a1.25e-3`
(GPU 6), `ds15b_es-decode_r1_T0_b64n1_N32rand_sig1e-3_a1.25e-3` (GPU 7), launched 2026-09-07 22:10.
BP reference = the 2026-08-31 run (64 × 4 prompts, all params, Adam 1e-6; its evals are n=2 @ T=0.6:
MATH-500 0.751 → 0.859 @120 / 0.861 @180, AIME24 0.15 → 0.40 @180 — a greedy re-score of its final
checkpoint is needed for a like-for-like number).

**Results (greedy pass@1, MATH-500 / AIME24; both runs complete 2026-09-12; full tables in zo_opd.md):**

| arm | `es/d_std` steps 1–10 | peak MATH-500 (step) | final | gain vs own base, peak / final |
|---|---|---|---|---|
| es-prefill | **1.23e-3** | **0.808** (140) | 0.784 @260 | **+13.8 / +11.4 pp** |
| es-decode r1 | 0.40e-3 | 0.740 (120) | 0.674 @279 (common ruler 0.686) | +9.2 / +2.6 pp |
| BP @220 (T=1.0-trained, greedy re-score) | — | — | 0.678 | +0.8 pp |

**Key reads.** (1) With every setting matched, the clean-KV decode rail carries **⅓ of the prefill
rail's signal** (3.0× at T=0) and its accuracy curve peaks earlier and reverts to base by the end —
the 09-07 verdict holds without the confounds; es-decode has no niche. (2) es-prefill beating BP here
is the greedy protocol, not the method: BP's T=1.0-trained policy loops under greedy decoding (36 %
capped, 0.678), and on the sampled ruler BP 0.859 > es-prefill 0.829 still stands; a like-for-like
greedy reference needs BP trained at T=0. (3) The greedy gains are mostly "stop looping" (MATH-500
cap-hit 0.31 → 0.17). Both runs crashed once (Ray logs filled the root disk, 2026-09-08) and were
resumed; es-prefill's peak checkpoint was pruned (keep=1), its score is the in-run eval, which the
common ruler reproduces exactly at step 260.

## 2026-09-07 — es-decode (the `es_token` trainer with held rails): search done, closed out

> "es-token" runs = the `es_token` trainer. Two modes: **es-token-decode** (fresh rank-1 noise every
> token, 0831–0903) and **es-decode** (one noise per rail, held for the whole rollout, 0905–0907,
> `rail_mode=seq`). Both score rails on the clean rollout's KV. Detail and raw numbers:
> [es_profile_results.md](es_profile_results.md) §15–16.

**Setting (all arms).** R1-Distill-1.5B ← JustRL-1.5B, DAPO-Math, 64 prompts/step, N=32 rails
(16 antithetic pairs), σ=1e-3, 7168 tokens, k1 fitness, es-prefill's update rule, one GPU.
~330 s/step rank-1, ~1340 s full-rank. Ruler: MATH-500 n=2 @ T=0.6 (base 0.751).

**What was swept.** Step size (α 1.25e-3 z-scored; 2.8e-3 raw/adaptive), rails (N 32 → 128),
perturbation shape (rank-1 → full matrix). One real bug fixed on the way: a fused-kernel
out-of-range read that NaN'd the last rail.

**Results.**

| arm | @10 | @20 | @30 | @39 | peak |
|---|---:|---:|---:|---:|---:|
| es-decode rank-1, α 1.25e-3 | 0.773 | **0.786** | 0.785 | 0.772 | +3.5 pp |
| es-decode rank-1, raw α 2.8e-3 | 0.762 | 0.776 | 0.770 | 0.775 | +2.5 pp |
| es-decode rank-1, N=128 | 0.770 | stopped @12 | | | — |
| es-decode full-rank | | 0.779 | | | +2.8 pp |
| es-prefill C (ref) | | 0.803 | | 0.815 | **0.829 @60** |
| BP (ref) | | 0.823 | | 0.843 | **0.859 @120** |

Every arm learns a little, peaks by step 20–30, then turns over at ~5 % cumulative weight
displacement. None gets near es-prefill; none gets near BP.

**Why (measured, not guessed).** The held rail rides the *clean* KV, so its perturbation only
changes the current token — it never sees its own effect on the history. That shows up directly
in the ES signal: the per-rail fitness spread `es/d_std` is **0.5e-3 for every es-decode arm vs
1.5e-3 for es-prefill** at the same σ — a 3× weaker signal. z-scoring then spends the same
displacement budget on it, so ~⅔ of each step is random walk. The 3× is **the same for rank-1
and full-rank, N=32 and N=128, and both step rules** — no knob touches it, because it is set by
*where* the rail is evaluated, not *how* it is perturbed.

> **Caveat (2026-09-07 audit, [zo_opd.md § "2026-09-07 BP es-prefill and es-decode setting audit"](zo_opd.md)):**
> the two trainers share the loss and step rule exactly, but es-prefill perturbs *all* 1.78 B params
> (embed + untied lm_head included; relative step 6.0e-3 vs es-decode's 7.8e-3 at the same α) and
> batches 16 prompts × n=4 vs es-decode's 64 × n=1, and trains at top_p=1 vs 0.95. The first two
> inflate `d_std` independently of the clean-KV effect, so the 3× is an upper bound on it until the
> control run in that page is done.

**es-prefill vs es-token, in one table.**

| | es-prefill | es-token-decode | es-decode |
|---|---|---|---|
| noise | 1 per rail, held; scored by a full prefill | fresh per token; rails on clean KV | 1 per rail, held; rails on clean KV |
| signal per rail (`d_std` @ σ 1e-3) | **1.5e-3** | per-token, not the same scalar | 0.5e-3 (⅓) |
| cost / seq @ N=32 | 5.0 s | 8.7 s (fused) | ~5 s rank-1, ~21 s full |
| headroom before the random walk bites | climbs to ~9–10 % | reverts at 4–5 % | reverts at ~5 % |
| ruler peak | **0.829 @60** | 0.803 @60 | 0.786 @20 |

**Bottom line.** es-prefill strictly dominates both es_token modes: more signal per rail, same or
lower cost, more headroom. The only way to get the full-history signal is to let the perturbation
propagate through the sequence — which *is* es-prefill. And es-prefill itself stays ~3 pp under
BP (the forward-only budget limit, 2026-09-01). Use es-prefill for forward-only OPD, BP when
backward is affordable. The es_token kernels (shared-KV attention, streaming LM head, fused rail
ops, packed-bit GEMV) are correct and fast and stay behind `rail_mode=seq`.

## 2026-09-03 — es-token-decode: fused Qwen2 kernels + the exact top-K loss arm

Two moves to give es-token-decode its best shot before closing it out:

**1. Fused kernels ported to Qwen2** (R1-Distill: no q/k-norm, 12 heads / 1536 hidden — both
non-power-of-2, needing padded+masked Triton ranges; gated bit-exact, max|d| = 0.0). Relaunched
`ds15b_es-token-decode_N32_sig1e-3_lr9e-3`: step 0 = **640 s** (decode 353 + assemble 272) vs
746 s with the separate rail op — **−14 %/step**.

**2. `loss_impl=topk` — the sampled-token estimator's three variance leaks removed at once.**
Per token the rails now score the truncated cross-entropy over the clean rail's top-16 ids
(`ℓ_{n,t} = −Σ_k π_n(k)·log q(k)`, fixed K set across rails → the rail finite-difference
estimates exactly this objective's gradient): no importance weight, no +1 score term, no
single-token sampling noise — the same top-K objective BP-OPD trains (`LOG_PROB_TOP_K=16`).
Teacher `log q` at arbitrary ids comes from a new eager HF teacher (vLLM `prompt_logprobs`
can't do it); gates: HF-vs-vLLM max|d| 0.12 (bf16 kernel noise), in-run K-gather consistency
7.6e-6, teacher cost ~1 s/step. Launched at lr 3e-3, recalibrated to lr 2.2e-2 by step-0 footprint.

**First result (2026-09-04): the truncated CE was the wrong objective — and the rails proved
it by optimizing it.** At footprint 5.5e-3/step the CE fell 36 % in 40 steps (0.822 → 0.530,
far beyond random walk — the exact estimator moves coherently) while greedy MATH-500 fell
68.0 → 60.0 → 54.2 and responses shortened 15 %. Diagnosis: `−Σ_k π(k)·log q(k)` is **linear
in π**, so its optimum is a delta on the teacher argmax — the arm was collapsing entropy, not
distilling. Fixed by adding the `π·log π` term (the payload already holds `log π` at the K
ids): the loss is now the truncated **reverse KL** `Σ_k π(k)(log π(k) − log q(k))`. Relaunched
as `..._topk16rkl`. Meanwhile the fused sampled-token arm (lr 9e-3, footprint 5.7e-3) runs
68.0 → 71.6 @20 → **75.2 @40** on the same ruler — the strongest es-token-decode curve so far.
Two learnings either way: the per-token exact estimator has enough signal-to-noise to *steer*
(the CE run moved fast and monotonically — in the objective's own direction), and objective
curvature (KL vs linear CE) matters more than estimator variance at this footprint.

**Close-out through @80 (2026-09-04 evening): the exact objective did not change the shape.**
Greedy MATH-500 curves (base 68.0/68.6 on this ruler):

| arm | loss | @20 | @40 | @60 | @80 | @100 | @120 | KL trend |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sampled fused (lr 9e-3, fp 5.7e-3) | sampled-token IW | 71.6 | **75.2** | 68.6 | 68.6 | 66.6 | 63.8 | 0.266 → 0.171 ✓ |
| topk CE (lr 2.2e-2) | truncated CE | 60.0 | 54.2 | killed | | | | its obj −36 % ✓ |
| topk **rkl** (lr 2.2e-2, fp 4.5e-3) | exact truncated reverse KL | **73.4** | 72.4 | 67.8 | 67.6 | 64.0 | 64.4 | 0.260 → 0.115 (−56 %) ✓ |

Every arm optimizes its own objective monotonically — and every accuracy curve does the same
one-jump-then-revert by ~4–5 % cumulative displacement. Removing all three variance leaks
(rkl: no IW, no +1 term, no single-token sampling noise) bought a faster KL slope and a
*non-declining* revert (67.6–67.8 vs the sampled arm's 63.8-and-falling) but no durable
accuracy gain. **Verdict: the estimator was never the binding constraint — the KL landscape's
accuracy transfer is.** "es-token-decode learns" is true on its own objective (reverse KL to
the teacher falls 43 % and keeps falling) and false on MATH-500 at any loss quality tried.
This matches the §16-style budget law: the coherent component steers the objective, the random
walk spends the displacement budget, and accuracy — which needs more than teacher-agreement —
reverts. The fused sampled run completed (final greedy 62.2 @140, 62.0 @149 — monotone
decline from the 75.2 peak); rkl followed it down (67.6 @80 → 64.0 @100, KL still falling,
−52 %). **The planned standard-ruler evals on the peak checkpoints cannot run**: the es_token
trainer's `_save_hf_checkpoint` prunes to the last 2 saves (`keep_last=2`), so @40/@20 were
deleted long before the peaks were identifiable — only declined-state checkpoints survive
(fused 140/149, rkl 80/100). The standing standard-ruler datum for es-token-decode remains the
earlier sampled arm's @60 = 0.803. Gotcha filed: copy peak checkpoints out mid-run (or raise
`keep_last`) whenever a peak-then-revert curve is possible.

**Both runs complete (2026-09-05 02:30).** Final curves: fused sampled 68.0 → **75.2 @40** →
62.0 @149; topk rkl 68.6 → **73.4 @20** → 56.6 @140 → 53.2 @149 (KL monotone to 0.115, −56 %).
The exact-objective arm ended *lower* than the sampled arm — at matched cumulative displacement
the decline is the displacement's doing, not the estimator's, and no loss quality changes it.
es-token-decode is closed on this setting: the machinery demonstrably optimizes any per-token
teacher objective it is given (three objectives, three monotone loss curves) and none of it
survives the random-walk damage on the accuracy landscape beyond a ~20–40-step transient.

**Head-to-head vs es-prefill (the closing comparison).** Same information budget (N scalars/step
— per-token fresh noise multiplies targets, not probes), but: cost ~1.7× per sequence even with
the fused kernels (8.7 vs 5.0 s/seq at N=32) and ~2.2× with the 0831 kernels; weaker signal per
scalar (detached-history direct term vs the prefill rail's full-trajectory gradient); and 3–4×
less displacement tolerance — the es-token arms reverted at ~4–5 % cumulative displacement where
es-prefill F was still climbing, holding its 0.82 plateau to ~9–10 %. Standard-ruler standing:
base 0.751 → es-token-decode 0.803 @60 → es-prefill 0.815–0.829 @60 → BP 0.846 @60, with BP also
~4× cheaper per sequence than es-prefill. es-prefill strictly dominates es-token-decode on every
measured axis; the per-token thread's residual value is the kernel work (shared-KV attention,
streaming LM head, fused rail ops).

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
| es-token-decode | 0.803 | step 60 | greedy peaks then reverts; dominated by es-prefill on cost too (2026-09-03) |
| es-decode rank-1 / full-rank | 0.786 / 0.779 | step 20 | turns over by ~39; rail signal 3× weaker than es-prefill's (2026-09-07) |

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
`rail_seq_kernels.py`; arms run 2026-09-05→07 on GPU 7, stopped; wandb `ds15b_es-decode_*`):
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

**Verdict (2026-09-07, es_profile_results.md §16): es-decode does NOT match es-prefill or BP, and is
strictly dominated.** Standard ruler MATH-500 (base 0.751): es-decode peaks **0.786** (r1, +3.5 pp)
vs es-prefill C **0.829** (+7.8) vs BP **0.859** (+9.5):

| arm (std ruler MATH-500) | @10 | @20 | @30 | @39 | peak |
| --- | --- | --- | --- | --- | --- |
| es-decode r1 zscore α1.25e-3 | 0.773 | **0.786** | 0.785 | 0.772 | +3.5 pp |
| es-decode r1 raw α2.8e-3 | 0.762 | 0.776 | 0.770 | 0.775 | +2.5 pp |
| es-decode r1 N=128 α1.25e-3 | 0.770 | stopped @12 | | | — |
| es-decode full-rank | | 0.779 | | | +2.8 pp |
| es-prefill C / BP | | 0.803 / 0.823 | | 0.815 / 0.843 | 0.829 @60 / 0.859 @120 |

Mechanism, measured in-run: the held rail
attends the CLEAN KV, so its per-rail k1 fitness spread `es/d_std` is **~0.5e-3 vs es-prefill's
~1.5e-3 (3×) at the same σ** — the detached-history rail carries ⅓ the coherent gradient; z-scoring
spends the same displacement budget for it, so ⅔ is random walk. The 3× gap is **independent of
perturbation rank (full = rank-1), N (32 = 128), α, and step normalisation** — signal-per-rail is set
by riding the clean KV, not by any knob. To keep the full-history signal the perturbation must
propagate → that IS es-prefill. Use es-prefill (forward-only) or BP; es-decode has no niche. Kernels
stay available behind `rail_mode=seq`.

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
