# es-prefill paper framing: loss justification, learning signal, OPSD extension

> Query-answer filed 2026-09-05: how to present es-prefill in a paper — SFT-vs-ES-loss
> justification, the mechanistic learning-signal story, and the OPSD (teacher-free) extension.
> Numbers cite [zo_opd_short.md](zo_opd_short.md) and [es_rails_formulation.md](es_rails_formulation.md).

## 1. The loss vs SFT

Fitness = frozen-trajectory surrogate J(W') = (1/T) Σ_t A_t log π_{W'}(y_t | y_<t),
y ~ π_W frozen, A_t = log q(y_t) − log π_W(y_t) frozen.

| axis | SFT | es-prefill / OPD |
|---|---|---|
| states | teacher-visited (y* ~ q) | **student-visited** (y ~ π) — no exposure bias |
| KL direction | forward KL(q‖π), mode-covering | **reverse KL(π‖q)**, mode-seeking: E[∇J] = −∇KL(π‖q) per visited state |
| credit | uniform +1 per token | **signed per-token advantage** A_t (= BP-OPD's token_reward_direct) |

ES-specific: the loss is a scalar functional evaluable by ONE teacher-forced prefill on a
frozen trajectory (SFT shares this; an RL return does not) — that is what makes
one-rail-one-prefill possible, and why y and A_t must be frozen (unfrozen fitness was
length-hacked in both directions; zo_opd_short "problems found" §2).

## 2. The learning signal (all measured)

1. Rail = directional derivative: d_i = (F₊−F₋)/2 = σ⟨g, ε_i⟩ + O(σ³). The update is the true
   BP-OPD gradient projected onto the N-dim rail subspace — same gradient, subsampled.
2. Works at D=1.5B because the landscape is low-rank: r_eff = tr(H)/κ_g ≈ 160 ≪ D; per-step
   progress ratio ≈ 1/(1 + r_eff/N). (Also why per-token noise bought nothing: information is
   N scalars regardless of B·T.)
3. Learning ends at the displacement budget, not signal exhaustion: random-walk displacement
   ~footprint·√S; tolerance ≈ 9–10 % RMS(W); turnover at S* ≈ (budget/footprint)² — the
   climb/plateau/soft-turnover curve shape is predicted.
4. Result: 0.751 → 0.815–0.829 MATH-500 (≈ 2/3–3/4 of BP's +9.5 pp) with seeds+scalars-only
   communication (distributed economics: es_rails_formulation.md §9).

## 3. OPSD extension (no external teacher)

> **[2026-09-07] This section's prediction was right, and the setting now exists as a paper.**
> *Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs*
> ([arXiv:2601.18734](https://arxiv.org/abs/2601.18734), `docs/papers/26_Self-Distilled Reasoner-*.pdf`)
> is exactly instantiation 1 below — privileged-context self-teacher — and it independently
> adopts caveat 3's fix (`--fixed_teacher`: the teacher is the **frozen initial policy**, kept as
> the LoRA-disabled base model). Built and running in this repo:
> [../OPSD/opsd_bp_vs_es.md](../OPSD/opsd_bp_vs_es.md).
> Two of this section's caveats already have measurements there: **σ had to be recalibrated**
> (caveat 1) but for a reason not anticipated here — with LoRA the binding constraint is the bf16
> forward's own noise floor setting σ from *below*, not SNR decay as q→π — and the **geometry**
> caveat (2) is satisfied by construction, since the paper's objective is forward KL with
> pointwise clipping rather than anything linear in π.

Applicable: the fitness only needs log q(y_t) teacher-forced evaluable. Instantiations:
- **Privileged-context self-teacher** (best fit): q = same weights conditioned on the
  ground-truth answer/hint — one extra prefill of the same model; one model per worker;
  seeds-only distributed story intact.
- **Frozen/EMA snapshot teacher**: evaluable, keeps an external anchor.
- **Reward-only** = es-rl, already measured (+5.7 pp @150 iters, teacher-free) → present as a
  spectrum: external teacher → privileged self-teacher → reward-only.

Caveats (each checkable with existing harness):
1. SNR: A_t shrinks as q→π; recalibrate σ, watch d_snr; run opd_curvature.py with q=self-teacher
   to measure the self-distillation r_eff before committing GPUs.
2. Geometry: keep the reverse-KL form — the topk-CE entropy collapse is the cautionary tale for
   linear-in-π self-referential objectives.
3. Anchor: a same-weights self-teacher is damaged by the same random walk as the student — the
   displacement-budget argument worsens without an external anchor; EMA/frozen teacher restores
   it (cheap, falsifiable ablation to propose).
