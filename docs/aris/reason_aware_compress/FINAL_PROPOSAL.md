# Final Proposal — TRACER: Transition-Reasoning-Aware Compression with Error-budgeted Rank allocation

**Supersedes** the earlier SRC-centric proposal. Re-architected after the literature survey (`LITERATURE.md`) revealed that "loss-aware structured SVD" (OBD-LLM, 2604.00821) and "sequential error-corrected SVD" (SAES-SVD, 2602.03051) are prior art. The defensible, less-contested wedge is the **mechanistic reasoning-transition / steering-subspace** angle.

## Title (working)
**Protect the Transition: Steering-Subspace Preservation for One-Shot Structured Compression of Reasoning LLMs**

## Abstract thesis (2 sentences)
Structured compression collapses reasoning not because it poorly matches *average* activations, but because it erases a small set of **low-variance, high-leverage transition directions** that control planning, backtracking, and the reasoning→answer handoff. We present a **one-shot** structured compressor that conditions its loss geometry on transition tokens, explicitly preserves mechanistic steering subspaces, and allocates heterogeneous rank by predicted final-loss increase — substantially improving math and long-context robustness at a fixed parameter budget.

## Problem anchor (frozen)
One-shot structured compression of Qwen3-4B→1.7B (0.36 retain): SVD-LLM-V2 on attention + Nystrom/MoDeGPT on MLP. Collapses MATH-500 to 0% (PPL 10³–10⁴) even with in-domain math calibration; iso-param SparseGPT (full-rank unstructured) keeps 45%. Single-module compression is ~free → failure is **distributed across depth**.

## Central thesis
Reasoning collapse under structured low-rank compression is caused by **erasing a transition-critical steering subspace** — directions that are *low-variance* (so variance-weighted truncation deletes them first) but *high final-loss leverage* (they drive inter-step reasoning transitions). Preserving that subspace, and allocating structured rank by **transition-conditioned loss sensitivity**, recovers reasoning one-shot and is not covered by OBD-LLM / SAES-SVD / PGSVD.

## Method — TRACER (3 components)

Let module *m* at layer ℓ have dense output `y = Wx`. Collect from the **dense** model on reasoning traces; define a transition-token set `T` = sentence-boundary tokens + tokens with high reasoning-steering score (SEAL transition class / Thought-Anchor planning-backtracking sentences).

**C1 — Transition-Conditioned Loss Geometry.** (user direction 1)
`X_{ℓ,m} = E_{t∈T}[x_t x_tᵀ]`, `G_{ℓ,m} = E_{t∈T}[g_t g_tᵀ]` with `g_t = ∂ℓ_t/∂y_t`. Compress by minimizing the second-order proxy `E(Ŵ) = ‖G̃^{½}(W−Ŵ)X^{½}‖_F²`.
*vs OBD-LLM (2604.00821)*: they use **global, average-token** K-FAC whitening; TRACER conditions the geometry on **transition tokens** — aligned to where reasoning actually breaks.

**C2 — Steering-Subspace Preservation.** (user direction 3)
Extract module-level reasoning directions `U_{ℓ,m}=[ũ_{ℓ,m,c}]` (difference-of-means, per "When Reasoning Meets Compression") with attribution-patching importances `Γ_{ℓ,m}=diag(I_{ℓ,m,c})`, behaviors c ∈ {planning/backtracking, uncertainty/reflection, example-testing, knowledge-addition}. Augment the output metric: `G̃_{ℓ,m} = G_{ℓ,m} + β U_{ℓ,m} Γ_{ℓ,m} U_{ℓ,m}ᵀ`. Solve weighted rank-r SVD on `A_{ℓ,m}=G̃^{½} W X^{½}` (attention); for MLP, run Nyström/MoDeGPT leverage-score sampling in this whitened space, map back to W. *Optional exactness*: deflate — keep `P_U W` at full rank, low-rank only `(I−P_U)W` (rank budget += dim U).
*vs When-Reasoning-Meets-Compression (2504.02010)*: they protect important **weights for quantization**; TRACER protects a **steering subspace inside structured low-rank/Nyström**.

**C3 — Loss-Budgeted Heterogeneous Rank Allocation.** (user direction 2)
Per-module structured loss curve `δ_{ℓ,m}(r) = Σ_{i>r} σ_i(A_{ℓ,m})²`. Global budget: `min_{r} Σ δ_{ℓ,m}(r_{ℓ,m}) s.t. Σ cost(r) ≤ B` — greedily keep singular directions with largest marginal loss-reduction-per-param; yields both heterogeneous rank **and** module compress/skip decisions.
*vs PGSVD (2510.05544)*: they equalize **activation-weighted reconstruction** (scalar grad-norms, dense-referenced); TRACER equalizes **predicted final-loss increase on transition tokens** with steering protection.

## Demoted / optional (collision-driven)
- **B (sequential re-linearization)** → **ablation only** (core idea = SAES-SVD prior art). Not in title/abstract.
- **D (bi-whitened SVD)** → the C1 **baseline to beat** (= OBD-LLM). Not a standalone claim.
- **A (low-rank + tiny OBS sparse residual)** → optional cheap recovery **"TRACER+Patch"** (= LoSparse/OATS family), applied only to modules with steepest remaining `δ(r)`. Add-on, not core.

## One-shot vs recovery
**Core = fully one-shot** (TRACER). TRACER+Patch is a cheap one-pass add-on. Existing OPD recovery (+2pp/138 steps on the SparseGPT student) confirms retraining is **not** the headline.

## Contribution ranking
1. **Headline**: one-shot structured compression fails by destroying a transition-critical steering subspace; protecting it + transition-conditioned loss geometry + loss-budgeted hetero-rank preserves reasoning at fixed budget.
2. Global average reconstruction / global loss-aware whitening (OBD-LLM) is **insufficient** for reasoning — fails specifically on transition tokens.
3. The failure cliff localizes to **transition tokens/sentences**, not just generic AR compounding (engage 2505.24187). (user direction 4)
4. **Long-context stress magnifies** the transition failure (M5); prior structured work only tested short context. (user direction 5)
5. A tiny sparse patch helps but is **not required** for the main effect.

## Top-3 reviewer risks & defenses
1. *"Just OBD-LLM + a steering regularizer."* → Object of study is **transition-conditioned reasoning robustness**; show global loss-aware baselines still fail on transition tokens while TRACER shifts the cliff and preserves steering energy (SER).
2. *"Just SAES-SVD renamed."* → Don't lead with M3/sequential; cumulative-error handling is prior art and orthogonal; novelty is the **mechanistic target** (transition subspace driving budget allocation).
3. *"Steering vectors are brittle / need supervision."* → Small fixed on-policy dense-trace calibration, no retraining; robustness plots vs #steering-vectors, behavior families, calibration size; show the protected subspace **transfers** across MATH/AIME/Olympiad. Also separate genuine mid-layer transition directions from the shallow last-layer "To/Step" token-prior (Sinii 2509.06608).

## Immediate next action
Run **Block 0** (reproduce collapse + steering-energy-retention probe `cos(W_comp ũ, W_dense ũ)` per layer, single-layer vs all-layer) — cheapest decisive test of the steering-subspace thesis. See `EXPERIMENT_PLAN.md`.
