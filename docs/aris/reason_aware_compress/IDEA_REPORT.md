# Idea Discovery Report — Reasoning-Aware Structured Pruning

> **⚠ v2 UPDATE (2026-06-02, post-literature-survey).** The original recommendation below was **SRC** (sequential re-linearized compression). A 3-agent literature survey (`LITERATURE.md`) then found prior-art collisions: **SAES-SVD (2602.03051)** already does sequential cumulative-error SVD (kills SRC as headline), **OBD-LLM (2604.00821)** already does loss-aware bi-whitened SVD (kills idea D), **PGSVD (2510.05544)** already does equalize-per-layer-error hetero-rank. The contribution was re-architected via a GPT-5.4 refinement pass into **TRACER** — a *transition-critical steering-subspace preservation* method. **The current recommendation lives in `FINAL_PROPOSAL.md` (TRACER); `EXPERIMENT_PLAN.md` and `IDEA_CANDIDATES.md` are updated to match.** The diagnosis and idea-generation body below is preserved as the audit trail; SRC/A/D are demoted to ablations/baselines.

**Direction**: Extend reasoning-aware calibration (RAC) from *unstructured* pruning (SparseGPT) to *structured* compression (SVD-LLM-V2 on attn + Nystrom/MoDeGPT on MLP). Diagnose why structured compression collapses reasoning at the 4B→1.7B (0.36 retain) one-shot budget, and propose methods to fix it.
**Date**: 2026-06-02
**Pipeline run (v1)**: code-grounded diagnosis → GPT-5.4 idea generation (10 ideas) → GPT-5.4 adversarial review → code-verified feasibility. **Pipeline run (v2)**: 3-agent literature survey (structured-pruning / reasoning-mechanism / efficient-reasoning+long-context) → GPT-5.4 collision-resolving refinement → TRACER. All four original anchor papers read in full; survey added ~40 verified arXiv references (`LITERATURE.md`).

## v2 Recommendation (TRACER) — see FINAL_PROPOSAL.md

Reasoning collapse under structured low-rank is caused by **erasing a transition-critical steering subspace** — directions that are *low-variance* (variance-weighted truncation deletes them first) but *high final-loss leverage* (they drive inter-step reasoning transitions: backtracking, verification, answer-commit). **TRACER** = (C1) transition-token-conditioned loss geometry + (C2) steering-subspace preservation + (C3) loss-budgeted heterogeneous rank. One-shot. Differentiated from OBD-LLM (avg-token, no reasoning), SAES-SVD (recon-only, unstructured), PGSVD (scalar-grad, dense-ref), and "When Reasoning Meets Compression" (protects weights for *quantization*, not a subspace inside *structured low-rank*). Killer diagnostics: phase-transition localization (TEL/FID/cliff-r*) and long-context-degrades-more (RULER/SER-vs-length).

---

## (v1 audit trail below — superseded by TRACER)

## Executive Summary

The collapse is **not** a calibration-domain problem (the user already uses in-domain math traces, which rescues SparseGPT but not SVD/Nystrom) and **not** a single-fragile-layer problem (per-layer sweep: any one module to retain-0.5 is ~free). It is **cross-depth error compounding under fixed-rank, variance-weighted, layer-independent reconstruction**. The codebase compresses every layer against the **dense** model's activations and never lets a layer see its compressed upstream (verified in `src/compress/compress_model.py:544-572`) — the textbook condition for uncorrected accumulation.

**Recommended central contribution**: **Sequential Re-Linearized Structured Compression (SRC)** — compress in depth order, recomputing each layer's input *and* OPD-gradient covariance through the **already-compressed** prefix before compressing it. It is the only proposal that attacks the *verified* failure mechanism, stays one-shot (no SGD), and reuses existing tools. Strengthen with a **low-rank + tiny sparse residual (LR+OBS)** to close the residual full-rank gap that SparseGPT exploits.

## Diagnosis (full version in `DIAGNOSIS.md`)

Five mechanisms, ranked by data support:
- **M1 rank-deficiency** (strongest): low-rank truncation repeatedly projects the residual stream onto a subspace; off-subspace contributions are lost and compound multiplicatively over 36 layers. SparseGPT stays full-rank → no rank floor problem.
- **M2 wrong-objective**: input-**variance**-weighted reconstruction (`‖(W−Ŵ)X‖`) sacrifices low-variance high-leverage reasoning directions. Fix = output-gradient / OPD-loss weighting.
- **M3 uncorrected-accumulation** (**code-verified**): all covariances collected once on the dense model; no layer sees the compressed upstream. → SRC.
- **M4 autoregressive-amplification**: small per-token logit error compounds over a 2k-token CoT → "loops, never boxes". Explains why C4 PPL hides the collapse.
- **M5 numerical**: ruled out (results doc: "clean, no NaNs").

## Ranked Ideas

### 🏆 Idea B — Sequential Re-Linearized Structured Compression (SRC) — RECOMMENDED CORE
- **Attacks**: M3 (primary), M2, M4. **Reviewer score: 8/10.**
- **Method**: Compress layers `ℓ=0..35` in order. Before compressing layer ℓ, run the calibration batch through the **already-compressed** layers `0..ℓ−1` (dense thereafter) and recollect `XᵀX_ℓ` (for SVD-V2 whitening / Nystrom C_σ) and `C_dy,ℓ` (OPD backward cov). Layer ℓ thus reconstructs against the activation distribution it will *actually* receive, absorbing upstream error. Still one-shot, no gradient descent.
- **Novelty**: Sequential recomputation is standard for *quantization/sparsity* (GPTQ/SparseGPT do it implicitly within a layer). Applying re-collected forward **and OPD backward** covariances to **structured low-rank** compression across depth, motivated by a verified compounding diagnosis, is the new delta. Differs from MoDeGPT (independent modular decomp) and SVD-LLM-V2 (per-matrix, dense-referenced).
- **Novelty status**: CONFIRMED-likely-novel for the structured + reasoning + OPD-cov combination (no lit search run; flag for `/novelty-check` before write-up).
- **Pilot**: <2 GPU-hr (see plan, Block 1).
- **Risk**: greedy — if early layers irreversibly destroy features, later relinearization adapts to damaged representations rather than recovering them. (Mitigated by combining with LR+OBS.)

### 🥈 Idea A — Low-Rank + OBS Sparse Residual (LR+OBS) — RECOMMENDED COMBINER
- **Attacks**: M1. **Reviewer score: 6/10** (alone); higher as combiner.
- **Method**: `Ŵ = UV + S`. UV = structured low-rank at ~0.30 budget; `S` = SparseGPT/OBS-pruned residual of `R = W − UV` at ~0.06 budget, allocated by OPD-weighted residual energy. Restores a few exact full-rank "escape edges" pure low-rank kills.
- **Novelty**: low-rank+sparse decomposition exists; real delta is OPD-guided residual budgeting **and** fitting the residual against the *compressed-upstream* activations (synergy with B).
- **Pilot**: Block 2.
- **Risk**: reviewers say the sparse tail (not the structured part) does the work → dilutes the "we fixed *structured* compression" claim. **Keep S small and ablate it.**

### Idea D — OPD-Weighted Bi-Whitened SVD — BACKUP (objective upgrade)
- **Attacks**: M2. **Score: 6/10.** Bilateral `tr[C_dy^{½}(W−Ŵ)XᵀX(W−Ŵ)ᵀC_dy^{½}]` via truncated SVD of `C_dy^{½} W (XᵀX)^{½}`, `C_dy` from `calibration_opd_loss.py`. ~All machinery exists (`svd_compress_layer_backward`, `collect_both_covariances_from_loader`). Risk: "just the right weighting for SVD" — still pure low-rank, may not beat the family gap. **Use as an ablation axis inside B, not standalone.**

### Idea C — QK-Coupled / OV-Coupled Logit Preservation — BACKUP (attn-only)
- **Attacks**: M2/M4 (attention only). **Score: 5/10.** Joint bilinear `‖X W_qW_kᵀXᵀ − X Ŵ_qŴ_kᵀXᵀ‖`. Risk: optimizes the wrong bottleneck — MLP/residual damage likely dominates; OV preservation ignores softmax/head-mixing. **Optional attention-side ablation.**

### Eliminated / demoted
- "Spare the important layers" — **ruled out by the per-layer sweep** (loss is distributed).
- "Better calibration domain alone" — already done; necessary-not-sufficient (RAC).
- Ideas 5–10 from generation (headwise rank reservation, Fisher knapsack, anchor subspace, closed-form LoRA fit, cross-layer basis transport, rollout-stability loss): valid but second-order vs B+A; **anchor-subspace** and **closed-form LoRA error fit** are the best of these and are retained as Block-4 extensions.

## Recommended program

- **Central contribution (one-shot only)**: **SRC (B)**, with the OPD-bi-whitened objective (D) as one of its internal knobs.
- **Strongest combination (one-shot + cheap correction)**: **B + A**. B fixes compounding; A restores the full-rank tail. The synergy is real because A's residual is fit against B's compressed-upstream activations, not the dense model.
- **Escape hatch already known**: a cheap reasoning-targeted recovery (LoRA/OPD) — but OPD on the SparseGPT student gave only +2pp in 138 steps, so recovery is **not** the headline; the one-shot story must carry the paper.

## Next Steps
- [ ] Run Block 1 pilot (SRC, attention-only first) — go/no-go on the compounding hypothesis.
- [ ] `/novelty-check` on SRC before write-up (lit search was skipped).
- [ ] If Block 1 is positive → Block 2 (LR+OBS) → full B+A → `/experiment-plan` is already drafted in `EXPERIMENT_PLAN.md`.
- [ ] Proposal: `FINAL_PROPOSAL.md`. Plan: `EXPERIMENT_PLAN.md`.
