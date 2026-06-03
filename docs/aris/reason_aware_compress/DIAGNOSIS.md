# Why Structured Pruning Collapses Reasoning — Diagnosis

**Date**: 2026-06-02
**Scope**: Qwen3-4B (non-thinking base) → ~1.7B effective, one-shot, no recovery FT.
Target structured methods: **SVD-LLM-V2** (self_attn) + **Nystrom/MoDeGPT** (MLP).
Anchor result: `docs/results/compressed_opd.md`.

## The headline facts that must be explained

| Method (1.7B budget) | Calib | C4 PPL | MATH-500 |
|---|---|---:|---:|
| SparseGPT 64% unstruct | OpenThought3 (math) | 82.0 | **45.0%** |
| SparseGPT 64% unstruct | C4 | 34.5 | **0.0%** |
| SVD_V2 (all) | math | 33,464 | **0.0%** |
| SVD_V2 attn + Nystrom MLP | math | 4,980 | **0.0%** |
| **Per-layer, ONE module at a time, retain 0.5** | math | — | **75–89%** (≈ baseline) |

Two facts in tension that pin the diagnosis:

1. **Single-module structured compression is nearly free.** Compressing *any one* attn block or MLP triplet to half params costs ~0 MATH (144/144 cells in 75–89%). No fragile layer, no depth trend.
2. **Whole-model structured compression at 0.36 retain is catastrophic** (PPL 10³–10⁴, 0% MATH) — under *both* C4 and in-domain math calibration. In-domain calibration rescues SparseGPT (0→45%) but **does NOT rescue** SVD/Nystrom (still 0%).

→ The failure is **not** a calibration-domain problem (RAC's thesis) and **not** a single-bottleneck-layer problem. It is **error accumulation across 36 layers of a rank-deficient (low-rank) approximation**, where the per-layer error is benign in isolation but compounds super-linearly once *every* layer is simultaneously rank-reduced.

## Five candidate mechanisms (ranked by how much the data supports them)

### M1 — Low-rank ≠ full-rank: rank deficiency destroys the residual stream geometry (STRONGEST)
SparseGPT keeps each layer **full-rank** (it zeros entries; the weight matrix still spans the full output space). SVD/Nystrom **truncate rank** to 0.36 of min(d_in,d_out). The per-layer sensitivity sweep shows a *single* truncated layer is fine — the 35 untouched full-rank layers absorb the error. But with *all* layers truncated, the residual stream is repeatedly projected onto a low-rank subspace; the off-subspace component of each layer's contribution is irrecoverably lost and the errors **compound multiplicatively** down the depth. This is why SparseGPT ≫ SVD at equal param budget, and why in-domain calibration (which only reshapes *which* subspace is kept, not the *rank*) rescues SparseGPT but not SVD. **This is the central mechanism.**

### M2 — Reconstruction objective is input-covariance-weighted, not output/loss-weighted (STRONG, RAC-adjacent but deeper)
Both SVD-LLM-V2 (whitening Phi from input cov XᵀX) and Nystrom (neuron scores from input cov C_σ) minimize **input-activation reconstruction** `‖(W−Ŵ)X‖`. RAC's insight is to fix *which X* (decode-time CoT activations). But even with the right X, the objective weights all input directions by their *variance*, not by their *downstream effect on the next-token / reasoning loss*. Reasoning depends on low-variance, high-leverage directions (e.g. the directions that carry "carry the 1", backtracking signals, the `\boxed{}` answer commit). Variance-weighted reconstruction systematically sacrifices exactly these. **The repo already has the tools to test this**: `collect_backward_covariances_from_loader` (output-grad cov C_dy) and `svd_compress_layer_backward` (minimizes `‖(W−Ŵ)ᵀΦ‖` with Φ from C_dy), plus `calibration_opd_loss.py` (a teacher-driven OPD gradient).

### M3 — Error accumulation is uncorrected because compression is layer-independent (STRONG, fixable)
SVD/Nystrom compress each layer in isolation against the **dense** model's activations. SparseGPT's OBS step also does this, *but* its full-rank zeroing leaves enough capacity that small per-layer errors don't snowball. For low-rank, the standard fix is **sequential/online calibration**: recompute layer ℓ's input activations through the *already-compressed* layers 1..ℓ−1, so layer ℓ corrects for upstream error (this is what GPTQ/SparseGPT do implicitly for quant/sparsity, and what SliceGPT/MoDeGPT do partially). The current pipeline collects *all* covariances in one dense forward pass (`collect_covariances_from_loader` runs on the dense model), so **no layer ever sees the compressed upstream** — accumulation is completely uncorrected.

### M4 — Generation-time autoregressive amplification (MODERATE, explains the "loops and never boxes")
MATH is decode-dominated and autoregressive: a small per-token logit error compounds over a 2k-token CoT. The C4-SparseGPT model "loops and never boxes an answer" (per the results doc) — a behavioral collapse, not a perplexity collapse (its C4 PPL is the *best* of the compressed set, 34.5). This is the same phenomenon RAC reports (heavy pruning → longer, worse CoTs). It explains why **PPL hides the collapse** and why a token-level / outcome-level metric is mandatory. For SVD the PPL itself is destroyed (10³–10⁴), so M4 is secondary to M1 there, but M4 is the dominant *symptom channel* for borderline models.

### M5 — Numerical / regularization artifacts (WEAK — already controlled)
Whitening uses eigh with `regularize_eps=1e-4`; Nystrom uses adaptive Cholesky ridge. The results doc explicitly notes "decomposition is clean (no NaNs)". So the collapse is **algorithmic, not numerical**. (Worth a cheap sanity check that eps isn't over-regularizing the small singular values that M2 says matter — but not the main story.)

## What this rules in / rules out for method design

- **Rule OUT**: "find and spare the important layers" — the sensitivity sweep proves loss is *distributed*, not localized. Per-layer rank allocation (SVD-LLM-V2's selling point) can help at the margin but cannot rescue 0.36 retain alone.
- **Rule OUT**: "better calibration domain alone" — in-domain math calib already used; rescues SparseGPT, not SVD.
- **Rule IN**: anything that (a) **raises effective rank per param** (hybrid sparse+low-rank, N:M, joint module decomposition with residual), (b) **changes the reconstruction objective from input-variance to loss/output-gradient weighting** (backward-cov whitening, OPD-gradient-weighted truncation — partially built), (c) **corrects accumulation sequentially** (online/compressed-upstream calibration), or (d) **a tiny recovery step** (LoRA/SFT/OPD) that the results doc already flags as the known escape hatch — the open question is making it *cheap* and *reasoning-targeted*.

## The sharpest research framing

> Structured (low-rank) compression fails on reasoning not because it picks the wrong layers or the wrong calibration text, but because **variance-weighted, per-layer-independent, fixed-rank** reconstruction discards the low-variance high-leverage directions that reasoning rides on, and the loss **compounds across depth** in a way SparseGPT's full-rank zeroing avoids. Fix the *objective* (loss-aware), the *accumulation* (sequential), and the *rank floor* (hybrid), and you should close most of the SparseGPT↔SVD gap **without** retraining — or close all of it with a reasoning-targeted micro-recovery.
