# Literature Review — Reasoning-Aware Structured Compression

**Date**: 2026-06-02. Three parallel survey agents (structured-pruning; reasoning-mechanism/steering; efficient-reasoning/long-context). All arXiv IDs were fetched/verified live; IDs dated past Jan-2026 are flagged ⚠ (real preprints, but verify title/authors before camera-ready). The OPD paper's own ID `2604.13016` is unrelated to this survey.

---

## 1. Prior-art collisions (READ FIRST — these reshaped the contribution)

| Paper | arXiv | What it already does | Collides with | Our remaining wedge |
|---|---|---|---|---|
| **OBD-LLM** ⚠ | 2604.00821 | K-FAC **bidirectional whitening** `W̃ = Lgᵀ W Lx` (input cov ⊗ output-grad cov) → SVD. Second-order loss-aware truncation. | **Idea D** (OPD bi-whitened SVD) — essentially identical | Per-layer independent, **uniform** ratio, **no reasoning/math eval**, no transition-conditioning. We condition the geometry on **transition tokens** + add steering-subspace protection. |
| **SAES-SVD** ⚠ | 2602.03051 | **CEALC**: local recon + weighted **cumulative-error compensation**, closed-form, aligns each layer output to FP counterpart; **evaluates math**. | **Idea B** (SRC / sequential M3 fix) — captures the core idea | Unstructured SVD, **reconstruction-only** (no loss/sensitivity weighting, no reasoning-direction story). M3/sequential is now **prior art → demote B to an ablation**. |
| **PGSVD / Activation-Informed Pareto** | 2510.05544 | Proves uniform per-layer **error tolerance ⇒ heterogeneous rank** (Pareto-optimal); ALS factor refit. *(the "equalize per-layer error" paper the user named)* | Heterogeneous-rank allocation | Activation-Frobenius weighted by **scalar** grad-norms (not directional/Hessian), **dense-referenced** ALS (the M3 failure), no reasoning eval. We equalize **transition-token final-loss increase**. |
| **SlimGPT** | 2412.18110 | Structured + OBS Hessian + **Incremental Pruning Ratio** for accumulation. | Structured + accumulation | Pruning, not low-rank; accumulation handled by a ratio schedule, not activation re-referencing or subspace protection. |
| **RAC** | 2509.12464 | On-policy **CoT calibration** for SparseGPT; `e_t` token-error heatmap: error blows up in the **decode region**. | Calibration-domain (already in our pipeline) | Necessary-not-sufficient for structured (our setting: in-domain calib already used, SVD still 0%). |

**Net effect**: "loss-aware structured SVD" (D) and "sequential error-corrected SVD" (B) are **taken**. The defensible, less-contested space is the **mechanistic reasoning-transition / steering-subspace** angle (the user's directions 1–3).

## 2. Structured pruning / low-rank landscape (selected)

| Method | arXiv | 1-line | Loss-aware? | Hetero-rank? | Sequential? |
|---|---|---|---|---|---|
| SVD-LLM / V2 | 2403.07378 / 2503.12340 | Cholesky/whitened SVD; V2 adds per-matrix ratio + 2-SVD truncation | recon | V2: yes | no |
| ASVD / FWSVD / GFWSVD | 2312.05821 / 2207.00112 / 2505.17974 | activation-scaled / Fisher-weighted / K-FAC-Fisher SVD | FWSVD,GFWSVD: yes | no | no |
| Dobi-SVD / ARA / Bolaco | 2502.02723 / 2510.19389 / 2405.10616 | differentiable / learned-mask / BO rank allocation | partial | yes | no |
| MoDeGPT | 2408.09632 | joint module decomp (Nyström/CR/SVD), reduces hidden dim | output-recon | module | no |
| SliceGPT / LLM-Pruner / ShortGPT / FLAP | 2401.15024 / 2305.11627 / 2403.03853 / AAAI'24 | dim-slice / grad-coupled / layer-drop / fluctuation | varies | varies | no |
| **LoSparse / OATS** | 2306.11222 / 2409.13652 | **low-rank + sparse hybrid** (one-shot, outlier-preserving) | recon / 2nd-moment | — | no |
| AlphaPruning / OWL | 2410.10912 / 2310.05175 | per-layer **sparsity budget** by spectral-shape / outlier-ratio | spectral | yes | no |
| EoRA | 2410.21271 | training-free residual error compensation in eigenspace | eigen | — | post-hoc |
| FISTAPruner | 2408.03728 | convex ℓ1 + **intra-layer cumulative error correction** | recon | — | yes (intra) |
| SparseGPT | ICML'23 | OBS one-shot unstructured; full-rank → avoids M1 | OBS Hessian | — | column-seq |

**Takeaways for us**: (i) loss-aware (OBD-LLM/GFWSVD/FWSVD) and hetero-rank (PGSVD/V2/ARA) and sequential (SAES-SVD/FISTAPruner) each exist individually — *combining them is not enough novelty*; (ii) LoSparse/OATS are the established **M1 (rank-deficiency) remedy** = low-rank + small sparse residual (our Idea A, now demoted to optional "TRACER+Patch"); (iii) MoDeGPT/SVD-LLM evaluated **short-context only** (128×2048, WikiText PPL) — a documented long-context gap (M5).

## 3. Reasoning mechanism + steering vectors (the new core)

- **When Reasoning Meets Compression** (2504.02010, ICLR'26): the blueprint. Difference-of-means steering vector per module/layer/behavior:
  `u^m_{ℓ,c} = mean_{D+} ā^m_{ℓ,c} − mean_{D−} ā^m_ℓ`, behaviors c ∈ {backtracking, uncertainty, example-testing, adding-knowledge}; normalized `ũ`. Weight importance via **attribution patching** `I^m_{ℓ,c} ≈ |Σ (ũ^m_{ℓ,c})ᵀ ∂L/∂a^m_ℓ|`. Findings: **final-layer `mlp.up_proj` critical** (quantizing it alone, 0.7% of weights, −16.3% acc); **protecting ~2% of weights → +6.57%** (up to +23.17%). Code: `psunlpgroup/Compression-Effects`. They built a **protect-list for quantization** — we make the **structured-low-rank analogue** (protect the *subspace*, not full-precision matrices).
- **Reasoning rides on linear directions**: Arditi (2406.11717, refusal = 1 direction, rank-1 edit), Marks & Tegmark (2310.06824, DiM directions are causal, low-variance), RepE (2310.01405), ActAdd (2308.10248).
- **Reasoning-specific**: Venhoff (2506.18167) & Ward (2507.12638) — backtracking is a steerable direction *present in the base model*; SAE reasoning features at ~layer 19 (2503.18878, ReasonScore); **SEAL** (2504.07986) — reasoning = execution/reflection/**transition** thoughts, linearly separable, one steering vector controls them (+11% acc, −12–50% tokens); **Thought Anchors** (2506.19143) — high-leverage sentences are planning/backtracking (transitions); **Reasoning-Focus Heads** (2509.23676) — mid-layer heads track the reasoning→answer transition.
- **Caveat to engage**: Sinii (2509.06608) — part of the *last-layer* reasoning effect is a shallow token-substitution bias ("To"/"Step"); our objective must target genuine **mid-layer** transition directions, not the trivial last-layer token prior.

**Supports the transition hypothesis** (LM fluency survives *within* a step; failure is at inter-step transitions): SEAL, Thought Anchors, RFHs, base-model backtracking direction. **Gap we fill**: nobody has shown "preserve transition steering vectors ⇒ recover reasoning under *structured low-rank*."

## 4. Phase transition & long-context (the diagnostics)

- **Sharp pruning cliff**: 2504.02010 (AIME collapses 40–50% sparsity; quant gradual to ~W3); 2505.11574 ("first vulnerable step" cascades; near-total breakdown at 2-bit); 2504.04823 (errors **accumulate over CoT**; harder tasks drop up to 4×); RAC 2509.12464 (`e_t` blows up in decode region).
- **Counter-evidence (must engage)**: **Beyond Exponential Decay** (2505.24187) — clean-model AR error is **sub-linear** (only ~5–10% "key tokens"), *no single catastrophic transition*. → Our M4 claim must be reframed: **compression raises the per-token error floor until self-correction can't keep up over ~2000 tokens** (converts their sub-linear curve into a cliff). This reframing is itself a contribution.
- **Snowballing**: 2305.13534 (early over-commitment → coherent-but-wrong).
- **Long-context × compression**: **ACBench** (2505.19433) — 4-bit keeps short tasks within 1–3% but **long-context/NIAH degrades far more**, ~32K cliff; **MaCa** (2602.07465) & **PM-KVQ** (2505.18610) — short calibration mis-estimates rare long-sequence channels; **RULER** (2404.06654) is the right instrument (not vanilla NIAH); calibration-content matters (2410.07461, 2410.17170, 2311.09755). **MoDeGPT/SVD-LLM tested short-context only** → clean M5 opening.

## 5. Implications (folded into the refined proposal)

1. Pivot the headline from "loss-aware/sequential SVD" (taken) to **transition-critical steering-subspace preservation** (open).
2. Use a **transition-token-conditioned, loss-oriented** covariance (dirs 1+2) — strictly more targeted than OBD-LLM's average-token K-FAC and PGSVD's scalar-grad Frobenius.
3. Make **phase-transition localization** (dir 4) and **long-context-degrades-more** (dir 5) headline diagnostics, with the metrics in `EXPERIMENT_PLAN.md` (TEL, FID, SER, cliff-point r*).
4. Demote B (sequential, → SAES-SVD) and D (bi-whitened, → OBD-LLM) to **ablations**; keep A (low-rank+sparse, → LoSparse/OATS) as optional **TRACER+Patch**.
5. Cite OBD-LLM, SAES-SVD, PGSVD, RAC, When-Reasoning-Meets-Compression as the precise precedents and beat/contrast each.
