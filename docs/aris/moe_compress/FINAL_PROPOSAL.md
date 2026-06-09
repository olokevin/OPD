# Final Proposal — Does Short Recovery Reorganize MoE Expert-Compression Quality at the Family Level?

> **Problem anchor (frozen):** In a controlled experts-only compression study of the small no-shared-expert MoE OLMoE-1B-7B-Instruct, does short recovery training reorganize compression-method quality at the **FAMILY** level rather than the step-0 score level — making one-shot rankings unreliable — and do MoE-specific step-0 recoverability diagnostics predict post-recovery value better than naive reconstruction error?

**Positioning (verbatim):** *"In a controlled experts-only compression study of small MoE models, short recovery reorganizes method quality primarily at the compression-FAMILY level rather than the step-0 score level, making one-shot rankings unreliable and favoring MoE-specific recoverability diagnostics over naive reconstruction metrics."*

**Verdict:** PROCEED-WITH-CAUTION (novelty 5.5/10; design defensible after the 5 fixes). **READY to plan.**

## Method thesis (one sentence)
Treat "which expert-compression method is best" as a *trajectory* question: hold the model, budget, calibration, and recovery protocol fixed; vary only the compression method across four families; and measure whether the training-free ranking **inverts** during short recovery and at what granularity (family vs criterion).

## Dominant contribution
A **controlled cross-family recovery atlas** for MoE expert compression that establishes (or refutes) a **family-level inversion**: the step-0 best *family* is not the post-short-recovery best family, with the family effect dominating the within-family criterion effect. This is the axis SlimQwen (whole-expert only, 400B tokens, co-varied depth+width) and *A Free Lunch* (dense, unstructured-only, within-family) both leave open.

## Supporting contribution (NOT co-equal)
A **step-0 recoverability diagnostic**: among pre-recovery metrics, MoE-specific structure-preservation signals (routed-token curvature, inter-expert diversity-retention) predict post-recovery value (AURC₀₋₂ₖ) better than naive reconstruction error or step-0 accuracy — validated leave-one-family-out. Mechanistic support for the dominant contribution; not a standalone paper.

## Explicitly rejected complexity
- **No new compressor.** SlimQwen says fine-grained criterion matters little post-training; a new criterion is the weakest bet. (Fisher-weighted reconstruction / heterogeneity-aware hybrid demoted to optional probes.)
- **Effective rank dropped from the headline** — 2602.20433 shows it doesn't predict LM performance; kept only as a negative control.
- **No depth/width pruning, no quantization** — experts-only isolates the variable; everything else frozen.

## What's frozen vs varied
| Frozen | Varied |
|---|---|
| Model (OLMoE-1B-7B-Instruct), attention, **router weights at compression (step 0)** | Compression **method** (6 focal, 3 families × 2) |
| Calibration corpus + total tokens (standardized) | Compression **family** (the headline axis) |
| Recovery data (OpenThoughts3), optimizer, LR schedule | Retain ratio {0.75, 0.50} |
| Recovery protocol (experts+router trainable, attn frozen) | Seed {×3} |

## Key claims (must prove)
- **C1 (dominant):** The training-free method ranking is not rank-stable under short recovery, and the reorganization is primarily **family-level** (between-family variance > within-family criterion variance in the AURC model).
- **C1-sharp (undeniable-novelty target):** A **robust cross-family inversion** — step-0 winner family ≠ 2k-step winner family, sign-flip 95% hierarchical-bootstrap CI excludes 0, holds in ≥75% of task×seed cells, at both retain ratios.
- **C2 (supporting):** A MoE-specific step-0 diagnostic predicts AURC₀₋₂ₖ better than reconstruction error / step-0 accuracy (leave-one-family-out).

## Must-run ablations
- Retain 0.50 (severity robustness) — never claim robustness from one severity.
- **Router-frozen ablation** (the user's original protocol) — quantifies the router-repair channel per family; ties to *Is Retraining-Free Enough?* (2603.02217).
- Native-calibration sensitivity (appendix) — show family inversion survives the standardized→native calibration swap.

## Remaining risks
- **Identifiability** (the reviewer's #1 concern): mitigated by 2 methods/family + dual budget axes + trainable router. If a family collapses to one usable method (e.g. only one merge variant ports cleanly), the family claim degrades to method-level — flagged as a gate.
- **Budget non-commensurability**: an inversion at "0.75 storage" could be an active-capacity artifact → report both budget axes; an inversion that holds under *both* is the strong result.
- **Null result**: most likely failure = rankings stay stable. This is a *well-powered negative* against the stitched-benchmark thesis (still publishable); C2 may survive even then (predicts recovery *magnitude*).

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the claim-driven roadmap and [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) / [NOVELTY_CHECK.md](NOVELTY_CHECK.md) for the provenance of every locked decision.
