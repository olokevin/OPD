# Novelty Check — MoE Expert Compression × Recovery Trajectory

> 2-agent multi-source search + GPT-5.4 cross-verification (2026-06-08). **Verdict: 5.5/10, PROCEED-WITH-CAUTION.** The 3-pillar program survives only if reframed: **Claim 1 (cross-family crossover + family>criterion) is the paper; Claim 2 (predictive diagnostic) is demoted to a supporting mechanism; effective-rank dropped from the headline.**

## Proposed work
Controlled study compressing ONLY OLMoE-1B-7B experts at retain 0.75, across all 4 families (whole-expert prune / merge / intra-expert low-rank / unstructured-SparseGPT), measuring the full short-recovery trajectory (8 eval points) during SFT on the original data.

## Claim-by-claim

### Claim 1 — Cross-family crossover + family>criterion variance — Novelty: MEDIUM
- **Closest prior:** **SlimQwen** (2605.08738) owns "one-shot MoE rankings don't survive recovery" (structured-only, 400B/80B, co-varies depth+width, no curve, no crossover). **A Free Lunch** (2510.14444) owns recovery-curve + *within-family* criterion crossover (dense, unstructured-only, implicit).
- **What's unoccupied** (verified across 2 searches, ~19 papers): cross-family scope incl. **low-rank + unstructured** (SlimQwen omits both), **explicit family-vs-criterion variance decomposition** (none found), experts-isolated (attn+router frozen), small no-shared-expert MoE, **short-horizon** trajectory with the curve itself.
- **The bare "methods converge" headline is TAKEN.** Do not lead with it.
- **The single sharpest result that makes it undeniably novel (per GPT-5.4):** a **systematic cross-family INVERSION** — the step-0 winner *family* is consistently NOT the step-k winner family across tasks/seeds, with large margins, family-effect dominating criterion-effect throughout recovery. If "winner becomes loser" is robust (not anecdotal), it stops looking like a stitched benchmark.
- **Strongest rejection to defend against:** "ablation-heavy benchmark on one small OLMoE at one retain ratio reproducing the known fact that post-training changes rankings; variance decomposition is bookkeeping, not insight."

### Claim 2 — Step-0 property predicts recoverability — Novelty: LOW as flagship, MEDIUM as MoE-specific support
- **The GENERAL idea is OLD:** "recoverability" predicting post-FT accuracy is a named concept in vision (PRACTISE, 2303.00972, feature-distance); "reconstruction error is misaligned with downstream perf" already shown in LLMs (2406.15524, EMNLP'24); gradient-alignment recovery proxy exists in LLMs (TraceNAS, 2602.02891); RankAdaptor (2406.15734) predicts pruned-LLM post-FT accuracy from rank-config.
- **⚠️ DROP effective rank from the headline:** 2602.20433 (2026, across 108 OLMo models) actively shows effective rank does NOT reliably predict LM performance. Keep only as a negative-control row.
- **The ONE open combination:** **inter-expert diversity-retention / routed-token curvature as a step-0 MoE recovery predictor, benchmarked head-to-head vs reconstruction error** — no precedent found. MoE-specific structure-preservation metrics are the defensible angle, but **only as a supporting mechanism for Claim 1, not a standalone paper.**

## Closest prior work table
| Paper | arXiv | Overlap | Key difference (our edge) |
|---|---|---|---|
| SlimQwen | 2605.08738 | "MoE one-shot rankings converge after training" | structured-only; we add low-rank+unstructured, experts-isolated, short-horizon curve, crossover+variance analysis |
| A Free Lunch | 2510.14444 | recovery curve + within-family criterion crossover | dense, unstructured-only; we do MoE + cross-FAMILY crossover + variance decomp |
| REAP | 2510.13999 | MoE experts-only prune-vs-merge, "pruning prevails" | strictly one-shot (no FT); their ranking is exactly what we test for stability |
| Is Retraining-Free Enough | 2603.02217 | MoE recovery | recovers ROUTER only, bars not curves |
| PRACTISE | 2303.00972 | "recoverability" predicts post-FT acc | vision, feature-distance predictor, not MoE/curvature/diversity |
| recon-error pitfall | 2406.15524 | recon error ≠ downstream perf (LLM) | establishes half of Claim 2; no predictor proposed |
| TraceNAS | 2602.02891 | gradient-alignment recovery proxy (LLM) | one-shot selector, not head-to-head vs recon error |
| effective-rank-no-predict | 2602.20433 | effective rank ⊀ LM perf | **contraindicates effective rank** — drop from headline |
| RankAdaptor | 2406.15734 | predicts pruned-LLM post-FT acc | from rank-config, not intrinsic step-0 property |

## Overall assessment
- **Novelty score: 5.5/10.** **Recommendation: PROCEED-WITH-CAUTION.**
- **Restructure:** Claim 1 = the paper (narrowed to *controlled post-recovery selection-bias in experts-only MoE compression*); Claim 2 = supporting mechanism; effective rank = negative control only.
- **Re-run this search near submission** — 5+ MoE-compression papers landed Feb–Jun 2026; a concurrent SlimQwen/REAP extension to low-rank+unstructured-with-trajectories is plausible.

## Suggested positioning (GPT-5.4's one-liner, adopt verbatim)
> *"In a controlled experts-only compression study of small MoE models, short recovery reorganizes method quality primarily at the compression-FAMILY level rather than the step-0 score level, making one-shot rankings unreliable and favoring MoE-specific recoverability diagnostics over naive reconstruction metrics."*

The undeniable-novelty target: a **robust cross-family inversion** (step-0 winner family loses by step-k), not just "rankings change."
