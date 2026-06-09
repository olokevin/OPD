# Research Idea Report — MoE Expert Compression × Training-Based Recovery

**Direction**: Which MoE expert compression/pruning/merging method really matters for *training-based* compression? Does a smart training-free method keep its lead after SHORT recovery training, and what is the recovery trajectory?
**Generated**: 2026-06-08 (idea-discovery Phase 2)
**Pipeline**: research-lit → **idea-creator** → novelty-check → research-review → research-refine
**Ideas**: 10 generated (GPT-5.4 xhigh via codex) → 4 survive filtering → converge into **1 coherent paper (3 pillars)** + 2 optional method-pivots.

---

## Landscape summary (from Phase 1)

The literature cell "which MoE *expert*-compression method matters AFTER short training" is empirically empty. The closest prior, **SlimQwen** (2605.08738), already asserts the broad claim ("no single one-shot expert prune/merge method establishes consistent superiority; differences are marginal after continual pretraining") — but only at **400B-token / 80B-model** scale, only **whole-expert prune/merge** (no intra-expert low-rank, no unstructured), **co-varying depth+width+expert**, on a model **with a shared expert**. The dense-model analog **A Free Lunch** (2510.14444) predicts the gap closes and closes *more* at scale, via local reconstruction shrinking the criterion's importance. **Is Retraining-Free Enough?** (2603.02217) recovers the *router*; we recover the *whole model on the original SFT data*.

**So the question is no longer "does the gap close?" (SlimQwen: yes, eventually).** The open science is: **how FAST does it close, is the ranking rank-stable during short recovery or does it CROSS OVER, across ALL families, with experts isolated, on a small no-shared-expert MoE — and what step-0 property predicts the post-training winner?**

**De-risking finding (Phase 2):** in the **verl env (transformers 4.56.1)** OLMoE experts are per-expert `OlmoeMLP` (`gate/up/down` `nn.Linear`), so the repo's `find_mlp_triplets` / Nyström / SVD-LLM-V2 / SparseGPT tooling applies to expert MLPs **directly** — the fused-3D blocker is sft-env-only. Expert params = **93.4%** of the 6.9B model, so retain-0.75-on-experts ≈ 23% total reduction. Whole-expert drop = 64→48 experts/layer; intra-expert shrink = intermediate 1024→768. (See memory `olmoe-experts-per-linear-in-verl-env`.)

---

## Recommended program — "The Recovery-Trajectory Atlas of MoE Expert Compression"

> **⚠️ Restructured after Phase 3 novelty-check (5.5/10, PROCEED-WITH-CAUTION; see [NOVELTY_CHECK.md](NOVELTY_CHECK.md)):** Pillar A+B (now "Claim 1") **is the paper**, narrowed to a *controlled post-recovery selection-bias study*. The undeniable-novelty target is a **robust cross-family INVERSION** (step-0 winner family ≠ step-k winner family, large margins, robust across tasks/seeds). Pillar C (predictive diagnostic) is **demoted to a supporting mechanism** inside the paper, NOT a co-equal contribution. **Effective rank is dropped from the headline** (2602.20433 contraindicates it — keep only as a negative-control row). Adopt the positioning one-liner from NOVELTY_CHECK.md verbatim.

The three surviving pillars are **not competing ideas** — they are one paper. Ship them together.

### 🏆 Pillar A (backbone): Recovery-Trajectory Atlas
- **Hypothesis**: The training-free ranking is **not rank-stable** under short recovery; some methods start better but are overtaken within 500–2k steps because they preserve *accuracy* but not *trainability*, while gentler distortions preserve an easier optimization basin.
- **Experiment**: 6–8 representatives at the SAME 0.75 expert retain ratio — {SlimQwen prune (freq/REAP), SlimQwen merge + partial-preservation, per-expert SVD-LLM-V2, per-expert Nyström, per-expert SparseGPT, random-drop control, magnitude control}. Eval MMLU + GSM8K at steps {0, 100, 250, 500, 1k, 2k, 5k, 10k} on OpenThoughts3 SFT. Primary outcomes = **Area-Under-Recovery-Curve (AURC)** + **crossover time** + endpoint, NOT step-0 alone.
- **Contribution**: empirical-finding. **Risk: LOW** — even "no crossovers / ranking is stable" is a clean publishable negative.
- **Effort**: 1–2 weeks. **Reviewer objection**: "just a benchmark" → rebutted by the crossover analysis + Pillars B/C turning it into science.

### 🥈 Pillar B (headline analysis): Family > Criterion variance decomposition
- **Hypothesis**: Compression **family** (drop / merge / low-rank / unstructured) explains far more post-recovery variance than the exact scoring rule *within* a family; within-family criterion differences collapse fast, family-level structure persists because each family loses specialization differently.
- **Experiment**: 2–3 methods per family at matched calibration budget + exact retain ratio; variance decomposition (family vs criterion vs layer-allocation) on step-0, 2k, 10k results.
- **Contribution**: empirical-finding. **Risk: LOW-MEDIUM**. **This is the axis SlimQwen literally does not cover** (it omits low-rank + unstructured entirely).

### 🥉 Pillar C (diagnostic payoff): What step-0 property predicts recoverability?
- **Hypothesis**: Step-0 task score + plain reconstruction error predict *immediate* damage but **not** short-recovery value; **routed-token curvature / gradient-alignment / diversity-retention** metrics predict the trajectory because they capture what SGD can cheaply repair.
- **Experiment**: from the 20–30 checkpoints generated by A+B, compute step-0 diagnostics {reconstruction err, MoE-block output-KL, routed-token Fisher-weighted error, gradient alignment, inter-expert diversity, effective rank}; regress against early-slope and AURC.
- **Contribution**: diagnostic. **Risk: MEDIUM** (correlation-fishing risk — pre-register a small metric set, hold out methods for validation). **Highest upside**: "step-0 eval is the wrong selection target; metric M predicts the winner" is more valuable than another compressor.

**Why this program survives SlimQwen**: it (1) measures the short-horizon *trajectory* SlimQwen never charts, (2) covers the two families SlimQwen omits, (3) isolates experts (attn+router frozen) so any ranking change is attributable to the expert-compression method alone, (4) operates in the small no-shared-expert regime where the "plain catches up" effect may differ.

---

## Optional secondary directions (fold in if a pillar yields a sharp signal; do NOT lead with these)

| # | Idea | When to promote | Risk |
|---|---|---|---|
| 3 | **Rerouting vs Relearning** — decompose recovery into router-reallocation vs surviving-expert takeover vs damaged-expert relearning (full-update vs router-frozen vs experts-frozen) | if Pillar A shows crossovers → explains the *mechanism* | MEDIUM (changes recovery protocol) |
| 6 | **Diversity retention predicts GSM8K (not MMLU)** recovery | if Pillar C flags diversity as a strong predictor | MEDIUM |
| 10 | **Partial-preservation is the hidden variable** — cross preservation fraction {0,small,med} × families | cheapest sharp ablation; converts SlimQwen's endpoint claim into a mechanism | LOW-MEDIUM |
| 5 | **Layer-allocation dominates method choice** (uniform vs sensitivity-weighted per-layer budget) | if you want a "light new method" angle | MEDIUM |

## Eliminated / demoted ideas

| Idea | Reason |
|---|---|
| 8 — Fisher-weighted local reconstruction (new compressor) | The framing explicitly says **don't add yet-another-compressor**; SlimQwen says fine-grained criterion matters little post-training, so a new criterion is the weakest bet. Keep only as a Pillar-C-motivated *probe*, not the headline. |
| 9 — Heterogeneity-aware hybrid (per-expert family) | Same: "mixture of tricks" reviewer risk; only worth it if Pillar B shows family matters AND buckets are cleanly separable. Defer. |
| 7 — One-step recoverability as selection rule | Subsumed by Pillar C (it's one candidate predictor among many); leaking-recovery-distribution objection. Run *inside* C, not standalone. |

## Pilot plan (Phase 2.5 — to run next)

The cheapest decisive pilot is **not** a training run — it's proving the **compress→reload→short-SFT→eval loop works end-to-end on OLMoE**, then a *single* mini-trajectory (2 methods × 3 checkpoints: step 0 / 250 / 1k) to confirm (a) tooling works, (b) recovery curves are measurable, (c) a crossover is at least plausible. Est: ~2–4 GPU-hr. Defined success: two methods show *different* step-0 MMLU and we can read both off at step 0 and step 1k.

## Next steps
- [ ] Phase 3: `/novelty-check` on Pillar A+B combined claim and Pillar C predictive claim
- [ ] Phase 4: `/research-review` on the 3-pillar program (honest about SlimQwen overlap)
- [ ] Phase 2.5 pilot: end-to-end OLMoE compress→SFT→eval smoke + mini-trajectory
- [ ] Phase 4.5: refine → EXPERIMENT_PLAN.md
