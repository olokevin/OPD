# Idea Discovery — Output Manifest

Direction: reasoning-aware structured pruning (SVD-LLM-V2 attn + Nystrom/MoDeGPT MLP), Qwen3-4B→1.7B one-shot.
Date: 2026-06-02.
- **v1 pipeline**: code-grounded diagnosis → GPT-5.4 idea gen → GPT-5.4 adversarial review (recommended SRC).
- **v2 pipeline**: 3-agent literature survey → GPT-5.4 collision-resolving refinement → **TRACER** (current recommendation). Prior-art collisions (SAES-SVD, OBD-LLM, PGSVD) demoted SRC/D to ablations.

| File | What it is | Version |
|---|---|---|
| DIAGNOSIS.md | 5 ranked failure mechanisms (M1–M5); analytical core | v1 (still valid) |
| LITERATURE.md | 3-agent survey: prior-art collisions + ~40 verified arXiv refs (structured pruning, steering/reasoning mechanism, efficient-reasoning + long-context) | **v2** |
| FINAL_PROPOSAL.md | **TRACER** — title, abstract thesis, C1/C2/C3 method math, prior-art differentiation, reviewer risks | **v2 (current)** |
| EXPERIMENT_PLAN.md | Block 0–5: SER probe, TRACER ablation ladder, phase-transition localization (TEL/FID/cliff-r*), long-context (RULER/SER-vs-length), headline ablation matrix | **v2 (current)** |
| IDEA_CANDIDATES.md | Compact TRACER component table + active thesis (session recovery) | **v2 (current)** |
| IDEA_REPORT.md | Full report; v2 header points to TRACER, v1 diagnosis/ideas preserved as audit trail | v1+v2 |
| MANIFEST.md | This file | v2 |
