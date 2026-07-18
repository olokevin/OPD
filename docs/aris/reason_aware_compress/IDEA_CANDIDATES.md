# Idea Candidates — Reasoning-Aware Structured Pruning (v3, post-falsification)

> **⚠ v3 (2026-06-02).** TRACER's central thesis (C2 steering-subspace preservation) is **FALSIFIED** (`EXPERIMENT_RESULTS.md` Block 0: steering directions are the *best-preserved*, not eroded; "low-variance" premise false — DoM dirs ~95× *higher* variance). Consequence: **A/B/D are re-promoted from ablations to first-class one-shot method candidates** — they are the direct fixes for the now-live mechanisms M1/M2/M3, and were only demoted on prior-art-novelty grounds. Full experiment design: `EXPERIMENT_PLAN.md` § "Blocks A/B/D". The TRACER scaffold (C1, the diagnostics in Blocks 2–3) is kept for characterizing whichever fix wins.

| Component | Role (v3) | Mechanism | Closest prior art | Status |
|---|---|---|---|---|
| **D** OPD-weighted bi-whitened SVD | **method candidate (run FIRST, no new code)** | M2 (objective) | OBD-LLM 2604.00821 | **RE-PROMOTED → Block D** |
| **A** low-rank + sparse residual | **method candidate (highest M1 ceiling)** | M1 (rank floor) | LoSparse 2306.11222 / OATS 2409.13652 | **RE-PROMOTED → Block A** |
| **B** sequential re-linearized (SRC) | **method candidate (attacks accumulation)** | M3 (accumulation) | SAES-SVD 2602.03051 | **RE-PROMOTED → Block B** |
| C1 transition-conditioned loss geometry | folds into Block D4 | M2 (transition) | OBD-LLM 2604.00821 (avg-token) | KEPT (as objective variant) |
| ~~C2 steering-subspace preservation~~ | **DROPPED** | — | When-Reasoning-Meets-Compression 2504.02010 | **FALSIFIED** |
| C3 loss-budgeted hetero-rank | optional knob on A/B/D | M2 | PGSVD 2510.05544 | KEPT (rank-allocation knob) |

## Active thesis (v3): close the SparseGPT↔SVD gap by fixing M1/M2/M3 — one-shot
- **Hypothesis**: structured low-rank collapses reasoning via (M1) a **rank floor** that drops the full-rank "escape edges" SparseGPT keeps, (M2) a **variance- not loss-weighted** reconstruction objective, and (M3) **uncorrected cross-depth error accumulation**. Fix the objective (D), restore the rank floor (A), and correct accumulation (B) → recover reasoning **one-shot**, no SGD.
- **Decisive cheap test (run FIRST)**: **Block D** — OPD bi-whitened SVD (zero new code) tells us whether the lever is "better objective" (D wins) vs "more rank" (A wins); **Block A** is the causal M1 test (add back *full-rank* sparse edges, not just the low-rank tail that the attention tail-rescue already showed is insufficient at 0→4%).
- **Killer diagnostics (kept from TRACER scaffold)**: phase-transition localization (TEL, FID, cliff-r*; engage 2505.24187) + long-context-degrades-more (RULER, SER-vs-length) — now used to *characterize* whichever A/B/D fix wins.
- **One-shot vs recovery**: all of A/B/D are one-shot; A's +Patch and B3's per-layer patch are cheap one-pass add-ons; OPD recovery (+2pp/138 steps) is NOT the headline.
- **Next**: Block **D → A → B** (`EXPERIMENT_PLAN.md` § "Blocks A/B/D"); subsystem ablation reprioritizes target subsystem only. Full plan `EXPERIMENT_PLAN.md` · proposal `FINAL_PROPOSAL.md` · lit `LITERATURE.md` · diagnosis `DIAGNOSIS.md` · results `EXPERIMENT_RESULTS.md`.

## (Audit trail — FALSIFIED) TRACER — Protect the Transition
- **Hypothesis (FALSIFIED at Block 0)**: structured low-rank erases a **low-variance, high-loss-leverage transition steering subspace**. Block-0 SER probe showed the opposite — steering directions are the *best-preserved* (cos 0.99 vs random 0.72) and are ~95× *higher* variance, not low. The plan's explicit falsifier fired → pivot to M1/M2/M3 (above). Preserved here as the audit trail; C2 is dropped, C1/C3 retained as objective/rank-allocation knobs on A/B/D.
