# Initial Experiment Results — A/B/D mechanism-fix track

**Date**: 2026-06-03 · **Plan**: `EXPERIMENT_PLAN.md` §"Blocks A/B/D" · **Status**: RUNNING (D in progress)
**Operating point**: retain 0.8, last decoder layer (35) dense, MATH-500/100 greedy + C4 PPL, OpenThought3 calib (128×2048), bf16, 1×H100.

**Standing references** (same eval contract): dense 4B **80.5% / 19.9** · native 1.7B 50.0% / 15.4 · SparseGPT+math 45.0% / 82.0 · SVD+Nystrom collapse @0.36 **0.0% / 4,980**.

## M0: Sanity — PASSED
D0 @0.8, MATH/16, 32-seq calib: **nz 3.316B · C4 PPL 55.7 · MATH 81.25%**. Pipeline validated end-to-end (compress + last-layer-skip + dataset-gold grading + C4 PPL + JSON). The 0.8 baseline sits at/above dense (80.5%) → mild perturbation, full dynamic range to rank methods (as the plan predicted).

## M1 — Block D (OPD-weighted bi-whitened SVD, mechanism M2 = objective) — RUNNING
| Cell | System | nz params | C4 PPL | MATH-500/100 | Status |
|---|---|---|---|---|---|
| D0 | fwd-only input whitening (= A0/B0 baseline) | 3.316B | 52.14 | **73.00%** | DONE |
| D1 | backward-only (CE grad) | 3.316B | 293.99 | **0.00%** | DONE |
| D2 | bilateral, C_dy from CE (= OBD-LLM baseline) | 3.316B | 52.10 | **70.00%** | DONE |
| D3 | bilateral, C_dy from OPD/teacher | — | — | — | **DEFERRED** (needs distinct teacher) |

> **D-block conclusion (M2 falsifier essentially fires).** Ordering: **D0 (73%) ≈ D2 (70%) ≫ D1 (0%)**. The bilateral CE-gradient objective (D2 = OBD-LLM-style) gives **no gain over plain input-whitened SVD** (D0) — within ±3% noise on 100 problems. Backward-only whitening (D1) is destructive (0% / PPL 294): grad-weighting without input whitening picks the wrong subspace for attention. → **M2 (objective) is not a separable lever** with the CE gradient; the plan's fork resolves toward **M1 (rank floor, Block A) as the headline**. The only surviving M2 hope is D3 (OPD/teacher gradient, deferred) — but the CE-bilateral null makes a large OPD-gradient effect unlikely.

## M2 — Block A (low-rank + sparse residual, mechanism M1 = rank floor) — PENDING (after D)
| Cell | System | nz params | C4 PPL | MATH-500/100 | Status |
|---|---|---|---|---|---|
| A0 | pure SVD-V2 (= D0) | 3.316B | 52.12 | **72.00%** | DONE |
| A1 | LR + sparse-residual vs DENSE acts | 3.316B | 42.35 | **80.00%** | DONE |
| A2 | LR + sparse-residual vs COMPRESSED-upstream acts (claim; refine_passes=1) | 3.316B | 42.41 | **82.00%** | DONE |

> **Block A = the M1 headline. Success criterion met: A2 (82%) ≥ A1 (80%) > A0 (72%).** Adding a small (~6% density) FULL-RANK sparse residual to the low-rank attention factors jumps MATH **72→82%** (**beats dense 4B 80.5%**) and drops PPL **52→42**, at the same total budget. Fitting R against the deployed model's COMPRESSED-upstream activations (A2) is ≥ fitting against dense (A1). This is exactly where attention **tail-rescue failed** (0→4% re-adding only *low-rank* tail) — the **full-rank escape edges** (M1) are the missing ingredient, **confirmed causally**. M1 is the headline; M2 (Block D) was null.

Success: A2 ≥ A1 and A2 holds accuracy to a lower ratio than A0 → full-rank escape edges are the missing ingredient (M1 confirmed causally).

## M3 — Block B (sequential re-linearized / SRC, mechanism M3 = accumulation) — PENDING (after A)
| Cell | System | nz params | C4 PPL | MATH-500/100 | Status |
|---|---|---|---|---|---|
| B0 | dense-pass layer-independent (= D0) | 3.316B | 52.13 | **71.00%** | DONE |
| B1 | SRC, fwd cov on compressed prefix | — | — | — | RUNNING |
| B2 | SRC + OPD-backward cov (teacher=Keven16-RL-Math) | — | — | — | RUNNING (GPU 2, after D3) |

> B0 (71.0%/52.1) reproduces the D0/A0 dense-pass baseline. B1 (SRC re-linearization, M3) running on GPU 5. **D3/B2 un-deferred** — running both with a genuine teacher (Keven16/Qwen3-4B-RL-Math-Step500; vocab/arch verified identical to student). First D3 launch on GPU 4 hit a CUDA OOM — **cause was external GPU contention** (a foreign 43GB job grabbed GPU 4 mid-run, leaving <50GB for D3's dual-model + OPD-backward footprint), not a code bug. Relaunched on a fully-free GPU 2 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--calib-batch-size 1`.

Success: B1 > B0 (accumulation matters); B2 ≥ B1 is the OPD-on-SRC claim (deferred).

## Block T — trace probe set FROZEN
`results/blockT/trace_probe_set.json`: 5 dense-correct MATH probes (pids 0–4), dense traces + gold frozen. Per-method diffs (`generate_traces`) run after the method cells land, at 2 ratios/method.

## Deferred (review-gated)
**D3 / B2** (OPD-weighted claim cells): GPT-5 review found teacher==student → OPD KL≡0 → degenerate. Fail-fast guard added. User chose to defer; needs a distinct teacher (candidates on disk: `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, `Qwen3-8B-Base`).

## Plan update (2026-06-03, user direction)
- **Block B (re-linearization) SKIPPED** — B1 killed mid-run; B2 dropped. B0 baseline (71%) kept for the record. M3 not pursued this pass.
- **Forward-only RATIO SWEEP added** (`ratio_sweep_trace.py`, GPU 5): SVD-V2 input-whitening attn + Nystrom MLP across retain **0.8/0.7/0.6/0.5/0.4/0.36**, last layer dense, MATH/100 + C4 PPL per ratio — the plain structured-compression cliff.
- **Reasoning-trace diff per ratio** (Block T, folded into the sweep): the 5 frozen dense-correct probes regenerated on each compressed model, first-divergence localized — **where does the trace break as the ratio drops.**
- **D3** (OPD/teacher bi-whitened SVD) still running on GPU 2 with the real teacher (Keven16 RL-Math); B2 link will be skipped.

## M4 — Forward-only ratio sweep + trace breakdown — RUNNING
| ratio | nz | C4 PPL | MATH/100 | probes✓/5 | median first-div frac |
|---|---|---|---|---|---|
| 0.8 | — | — | — | — | — |
| 0.7 | — | — | — | — | — |
| 0.6 | — | — | — | — | — |
| 0.5 | — | — | — | — | — |
| 0.4 | — | — | — | — | — |
| 0.36 | — | — | — | — | — |

> `div_frac` = fraction into the dense trace where the compressed trace first departs (1.0 = identical to the end). As ratio ↓, expect div_frac ↓ (breaks earlier) and probes✓ ↓.

## Next
→ collect the sweep + D3; fill tables; analyze the cliff + trace-break localization; then `/auto-review-loop`.
