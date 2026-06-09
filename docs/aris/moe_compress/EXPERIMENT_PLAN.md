# Experiment Plan — MoE Expert-Compression Recovery Atlas

> Claim-driven roadmap for [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md). Every locked decision traces to [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) (design) and [NOVELTY_CHECK.md](NOVELTY_CHECK.md) (positioning). Scripts → `scripts/moe_compress/`, python → `src/moe_compress/`. Reuse-first: build glue over `src/compress` + LlamaFactory, not greenfield.

## Claims → experiments map
| Claim | Experiment block | Primary outcome | Pass criterion |
|---|---|---|---|
| **C1** ranking not rank-stable; reorg is family-level | Block 2 (36-run atlas) + Block 3 (stats) | between-family vs within-family variance in AURC model | between-family variance > within-family; family×checkpoint interaction significant |
| **C1-sharp** robust cross-family inversion | Block 2 + Block 3 pre-registered test | sign-flip of family contrast | step-0 winner family ≠ 2k winner; 95% bootstrap CI excludes 0; ≥75% task×seed cells; both retains |
| **C2** MoE diagnostic predicts recovery > recon error | Block 4 (diagnostics + LOFO regression) | LOFO R²/rank-corr of AURC₀₋₂ₖ | MoE-specific predictor beats recon-error & step-0-acc out-of-family |

---

## Phase 0 — End-to-end smoke (PROVE THE LOOP) · 0.5 day · 1 GPU
**Goal:** one method, end-to-end on OLMoE, before any matrix. De-risks tooling, not science.
1. **Download** `allenai/OLMoE-1B-7B-0924-Instruct` (`HF_HOME=/data/yequan/huggingface huggingface-cli download`). *(Only base -0924 is local.)*
2. **Compress (verl env, tfm 4.56 — per-Linear experts, no fused-3D blocker):** `src/moe_compress/compress_olmoe.py`, mirroring `scripts/compress_sft/build_svd_nystrom_student.py`. Method = per-expert SVD-LLM-V2 @ retain 0.75. Enumerate `model.layers[i].mlp.experts[j].{gate,up,down}_proj` via `find_mlp_triplets`-style walk; **router `.gate` untouched**. Materialize to a dense smaller HF ckpt that reloads.
3. **Recovery SFT (sft env):** load the compressed dense ckpt; adapt `LlamaFactory/examples/compress_train/olmoe_compressed_fwd_sft.yaml` → `finetuning_type: full`, freeze attention, keep experts+router trainable, OpenThoughts3, ~200 steps.
4. **Eval:** MMLU + GSM8K at step 0 and step 200.
- **Gate G0:** loop runs end-to-end; two distinct step-0 scores readable; recovery moves the metric. If the unfuse/compress path fails → fall back to fused-tensor compression in the sft env (documented unfuse path #2). **STOP and report if G0 fails.**

## Phase 1 — Method-slate implementation · 2-3 days · CPU/1 GPU
Build the 6 focal compressors as `src/moe_compress/methods/` plugins (shared `compress(model, retain, calib, seed) -> HF ckpt` interface; router weights never touched). **All in verl env.**
| Family | Method | Reuse | New glue |
|---|---|---|---|
| expert-removal | random-expert-drop | — | drop 16/64 experts/layer (seeded); renormalize router rows for dropped experts only at *eval-time mapping*; resize `experts` ModuleList + `gate.weight` rows |
| expert-removal | REAP-saliency drop | REAP score `Σ gⱼ‖Eⱼ(x)‖₂` over calib | saliency hook + drop-lowest |
| merge | SlimQwen-merge + partial-preservation | — | freq/REAP importance; cosine-sim pairing; importance-weighted convex merge; keep top-half intact |
| merge | 2nd merge variant (HC-SMoE output-cluster) | — | agglomerative cluster on mean expert outputs over calib; freq-weighted merge |
| weight-approx | per-expert SVD-LLM-V2 | `svd/svd_llm_v2.py` | enumerate experts; intermediate 1024→768 @0.75 |
| weight-approx | per-expert SparseGPT | `unstructured/sparsegpt.py` | 25% unstructured per expert matrix; **note: storage-only, no FLOP saving** |
| (aux control) | magnitude prune | — | global magnitude mask |

**Standardized calibration (primary):** same corpus (OpenThoughts3 + C4 mix) + same total tokens **256×2048** for every method. Native per-paper recipes deferred to Block 5 appendix.

**Budget accounting (`src/moe_compress/budget.py`):** for every compressed ckpt log BOTH axes — (1) storage/nonzero params; (2) routed **active-capacity** (expected active expert-MLP nonzeros/FLOPs per token under top-8). Tabulate so an inversion can be checked against each. *Known asymmetry to report honestly:* expert-drop preserves active-capacity (still top-8 over 48), width-shrink cuts it, unstructured cuts storage-nonzeros but not dense FLOPs.

## Phase 2 — recovery atlas · ~1-1.5 GPU-days train + eval-dominated · 8 GPU
> **Slate expanded (2026-06-08):** weight-approx family now has **5 methods** — `nystrom` (forward-only Nyström, was mislabeled `svd_llm_v2`), `nystrom_combined` (fwd+bwd CE-calibrated joint Nyström kernel), `svd_llm_v2` (per-matrix whitening SVD, the REAL SVD-LLM-V2), `sparsegpt`, `mobe` (Mixture-of-Basis-Experts, shared cross-expert basis). All training-free. Full slate = expert-removal{random,reap} + merge{slimqwen,hcsmoe} + weight-approx{nystrom,nystrom_combined,svd_llm_v2,sparsegpt,mobe} + control{magnitude} = **10 methods**. Recovery matrix = 10 × 2 retains × 3 seeds = **60 runs** (or hold to a focal subset). Note: svd_llm_v2, mobe, and nystrom_combined-on-down store dense-but-low-rank weights (factored U@V materialized); budget reported as factor-retain, not nonzero count.

- **Retain 0.50 pilot first** (6 methods, 1 seed): if ≥2 methods degenerate (step-0 collapse, unrecoverable) → swap 0.50 → **0.625**. Gate G2a.
- **Recovery protocol (locked):** experts+router trainable, **attention frozen**; OpenThoughts3; checkpoints saved at optim-steps **{0, 100, 500, 2000}** (primary). ⚠️ **Sample/step accounting:** at effective batch 8, 10k samples ≈ 1250 steps < 2000 — so set recovery length to **≥16k samples** (≥2000 steps) OR lower eff-batch; primary horizon is **step 2000**, with 5k/10k-step supplementary on a subset (best+worst family × 0.75 only).
- **Seeds re-sample** stochastic compression choices + recovery data order.
- **GPU allocation:** 1 run/GPU, 8 concurrent (use the project's CUDA-pin pattern, memory `opd-concurrent-runs-gpu-pinning`). Train cost ~1-1.5 GPU-h/run to step 2000 → ~36-54 GPU-h ≈ **5-7 wall-hours on 8 GPU**.

## Phase 3 — Statistics & the inversion test · 0.5 day · CPU
`src/moe_compress/stats.py` + `scripts/moe_compress/analyze_atlas.py`:
- Trajectory mixed model: `score ~ family*checkpoint + retain + task + (1|method) + (1|seed)`.
- Summary model: `AURC_0-2k ~ family + retain + task + (1|method) + (1|seed)`; extract **between-family vs within-family-method variance**. *Do NOT use layer as pseudo-replication.*
- **Pre-registered inversion test (register BEFORE looking at full results):** (a) step-0 winner family ≠ 2k winner family; (b) pairwise family contrast sign-flips, 95% **hierarchical-bootstrap** (resample task & seed) CI excludes 0; (c) flip in ≥75% of task×seed cells; (d) holds at both retains.
- Output: recovery-curve figure (acc vs step, by family, CI bands) + variance-decomposition table + inversion verdict.

## Phase 4 — Step-0 diagnostics & predictive analysis (Claim 2, supporting) · 1 day · 8 GPU
- **Inflate n:** the 36 ckpts + add retain {0.625 or 0.50 variant} + extra seeds → **≥36, target 48** compressed instances as independent units.
- **Pre-registered predictor set** (`src/moe_compress/diagnostics.py`, computed at step 0): MoE-specific = {routed-token curvature (Hessian-diag/GGN on routed tokens), inter-expert diversity-retention (pairwise expert-output similarity vs uncompressed)}; baselines = {reconstruction error, step-0 task accuracy}; **negative control = effective rank** (expect null, per 2602.20433).
- **Validation = leave-one-family-out**: train predictor → AURC₀₋₂ₖ on 2 families, test on held-out family; report out-of-family rank-corr/R². Small predictor set in the main claim; no all-6-metrics pooled p-value fishing.

## Phase 5 — Ablations & robustness (appendix) · as budget allows
- **Router-frozen ablation** (user's original protocol): rerun best+worst family × 0.75, router frozen → quantify router-repair channel; ties to *Is Retraining-Free Enough?* (2603.02217).
- **Native-calibration sensitivity:** rerun the 6 methods @0.75 with their own paper recipes; show family inversion survives the calibration swap.
- **Supplementary long horizon:** 5k/10k-step on a subset (does the inversion persist toward SlimQwen's endpoint regime?).

---

## Run order & decision gates
1. **G0** (Phase 0): loop works end-to-end on OLMoE → else fix tooling / fused-fallback, STOP-report.
2. **Phase 1**: all 6 methods produce reloadable ckpts at matched budget.
3. **G2a** (Phase 2 pilot): 0.50 not degenerate (else → 0.625).
4. **Phase 2** full 36-run matrix.
5. **G3** (Phase 3): inversion verdict (i/ii/iii) → selects the claim (matrix below).
6. **Phase 4** diagnostics; **Phase 5** ablations as budget allows.

## Compute budget (8× A800 80GB)
| Phase | GPU-h | Wall (8 GPU) |
|---|---|---|
| 0 smoke | ~2 | 0.5 day incl. download |
| 1 methods | ~4 | 2-3 days (mostly code) |
| 2 atlas (train) | ~36-54 | 5-7 h |
| 2 + 4 **eval** (dominant: 4 tasks × 4-6 ckpts × 36+ runs; GSM8K CoT gen is the cost) | ~120-200 | 2-3 days |
| 3 stats | 0 | 0.5 day |
| 5 ablations | ~40-60 | 1 day |
| **Total** | **~210-320 GPU-h** | **~1.5-2 weeks** |
*Eval, not training, is the budget. Cache step-0 generations; batch eval across checkpoints; use vLLM for GSM8K CoT.*

## Results-to-claims matrix
| Outcome (G3) | Claimable |
|---|---|
| **(i) inversion robust** | One-shot rankings unreliable in OLMoE-1B experts-only short-recovery; recovery reorganizes at the FAMILY level. + MoE step-0 diagnostics > recon error (if Phase-4 LOFO passes). Bound to this arch/retain-range/protocol. |
| **(ii) weak / task-specific / within-family only** | No family-level inversion claim. Only: recovery *can* reorder methods, but method/task/budget-specific; taxonomy does not dominate. |
| **(iii) no inversion** | Well-powered **negative result**: short recovery preserves step-0 ordering in this controlled setup (refutes stitched-benchmark thesis). C2 may still hold if a step-0 diagnostic predicts recovery *magnitude*. |

## Honest caveats to surface in the paper
- Single architecture (OLMoE-1B, no shared expert) — do not overgeneralize.
- Unstructured (SparseGPT) gives storage but not FLOP savings without 2:4 kernels — report under the storage budget axis, flag the active-capacity asymmetry.
- "Family" inference is only as strong as 2 methods/family allows; if a family ports to 1 usable method, downgrade to method-level claim (gate).
