# Research Review — Experimental Design (MoE Expert Compression × Recovery)

> GPT-5.4 senior-reviewer critique of the EXPERIMENTAL DESIGN (2026-06-08). **Design score: 4.5/10 as originally specced** — three *identifiability* killers must be fixed before any runs. "The single biggest problem is not compute. It is identifiability." All fixes are cheap and folded into the EXPERIMENT_PLAN.

## The three identifiability killers (fix before running anything)

1. **Family claim under-identified — n=1/family means an inversion is a *method* effect wearing a *family* label.** Family and method are collinear with one method per family.
   - **Fix:** ≥2 methods per family. **3 families × 2 methods is the floor; 4×2 better.**
2. **"Retain 0.75" is NOT a common budget across families.** In a top-8/64 MoE: expert-drop cuts storage + candidate support but preserves *active dense capacity/token*; width-shrink cuts active capacity; unstructured sparsity cuts nonzeros but not dense FLOPs (no sparse kernels). A cross-family inversion at "0.75" could be a pure budget artifact.
   - **Fix:** Predefine **two budget axes** — *primary:* storage/nonzero-param budget; *required sensitivity:* routed **active-capacity** budget (expected active expert-MLP nonzeros / FLOPs per token under top-8). **Standardize the calibration token budget** across methods for the main comparison; native paper recipes → appendix sensitivity table.
3. **Frozen router is a family-specific confound.** Expert-drop removes routable destinations, so a frozen router tests "compression + routing mismatch," not recoverability; low-rank/unstructured keep all 64 experts and the router's support intact.
   - **Fix:** **Main recovery protocol = experts + router trainable, attention frozen.** Frozen-router → ablation only. (Note: this *relaxes* the user's original "router frozen" instruction — flag for user sign-off; the user froze the router to isolate experts, but the reviewer shows that *during recovery* this structurally penalizes expert-drop families. Compromise: experts compressed only [router weights untouched at step 0], but router trainable *during recovery*.)

## Two more must-fixes

4. **Task/seed design too thin.** MMLU+GSM8K can't support "across tasks"; GSM8K is noisy at 1.3B-active. → **≥4 tasks** (MMLU, GSM8K, ARC-C, HellaSwag/Winogrande) × **3 end-to-end seeds**; inversion defined by hierarchical-bootstrap CIs over task×seed.
5. **Claim 2 underpowered + metric-fishing risk.** Independent unit = the compressed *checkpoint*, not every recovery checkpoint; 6 methods × few runs is too small for 6 correlated diagnostics. → inflate n via **retain × seeds to 36-48 model instances**; pre-register 2 MoE-specific predictors + 2 baselines + 1 negative control; **leave-one-family-out** prediction of AURC₀₋₂ₖ, not pooled p-values.

## Minimum-viable-but-rigorous design (fits ~1-2 GPU-weeks on 8×A800)

- **Slate (6 focal, 3 families × 2 methods):**
  - *expert-removal:* random-expert-drop + one importance-based expert-drop (REAP saliency).
  - *structural expert-compression:* SlimQwen-merge+partial-preservation + (second structural method — e.g. HC-SMoE-style output-cluster merge, or a second SlimQwen merge variant).
  - *per-expert weight-approximation:* SVD-LLM-V2 + SparseGPT.
  - magnitude = auxiliary control only (not a family unless paired).
- **Severities:** 2 retain ratios — **0.75 and 0.50** (pilot 0.50; if degenerate for several methods, use 0.625). Never claim robustness from one severity.
- **Calibration:** standardized budget (same corpus mix + same total tokens) for the main comparison; native recipes → appendix.
- **Recovery:** experts+router trainable, attention frozen. Checkpoints at **0, 100, 500, 2000** (primary); 5k/10k supplementary on a subset.
- **Outcomes:** AURC₀₋₂ₖ + 2k-step accuracy (primary).
- **Seeds:** 3 end-to-end per method×retain (re-sample stochastic compression + recovery data order).
- **Scale:** 6 × 2 × 3 = **36 primary recovery runs** — right size for the budget and the claim.

## What to fit (statistical models)
- Trajectory: `score ~ family*checkpoint + retain + task + (1|method) + (1|seed)`.
- Summary: `AURC_0-2k ~ family + retain + task + (1|method) + (1|seed)`.
- Variance question: between-family variance vs within-family method variance from the summary model. **Do NOT use layer as pseudo-replication.**
- **Pre-registered inversion criterion (all must hold):** (a) step-0 winning family ≠ 2k-step winning family; (b) the pairwise contrast between those two families flips sign with 95% hierarchical-bootstrap CI excluding zero; (c) the flip appears in ≥75% of task×seed cells; (d) holds at both retain ratios.
- Claim 2: predict AURC₀₋₂ₖ from pre-registered step-0 diagnostics with leave-one-family-out validation; small predictor set in the main claim.

## Results-to-claims matrix
| Outcome | What you may claim |
|---|---|
| **(i) Inversion robust** | One-shot rankings unreliable in this OLMoE-1B experts-only short-recovery regime; recovery reorganizes performance at the coarser FAMILY level. + MoE-aware step-0 diagnostics > naive reconstruction (if LOFO predictive test passes). Bound to this arch / retain range / protocol. |
| **(ii) Inversion weak / task-specific / within-family only** | Cannot claim family-level inversion. Only: recovery can reorder methods, but the effect is method/task/budget-specific. Step-0 rankings are *sometimes* unstable; taxonomy does NOT dominate. |
| **(iii) No inversion (rankings stable)** | Under this controlled setup, short recovery mostly preserves step-0 ordering — a **well-powered negative result** against the stitched-benchmark thesis. Claim 2 may still survive if some step-0 diagnostics predict recovery *magnitude* even when ranking is stable. |

## Net
Original design 4.5/10 → with the 5 fixes it becomes a defensible study at ~36 runs. The fixes cost design discipline, not compute. **Carry every fix into the EXPERIMENT_PLAN.**
