# Experiment Plan — Reasoning-aware structured compression (M1/M2/M3 method search)

**History**: started SRC-centric → re-architected to TRACER (steering subspace) → **TRACER C2 falsified at Block 0** (`EXPERIMENT_RESULTS.md`). Current primary track = **mechanism-fix methods A/B/D** (M1 rank floor, M2 objective, M3 accumulation); TRACER's C1/C3 survive as knobs and Block 2 (transition localization) survives as a diagnostic. Aligned with `FINAL_PROPOSAL.md` (audit trail) and `LITERATURE.md`.

**Claim to prove**: structured low-rank compression collapses reasoning via the **rank floor (M1)** — dropping the full-rank "escape edges" SparseGPT keeps — compounded by a **variance- not loss-weighted objective (M2)** and **uncorrected cross-depth accumulation (M3)**; restoring the rank floor (A), fixing the objective (D), and correcting accumulation (B) recovers reasoning **one-shot** and holds accuracy to a lower retain ratio. Failure localizes to reasoning-transition tokens/sentences (Block 2, Block T).

**Operating point (whole plan)**: retain **0.8 first → sweep down** (Block 2: 0.8/0.7/0.6/0.5); **last decoder layer's linears never compressed** (`model.layers.{N-1}.*`). The 0.36 figure is retained only as the known-collapse reference.

**Standing eval (every cell reports BOTH)**: MATH-500 greedy, 100 (pilot)/200 (headline) problems, `max_new_tokens=2048`, `ttrl_math` grader; **and** C4 sliding-window PPL, seqlen 2048, seed 0. 1× H100, bf16. References: 4B dense 80.5% / 19.9; native 1.7B 50.0% / 15.4; SparseGPT+math 45.0% / 82.0; SVD+Nystrom (dense-ref) @0.36 0.0% / 4,980. Calibration held fixed (OpenThought3 / V2 fresh-gen, 128×2048, seed 3) unless a block varies it explicitly.

---

> Operating point (0.8-first, last layer skipped) applies to **every** block — full detail in "Operating point & protocol for A/B/D" below. Block-0 DONE results stay as-run at 0.36; the protocol applies to any re-run + Blocks 1/2 + the method blocks.

## Block 0 — Reproduce + steering-subspace probe (≈0.7 GPU-hr) [DONE — thesis falsified, see EXPERIMENT_RESULTS.md]
1. Reproduce SVD_V2+Nystrom at the operating ratio (0.8 first; the 0.36 collapse stays the reference) math-calib.
2. **Steering-Energy-Retention (SER) probe** — the cheapest decisive test:
   - Extract dense steering vectors `ũ^m_{ℓ,c}` (difference-of-means on reasoning traces, 4 behaviors) via the `psunlpgroup/Compression-Effects` recipe.
   - For (a) a *single-layer* compressed module (harmless, retain 0.5) and (b) the *all-layer* collapse, measure per layer:
     `SER = ‖U^T W_comp x‖² / ‖U^T W_dense x‖²` and `cos(W_comp ũ, W_dense ũ)`.
   - **Prediction (thesis)**: SER stays ≈1 for single-layer; for all-layer it **decays with depth / drops sharply on the steering subspace specifically** (much more than on random directions of equal variance). A random-direction control is mandatory.
   - **Falsifier**: if the steering subspace is preserved as well as random low-variance directions in the collapsed model, the steering thesis is wrong → fall back to M1 (hetero-rank + LoSparse patch) as headline.
3. **Variance-vs-leverage check**: confirm `ũ` directions are low input-variance but high attribution-importance `I^m_{ℓ,c}` — the premise of "variance truncation deletes them first."
- **Go/No-go**: steering subspace selectively eroded → TRACER C2 justified → proceed.

## Block 1 — TRACER core, one-shot (≈2 GPU-hr) [DEMOTED — C2 falsified; C1/C3 survive as knobs on A/B/D]
> C2 (steering preservation) is dropped (`EXPERIMENT_RESULTS.md`). What remains live from this block: **C1** (transition-conditioned loss geometry) folds into Block D4, and **C3** (loss-budgeted hetero-rank) is an optional rank-allocation knob on A/B/D. The ablation ladder below is retained for the record; run only the C1/C3 cells, and at the new operating point.
Cells @ **0.8 first → sweep down, last decoder layer skipped**, MATH/100 + PPL:
- T0 baseline: SVD_V2+Nystrom dense-ref (= Block 0) at the operating ratio.
- **T-C1**: transition-conditioned loss geometry only (`G̃` w/o steering term) — vs **OBD-LLM-style** average-token K-FAC (the prior-art baseline to beat). *(live → D4)*
- ~~**T-C2**: + steering-subspace preservation~~ — **dropped (falsified)**.
- **T-C3**: + loss-budgeted heterogeneous rank (vs uniform rank). *(live → A/B/D knob)*
- **Success**: C1 and/or C3 measurably beat the dense-ref baseline along the ratio sweep (each component adds). Stretch: approach SparseGPT's one-shot accuracy.

## Block 2 — Killer diagnostic: phase-transition localization (≈1.5 GPU-hr) [makes the paper] (user dir 4)
Retain-ratio sweep **`r ∈ {0.8, 0.7, 0.6, 0.5}`**, **last decoder layer skipped**, on MATH/AIME: dense-ref baseline vs the surviving method candidates (D/A/B, + C1/C3 knobs). Segment traces into sentences; mark transition sentences (steering-score peaks / SEAL transition-reflection class). Metrics:
- **TEL (Transition Excess Loss)**: NLL(compressed)−NLL(dense) on post-transition tokens ÷ same on execution tokens. Thesis: TEL ≫ 1 and spikes before generic quality collapses.
- **FID (First Irreversible Divergence)**: first sentence boundary after which k rollouts never reach the correct answer. Localizes "where generation stops being salvageable." *(Complements Block T's first-divergence read at the trace level.)*
- **Sentence Survival Curve** `P(FID > s)` over step index s.
- **Cliff point r\***: smallest retain (within the sweep) where median FID collapses before the first successful backtracking/planning transition.
- **box-rate vs response-length crossing** (RAC's looping signature: length↑ while acc↓).
- **AR-floor reframe (engage 2505.24187)**: fit per-position L2 logit error(t) on teacher-forced CoTs to {linear, exp, sub-linear/key-token}; show **compression raises the per-token error floor** until it crosses self-correction capacity (turns sub-linear curve into a cliff). Compare dense vs compressed vs best method candidate.

## Block 4 — Headline table + ablation matrix (≈2 GPU-hr) [PAPER MAIN]
Best method candidate at its chosen operating ratio (from the 0.8→0.5 sweep), **last decoder layer skipped**, **MATH/200 + AIME24 + AMC23 + OlympiadBench** (JustRL `gen_vllm.py`/`grade.py`) + C4 PPL, vs:
- dense, native 1.7B, SparseGPT+math, SVD+Nystrom dense-ref (the within-block baseline),
- **OBD-LLM-style** (avg-token bilateral whitening = Block D2), **SAES-SVD-style** sequential (= Block B1), **PGSVD-style** hetero-rank (= C3 knob),
- **D** (OPD bi-whitened, M2), **A** (LR + sparse residual / +Patch, M1), **B** (SRC, M3), **B3** (best one-shot combination), + the C1 transition-conditioned objective (D4).
This matrix is what makes the novelty legible vs each prior-art collision.

## Block 5 — Robustness (defuse reviewer risk, only if the method blocks are positive)
Best method candidate's MATH vs: calibration size, calibration domain, behavior/transition-set choices (where C1/D4 is used); **transfer** (calibration from MATH traces → AIME/Olympiad). For any block that uses transition-token conditioning, separate genuine mid-layer transition directions from the shallow last-layer "To/Step" token prior (Sinii 2509.06608).

---

# Blocks A/B/D — Mechanism-fix method candidates (post-falsification re-promotion)

> **Why these blocks exist.** The original pipeline demoted ideas **A** (low-rank + sparse residual), **B** (sequential re-linearized compression / SRC), and **D** (OPD-weighted bi-whitened SVD) to ablations/baselines purely on **prior-art-novelty** grounds (LoSparse/OATS, SAES-SVD, OBD-LLM respectively), in favor of TRACER's steering-subspace thesis. That thesis is now **falsified** (`EXPERIMENT_RESULTS.md` Block 0: steering directions are the *best-preserved* part; the "low-variance" premise is empirically false — DoM directions are ~95× *higher* variance). With C2 dead, the live mechanisms are **M1 (rank-deficiency)**, **M2 (variance- not loss-weighted objective)**, **M3 (uncorrected cross-depth accumulation)** — and A/B/D are exactly the three direct fixes for them. They are re-promoted here from "ablation cell" to **first-class one-shot method candidates**, while the TRACER scaffold above is kept only for the **Block 2 transition-localization diagnostic** and the C1/C3 knobs.
>
> **These blocks are self-contained and can launch now** (they don't depend on TRACER C2). Run them at the operating point below (retain 0.8 first → sweep down, last decoder layer skipped).
>
> **Mechanism → method map**: A → M1 (restore full-rank escape edges). B → M3 (correct accumulation sequentially). D → M2 (loss/OPD-weighted reconstruction objective). All three are **one-shot, no SGD**, eval-dominated (compression is minutes).

## Operating point & protocol for A/B/D (start here)

- **Retain ratio = 0.8** (keep 80% of params) as the **starting point**, not 0.36. Rationale: at 0.36 the model is *fully collapsed* (0% MATH, PPL ~5k) — a floor that hides method differences. At 0.8 the dense model is only mildly perturbed, so each method's effect on a *working* model is legible and the ablation ladder has dynamic range. **Sweep down later** (0.8 → 0.7 → 0.6 → … → 0.36) once the 0.8 ordering is established, to find each method's cliff. Every cell below that says "@0.36" is **re-anchored to @0.8 for the first pass** unless it explicitly references the collapse baseline.
- **Skip the last decoder layer's linear layers.** Never compress any linear in `model.layers.{N-1}.*` (N = `config.num_hidden_layers`, so layer 35 for Qwen3-4B) — neither attention (q/k/v/o_proj) nor MLP (gate/up/down_proj). Final-layer linears feed the LM head directly; truncating them injects error with no downstream depth to absorb it.
  - **Implementation note (the code can't do this via `skip_layers` alone)**: `skip_layers` matches on the **leaf name only** (`name.split(".")[-1] in skip_layers`, verified in `compress_model.py:194`, `svd_llm_v2.py:253`, `nystrom.py:28`, `pruning.py:116/181`, `calibration.py:117/161/210/318/550`), so passing `"layers.35"` does nothing and passing `"q_proj"` would skip that proj in *every* layer. To skip exactly the last layer, the A/B/D drivers must add a `name.startswith(f"model.layers.{N-1}.")` filter around the compression + covariance-collection calls (or a one-line patch to each skip predicate to also test `any(name.startswith(p) for p in skip_prefixes)`). Spell this out in the driver; it is **not** free.

**Shared standing baselines for A/B/D** (same eval contract — MATH-500 + C4 PPL): dense 4B 80.5% / 19.9 · native 1.7B 50.0% / 15.4 · SparseGPT+math 45.0% / 82.0 · **collapse reference = SVD_V2-attn + Nystrom-MLP dense-ref @0.36 = 0.0% / 4,980**. At the new **@0.8** operating point the *first thing every block reports is the plain dense-ref baseline @0.8 + last-layer-skipped* (T0.8) — expected to be much closer to dense; the method's job is to keep T0.8 as high as possible and to **stay high as the ratio sweeps down**. "Headline" bar unchanged: approaching/beating SparseGPT's one-shot accuracy at the aggressive end of the sweep.

## Block D — OPD-Weighted Bi-Whitened SVD (≈1 GPU-hr) [cheapest, run FIRST]
**Mechanism**: M2. Replace input-variance whitening with a **bilateral** objective `min ‖C_dy^{½}(W−Ŵ)(XᵀX)^{½}‖_F²` (truncated SVD of `C_dy^{½} W (XᵀX)^{½}`), with the output-grad covariance `C_dy` driven by the **OPD/teacher** loss rather than next-token CE.
- **Infra: already fully wired** (verified). `collect_both_covariances_from_loader_opd(student, teacher, calib_loader, …)` → `(C_x, C_dy)`; then `svd_llm_v2_compress_model(model, fwd_cov, objective="combined", backward_covariances=bwd_cov, …)` dispatches to `svd_compress_layer_combined` per layer. No new core code — just a driver script.
- **Cells @0.8 first (then sweep 0.7/0.6/0.5/0.36), last layer skipped, MATH/100 + PPL** (attention modules; MLP held at the standing Nystrom path unless a cell says otherwise):
  - **D0** = dense-ref input-only SVD-V2 (T0.8) — the within-block baseline.
  - **D1** = backward-only whitening (`objective="backward"`, `C_dy` from next-token CE) — isolates "grad-weighted vs input-weighted".
  - **D2** = bilateral, `C_dy` from next-token CE (`objective="combined"`) — the **OBD-LLM-style prior-art baseline to beat**.
  - **D3** = bilateral, `C_dy` from **OPD/teacher** loss (`collect_both_covariances_from_loader_opd`) — the actual Idea-D claim.
  - (optional **D4**) calibrate `C_dy` on **transition tokens only** — the bridge to TRACER C1, reuses transition masking from Block 1.
- **Trace-diff** (Block T, below): run on D0 vs D3 at the *first ratio where they diverge in accuracy* — does OPD bilateral whitening change *where* the trace breaks?
- **Success**: along the sweep, D3 > D2 ≥ D1 > D0 (ordering should appear before full collapse); the *ratio at which each cell falls off the dense accuracy* ranks the objectives. **Falsifier**: D3 ≈ D0 at every ratio → M2/objective is **not** a separable lever; the rank floor (M1) dominates → A becomes the headline.

## Block A — Low-Rank + Sparse Residual (LR+OBS / "+Patch") (≈1.5 GPU-hr) [highest-ceiling for M1]
**Mechanism**: M1 directly. `Ŵ = UV + S`: `UV` = structured low-rank at a *reduced* budget, `S` = SparseGPT/OBS-pruned residual of `R = W − UV` at the leftover budget, allocated by OPD-weighted residual energy. Restores the exact, full-rank "escape edges" that pure low-rank truncation kills — the mechanism SparseGPT (full-rank zeroing) keeps and SVD loses. The **total** stays at the block's retain ratio: at 0.8, split e.g. LR 0.74 + S 0.06; the split *fraction*, not the absolute budgets, is what A3 sweeps.
- **Infra: SVD entry (`svd_compress_layer` / `svd_llm_v2_compress_model`) + SparseGPT (`SparseGPT.add_batch`/`fasterprune`, `sparsegpt_prune`) both exist.** **NEEDS WRITING**: a thin orchestrator `hybrid/lr_sparse.py` — after SVD, materialize `Ŵ_lowrank`, form `R = W − Ŵ_lowrank`, wrap `R` in a temp `nn.Linear`, run `SparseGPT` on it against the **compressed-upstream** activations, store `S` as a sparse residual added at forward. ~80–120 LoC; no new math.
- **Cells @ iso-param, retain 0.8 first then sweep down, last layer skipped, MATH/100 + PPL**:
  - **A0** = pure SVD-V2 at the block ratio (= D0) — the no-patch baseline.
  - **A1** = LR + sparse-residual (≈¾:¼ of the *compressed* budget to S, i.e. small S), residual fit against **dense** activations (cheap baseline).
  - **A2** = same split, residual fit against **compressed-upstream** activations (the real claim; **synergy with B** — fits R against B's already-compressed prefix).
  - **A3** = budget-split *fraction* sweep at fixed total (S taking {⅓, ¼, ⅛} of the compressed-away mass) to find where the full-rank tail earns its params.
  - **Scope note**: applies cleanly to the **attention SVD path**; Nystrom-MLP is neuron-subsampling (no SVD residual) — for MLP, the "+Patch" is a SparseGPT residual on the *reconstructed* MLP weights, flagged as a separate sub-cell A-MLP.
- **Trace-diff** (Block T): A0 vs A2 at the ratio where A2 first beats A0 — do the restored full-rank edges fix a *specific* failure in the trace (e.g. a dropped backtrack, a botched arithmetic step)?
- **Success**: across the sweep A2 ≥ A1 and A2 holds dense-level accuracy to a *lower* ratio than A0 → the full-rank escape edges are the missing ingredient (M1 confirmed *causally*, succeeding where attention tail-rescue's 0→4% failed because that test only re-added **low-rank** tail, not **full-rank** sparse edges). **Reviewer-risk control**: keep `S` small and **ablate it** (A0 vs A2) so the claim stays "we fixed *structured* compression", not "a sparse tail did the work".

## Block B — Sequential Re-Linearized Structured Compression (SRC) (≈2 GPU-hr) [attacks M3]
**Mechanism**: M3 (code-verified: `compress_model.py:~520–543` collects *all* covariances in one **dense** pass — no layer ever sees its compressed upstream). Compress layers `ℓ=0..35` in depth order; before compressing layer ℓ, push the calibration batch through the **already-compressed** prefix `0..ℓ−1` and recollect `XᵀX_ℓ` (and `C_dy,ℓ` for the bi-whitened variant) so layer ℓ reconstructs against the distribution it will *actually* receive. Still one-shot, no SGD.
- **Infra: per-layer compress primitives all exist** (`svd_compress_layer`, `svd_compress_layer_combined`, `svd_als_compress_layer`; `nystrom_compress_mlp` for MLP). **NEEDS WRITING**: `sequential/relinearized.py` — the depth-ordered loop + a per-layer hook `collect_layer_input_covariance(model, layer_name, loader)` that runs forward through the partially-compressed model and grabs one layer's input cov. Moderate effort (the loop is the only new control flow; the covariance hook is a single-layer reuse of existing forward hooks).
- **Cells @0.8 first (then sweep down), last layer skipped, MATH/100 + PPL**:
  - **B0** = dense-pass layer-independent (= D0) — the current pipeline at this ratio.
  - **B1** = SRC with **forward** cov re-collected on the compressed prefix (the SAES-SVD-style accumulation fix — the prior-art **baseline to beat**, run honestly).
  - **B2** = SRC with **forward + OPD-backward** cov re-collected on the compressed prefix (the Idea-B delta: re-linearization with the OPD-faithful objective — combines B's accumulation fix with D's objective).
  - **B3** (synergy) = SRC + sparse residual from A fit at each layer against the compressed prefix (B2 + A2). The strongest one-shot combination if A and B each move the needle.
- **Trace-diff** (Block T): B0 vs B1 at the ratio where they diverge — does correcting accumulation delay the trace breakdown to deeper into the CoT (a *later* first-divergence point)?
- **Success**: across the sweep B1 > B0 (accumulation matters) and B2 ≥ B1 (OPD objective adds on top of re-linearization); SRC should hold accuracy to a lower ratio than B0. **Risk to surface**: greedy — if early layers irreversibly destroy features, re-linearization adapts to *damaged* representations; B3 (full-rank patch per layer) is the hedge.

## Block T — Reasoning-trace diff: uncompressed vs compressed (≈0.5 GPU-hr) [inspiration / mechanism color]
**Purpose** (user-requested): for **each** compression method, look at *how the reasoning itself changes*, not just the accuracy number. Pick **5 fixed MATH-500 problems the uncompressed model solves correctly** (greedy), then for every method/cell of interest generate the trace on the compressed model under the **same prompt + greedy decoding** and diff it against the uncompressed trace. The point is qualitative inspiration: where does compression first bend the reasoning, and *how* (drops a step? hallucinates a number? loops? right approach, wrong arithmetic? abandons backtracking? never emits `\boxed{}`?).

**Protocol**:
1. **Fixed probe set**: run the uncompressed Qwen3-4B on MATH-500 (greedy, `enable_thinking=False`, `do_sample=False` — exactly as `eval_math500`), keep the first 5 problems graded **correct** by `ttrl_math.compute_score`. Freeze these 5 (problem id + prompt + dense trace + gold) as `trace_probe_set.json` so every method diffs against the same reference.
2. **Per method/cell**: regenerate the trace on the compressed model, same 5 prompts, same greedy settings, `max_new_tokens=2048`. Save `{problem_id, dense_text, comp_text, dense_correct=True, comp_correct, method, ratio}`.
3. **Diff & annotate** (cheap, mostly by reading + a light LLM-assisted pass): align the two traces; record (a) **first-divergence token/sentence** (where the compressed trace first departs from the dense one), (b) a **failure-mode tag** {dropped/early-stop, arithmetic slip, wrong plan, lost backtrack/verify, repetition/loop, never-boxes, fluent-but-wrong}, (c) whether the divergence is at a **reasoning transition** (ties back to the dead TRACER C1 intuition — does the trace break at planning/backtracking boundaries even though the *steering subspace* is preserved?).
4. **Run it at two ratios per method**: at **0.8** (mild — catches the *first* qualitative change before accuracy drops) and at the **ratio where that method's accuracy first falls off** (catches the failure that actually kills it).

**Infra**: lift the prompt-build + greedy `model.generate(**enc, max_new_tokens=…, do_sample=False, pad_token_id=…)` + `batch_decode(skip_special_tokens=True)` directly from `eval_math500` (`scripts/.../layer_sensitivity.py:181–224`) — but **capture the decoded string per example** (the function currently returns only aggregate accuracy, so write a thin `trace_diff.py` that keeps `(problem, generated_text, correct)` per item). MATH-500 prompts/gold come from `datasets/test_data/MATH-500/test.parquet` (`row["prompt"]`, `row["reward_model"]["ground_truth"]`). ~60–80 LoC, no model changes.

**Output**: a short `TRACE_DIFF.md` with the 5×methods grid + the failure-mode tags + 2–3 representative side-by-side excerpts. This is **diagnostic/inspiration, not a headline metric** — it tells us *which mechanism story the traces support* (e.g. if compressed traces fail by arithmetic slips that grow with depth → M1/accumulation; if they fail by abandoning backtracking → revisit a transition-specific objective). Feeds back into which of D/A/B to push and whether C1 (transition-conditioned objective) deserves resurrection on stronger evidence than the falsified C2.

## Run order & budget — A/B/D
Sequential by cost/wiring: **D (≈1 hr, no new code) → A (≈1.5 hr, ~100 LoC) → B (≈2 hr, new loop)**, plus **Block T trace-diff (≈0.5 hr, ~70 LoC)** run alongside each method at the two diagnostic ratios; ~5 GPU-hr to first signal on all three. Cells within a block parallelize across the 8-GPU node (eval dominates). **First pass at retain 0.8 with the last decoder layer skipped**, then sweep the ratio down to find each method's cliff. **D first** because it's a zero-new-code probe of the M2/objective lever and immediately tells us whether the fix is "better objective" (D wins) or "more rank" (A wins) — that fork sets the headline. **Build `trace_probe_set.json` once** (5 dense-correct MATH problems) before any method so all trace-diffs share a reference. Synergy cells (A2-on-B prefix, B3) only after both A and B independently show signal.

**Subsystem scope**: D/A/B run on the **full (attn + MLP)** model at the operating point (0.8 first, last layer skipped). Each method applies cleanly to the **attention SVD path**; for MLP, Nystrom has no SVD tail, so A's "+Patch" and D's bilateral whitening operate on the *reconstructed* MLP weights (flagged in Blocks A/D). No separate attn-only/mlp-only subsystem split is run.

## Run order & budget (whole plan)
**Diagnostic track**: Block 0 is DONE (thesis falsified); Block 1 is demoted to its surviving C1/C3 cells; Block 2 (transition localization) runs after a method candidate shows signal; Block 4 (headline table) after 2; Block 5 last.
**Mechanism-fix track (A/B/D)**: D → A → B (~5 GPU-hr to first signal incl. Block T; see "Run order & budget — A/B/D" above). With C2 falsified, the **A/B/D track is the primary method-search path**; Block 2 (transition localization) and Block T (trace-diff) characterize whichever mechanism-fix wins, and Block 1's C1 folds into D4. Fits the 8-GPU node; one-shot compression is minutes, eval dominates.

## Files to touch
- `src/compress/` — new `steering.py` (difference-of-means `ũ` + attribution-patching `I`, port from `psunlpgroup/Compression-Effects`); `calibration.py` already has fwd+bwd cov collection (`collect_both_covariances_from_loader`, `collect_both_covariances_from_loader_opd`) — add **transition-token masking** to the hooks; new `tracer.py` (C1 `G̃^{½}WX^{½}` weighted SVD + C2 steering augmentation/deflation + C3 budgeted rank knapsack); reuse `svd_compress_layer_backward`, `svd_compress_layer_combined`, `nystrom_compress_mlp`, `unstructured/sparsegpt.py` (for +Patch).
- **New for A/B/D**: `src/compress/hybrid/lr_sparse.py` (Idea A — SVD then SparseGPT-on-residual orchestrator, ~80–120 LoC); `src/compress/sequential/relinearized.py` (Idea B — depth-ordered re-linearization loop + `collect_layer_input_covariance` single-layer hook). Idea D needs **no new core code** — only a driver script wiring `collect_both_covariances_from_loader_opd` → `svd_llm_v2_compress_model(objective="combined", …)`.
- **Last-layer skip helper**: `skip_layers` is leaf-name-only, so add a shared `name.startswith(f"model.layers.{N-1}.")` filter (N = `config.num_hidden_layers`) applied in each A/B/D driver around the compress + covariance-collection calls (or a one-line `skip_prefixes` patch to the skip predicates). Default retain **0.8** in the drivers' `--ratio`, with a `--ratio-sweep` for the cliff search.
- `scripts/opd/math/compressed_opd/` — `--hetero-rank`, `--retain-sweep` flags; diagnostic scripts (SER probe [done], TEL/FID for Block 2); **new method drivers** `bi_whitened_svd.py` (D), `lr_sparse_residual.py` (A), `sequential_src.py` (B), and **`trace_diff.py`** (Block T — per-example greedy traces dense-vs-compressed, lifts prompt-build+generate from `eval_math500` but returns per-item `(problem, text, correct)`; writes `trace_probe_set.json` + `TRACE_DIFF.md`), each reusing `eval_math500` (from `layer_sensitivity.py`) + `evaluate_model_ppl` (`ppl_eval.py`).
- Submodule discipline: commit/push inside `src/compress` first, then bump the pointer in OPD (per CLAUDE.md).

## Reused / verified infra (lowers cost)
- fwd + **backward (output-grad) covariance** + both-in-one-pass already exist in `src/compress/calibration.py`; **OPD-faithful** variant `collect_both_covariances_from_loader_opd(student, teacher, …)` is wired (drives Idea D, Block B2).
- `svd_compress_layer_backward` minimizes `‖(W−Ŵ)ᵀΦ‖` with Φ from grad cov (C1 backbone); `svd_compress_layer_combined(whitening_input, whitening_gradient)` already does the **bilateral** objective (Idea D core); `svd_llm_v2_compress_model(objective="combined"/"backward")` dispatches them full-model.
- `unstructured/sparsegpt.py` (`SparseGPT.add_batch`/`fasterprune`, `sparsegpt_prune`) prunes against collected Hessian — reusable on an arbitrary residual `R` (Idea A).
- `calibration_opd_loss.py` gives the OPD/teacher gradient direction for `g_t` (C1, Idea D `C_dy`).
- `eval_math500` (`scripts/.../layer_sensitivity.py`) + `evaluate_model_ppl` (`src/compress/ppl_eval.py`) are the standing eval entry points for every A/B/D cell.
- Dense-ref collapse (T0) and per-layer harmlessness already established in `docs/results/compressed_opd.md`.
- **Caveat (Nystrom MLP)**: Nystrom is neuron-subsampling, **not** SVD — it has no ordered singular tail. So A's "+Patch" and the tail-rescue intuition apply to the **attention SVD path**; the MLP path needs a SparseGPT residual on reconstructed weights or a low-rank-MLP variant (flagged in Blocks A/D).
