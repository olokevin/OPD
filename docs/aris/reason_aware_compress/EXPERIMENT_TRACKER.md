# Experiment Tracker — TRACER

Run IDs map to `EXPERIMENT_PLAN.md` blocks. Status: PENDING / RUNNING / DONE / FAILED.

| Run | Block | Description | GPU | Status | Key result | Notes |
|---|---|---|---|---|---|---|
| B0-sanity | 0 | SER probe, 2 layers, 8 seqs, no attribution | 0 | DONE | steer_cos 0.986 vs rand 0.70 (all-layer); gap −0.29 | pipeline OK; **falsifier signal** — steering PRESERVED, random eroded more |
| B0-full | 0 | SER probe, 10 layers, 128 seqs/convs, +attribution | 0 | DONE | steer_cos 0.99 vs rand 0.72; gap −0.27; steering 95× HIGH-variance | **THESIS FALSIFIED** — steering subspace is best-preserved, not eroded. Pivot to M1. |
| TR-sanity | 0' | Tail-Rescue k=0, 8 problems | 0 | DONE | MATH=0.000, 1.697B | ✓ reproduces collapse (correctness check) |
| TR-full | 0' | Tail-Rescue k∈{0,4,8,16,32,64}, 100 problems | 0 | DONE | MATH 0→4% across k; k=64 (+47M attn) still 4% | **2nd negative**: attn-tail M1 not the cause (MLP held @0.36 masks it) |
| SA-* | 0'' | Subsystem ablation: attn-only vs mlp-only vs both | 0 | DROPPED | attn-only @0.36 = 3% (partial, kept for record) | **no longer run** — attn/mlp split removed from plan |
| TL-* | 2 | phase-transition localization (TEL/FID), sweep 0.8/0.7/0.6/0.5 | — | PENDING | — | runs after a method candidate shows signal |
| HL-* | 4 | headline table + ablation matrix | — | PENDING | — | best method candidate vs prior-art baselines |
| D0-sanity | D | D0 @0.8, MATH/16, 32-seq calib (pipeline check) | 7 | DONE | nz 3.316B, C4 PPL 55.7, **MATH 81.25%** | ✓ pipeline validated; @0.8 last-layer-skip baseline ≈ dense (80.5%) — full dynamic range |
| **D0–D2** | D | OPD-weighted bi-whitened SVD (objective fix, M2) | 5 | DONE | D0 73%/52.1 · D1 0%/294 · D2 70%/52.1 | **M2 null**: D2(bilateral CE)≈D0(fwd-only), D1(bwd-only) collapses → objective not a separable lever → M1/Block A is the headline. **D3 deferred** (distinct teacher) |
| **A0–A2** | A | Low-rank + sparse residual / +Patch (rank floor, M1) | 5 | DONE | A0 72%/52.1 · A1 80%/42.4 · **A2 82%/42.4** | **M1 HEADLINE**: A2≥A1>A0, full-rank sparse residual recovers 72→82% (beats dense 80.5%) at same budget — causal M1 confirm where tail-rescue failed. A3 sweep = nice-to-have |
| **B0–B1** | B | Sequential re-linearized compression / SRC (accumulation, M3) | 5 | RUNNING | — | B0/B1 @0.8. **B2 deferred** (needs distinct teacher) |
| T-probeset | T | Freeze 5 dense-correct MATH probes → `trace_probe_set.json` | 6 | DONE | 5 probes (pids 0–4), dense traces + gold frozen | reference for every per-method trace diff |
| **T** | T | Reasoning-trace diff: dense vs compressed, per method/cell | — | PENDING | — | `generate_traces()` runs after D/A/B cells land (2 ratios/method) |

> **A/B/D re-promotion (2026-06-02)**: TRACER C2 falsified → A/B/D (demoted only for prior-art-novelty) are now first-class one-shot method candidates attacking the live mechanisms M1/M2/M3. Self-contained, launch now. **Operating point: retain 0.8 first (last decoder layer's linears skipped), then sweep down (Block 2: 0.8/0.7/0.6/0.5).** Long-context block (old Block 3) and the attn-only/mlp-only subsystem split are **removed**. Block T trace-diff runs alongside each method for qualitative inspiration. See `EXPERIMENT_PLAN.md` § "Operating point & protocol" + "Blocks A/B/D".

## Code implemented
- `src/compress/steering.py` — difference-of-means steering vectors `ũ^m_{ℓ,c}` (input-space), attribution importance `I^m_{ℓ,c}`, per-module input covariance + MLP baseline collection. Behaviors via lexical cue proxy (GPT-4o-span labeling is a documented refinement).
- `scripts/opd/math/compressed_opd/block0_ser_probe.py` — Block 0 driver: reproduce collapse, extract steering, measure SER (attn: weight-action cos/energy; mlp: local directional-derivative at real baseline x0) for single-layer vs all-layer, with **variance-matched** random controls + variance-vs-leverage check.

### A/B/D mechanism-fix drivers (2026-06-03, experiment-bridge)
- `scripts/opd/math/compressed_opd/compress_common.py` — shared helpers: load_model, OpenThought3/C4 calib loader, **last-decoder-layer skip via `drop_protected_stats`** (drop stats whose layer idx ∈ protect → core leaves them dense; the repo-established `tail_rescue.py` pattern, since `skip_layers` is leaf-name-only), MATH-500 + C4 PPL eval contract, param counting, CLI.
- `scripts/.../bi_whitened_svd.py` — **Block D** (M2). D0 fwd-only / D1 bwd-only(CE) / D2 combined(CE)=OBD-LLM / D3 combined(OPD/teacher). Attn objective varies; MLP=Nystrom fwd. No new core code.
- `src/compress/hybrid/lr_sparse.py` + `scripts/.../lr_sparse_residual.py` — **Block A** (M1). `LRPlusSparse = UV + S`; SVD low-rank at reduced budget + SparseGPT-pruned residual of `R=W−UV`. A0 pure-SVD / A1 dense-acts / A2 compressed-upstream / A3 budget-split sweep. A2 fits R against the **true deployed UV+S prefix** via a refinement Hessian pass (`refine_passes`).
- `src/compress/sequential/relinearized.py` + `scripts/.../sequential_src.py` — **Block B** (M3). Depth-ordered re-linearization; `collect_layer_input_covariance` re-collects layer ℓ's input cov through the compressed prefix. B0 dense-pass / B1 SRC-fwd / B2 SRC+OPD-bwd.
- `scripts/.../trace_diff.py` — **Block T** diagnostic. `--mode build` freezes 5 dense-correct MATH probes → `trace_probe_set.json`; `generate_traces()` (importable) dumps per-item dense-vs-compressed traces, graded vs **dataset gold**.

## Code review (GPT-5.4 / GPT-5 high, A/B/D, 2026-06-03)
1 CRITICAL + 2 MAJOR, all addressed:
- **CRITICAL — D3/B2 degenerate when teacher==student** (OPD KL≡0 → zero backward cov → silent collapse to fwd baseline). Fix: drivers now **reject** teacher==student unless `--allow-degenerate-opd`. **User decision: skip D3/B2 this pass**, deploy teacher-free cells; OPD claim cells deferred until a distinct teacher is chosen (candidates on disk: `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, `Qwen3-8B-Base`).
- **MAJOR — A2 upstream was low-rank-only, not UV+S.** Fix: `refine_passes` re-fits residuals against the fully-installed `LRPlusSparse` prefix.
- **MAJOR — OPD/CE backward not response-only** (packed chat windows have all-ones masks → prompt tokens enter the loss). Documented limitation (shared with the standing OPD covariance pipeline); response-span-aware loader is the refinement. Affects D1/D2/(D3/B2).
- Verified clean: last-layer skip across all 4 paths, MATH grading uses dataset gold (eval + trace-diff), D1/D2 arg routing, stats-dict cloning (no aliasing), whitening/Cholesky numerical guards.

## Code review (GPT-5.4, applied)
2 CRITICAL + 3 MAJOR fixed: (1) MLP probe now local directional-derivative not MLP(u~); (2) random control variance-matched under input cov not L2; (3) attribution uses same assistant+behavior-span masking; (4) single-layer target = validated-harmless layer 20; (5) variance-vs-leverage logs `uᵀCu/‖u‖²` not `‖a_bar‖`.
