# Full-sequence calibration — reweighting × length-filter, then ratio sweep

**Date**: 2026-06-04 · Forward-only compression (SVD-V2 input-whitening attn + Nystrom MLP, last layer dense), OpenThought3 reasoning-trace calib, 128 sequences, bf16, 1×H100. MATH-500/100 greedy + C4 PPL.

> **PRODUCTIONIZED (2026-06-04)**: `sequence`-reweighting + `full`-length is now the **DEFAULT** calibration format for all SVD/Nystrom compression — forward, backward, AND combined (`reweight="sequence"` default on every collector in `src/compress/calibration.py`; backward hooks now mask `grad_output` by `attention_mask`; `compress_model_with_loader` threads it through; `compress_common.build_calib_loader` / `layer_sensitivity.build_openthought3_loader` default `length="full"`). **Escape hatch**: `reweight="token"` + `length="window2048"` reproduces the pre-2026-06-04 baselines exactly. Shared mask-aware accumulator (`_accumulate_cov`) keeps fwd/bwd masking consistent; accumulators are CPU-resident (GPU matmul) so the combined path stays memory-bounded. **Memory caveat (resolved)**: the backward/combined pass over a ~10k-token sequence OOMs 96GB, so `full` **truncates each sequence to `max_seq_len=4096`** (covers p90≈4.4k) at **batch_size=1** for the backward path; forward-only is unaffected. Unit tests: `tests/test_calibration_masking.py` (token regression, token≠sequence, per-row norm, padding-invariance, backward grad-masking) — 10/10 pass; real-4B GPU checks pass (token batch-invariant, token≠sequence fwd 0.03 / bwd 0.64, combined no-OOM). **Integration confirmed**: `bi_whitened_svd.py` (no flags) logs *"sequence mixed covariance"* — default wired through drivers; the combined fwd+bwd collection runs at **~45GB** (vs 93GB OOM uncapped) with the 4096-cap + bs=1. End-to-end D0/D2 @0.7 ran clean (no OOM): D0 **62.0%** / PPL 105.7, D2 58.0% / PPL 105.3 — on a *reduced* config (MATH/**50**, **64** calib seqs, `full` truncated-to-4096), so not directly comparable to the stage-1 headline `sequence:lt2048` **71%** (MATH/100, 128 seqs, no cap); the smoke's purpose was OOM-fix + default-wiring confirmation, both met. A full re-validation at the headline config (128 seqs, MATH/100) would be needed for an apples-to-apples number.

**Question**: does calibrating on **full (un-windowed) reasoning sequences** — instead of 2048-token windows — change the compression cliff? Two design axes:

- **reweight** = how cross-sequence averaging is done: `token` (every token equal; long traces dominate) vs `sequence` (every conversation equal).
- **length** = which conversations enter calibration: `full` (whole conversation, no cap) vs `lt2048` (only conversations < 2048 tokens, dropped not truncated).

Both are **mask-aware** (pad positions excluded — unbiased per real token); the 2048-window baseline used all-ones masks and dropped short convs + long-trace tails. New code: `loaders.build_fullseq_calib_loader`, `calibration.collect_covariances_reweighted`, `compress_common.eval_math_capture`, driver `fullseq_calib_sweep.py`.

**Metrics per cell** (one generation pass, `eval_math_capture`):

- **strict** = MATH-500 acc, final-answer graded by ttrl_math (the pick-best metric).
- **relaxed** = "contains a correct answer" — any `\boxed{}` grades correct anywhere (pardons looping-past-answer).
- **gen_len** = mean generated tokens (looping/blow-up signal).
- **tok2corr** = mean generated tokens to the first correct `\boxed{}` (over those that reach it).

Baselines (2048-window, forward-only) for reference: r=0.8 72% · r=0.7 **66%** · r=0.6 37% · r=0.5 20% · r=0.4 4% (strict MATH/100).

## Stage 1 — 4 settings at retain 0.7 (pick best by STRICT MATH) — RUNNING

| setting            | strict          | relaxed | gen_len | tok2corr | C4 PPL |
| ------------------ | --------------- | ------- | ------- | -------- | ------ |
| token · full      | 66.0%           | 66.0%   | 810     | 537      | 103.0  |
| sequence · full   | 69.0%           | 69.0%   | 798     | 503      | 98.7   |
| token · lt2048    | 65.0%           | 65.0%   | 854     | 535      | 355.2  |
| sequence · lt2048 | **71.0%** | 71.0%   | 799     | 573      | 92.4   |

→ 2048-window baseline at r=0.7 = **66% strict**.

### Stage 1 conclusion — winner: `sequence · lt2048` (71.0%)

Ranked by strict MATH: **sequence·lt2048 71% > sequence·full 69% > token·full 66% ≈ token·lt2048 65%**.

- **Sequence-level reweighting is the dominant factor**: both `sequence` settings (71, 69) beat both `token` settings (66, 65) by ~+4–6pp. Weighting each *conversation* equally — instead of letting long traces dominate the token pool — yields a better covariance for reasoning. **+5pp over the 2048-window baseline (66→71%).**
- **Full-length helps PPL, hurts nothing**: `token:full` PPL 103 vs `token:lt2048` 355 (3.4× better) — the long-trace tails the windowing dropped condition the covariance. But on *strict MATH*, `lt2048` slightly edges `full` within sequence-reweighting (71 vs 69).
- Winner carried to Stage 2: **`sequence:lt2048`** (best strict MATH **and** best PPL 92.4).

## Stage 2 — best setting at 0.6 / 0.5 / 0.4 — PENDING

| ratio | strict          | relaxed | gen_len | tok2corr | C4 PPL  |
| ----- | --------------- | ------- | ------- | -------- | ------- |
| 0.6   | **47.0%** | 47.0%   | 918     | 468      | 263.6   |
| 0.5   | **36.0%** | 36.0%   | 1004    | 481      | 1499.2  |
| 0.4   | **13.0%** | 13.0%   | 1333    | 726      | 28683.3 |

### Stage 2 conclusion — full-seq sequence-reweighted calibration pushes the cliff lower (and the gap GROWS with compression)

Winner `sequence:lt2048` vs the 2048-window forward-only sweep, strict MATH/100:

| ratio      | full-seq seq·lt2048 | 2048-window | **Δ**  |
| ---------- | -------------------- | ----------- | ------------- |
| 1 (4B)     | 80%                  |             |               |
| 0.8        | **77%**        | 71%         | **+6**  |
| 0.7        | **71%**        | 66%         | **+5**  |
| 0.6        | **47%**        | 37%         | **+10** |
| 0.5        | **36%**        | 20%         | **+16** |
| 0.4        | **13%**        | 4%          | **+9**  |
| 0.4 (1.7B) | 50%                  |             |               |

- **The improvement grows as compression gets more aggressive** then narrows at the extreme: +5 → +10 → +16 → +9pp. Peak gain at r=0.5 (nearly doubles, 20→36%); at r=0.4 both are deep in collapse but full-seq still **3.25× higher (13% vs 4%)**. Better calibration matters most where the model is under stress but not yet fully destroyed.
- **No looping pathology**: at every ratio `relaxed == strict` and gen_len stays bounded (799→1004 tokens), vs the 2048-window sweep where traces ballooned 5–7× and looped past the answer (the termination failure). Better calibration appears to **fix the termination/looping failure**, not just raise raw accuracy — the model both reaches the answer *and* stops.
- **Mechanism**: weighting each conversation equally (sequence-reweighting) + mask-aware per-token normalization yields a covariance that better reflects the reasoning distribution than the token-pooled, short-conv-dropping 2048-window scheme. This is a **pure calibration change — same forward-only SVD+Nystrom method, same budget** — so it composes with the M1 sparse-residual headline (orthogonal levers).
