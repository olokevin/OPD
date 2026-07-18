# Experiment Results — TRACER / Block 0

**Date**: 2026-06-02
**Plan**: `EXPERIMENT_PLAN.md` · **Status**: Block 0 DONE → **central thesis falsified, pivot triggered**

## Block 0 — Steering-Energy-Retention (SER) probe [GATING]

**Setup**: Qwen3-4B (non-thinking base), OpenThought3 math-trace calibration (128×2048). Difference-of-means steering vectors `ũ^m_{ℓ,c}` (4 behaviors, input space) extracted from the dense model + attribution importance. Measured per-module preservation of the weight's action along `ũ` — `cos(W_dense ũ, W_comp ũ)` and energy ratio — under (a) single-layer compression @ retain 0.5 (validated-harmless layer 20) vs (b) all-layer @ retain 0.36 (the known 0%-MATH collapse). Control = **variance-matched** random directions (`qᵀCq = ũᵀCũ` under the module input covariance). Code reviewed by GPT-5.4 (2 CRITICAL + 3 MAJOR fixed before run). 200 cells, all depths L0–L35.

### Results

| Condition | steering cos | random cos | steering energy | random energy |
|---|---:|---:|---:|---:|
| single-layer (0.5) | 0.994 | 0.843 | 0.989 | 0.660 |
| **all-layer (0.36, the collapse)** | **0.993** (attn) / 0.953 (mlp) | **0.721** | 0.986 / 0.915 | 0.521 |

- **Steering-vs-random cos gap = −0.27** (thesis predicted **> 0**). Steering directions preserved at **every depth** (L0…L35 all cos ≈ 0.99), no depth decay.
- **Variance-vs-leverage check**: difference-of-means steering directions are **~95× HIGHER** directional variance than the average direction (`dir_var` median 11.2 vs `mean_diag_var` 0.098).

### Verdict: TRACER central thesis FALSIFIED (the plan's explicit falsifier condition)

The thesis was: structured compression collapses reasoning by **erasing a low-variance, high-leverage steering subspace**. The data show the opposite:

1. **Steering directions are the BEST-preserved part**, not the most eroded — preferentially preserved vs variance-matched random directions, at all depths.
2. **The "low-variance" premise is empirically false**: difference-of-means steering vectors are ~95× *high*-variance. Activation-aware SVD/Nystrom preferentially preserve high-mass directions of the activation-weighted operator, so these directions survive — random directions (spread across the full spectrum incl. the truncated tail) lose ~half their energy.

**Scope of the claim (per GPT-5.4 second-opinion — do not over-read):**
- ✅ Falsified: "compression collapses reasoning by first-order destruction of the extracted DoM steering directions."
- ❌ NOT claimed: "steering is irrelevant to reasoning" (preservation is necessary-not-sufficient; behavior-critical info could live in a tiny *truncated tail* of `ũ` that the whole-vector cos/energy metric misses), nor "M1 is proven."

## Pivot

→ **Drop TRACER's C2 (steering-subspace preservation).** The leading live mechanism is now **M1 (rank-deficiency)**: low-rank truncation discards off-subspace / tail mass the residual stream needs, compounding over depth. Random directions losing ~50% energy while the top (steering) directions are kept is exactly the M1 signature.

**This Block-0 negative result is publishable** as a pivot point: an intuitive, literature-grounded hypothesis (steering-subspace erosion) predicted selective damage; the data show the supposedly-fragile directions are among the best preserved. Paired with one positive follow-up on the real mechanism, it redirects the field's mechanism search toward rank allocation / tail restoration.

## Next experiment (gated): Tail-Rescue (causal M1 test) — supersedes old Block 1

GPT-5.4-recommended sharpest/cheapest confirmer (more causal than more SER analysis):

> Take the collapsed model; **add back a tiny number `k` of the discarded singular components** (the truncated tail) per module — preferentially in the modules with the largest activation-weighted residual — and sweep small `k`. **If MATH recovers monotonically with tiny tail reinjection → M1 confirmed** (the lost tail mass is the culprit). If not → "missing tail" is not the story, look at cross-layer accumulation / a few under-ranked layers.

This becomes the new gating experiment before any method-building. Tests the truncated tail directly (addresses the "cos misses the tail" confound 1b) and is one-shot, <1 GPU-hr.

## Tail-Rescue — causal M1 test [GATING, second negative]

**Setup**: all-layer compress @ 0.36 (attn SVD-V2 + MLP Nystrom). On the **attention SVD modules only**, reinject `k` discarded singular components per module (rank `r → r+k`); MLP held at Nystrom 0.36 throughout. Sweep `k`, eval MATH-500 (100, greedy, verl env). k=0 sanity reproduced the exact 0% / 1.697B collapse ✓.

| k | MATH-500 | total params | +attn tail |
|---:|---:|---:|---:|
| 0 | 0.00 | 1.697B | — |
| 4 | 0.00 | 1.700B | +2.9M |
| 8 | 0.01 | 1.703B | +5.9M |
| 16 | 0.03 | 1.709B | +11.8M |
| 32 | 0.01 | 1.721B | +23.6M |
| 64 | **0.04** | 1.744B | +47.2M |

### Verdict: simple attention-tail M1 NOT confirmed (second clean negative)

- Reinjecting the attention tail gives only a **weak, noisy** rise (0 → ~4%), **not** the monotonic climb toward 45% that strong-M1 predicts. On 100 problems (±2–4% binomial noise) `0/0/1/3/1/4%` is "still collapsed."
- At **k=64 attention is nearly un-truncated (+47M)** yet MATH is 4% → **the attention low-rank tail is not the dominant cause.**
- **Crucial confound (by design)**: MLP/Nystrom was held at 0.36 the whole sweep, so a dead MLP path masks any attention benefit. This experiment proves attention-tail loss is *not sufficient* to explain collapse; it does **not** rule out attention mattering *in combination*.

**Refined posterior** (GPT-5.4-concurred): (1) most likely **MLP/Nystrom is the primary bad actor**; (2) next, **joint/interaction degradation**; (3) least likely, attention-tail rank-deficiency alone.

## Next experiment (gated): Subsystem Ablation — supersedes more tail-rescue

Single most diagnostic test per GPU-hour (GPT-5.4-recommended over MLP tail-rescue):
compress **attn-only @0.36 (MLP dense)** vs **MLP-only @0.36 (attn dense)** vs **both** (existing), eval MATH-500. Disambiguates in one shot:
- MLP-only ≈ both ≈ 0, attn-only healthy → **MLP/Nystrom is the killer**.
- attn-only & MLP-only both OK, both collapses → **interaction / cross-layer composition**.
- attn-only ≈ 0 too → attention low-rank *structure* (not missing tail) is harmful.

## Files / repro
- Results: `scripts/opd/math/compressed_opd/results/block0/ser_probe.json`
- Log: `logs/compressed_opd_v2/block0_full.log`
- Code: `src/compress/steering.py`, `scripts/opd/math/compressed_opd/block0_ser_probe.py`
- Repro: `CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl python3 scripts/opd/math/compressed_opd/block0_ser_probe.py --layers 35,33,30,25,20,15,10,5,2,0 --single-layer-target 20 --calib-num-seqs 128 --steer-num-convs 128 --out results/block0/ser_probe.json`
