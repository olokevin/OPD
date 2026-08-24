# ES on math reasoning — Qwen2.5-Math-7B

> Evolution Strategies (weight-space, forward-only) fine-tuning of **Qwen2.5-Math-7B** on
> **MATH lvl 3–5**, evaluated on **MATH-500**. Six perturbation subspaces are compared:
> full-weight ES (paper baseline), ZO-Act r=1, activation-magnitude input sparsity,
> FuRA (full-rank BTT, small core only), and two **fixed-spectrum (ISO)** variants that
> move only the singular *frames* — see [§10](#10-iso-fixed-spectrum-es).
> wandb project: **`ES-q2p5-7b`** · code: `verl/verl/trainer/es/`, `scripts/es/`

Status: **runs in flight** (dense/zoact/insparse/fura started 2026-08-20; the two ISO
runs started 2026-08-21). This page is updated as evals land.

---

## Summary: Aug 24

### Where we are

**ES, main comparison — done (6 arms × 150 steps).** Freezing the entire
singular-value spectrum costs nothing: `iso` +0.44 ± 0.31 pp and `isobtt`
−0.37 ± 0.48 pp vs unconstrained dense ES, both statistically level. All ~21 pp of
gain (51.6 → 72.4) is **singular-frame rotation** ([§7](#7-results), [§10](#10-iso-fixed-spectrum-es)).

**What actually decides the ranking is step size, not subspace.** `fura` moved
−12.25 → +0.82 pp on footprint alone, and σ — not α — is the operative knob
([§11.3](#113-answer-yes--but-scale-σ-not-α)). Only rank-1 `zoact` is genuinely
subspace-limited (−2.61 pp, and *worse* with a bigger step). FuRA at matched
footprint is ~1 pp over dense, not the ~3 pp one checkpoint suggested ([§11.5](#115-correction-furas-edge-over-dense-is-1-pp-not-3)).

**Setup bug found and fixed ([§12](#12-alignment-with-the-official-implementation)).**
The official recipe resamples a **1024-problem batch every iteration**; we reused one
fixed 64-problem batch for all 150, which ES memorised (train 66.9 → 77.7 while
held-out went 70.8 → 71.6). With batch-128 resampling, dense ES reaches its plateau
in **~10 iterations instead of ~40**, and a σ sweep puts the optimum at **2e-3–4e-3**,
not the paper's 1e-3:

| σ (aligned, 20 it) | 1e-3 | **2e-3** | **4e-3** | 8e-3 |
|---|---|---|---|---|
| MATH-500 @10 / @20 | 71.2 / 70.8 | **73.6 / 72.8** | 72.0 / **73.2** | 72.0 / 69.0 |

**BP leg — half done, not yet conclusive.** `isobtt` and `isobtt_mix` finished
138/138, but **`dense` (SIGTERM @22) and `iso` (CUDA illegal memory access @20) died
early**. On what did run, BP reaches ES-level accuracy in ~1 h vs ~15 h, `iso` again
tracks `dense`, and the orthogonal input mixer is the best and most stable arm while
`isobtt` collapsed mid-run ([§13.5](#135-results--half-the-arms-landed)).

⚠️ **ES and BP numbers are not comparable as they stand**: ES evaluates greedy (n=1),
BP evaluates mean@4 at T=1.0. The same base model reads **51.6 greedy vs 19.4 sampled**.

### Next steps

1. **Re-run BP `dense` and `iso`** to 138 steps — two of four arms are missing, so the
   BP comparison answers nothing yet. Run `iso` under `CUDA_LAUNCH_BLOCKING=1` to
   localise the illegal memory access (the launcher currently forces it to 0).
2. **Sweep the BP learning rate.** `isobtt`'s collapse is most likely step size: the
   ISO LR was matched *analytically*, never swept — and §11/§7 have now shown twice
   that step size dominates this task.
3. **Re-do the ES headline on the aligned (resampled) setup.** The §7 comparison was
   run on the memorising fixed batch; §12 changes the operating point (σ 2e-3–4e-3,
   ~10× faster convergence), so the `iso`/`isobtt`-vs-dense result should be
   re-confirmed there before it is quoted.
4. **Match the eval protocols** (greedy both sides, or mean@4 both sides) before
   making any ES-vs-BP claim.
5. **Re-run the crashed `zoact` σ-matched control** (died 73/150), or replace it with
   an intermediate-σ sweep to locate where rank-1 stops absorbing the step.
6. **Broader benchmarks** (AIME24, AMC23, Minerva, OlympiadBench) — five ES arms now
   sit within ±1 pp on MATH-500, so a second axis is needed to separate them.

## 1. What we are reproducing

*Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*
(`docs/papers/26_ICML_Evolution Strategies at Scale...pdf`), §4.3 + Appendix A.6:

| Paper setting      | Value                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Base model         | Qwen2.5-Math-7B                                                             |
| Train data         | MATH, difficulty**3–5**                                              |
| Template           | **Qwen-Math** (Table 7)                                               |
| Reward             | binary `\boxed{}` correctness, **no format reward**, OatZero grader |
| Response budget    | max**3,000** tokens                                                   |
| ES hyperparameters | σ = 0.001,**α = σ/2 = 0.0005**, N = 30                             |
| Headline result    | MATH-500**53.0 → 78.0** (ES-CHKPT-3, 192 steps)                      |

> ⚠️ The prompt quoted "α = σ". The PDF actually reads **α = σ⁄2** (a stacked fraction);
> Appendix A.2/A.4 confirm it (σ=0.001→α=5e-4, σ=0.0015→α=7.5e-4). We use **α = 0.0005**.

The ES update itself (Algorithm 2) is z-score-normalised OpenAI-ES:

```
θ_{t+1} = θ_t + (α/N) · Σ_n Z_n ε_n ,   Z_n = (R_n − mean R)/std R ,   ε_n ~ N(0, I)
```

## 2. The six runs

Runs 1–4 share **everything except which subspace ε lives in**. Every mode writes

```
W = W_base + P(C)          C = trainable coefficients (kept in fp32)
```

and ES perturbs / updates `C`, never `W` directly.

| # | Run (wandb name)                  | Subspace `P`                                                                                          | Trainable coeffs | % of 7.6B |
| - | --------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------- | --------- |
| 1 | `es-dense-full_…`              | `P(C) = C` — every parameter (paper baseline)                                                        | 7,615,616,512    | 100%      |
| 2 | `zoact-r1_…`                   | `P(C) = C·Vᵣ`, `Vᵣ` = top-1 right singular vector of the layer's **input activations**     | 1,390,592        | 0.018%    |
| 3 | `insparse-d0.01_…`             | `P(C)[:, idx] = C`, `idx` = top-1% input channels by **activation RMS**                       | 65,415,168       | 0.86%     |
| 4 | `fura-btt-smallcore_…`         | `P(C)[:, blkⱼ] = Aⱼ·Cⱼ`, from the full-rank BTT factorisation `Wⱼ = Aⱼ Rⱼ`                   | 97,771,520       | 1.28%     |
| 5 | `iso-fixedspec-b128_…`         | **multiplicative**: `W ← C_L W C_Rᵀ`, `C` orthogonal — fixed spectrum, both frames move    | 141,102,080 †   | 1.85%     |
| 6 | `isobtt-fixedspec-smallcore_…` | same constraint on the block-wise SVD:`Rⱼ ∈ O(b)` trained, `Aⱼ` and each block's spectrum frozen | 48,470,016 †    | 0.64%     |

† Runs 5/6 are not of the form `W = W_base + P(C)`: the perturbation is a *group action*,
not an additive coefficient, so the entry is the **dimension of the manifold ES searches
per step**, not a coefficient count. Storage differs: run 5 keeps an fp32 master `W`
(6.53 B), run 6 keeps 97,771,520 fp32 core entries (the same tensors as run 4) plus a
frozen bf16 `A`. See [§10](#10-iso-fixed-spectrum-es).

**Run 2 — ZO-Act** (arXiv:2607.01125, r=1 setting). One-shot: a single forward pass over a
calibration set gives `X_ℓ ≈ U_r D_r V_rᵀ`; the effective weight is `W_eff = W + V_r B`
(x·W convention) → in PyTorch's `(out, in)` layout `ΔW = Bᵀ V_rᵀ`, i.e. the **row space of the
update is frozen to the dominant activation direction** and only the `out`-dim coefficient is
free. Paper r=1 values used: μ (=σ) = 1e-3, calibration on training examples, all linear layers
except embeddings/head.

**Run 3 — input sparsity.** Same "one column of freedom per output" idea but with the
*canonical* basis instead of the SVD basis: keep the top-k input channels by calibration
activation RMS (Wanda/AWQ salient-channel criterion), k = 1% of `in_features`.
It is the natural ablation of run 2 — *does the activation-informed **direction** matter, or
just hitting the large-magnitude input channels?*

**Run 4 — FuRA.** Per input block *j* of size *b*, `W[:, blkⱼ] = Uⱼ diag(Sⱼ) Vhⱼ` (exact,
full rank since `b ≤ out`). That is BlockTT with `decomp_mode=output_one_block`,
`blocktt_rank=full`, `s_merged_to=keep_frozen`, `train_position=small`: the large core
`A = U·S` is frozen, the small core `R = Vh` is perturbed, so
`ΔW[:, blkⱼ] = Aⱼ ΔRⱼ` — the update is confined to each block's own left singular subspace
and re-weighted by its singular values. Block shapes come from `_closest_factor_pair`, same as
`btt_layer.py` (e.g. 3584 → 56×64, 18944 → 128×148).

## 3. Setup actually used (and deviations)

| Knob                       | Value                                                         | Note                                                                                                                             |
| -------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| σ / α / N                | 1e-3 / 5e-4 / 30                                              | paper (runs 1–4)                                                                                                                |
| σ / α (ISO runs 5–6)    | **5e-2 / 2.5e-2**                                       | σ is a*relative footprint* there, not a noise std — [§10.4](#104-scale-convention--σ-is-a-relative-footprint-not-a-noise-std) |
| Template / reward / grader | Qwen-Math / binary `\boxed{}` / `ttrl_math` `fast=True` | paper (`fast=True` **is** the OatZero grader)                                                                            |
| Decoding                   | greedy (T=0)                                                  | paper (countdown/sudoku); makes rewards a deterministic function of the seed                                                     |
| Train batch                | **64** fixed problems (shuffled seed 0)                 | paper used 200 for countdown; math batch unspecified                                                                             |
| Train token budget         | **1,536**                                               | ⚠️ deviation from 3,000 — see below                                                                                           |
| Eval token budget          | **3,000**                                               | paper                                                                                                                            |
| Eval                       | full MATH-500, every 10 iterations                            |                                                                                                                                  |
| Iterations                 | 150                                                           | paper's best MATH-500 ckpt was at 192                                                                                            |
| Hardware                   | 1× H100 NVL per run, 1 vLLM engine                           | GPUs 1/2 (runs 1–4, 3/4 queued behind 1/2);**GPU 5** (run 5), **GPU 3** (run 6)                                     |

**Why 1,536 training tokens.** Measured on the base model over MATH-500 at a 3,000-token cap:
correct answers have p50 = 506, p90 = 988, **p99 = 2,030** tokens; wrong answers have p90 = 3,000
(non-terminating loops). Capping the *training* rollout at 1,536 keeps **98.4%** of the
model's correct answers (accuracy 51.2% → 50.4%) but nearly halves wall-clock, because decode
time is set by the longest sequence in the batch, not by the average. Held-out eval keeps the
paper's 3,000. Measured cost per generation pass (64 prompts, 1 H100):
3,000 → 21.5 s, 2,048 → 15.1 s, **1,536 → 11.4 s**; batch 32 vs 64 differs by only 10%
(decode is latency-bound), so the batch was kept at 64 for a better gradient estimate.

## 4. Base model check

Pipeline validation before any training — greedy, Qwen-Math template, 3,000 tokens, our grader:

|                         | MATH-500                                                          |
| ----------------------- | ----------------------------------------------------------------- |
| Paper (Qwen2.5-Math-7B) | 53.0                                                              |
| **This repo**     | **51.2** (standalone) / **51.6** (in-trainer, step 0) |

Within ~1.4 pp — the residual is grader/vLLM-version/precision, not a setup error.

## 5. Calibration (runs 2 & 3)

`scripts/es/calibrate_activations.py`: 256 training problems → base-model greedy rollouts →
prompt+rollout replayed through HF with forward hooks on all 196 linear layers.
Top-r right singular vectors come from **randomized subspace iteration on the never-materialised
Gram matrix** `G = XᵀX` (sketch width 8, 3 passes + a Rayleigh–Ritz pass), so memory is
O(d·8) per layer instead of O(d²) (`down_proj` alone would be a 1.4 GB Gram).

Artifacts: `datasets/es_math/calib_qwen2p5_math_7b.pt`, `datasets/es_math/calib_rollouts.jsonl`.

Findings: the top-1 activation direction carries **53.8% of activation energy on average**
(min 0.30, max 0.98), and it is extremely concentrated — **75.7% of its L2 mass sits in the top
1% of input channels**. This is the massive-activation/outlier-channel structure of LLMs, and
it is why runs 2 and 3 are near-neighbours in what they actually perturb.

## 6. Numerical health

`scripts/es/test_es_perturb_modes.py` — runs 1–4 (all PASS). The ISO runs have their own
suite, `scripts/es/test_iso_es.py`; see [§10.8](#108-verification--scriptsestest_iso_espy):

* perturb → restore is **bit-exact** for dense/zoact/insparse (W is recomputed from
  `base + C`, never add-then-subtract), and `(W⁺+W⁻)/2 = W` to bf16 round-off.
* the FuRA factorisation is **lossless** — reconstruction error on real Qwen weights is
  1.56e-3 relative, exactly the bf16 ULP floor (verified on 4 layers across depths).
* update matches `(α/N)·Σ Zₙεₙ` to 1e-9.

Two numbers worth carrying forward (relative ‖ΔW‖/‖W‖ at σ=1e-3):

| mode                   | ‖ΔW‖/‖W‖                                            | vs bf16 round-off (1.6e-3) |
| ---------------------- | -------------------------------------------------------- | -------------------------- |
| dense                  | 5.0e-2                                                   | 31×                       |
| insparse (10% test)    | 1.6e-2                                                   | 10×                       |
| **zoact r=1**    | **4.2e-3**                                         | **2.7×**            |
| **fura**         | **4.0e-3**                                         | **2.6×**            |
| **iso / isobtt** | **5.0e-2** (by construction, σ *is* this ratio) | 31×                       |

⚠️ At the paper's σ, the structured modes' weight-space footprint is only ~2.6× the bf16
quantisation floor of the vLLM rollout weights, so a non-trivial fraction of what the model
actually "sees" is rounding rather than the intended direction. **fp32 coefficient masters fix
the *update* side** (one ES step moves a coefficient by ~α/√N ≈ 1e-4, at or below one bf16 ULP
of a typical weight — a bf16-only accumulator would silently round most of it away), but the
rollout itself stays bf16. Health check in flight: `train/reward_std` (see below).

## 7. Results

Base = MATH-500 51.6% at step 0 for every run (identical, as expected).

<!-- AUTO:RESULTS BEGIN -->

| # | Run                          | trainable coeffs | ‖ΔW‖/‖W‖ | mean reward σ (last 10 it) | MATH-500 best  | best @ step | steps done |
| - | ---------------------------- | ---------------- | ------------- | --------------------------- | -------------- | ----------- | ---------- |
| 1 | dense (paper ES)             | 7,615,616,512    | 5.0e-2        | 0.027                       | **73.4** | 40          | 150        |
| 2 | zoact r=1                    | 1,390,592        | 4.2e-3        | 0.020                       | **72.2** | 130         | 150        |
| 3 | insparse d=1%                | 65,415,168       | 1.6e-2*       | 0.024                       | **73.4** | 80          | 150        |
| 4 | fura small-core              | 97,771,520       | 4.0e-3        | 0.025                       | **63.4** | 40          | 42         |
| 5 | iso fixed-spectrum           | 141,102,080†    | 5.0e-2        | 0.030                       | **74.0** | 60          | 150        |
| 6 | isobtt fixed-spec small-core | 48,470,016†     | 5.0e-2        | 0.028                       | **73.4** | 120         | 150        |

\* insparse ‖ΔW‖/‖W‖ measured at the 10% test density, not the 1% run density.

† manifold dimension searched per step, not a coefficient count — the ISO modes perturb by a group action, not an additive coefficient. They run at σ = 5e-2 / α = 2.5e-2 (footprint-matched to run 1, *not* the paper's nominal σ); see [§10](#10-iso-fixed-spectrum-es).

Fixed-spectrum constraint health (worst value seen; ‖W‖_F drift for `iso`, max|RᵀR − I| for `isobtt` — both should stay at fp32 round-off): **iso fixed-spectrum** 5.1e-04, **isobtt fixed-spec small-core** 5.8e-05.

### MATH-500 curve (eval every 10 iterations, 3,000-token budget)

| step | dense (paper ES) | zoact r=1 | insparse d=1% | fura small-core | iso fixed-spectrum | isobtt fixed-spec small-core |
| ---- | ---------------- | --------- | ------------- | --------------- | ------------------ | ---------------------------- |
| 0    | 51.6             | 51.6      | 51.6          | 53.2            | 51.6               | 53.2                         |
| 10   | 70.8             | 58.8      | 66.4          | 55.2            | 70.2               | 68.2                         |
| 20   | 71.4             | 66.0      | 68.4          | 60.2            | 72.4               | 68.0                         |
| 30   | 72.4             | 66.4      | 71.0          | 60.2            | 71.4               | 71.2                         |
| 40   | 73.4             | 70.2      | 71.4          | 63.4            | 71.8               | 70.8                         |
| 50   | 69.6             | 69.8      | 71.8          |                 | 72.2               | 71.0                         |
| 60   | 71.6             | 70.4      | 72.2          |                 | 74.0               | 72.8                         |
| 70   | 73.0             | 70.6      | 72.4          |                 | 72.2               | 71.0                         |
| 80   | 72.0             | 70.0      | 73.4          |                 | 72.6               | 72.6                         |
| 90   | 73.2             | 69.0      | 73.0          |                 | 73.2               | 70.6                         |
| 100  | 71.4             | 69.4      | 72.2          |                 | 71.8               | 72.2                         |
| 110  | 70.6             | 70.4      | 72.2          |                 | 71.0               | 71.4                         |
| 120  | 72.2             | 71.8      | 72.2          |                 | 72.2               | 73.4                         |
| 130  | 70.4             | 72.2      | 71.6          |                 | 72.4               | 72.6                         |
| 140  | 72.8             | 70.8      | 71.6          |                 | 73.2               | 72.2                         |
| 150  | 71.6             | 71.4      | 70.8          |                 | 72.4               | 72.8                         |

Timing: ~363 s/iteration (30 perturbations x 64-prompt rollout + grading).

<!-- AUTO:RESULTS END -->

### Headline (final, 150/150): freezing the entire spectrum costs nothing

Five runs completed the full 150 iterations. Paired against run 1 (unconstrained dense ES)
over all **15 shared eval steps** (10…150), in pp of MATH-500 accuracy:

| vs dense                                   | σ      | paired Δ                | t       | verdict                             |
| ------------------------------------------ | ------- | ------------------------ | ------- | ----------------------------------- |
| **`iso` fixed-spectrum**           | 5e-2    | **+0.44 ± 0.31**  | +1.40   | **indistinguishable**         |
| **`isobtt` fixed-spec small-core** | 5e-2    | **−0.37 ± 0.48** | −0.78  | **indistinguishable**         |
| `insparse` d=1%                          | 1e-3    | −0.39 ± 0.47           | −0.82  | indistinguishable                   |
| `fura` σ-matched (n=11, →110)          | 1.25e-2 | +0.82 ± 0.53            | +1.53   | indistinguishable                   |
| `zoact` r=1                              | 1e-3    | −2.61 ± 0.87           | −3.02  | **worse**                     |
| `zoact` r=1 σ-matched (n=7, →70)       | 1.2e-2  | −4.23 ± 0.79           | −5.37  | **worse** (diverging)         |
| `fura` original                          | 1e-3    | −12.25 ± 1.20          | −10.18 | **worse** (step-size starved) |

**The result.** Every singular value of every weight matrix held exactly fixed, and the
model still goes **51.6 → 72.4** on MATH-500, statistically level with unconstrained ES.
All ~21 pp is singular-frame rotation. This is ISO's spectral-inheritance claim tested in
its strongest form — *exactly* fixed rather than approximately — and in a **forward-only
ES** setting the paper does not cover. `isobtt` reaches the same place with the per-block
spectrum frozen, a **48.5 M-dimensional** search manifold (0.64% of the model) and 97.8 M
stored trainable entries instead of a 6.53 B fp32 master.

### It is step size, not subspace — except for `zoact`

The one confound in the original design was that `fura` and `zoact` ran at a 12× smaller
weight-space footprint than `dense`/`iso`. The σ-matched controls settle it, and they
settle it *differently* for the two modes:

* **`fura` was purely step-size starved.** Same factorisation, same trainable tensors,
  only ‖ΔW‖/‖W‖ changed 4.0e-3 → 1.25e-2: **−12.25 pp → +0.82 pp** vs dense. The
  block-wise BTT subspace was never the problem.
* **`zoact` is genuinely subspace-limited.** Raising σ made it *worse*, not better
  (−2.61 → −4.23 pp), and the run was actively collapsing — train accuracy fell 78% → 50%
  between steps ~40 and 73 — before it died with `Aborted (core dumped)` at step 73. A
  rank-1 subspace concentrates the entire footprint into one activation direction, so a
  12× larger step there is destabilising rather than helpful. `zoact` is the only mode
  significantly below dense at *any* step size.

This also explains the ordering at matched footprint: what matters is how *widely* the
perturbation energy is spread, not which structured basis it lives in. `dense` (full),
`iso` (all of O(m)×O(n)), `isobtt` (per-block O(b)), `insparse` (1% of channels) and
σ-matched `fura` all land within ~1 pp of each other; only the rank-1 mode falls away.

Supporting α-sweep on `fura` at fixed σ=1e-3 (`logs/es/sweep_fura_*.log`): α×12.5 → 70.0,
α×40 → 68.2 then divergence to 61.0. Raising α *alone* does not recover the deficit —
σ and α have to move together, which is what the footprint-matching convention of
[§10.4](#104-scale-convention--σ-is-a-relative-footprint-not-a-noise-std) enforces.

**Run 5/6 mechanics.** `train/reward_std` at iteration 1 was 0.075 (`iso`) and 0.058
(`isobtt`) vs dense's 0.084 — a purely rotational perturbation moves reward about as much
as an unconstrained one. Final train accuracy on the fixed 64-problem batch: 77.3%
(`iso`), 80.1% (`isobtt`), 77.7% (dense). Iteration time 395 s / 380 s vs dense's 360 s
(**+10% / +6%**).

**Runs 1 and 2 are complete (150/150).** Headline:

|                  | base | best                 | final (150) | plateau mean (steps 100–150) | trainable coeffs |
| ---------------- | ---- | -------------------- | ----------- | ----------------------------- | ---------------- |
| dense (paper ES) | 51.6 | **73.4** @ 40  | 71.6        | **71.50** ± 0.92       | 7.62B            |
| ZO-Act r=1       | 51.6 | **72.2** @ 130 | 71.4        | **71.00** ± 1.02       | 1.39M            |

**ZO-Act r=1 matches full-weight ES.** The plateau gap is **0.50 pp**, against a ±2.0 pp
binomial SE on a single 500-problem eval (≥0.8 pp even for the 6-eval mean, and the evals are
correlated, so that is a floor). The two are statistically indistinguishable from step ~100 on
— from **5,500× fewer trainable coefficients** (0.018% of the model). ZO-Act simply takes
longer to get there: 58.8 vs 70.8 at step 10, converging by ~step 100, exactly the
step-size lag the 12×-smaller ‖ΔW‖ predicts. **The rank-1 activation subspace is not a
capacity bottleneck for ES on this task.**

**Reproduction verdict: 51.6 → 73.4** against the paper's 53.0 → 78.0. Both curves flatten by
step 40 and oscillate ±1.5 pp thereafter, while the train reward spread decays to 0.027 / 0.020
— the fixed 64-problem batch is being exhausted as a signal source, so the residual 4.6 pp is
most plausibly a **batch** limit (paper's math batch is unspecified; countdown used 200 vs our
64) plus the 1,536-token training rollout cap, not an iteration-count limit. Best coefficients
are checkpointed at `/data/yequan/es/ES-q2p5-7b/<run>/es_train_*/es_coef_best.pt`.

**Training dynamics (run 1).** Train accuracy on the fixed 64-problem batch climbs
58.6 → ~70% over 20 iterations while the reward spread across the 30 perturbations decays
0.084 → ~0.030. The batch is *not* saturated (the best perturbation still reaches 78%), so the
signal is weaker but alive — consistent with held-out MATH-500 rising fast (51.6 → 70.8 by step
10) and then more slowly (71.4 by step 20). Watch `train/reward_std`: if it collapses toward
zero the fixed batch has been solved and the run needs a larger or resampled batch, not more
iterations.

**Run 3 (insparse) also reaches dense parity — and faster than ZO-Act.** 66.4 @ 10 →
71.0 @ 30 → **71.8 @ 50**, i.e. at dense's 71.5 plateau by step 30–50, where ZO-Act needed
~100–130 steps. That ordering is exactly what the footprints predict (insparse 1.6e-2 vs
ZO-Act 4.2e-3), not evidence of a better subspace. **The interesting comparison is
per-parameter:** insparse buys its faster convergence with **65.4M** free coefficients against
ZO-Act's **1.39M** — 47× more — for the same endpoint. So the activation-informed *direction*
is not doing work that the canonical top-magnitude input channels cannot; what the SVD basis
buys is **parameter efficiency**, not reachable accuracy.

**Run 4 (FuRA) at the paper's α was step-size-starved**, not capacity-limited: 55.2 @ 10 →
60.2 @ 20 → 63.4 @ 40, still climbing but far off dense's pace, while its reward spread stayed
healthy at 0.020–0.028. It was stopped at step 42 and replaced by the learning-rate search in
[§11](#11-fura-learning-rate-search); the `fura small-core` column above is that stopped
α=5e-4 run and does not continue past step 40. Note FuRA's step-0 base reads **53.2, not
51.6**: it re-materialises W from the BTT factorisation at init, and the 1.56e-3 bf16 round-off
(§6) flips a handful of greedy trajectories. Harmless, but FuRA's deltas must be read against
its own 53.2 baseline.

**All modes carry signal.** Train reward spread over the 30 perturbations is clearly
non-degenerate (dense 0.08→0.030, ZO-Act 0.049→0.027 as training progresses), i.e. the σ=1e-3
perturbation moves rewards even in the rank-1 subspace — the bf16 rollout floor flagged in §6
did **not** materialise as a dead gradient for either run.

Timing: **~360 s/iteration** (30 perturbations × 11.4 s + grading), ~15 h for 150 iterations.

## 8. Where things live

| What                           | Path                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Launcher (all 6 modes)         | `scripts/es/run_es_math.sh` (`PERTURB_MODE=dense\|zoact\|insparse\|fura\|iso\|isobtt`)                       |
| Auto-queue next mode on a GPU  | `scripts/es/chain_next_run.sh`                                                                            |
| Data prep (Qwen-Math template) | `scripts/es/prepare_qwen_math_data.py`                                                                    |
| Activation calibration         | `scripts/es/calibrate_activations.py`                                                                     |
| Numerical tests                | `scripts/es/test_es_perturb_modes.py` (runs 1–4), `scripts/es/test_iso_es.py` (runs 5–6)              |
| Perturbation kernels           | `verl/verl/workers/rollout/vllm_rollout/es_worker_extension.py` (`StructuredESMixin`; ISO = `_iso_*`) |
| Trainer                        | `verl/verl/trainer/es/ray_trainer.py`, config `verl/verl/trainer/config/es_trainer.yaml`                |
| Reward / prompt                | `verl/verl/trainer/es/task_utils.py` → `task_type=qwen_math`                                           |
| Train / eval parquet           | `datasets/es_math/*.parquet`                                                                              |
| Logs                           | `logs/es/run{1..6}_*.log`                                                                                 |
| Best-eval coefficients         | `/data/yequan/es/ES-q2p5-7b/<run>/es_train_*/es_coef_best.pt`                                             |

## 9. Next steps

_(to be revised once the runs land)_

1. **If a structured mode flat-lines** — first suspect is the bf16 rollout floor (§6), not the
   subspace. The scale-matched control is σ chosen so ‖ΔW‖/‖W‖ matches dense
   (≈1.2e-2 for zoact/fura), with α = σ/2 kept.
2. **Budget-matched run 3.** At d=1% run 3 has 47× more free parameters than run 2. The exact
   ablation is `k = 1` (one input channel per layer) — then run 2 and run 3 have *identical*
   parameter counts and differ only in canonical-vs-SVD basis.
3. **Rank sweep for ZO-Act** (r = 1, 4, 16) — the top-1 direction only carries 54% of activation
   energy, so r>1 should matter more here than in ZO-Act's classification tasks.
4. **FuRA orientation ablation** — `input_one_block` (large core trainable) vs the current
   `output_one_block`, and `s_merged_to=keep_trainable`, matching the repo defaults used for
   gradient-based FuRA.
5. **Broader benchmarks** for whichever mode wins: AIME24, AMC23, Minerva, OlympiadBench
   (parquets already in `datasets/test_data/`), to reproduce the paper's Figure 2 panel.
6. **Longer horizon.** Paper's best MATH-500 checkpoint was at 192 steps; we stop at 150.
7. **Decorrelated-noise `dense` control.** `_es_noise` reseeds per layer with the bare seed,
   so same-shaped layers get identical noise (all 28 `q_proj` move together) — see
   [§10.9](#109-one-deviation-worth-flagging). Runs 5/6 do not have this. A `dense` rerun
   with layer-mixed seeds isolates it, and would tell us whether the paper baseline is
   leaving a 28× search-dimension factor on the table.
8. **ISO σ / block-size sweep.** σ is now a relative footprint, so the natural sweep is
   σ ∈ {2.5e-2, 5e-2, 1e-1} and `ISO_BLOCK_SIZE` ∈ {64, 128, 512} (cost is linear in b,
   search dimension is linear in b). `ISO_PERM=false` is the fixed-subgroup control.
9. ~~**Test spectral inheritance directly.**~~ **Answered (150/150).** `iso` +0.44 ± 0.31 pp
   and `isobtt` −0.37 ± 0.48 pp vs dense — the singular values of Qwen2.5-Math-7B are
   ES-irrelevant on MATH; the whole 21 pp is frame rotation. The open follow-up is *how
   far* it goes: a rank-truncated `Σ₀`, or a spectrum swapped in from a different
   checkpoint, would test whether the values matter at all or merely happen to be adequate.
10. ~~**`fura` vs `isobtt` at matched footprint.**~~ **Answered.** `fura` at σ=1.25e-2 goes
    from −12.25 pp to +0.82 pp vs dense: its deficit was step size, not subspace. `zoact`
    goes the other way (−2.61 → −4.23 and diverging), so rank-1 is a real subspace limit.
11. **Rerun the crashed `zoact` σ-matched control** (`Aborted (core dumped)` at step 73,
    `logs/es/run2b_zoact_sigmatched_long.log`). The trend was already clear — train accuracy
    78% → 50% before the crash — but the run should either be reproduced to confirm the
    divergence or replaced by an intermediate-σ sweep (σ ∈ {3e-3, 6e-3}) to locate where
    the rank-1 subspace stops absorbing the step.
12. **Broader benchmarks for `iso`/`isobtt`** (AIME24, AMC23, Minerva, OlympiadBench) — the
    MATH-500 plateau is now tight enough across five modes (±1 pp) that a second axis is
    needed to separate them at all.

## 10. ISO: fixed-spectrum ES

> Runs 5 and 6. *ISO: An RLVR-Native Optimization Stack* ([arXiv:2607.19331](https://arxiv.org/abs/2607.19331))
> observes **spectral inheritance**: RLVR reuses the base model's singular *values* and
> acquires new behaviour by rotating the singular *frames*. ISO turns that into a
> constraint. This section derives the ES analogue and the block-wise-SVD version.

### 10.1 What ISO does, and why ES cannot copy it directly

ISO constrains every 2-D weight to the **fixed-spectrum family** (their Eq. 2)

```
F(W₀) = { U Σ₀ Vᵀ : U ∈ St(m,q), V ∈ St(n,q) },     q = min(m,n)
```

with `Σ₀` frozen at the base checkpoint's spectrum. ISO-Optimizer stores `U, V`, steps them
with a base optimizer, and then **restores feasibility with a polar retraction**
(Eq. 30–31, 34–35): `G_U = G_W V Σ₀`, `G_V = G_Wᵀ U Σ₀`, `(Ū, V̄) = Opt(·)`, then
`U⁺ = polar(Ū)`, `V⁺ = polar(V̄)`, `W⁺ = U⁺ Σ₀ (V⁺)ᵀ` — implemented as an **fp64 SVD of
each frame, every step**. (Their Prop. 4.1 / Eq. 16, `diag(Uᵀ Ẇ V) = 0`, is a statement
about *tangent directions*; the algorithm's spectrum is exact because `W` is rebuilt from
`Σ₀` each step.)

That stack does not survive the transfer to ES. A gradient step pays **one** retraction
pass per iteration; **ES pays N = 30 feasible perturbations plus the update = 31**. Cost
of one pass, measured on this H100 for Qwen2.5-Math-7B's actual frame shapes (fp64
`linalg.svd` + `P Qᵀ`, all 7 linears × 28 layers):

| frame                               | shape       | s / polar       |
| ----------------------------------- | ----------- | --------------- |
| `q_proj`/`o_proj` `U`,`V`   | 3584×3584  | 2.37            |
| `gate_proj`/`up_proj` `U`     | 18944×3584 | 2.91            |
| `down_proj` `V`                 | 18944×3584 | 2.83            |
| **one full feasibility pass** |             | **744 s** |

→ **6.4 h per ES iteration**, i.e. **65× the 354 s rollout it is meant to serve**. Feasibility
has to be *free*, not *restored*. (Measured while run 5 was training on the same GPU, so
treat it as an upper bound on a quiet card; the order of magnitude is not in doubt.)

### 10.2 The orbit form — feasibility without retraction

**Lemma.** `O(m)` acts transitively on `St(m,q)` for `q ≤ m` (any orthonormal q-frame
extends to an orthonormal basis), so

```
F(W₀) = { C_L W₀ C_Rᵀ : C_L ∈ O(m), C_R ∈ O(n) }.
```

*Proof.* `W₀ = U₀Σ₀V₀ᵀ`. For any `U ∈ St(m,q)` pick `C_L ∈ O(m)` with `C_L U₀ = U`;
likewise `C_R`. Conversely `C_L U₀ ∈ St(m,q)` since `(C_L U₀)ᵀ(C_L U₀) = U₀ᵀU₀ = I`. ∎

Three consequences, all of which the polar-retraction formulation gives up:

1. **No SVD anywhere, and no frame storage.** `U`, `Σ₀`, `V` are never formed or stored —
   the state is just `W` itself. ISO's `(U, V)` are ~1.19× `|W|` plus optimizer state.
2. **Feasibility is never lost, so nothing has to be projected back.** `σ(C_L W C_Rᵀ) = σ(W)` is an identity. ISO's polar retraction also lands on an exact spectrum, but only
   *after* an fp64 SVD pulls `Ū` back onto the manifold, and that projection perturbs the
   realised step by `O(‖ξ‖²)` relative to the tangent direction the optimizer asked for.
   Here the perturbation *is* the group element, so there is no gap to close.
3. **`‖W‖_F` is an exact invariant**, so it is a free online proof that the constraint
   still holds. Logged as `iso/frob_drift`.

### 10.3 Perturbing so the frames stay orthonormal

For skew `Ω` the **Cayley transform**

```
Cay(X) := (I − X/2)⁻¹ (I + X/2)
```

is exactly orthogonal — `I − X/2` is always invertible because `eig(X) ⊂ iℝ` — so

```
W(ε; σ) = Cay(σΩ_L) · W · Cay(σΩ_R)ᵀ  ∈ F(W₀)   exactly.
```

The perturbed frames are `U ← Cay(σΩ_L)U` and `V ← Cay(σΩ_R)V`, orthonormal by
construction. Expanding, `W(ε;σ) = W + σ(Ω_L W − W Ω_R) + O(σ²)`, and since `Uᵀ Ω_L U`
and `Vᵀ Ω_R V` are skew (zero diagonal),

```
diag(Uᵀ Ẇ V) = diag(UᵀΩ_L U Σ₀) − diag(Σ₀ VᵀΩ_R V) = 0,
```

recovering ISO's Prop. 4.1 / Eq. (16) as a corollary rather than a design goal.

**Tractable generator.** A dense `Ω` costs `O(m³ + m²n)` per perturbation. Two structures
make `Cay` cheap, and they are not equivalent:

| generator                                          | cost  | search dim / layer | `‖ΔW‖/‖W‖` at scale σ |
| -------------------------------------------------- | ----- | ------------------ | ----------------------------- |
| low-rank `Ω = PQᵀ − QPᵀ`, rank 2k (Woodbury) | `6k· | W                  | `                             |
| block-diagonal, block size b (batched b×b solve)  | `b·  | W                  | `                             |

The low-rank generator confines the move to a 2k-dimensional subspace, so at 7B
(`m ~ 10⁴`, k ~ 8) it attenuates the step by ~50× — pushing it *below* the 1.6e-3 bf16
rollout floor of §6. **Block-diagonal is the only cheap generator that keeps a full-strength
step**, which is also why the BTT/block factorisation of §10.6 is a natural fit rather
than only a parameter-count trick.

So: `Ω_L = Πᵀ blkdiag(Ω₁,…,Ω_{m/b}) Π`, `Ω_j = (E_j − E_jᵀ)/√(2b)`, `E_j ~ N(0,1)^{b×b}`.
The permutation `Π` is **re-drawn every seed**, so the group generated across iterations is
still all of `O(m) × O(n)` and not a fixed block-diagonal subgroup. Rows are permuted
*within each fused vLLM output segment* (`qkv_proj → [3584,512,512]`,
`gate_up_proj → [18944,18944]`) so a rotation never mixes q/k/v or gate/up channels; the
launch log prints the detected segments, and a silent fallback would be visible there.

### 10.4 Scale convention — σ is a *relative footprint*, not a noise std

With `Ω_j` entries `~ N(0,1/b)` one has `E‖Ω_j F‖_F ≈ ‖F‖_F`, so `‖ΔW‖_F/‖W‖_F ≈ σ`
(both sides share `σ/√2`). **σ therefore means the relative weight-space displacement**,
directly comparable to the `‖ΔW‖/‖W‖` column of §6 — measured 4.94–5.01e-2 at σ = 5e-2 on
real Qwen weights.

This forces a deviation from the paper's nominal σ: at σ = 1e-3 the ISO modes would move
the weights by 1e-3 relative, i.e. **below the 1.6e-3 bf16 floor** flagged in §6 — a dead
perturbation. We instead **footprint-match the dense baseline** (5.0e-2), which is the
scale-matched control §9.1 already called for. With `α = σ/2` the per-iteration motion
`α/√N` then matches dense ES *exactly*: dense moves `α‖ε‖_F/√N = 5e-4·50·‖W‖/√30 = 4.6e-3‖W‖`; ISO moves `α/√N·‖W‖ = 2.5e-2/√30 = 4.6e-3‖W‖`.

### 10.5 The ES update on the group

The ES gradient lives in the Lie algebra. With z-scored rewards `Z_n`,

```
Ω̄_L = (1/N) Σ_n Z_n Ω_L⁽ⁿ⁾ ,     W ← Cay(α Ω̄_L) W Cay(α Ω̄_R)ᵀ.
```

Because each seed uses its own permuted block basis, `Ω̄` is not block-diagonal, so we
realise `Cay(αΩ̄)` as the **ordered product of the N individual Cayley factors** at scale
`(α/N)Z_n`. That product is exactly orthogonal (a product of orthogonals) and equals the
single Cayley up to `O(α²/N) ≈ 2e-5` relative — far below the ES noise floor. Cost is
N× the perturbation cost, once per iteration.

### 10.6 Run 6 — the same constraint on the block-wise SVD

Run 5 keeps a **6.53 B fp32 master** `W`. Run 6 removes that by putting the constraint on
the block-wise SVD already used by FuRA (run 4). Per input block *j*,

```
W[:, blkⱼ] = Aⱼ Rⱼ ,    Aⱼ = Uⱼ diag(Sⱼ) ∈ ℝ^{m×b}  frozen (bf16),
                        Rⱼ = Vhⱼ ∈ O(b)            trained (fp32).
```

`R` is *already* orthogonal at initialisation — it is `Vh` from an exact SVD — so the only
change from run 4 is to **keep it there**: perturb `Rⱼ ← Cay(σΩⱼ) Rⱼ` instead of adding
free coefficients. Then

```
W[:, blkⱼ]ᵖᵉʳᵗ = Uⱼ (Σⱼ Cⱼ) Vhⱼ ,   and  σ(Σⱼ Cⱼ) = σ(Σⱼ),
```

so **each block's spectrum is fixed exactly**, and the trainable state drops from a 6.53 B
fp32 master to 97.8 M fp32 core entries (1.28% of the model; manifold dimension 48.5 M =
0.64%, since `dim O(b) = b(b−1)/2`, half the stored `b²`). Run 4 vs run 6 is therefore a
**clean one-variable ablation**: identical factorisation, identical trainable tensors,
the only difference is whether the small core is constrained to the orthogonal group.

*Caveat.* `A` is stored bf16 (as in run 4), so the preserved spectrum is that of the
bf16-rounded `A`, which differs from `σ(W_orig)` by the 1.6e-3 reconstruction floor of §6.
The constraint itself is exact; its reference point is 1.6e-3 off.

### 10.7 Cost

`iso` costs `2b·|W|` fp32 flops per perturbation (`b = 128`), measured on the real shapes:

| param                             | shape       | ms / perturbation |
| --------------------------------- | ----------- | ----------------- |
| `qkv_proj`                      | 4608×3584  | 3.63              |
| `o_proj`                        | 3584×3584  | 3.35              |
| `gate_up_proj`                  | 37888×3584 | 10.03             |
| `down_proj`                     | 3584×18944 | 6.42              |
| **whole model (28 layers)** |             | **656 ms**  |

→ **≈20 s / iteration** (30 perturbations + the update) on top of ~354 s of rollout: **+5.7%**.
Against the 6.4 h/iteration a faithful port of ISO's polar retraction would cost
([§10.1](#101-what-iso-does-and-why-es-cannot-copy-it-directly)), the orbit formulation is
**~1,100× cheaper** at the same constraint. `isobtt` reuses run 4's reconstruction path and
adds only a batched b×b solve, so its overhead is negligible.

*In the live run* iteration 1 took **403.7 s vs dense's ~354 s (+14%)** — about 2.5× the
isolated kernel time, the remainder being the bf16 write-back into the vLLM parameter and
`empty_cache()` between layers. 150 iterations ≈ **16.8 h**. Memory: the fp32 master is
6.53 B params = 26.1 GB, so run 5 uses `gpu_memory_utilization=0.45` like `dense`
(67.4 GB / 95.8 GB total on the H100 NVL); run 6 keeps `A` in bf16 and uses 0.55.

### 10.8 Verification — `scripts/es/test_iso_es.py`

All checks PASS. The two that matter:

* **The kernels are the operator they claim to be.** The batched permute+bmm path is
  compared against an explicitly materialised `Πᵀ blkdiag(Cⱼ) Π` — agreement 9e-17 in fp64,
  and that dense operator is orthogonal to 5.6e-16.
* **The algebra is exact; the fp32 residual is round-off.** Running the identical
  perturbation in fp64 shrinks `‖Δσ‖/‖σ‖` from 2.3e-8 to 2.8e-14 — **8.3·10⁵× tighter**,
  i.e. it tracks machine epsilon, not a modelling error.

On real Qwen2.5-Math-7B weights, at the run's σ = 5e-2 (svdvals taken in fp64 — `q_proj`
has condition number 5·10⁶, so an fp32 SVD adds ~1e-4 of its own noise and would swamp the
quantity under test):

| layer                             | shape        | `‖ΔW‖/‖W‖` | `‖Δσ‖/‖σ‖` |
| --------------------------------- | ------------ | ----------------- | ------------------- |
| `layers.0.self_attn.q_proj`     | 3584×3584   | 5.005e-2          | **2.50e-8**   |
| `layers.13.mlp.gate_proj`       | 18944×3584  | 4.974e-2          | **2.41e-8**   |
| `layers.27.mlp.down_proj`       | 3584×18944  | 4.977e-2          | **2.45e-8**   |
| `q_proj` (isobtt, per-block)    | n=56, b=64   | 4.954e-2          | **1.14e-7**   |
| `down_proj` (isobtt, per-block) | n=128, b=148 | 4.982e-2          | **1.25e-7**   |

Also verified: perturb→restore is bit-exact (`iso`); the same seed reproduces the same
perturbation; `(W⁺+W⁻)/2 = W + O(σ²)`; the committed update matches the first-order
estimator `(α/N)Σ Z_n Ω_n` to `O(α²)`; `Cay(sΩ)Cay(sΩ)ᵀ = I` to 8e-7 for s ∈ {1e-4, 5e-2, 1};
and `RᵀR = I` to 6e-6 after perturbing the `isobtt` cores.

Online, `iso/frob_drift` (run 5) and `iso/orth_err` (run 6) are logged every iteration.

### 10.9 One deviation worth flagging

The legacy `_es_noise` reseeds its generator with the **bare seed for every layer**, so any
two parameters with the same shape draw *identical* noise — all 28 `q_proj` matrices are
perturbed in the same direction. That ties the effective search dimension to ~1/28 of the
nominal one for runs 1–4. The ISO modes mix the layer id into the seed, so they do not
inherit it. This is a second difference between runs 5/6 and runs 1–4 beyond the subspace,
and it should be attributed carefully; the clean control is a `dense` rerun with
decorrelated noise, which is cheap to add and is now item 7 of §9.

### 10.10 fp32 accumulation drift — measured and corrected

`iso/frob_drift` on run 5 did **not** stay at the 3.5e-6 of iteration 1: it grew to
**4.3e-4 by iteration 126**, and a log-log fit gives slope **1.03** — *linear* in t, not the
√t of a random walk. Constant +3.43e-6 per iteration. That is a systematic bias, so it was
worth chasing rather than filing under round-off.

**What it is.** Reproduced offline on the real kernels (`allow_tf32` confirmed `False`, so
this is genuine fp32, not TF32):

| state dtype | drift after 40 iterations                                   |
| ----------- | ----------------------------------------------------------- |
| fp32        | **+2.365e-4** (exactly 5.91e-6 × t, always positive) |
| fp64        | 1e-16 (no accumulation at all)                              |

So it is pure round-off — but *biased*, which is why it compounds. Decomposing it:

```
after 40 it:  global gain g = 1.000236485
              per-mode σᵢ/σᵢ⁰ :  mean 1.000236438,  sd 7.6e-6,  min 0.99987, max 1.00039
              ‖σ/g − σ⁰‖/‖σ⁰‖ = 1.7e-7      (vs raw drift 2.4e-4)
```

**The drift is a pure isotropic gain.** Every singular value scales by the *same* factor;
the *shape* of the spectrum — which is what the ISO constraint is actually about — is
preserved to 1.7e-7. Removing one scalar per matrix takes the drift from 2.4e-4 to 1.7e-7,
a 1,400× reduction.

**Fix** (`_iso_recondition`, called from `_iso_commit` each update, cost ≈ the norm we were
already computing for the metric):

* `iso` — `state *= ‖W₀‖_F / ‖W‖_F`. Exact, because the error is isotropic.
* `isobtt` — one Newton–Schulz step `R ← R(1.5I − 0.5 RᵀR)`, which drives
  `‖RᵀR − I‖ = E` to `−0.75E²` (1e-5 → 3.6e-7 measured in the test suite).

Verified end-to-end over **300 ES updates** through the real `es_update` path:

| mode       | reconditioning | ‖Δσ‖/‖σ‖ @ 100 | @ 300                    | shape-only @ 300 |
| ---------- | -------------- | --------------------- | ------------------------ | ---------------- |
| `iso`    | **on**   | 3.4e-7                | **5.6e-7**         | 5.6e-7           |
| `iso`    | off            | 2.7e-4                | (linear)                 | 3.3e-7           |
| `isobtt` | **on**   | 1.63e-6               | **1.63e-6** (flat) | 4.0e-7           |
| `isobtt` | off            | 2.6e-4                | 7.6e-4                   | **3.0e-4** |

Two things to note. With the fix, `isobtt` is *perfectly flat* over 300 updates and `iso`
grows only as a √t round-off walk. And **`isobtt` is the mode that actually needed it**:
uncorrected, its *shape* error grows to 3.0e-4, a real constraint violation, whereas
`iso`'s uncorrected error was entirely the benign scalar.

**Confirmed live on the 7B model, over the full 150-iteration run.** Run 6 logs both sides
every step — `iso/frob_drift` is the worst-layer `‖RᵀR − I‖` the 30 Cayley factors left
behind, `iso/orth_err` is what survives the Newton–Schulz step:

|                    | step 1  | 10      | 50      | 100     | 150               |
| ------------------ | ------- | ------- | ------- | ------- | ----------------- |
| pre-fix            | 5.83e-5 | 9.18e-6 | 1.01e-5 | 1.22e-5 | 1.05e-5           |
| **post-fix** | 1.07e-6 | 9.54e-7 | 1.01e-6 | 1.01e-6 | **1.07e-6** |

The post-fix series is **flat** — max/first = 1.17× over 150 updates, i.e. no accumulation
at all, exactly as the offline 300-update test predicted. Run 5 (`iso`, uncorrected) ended
at `frob_drift = 5.13e-4`, matching the linear extrapolation 3.43e-6 × 150 = 5.1e-4.

**Does this invalidate run 5?** No. Run 5 ran *without* the correction, so its spectrum
picked up a per-matrix gain reaching ~5e-4 by step 150 — but the shape held at ~3e-7
throughout, the fixed-spectrum constraint was never meaningfully violated, and 5e-4 is 3×
below the 1.6e-3 bf16 quantisation the vLLM forward applies to `W` anyway. Run 6 and any
later run get the correction. The reason to care is horizon: uncorrected, the linear growth
would reach **3.4e-2 at 10k iterations**, which would matter.

## 11. FuRA learning-rate search

**Question:** FuRA's small-core subspace lagged badly at the paper's α (60.2 @ step 20 vs
dense's 71.4). Is that a *capacity* limit of the subspace, or just too small a step?

**Evidence it is step size, not direction.** FuRA's train reward spread at α=5e-4 was
0.020–0.028, comparable to dense's ~0.027 — the σ=1e-3 perturbation produces perfectly
resolvable reward differences, so the ES gradient *direction* is being estimated fine. What
differs is how far each step travels. The ES update `θ += (α/N)·Σ Zₙεₙ` has scale-free
z-scores, so per-step motion is **linear in α**, and the measured footprints (§6) give

```
dense  ‖ΔW‖/‖W‖ = 5.0e-2      fura  ‖ΔW‖/‖W‖ = 4.0e-3      ->  12.5x smaller
alpha_matched = 12.5 * 5e-4 = 6.25e-3
```

**Design** (`scripts/es/sweep_fura_lr.sh`, GPU 2, sequential): sweep α at fixed σ, bracketing
12.5× geometrically (4× / 12.5× / 40×), plus one control that scales **σ as well as α**
(σ=1.25e-2, α=σ/2) to test whether a larger *exploration radius* buys anything beyond a larger
step. 20 iterations each with eval every 5 — in the completed runs, dense / insparse / ZO-Act /
FuRA were already cleanly separated by step 10–20, so 20 is enough to rank. Configs run in
order of expected informativeness so an early stop still answers the question.

The α=5e-4 run was stopped at step 42 and its log archived as the 1× baseline
(`logs/es/run4_fura_alpha5e-4_baseline.log`; it reached 63.4 @ step 40).

<!-- AUTO:FURASWEEP BEGIN -->

| α      | ×paper | σ      | MATH-500 @5 | @10  | @15    | @20            | train acc @20 | reward σ (mean) | status                                  |
| ------- | ------- | ------- | ----------- | ---- | ------ | -------------- | ------------- | ---------------- | --------------------------------------- |
| 5e-4    | 1×     | 1e-3    |             | 55.2 |        | 60.2           | 63.0          | 0.030            | done                                    |
| 2e-3    | 4×     | 1e-3    |             |      |        |                |               |                  | _queued_                              |
| 6.25e-3 | 12.5×  | 1e-3    | 65.4        | 66.4 | 70.0   | 69.2           | 74.6          | 0.024            | done                                    |
| 2e-2    | 40×    | 1e-3    | 68.2        | 66.2 | 67.2   | 61.0           | 58.0          | 0.028            | done                                    |
| 6.25e-3 | 12.5×  | 1.25e-2 | 66.2        | 70.4 | 74.2   | 73.4           | 63.4          | 0.052            | done                                    |
| 5e-4    | —      | 1e-3    | 70.8†      | 70.8 | 71.4† | **71.4** | 68.3          | 0.053            | **dense reference** (first 20 it) |

† dense was evaluated every 10 steps, so its @5/@15 cells repeat the neighbouring eval; the sweep uses every 5.

<!-- AUTO:FURASWEEP END -->

### 11.1 The 64-problem batch is the ceiling, not the method

Config 1 (α = 6.25e-3, the footprint-matched step) rescued FuRA: **65.4 @ 5 → 70.0 @ 15**,
against 55.2 @ 10 / 60.2 @ 20 for the paper α. So the small-core subspace was step-size-starved,
as predicted — but it stops ~2 pp short of dense, and pairing each eval with the train accuracy
at the same step (`scripts/es/train_vs_heldout.py`) shows why that comparison is subtle.

**Held-out MATH-500, averaged over every eval whose train accuracy fell in 73.5–76.5%:**

| run              | held-out @ matched train acc | n |
| ---------------- | ---------------------------- | - |
| insparse d=1%    | **72.0**               | 2 |
| dense (paper α) | **71.8**               | 8 |
| ZO-Act r=1       | 69.9                         | 5 |
| FuRA 12.5× α   | 69.2                         | 1 |

FuRA at 12.5× is ~2.6 pp below dense *at the same level of training progress*, so it is not
merely earlier on the same trajectory — though with n=1 and a ±2 pp eval SE this is suggestive,
not established. The 40× and σ+α-matched configs will say whether it holds.

**The bigger finding is in the gap column.** Every method's held-out score plateaus at 71–73
while train accuracy keeps climbing:

| run                       | train acc → | held-out →  | gap at start → end     |
| ------------------------- | ------------ | ------------ | ----------------------- |
| dense, step 10 → 150     | 66.9 → 77.7 | 70.8 → 71.6 | **+3.9 → −6.1** |
| insparse, step 10 → 60   | 64.9 → 75.6 | 66.4 → 72.2 | **+1.5 → −3.4** |
| FuRA 12.5×, step 5 → 20 | 69.2 → 74.6 | 65.4 → 69.2 | −3.8 → −5.4          |

Dense ends with train accuracy **6.1 pp above** held-out, having started 3.9 pp below it. That
is textbook overfitting of a **64-problem** training set, and it is the quantitative version of
the earlier "the batch is the binding constraint" claim: past ~step 40 the ES runs are buying
train accuracy that does not transfer. **No learning rate fixes this** — the remaining ~5 pp to
the paper's 78.0 has to come from more training problems (or resampling them each iteration),
not from a bigger step or a different subspace.

### 11.2 Stability edge: 12.5× in, 40× out

α = 2e-2 (40×) **diverges**. Its train accuracy peaks at *step 3* and then falls monotonically:

```
52.2  61.6  71.4  70.7  69.7  69.2  65.9  63.8  62.3  64.3
62.0  67.4  67.3  64.6  66.9  60.8  58.8  58.4  59.0  58.0
```

with held-out following it down (68.2 @5 → 66.2 @10 → 67.2 @15 → 61.0 @20). The reward spread
stays healthy at ~0.028 throughout, so this is not a dead gradient — the step simply overshoots
and walks the weights back downhill. Note 40× still produced the *fastest early climb* of any
FuRA config (68.2 by step 5) before degrading, which is the classic too-large-LR signature.

So the footprint-matched prediction bracketed correctly: **12.5× sits inside the stable region,
40× outside**. The optimum is somewhere in between, but locating it precisely matters less than
the next question — 20 iterations compares *transients*, not plateaus. Dense needed ~40 steps to
reach its 71.5 plateau and 150 to show its −6.1 pp overfitting gap. **The decisive test is
running FuRA at α = 6.25e-3 for the full 150 iterations** and comparing plateau-to-plateau.

### 11.3 Answer: yes — but scale σ, not α

The σ+α-matched config (σ = 1.25e-2, α = σ/2 = 6.25e-3) is the winner, and it does not merely
match dense — it **beats it, in a fraction of the steps**:

|                                | @5     | @10  | @15            | @20  | train acc @20  | reward σ       |
| ------------------------------ | ------ | ---- | -------------- | ---- | -------------- | --------------- |
| FuRA, α-only 12.5× (σ=1e-3) | 65.4   | 66.4 | 70.0           | 69.2 | 74.6           | 0.024           |
| **FuRA, σ+α matched**  | 66.2   | 70.4 | **74.2** | 73.4 | **63.4** | **0.052** |
| dense (paper ES)               | 70.8† | 70.8 | 71.4†         | 71.4 | 68.3           | 0.053           |

**74.2 at step 15** exceeds dense's best over all 150 iterations (73.4 @ step 40) and sits
2.7 pp above dense's plateau (71.5).

**The two FuRA configs share α and differ only in σ, and their train/test relationship
inverts.** α-only reaches train 74.6 / test 69.2 (gap **−5.4**); σ-scaled reaches train 63.4 /
test 73.4 (gap **+10.0**) — *lower* train accuracy, *higher* held-out. Dense at step 150 sits at
−6.1. So along the α axis FuRA memorises the 64-problem batch; along the σ axis it does not.

**Why σ is the regulariser, not just an exploration radius.** ES does not optimise R(θ) — it
optimises the Gaussian-smoothed `E_{ε~N(0,I)}[R(θ + σε)]`. σ *is* the smoothing bandwidth, so
raising it changes the objective to a genuinely flatter one, while raising α only takes bigger
steps on the same sharp objective. The reward-spread column is the fingerprint: the σ-scaled run
is the only FuRA config whose spread (0.052) matches dense's (0.053), i.e. σ — not α — sets the
effective signal scale. This is exactly the coupling the paper's `α = σ/2` encodes, and it is
why scaling α alone was never going to reproduce dense behaviour.

**Consequence for the whole study.** The §11.1 conclusion ("the 64-problem batch is the ceiling")
needs qualifying: it is the ceiling *at σ = 1e-3*. A larger smoothing radius partially escapes
it — FuRA at σ = 1.25e-2 reaches 74.2 where every σ = 1e-3 method plateaued at 71–73. **σ is now
the most promising knob for closing the remaining gap to the paper's 78.0, ahead of batch size.**

Running now: `run4b_fura_sigmatched_long` — 150 iterations at σ = 1.25e-2, α = 6.25e-3, to
compare plateau-to-plateau against dense's 71.5 and check whether the +10.0 pp generalisation
margin survives long training.

### 11.4 At *matched* footprint, the structured subspace beats dense

There are two comparisons in §11.3 and they say different things:

| comparison                           | holds fixed                      | varies             | result                                              |
| ------------------------------------ | -------------------------------- | ------------------ | --------------------------------------------------- |
| FuRA σ=1e-3 vs σ=1.25e-2           | α = 6.25e-3, subspace           | **σ**       | 69.2 → 73.4 @20 —*smoothing*                    |
| **FuRA σ=1.25e-2 vs dense σ=1e-3** | **‖ΔW‖/‖W‖ = 5.0e-2** | **subspace** | **74.2 vs 73.4 best; 63.4 vs 68.3 train acc** |

The second row is the sharper claim. FuRA's footprint is 12.5× smaller than dense's per unit σ
(4.0e-3 vs 5.0e-2), so **σ = 1.25e-2 puts FuRA at exactly dense's weight-space perturbation
magnitude**. At that matched footprint FuRA reaches a *higher* held-out score from a *much
lower* train accuracy — so the advantage is not "a bigger perturbation", it is the **subspace
itself acting as a regulariser**: confining ΔW to each input block's own left-singular subspace,
re-weighted by that block's singular values, is a structural prior that unconstrained full-weight
ES does not have.

That reframes the whole comparison. The step-indexed curves rank methods by *footprint*
(§7); at equal footprint they rank by *subspace*, and the ranking flips in favour of the
structured one.

**Test in flight:** ZO-Act r=1 at its own footprint-matched σ (its footprint is 4.2e-3 per unit
σ, so σ = 1.2e-2, α = σ/2 = 6e-3), 150 iterations on GPU 1
(`run2b_zoact_sigmatched_long`). ZO-Act at the paper σ plateaued at 71.0. If footprint-matching
lifts it to ~74 as it did FuRA, the rule is general — **every structured subspace needs its σ
rescaled to the dense footprint, and once rescaled they match or beat full-weight ES**. If it
does not, the gain is specific to FuRA's block structure and the two findings are separate.

### 11.5 Correction: FuRA's edge over dense is ~1 pp, not 3

The 150-iteration confirmation run finished the picture, and it **tempers §11.3/§11.4**. The
74.2 @ step 15 seen in the 20-iteration sweep was near the top of a noisy band, not a new level:

```
step   0    10    20    30    40    50    60    70    80    90   100   110
acc  53.2  70.4  73.4  74.0  72.0  73.8  73.8  73.4  73.0  71.0  71.8  71.8
```

Plateau comparison over each run's post-rise window:

| run                | window        | plateau mean    | sd   | best |
| ------------------ | ------------- | --------------- | ---- | ---- |
| FuRA σ+α matched | steps 30–110 | **72.73** | 1.10 | 74.0 |
| dense (paper ES)   | steps 40–150 | **71.82** | 1.19 | 73.4 |

**Gap = +0.92 pp**, against per-eval SE of ~2 pp. So the honest statement is: FuRA at matched
footprint is **at least as good as full-weight ES, plausibly ~1 pp better, but not the ~3 pp the
single step-15 point suggested**. The §11.3 claim "beats dense" holds only for *best-checkpoint*
(74.0 vs 73.4), which is within noise; the *plateau* claim is a ~1 pp edge that this experiment
cannot resolve from zero.

What does survive strongly:

* **σ, not α, is the operative knob** (§11.3) — 69.2 vs 73.4 @20 at identical α is far outside
  eval noise, and the train/test gap inversion (−5.4 vs +10.0) is a mechanism, not a fluctuation.
* **FuRA reaches dense's level from 1.3% of the parameters**, and gets there faster
  (73.4 by step 20 vs dense's 71.4).
* The §11.4 "structured subspace regularises" reading is *directionally* supported (FuRA holds
  72.7 at much lower train accuracy) but the effect size is ~1 pp, not 3.

**Sweep verdict:** step size alone (α at fixed σ) rescues FuRA only partly and destabilises past
~20×; joint σ+α scaling matches and exceeds dense. The `output_one_block` / `keep_frozen`
orientation was never the limitation. The α=2e-3 (4×) config was cancelled once σ proved to be
the operative variable — the α-only ladder (1× / 12.5× / 40×) already shows rise-then-diverge.

## 12. Alignment with the official implementation

Source: [github.com/VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale)
(`es_at_scale/train.py`, `trainer/es_trainer.py`, `utils/worker_extension.py`,
`utils/reward_shaping.py`, `template_function/apply_template.py`), cloned and read 2026-08-22.

Their documented math command:

```
--task math --model-name Qwen/Qwen2.5-Math-7B --sigma 0.001 --population-size 30
--n-iterations 500 --eval-freq 5 --train-dataset datasets/train/math_lvl3to5_8k
--batch-size 1024 --mini-batch-size 1024 --max-tokens 3000 --n-vllm-engines 8
```

### What matched already

| item                          | official                                                                                                                     | ours                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Prompt template               | `qwen_math_template()` — literal `<\|im_start\|>system                                                                    |                                                                       |
| Please reason step by step…` | byte-identical (we render it via the tokenizer chat template; verified against their dataset's pre-rendered `input` field) |                                                                       |
| Reward shaping                | `z_score()`: `(r − mean)/(std + 1e-8)`                                                                                  | identical                                                             |
| α                            | `alpha = sigma/2` when unset                                                                                               | identical (**confirms the α = σ/2 reading, not α = σ**)     |
| Population                    | 30                                                                                                                           | 30                                                                    |
| Decoding                      | train & eval `T=0.0, top_p=1.0`, `seed = global_seed + iteration`                                                        | identical                                                             |
| Per-iteration seeds           | `np.random.default_rng(global_seed + iteration).integers(0, 2**30, N)`                                                     | identical                                                             |
| Population sees               | the*same* batch within an iteration                                                                                        | identical                                                             |
| Precision                     | `dtype="bfloat16"`                                                                                                         | bfloat16 (+ fp32 coefficient master — a strict improvement, see §6) |
| Grader                        | mathd +`math_verify` lineage                                                                                               | `ttrl_math` (same lineage); base 51.6 vs their 53.0                 |

### What did not match — and the fix

| item                     | official                                                                                               | ours (was)                                                  | status                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training batch** | **1024**, **resampled every iteration** from an 8.5k pool via `DataLoader(shuffle=True)` | **one fixed 64-problem batch for all 150 iterations** | **fixed** — `es.train_batch_size` now resamples per iteration (`_draw_batch`, sampling-without-replacement over shuffled epochs) |
| Train max tokens         | 3000                                                                                                   | 1536                                                        | fixed — aligned runs use 3000 for train*and* eval                                                                                        |
| Iterations               | 500                                                                                                    | 150                                                         | not affordable (see below)                                                                                                                  |
| eval-freq                | 5                                                                                                      | 10                                                          | aligned runs use 5                                                                                                                          |
| vLLM engines             | 8                                                                                                      | 1                                                           | hardware                                                                                                                                    |

**The batch is the headline discrepancy, and it is exactly the mechanism §11.1 measured.** A
fixed 64-problem batch is 16× smaller than the official one *and* never refreshed, so ES can
memorise it — which is precisely what we saw (dense: train 66.9 → 77.7 while held-out went
70.8 → 71.6, a gap swing of +3.9 → −6.1 pp). With per-iteration resampling that failure mode
cannot occur: no problem is seen often enough to be memorised.

**Why not simply run the official config.** Measured on 1×H100: a generation pass costs
`≈ 17 + 0.07·B` seconds at 3000 tokens, so batch 1024 × 30 perturbations ≈ 44 min/iteration →
500 iterations ≈ **17 GPU-days**. Their 8-engine setup does it in ~1/8 of that. We use
**batch 128 resampled**, which keeps the resampling fix (the important part) at 13 min/iteration.

<!-- AUTO:ALIGNED BEGIN -->

<!-- AUTO:ALIGNED END -->

## 13. BP counterpart — fixed-spectrum training with true gradients

> Four GRPO runs on the *same* task as §1–§7 (Qwen2.5-Math-7B, MATH lvl 3–5 →
> MATH-500, binary `\boxed{}` reward), optimised with AdamW + backprop instead of
> forward-only ES, so ES-vs-BP is a controlled comparison of the **optimiser**, not
> the task. wandb project **`BP-q2p5-7b`**. Started 2026-08-23, GPUs 6+7.

### 13.1 Getting the constraint for free under autograd

ES could enforce the fixed spectrum by *constructing* each perturbation inside
`F(W0)` ([§10.2](#102-the-orbit-form--feasibility-without-retraction)). BP cannot:
the optimizer proposes an arbitrary step. ISO's own answer is to step the frames
freely and project back with an fp64 polar retraction each step — affordable for
one gradient step, but it needs a Riemannian layer bolted onto the optimizer.

Instead we **parameterise the orthogonal factors**: every `C` is `Cay(Ω)` for a
*trainable skew* `Ω`, `Cay(X) = (I − X/2)⁻¹(I + X/2)`. Then

* `Ω = 0` at init ⇒ `C = I` ⇒ step 0 is the pretrained model **bit-exactly**;
* the constraint holds for **any** value the optimizer produces, so plain AdamW and
  plain FSDP work unchanged — no Riemannian optimizer, no retraction, no projection,
  and nothing to drift off (the `_iso_recondition` machinery ES needs has no
  analogue here);
* `Ω` is stored square and skew-symmetrised in the forward; the symmetric half is
  in the kernel of the map and receives *exactly* zero gradient, so the effective
  dimension is `b(b−1)/2` per block — half the stored count. Both are reported.

Crucially the base weight is **never materialised during training**. Each mode is
two cheap orthogonal transforms wrapped around the untouched frozen linear:

| mode           | forward                                             | trainable (stored / manifold) | % of 7.6 B   |
| -------------- | --------------------------------------------------- | ----------------------------- | ------------ |
| `dense`      | `F.linear(x, W)` — full FT baseline              | 7,615,616,512                 | 100%         |
| `iso`        | `blkrot_out( F.linear( blkrot_in(x), W0 ) )`      | 322,961,408 / 160,247,808     | 4.24 / 2.10% |
| `isobtt`     | `F.linear( blkrot_in(x), W0 )`, contiguous blocks | 117,039,104 / 58,458,112      | 1.54 / 0.77% |
| `isobtt_mix` | `isobtt` + orthogonal mixer `M ∈ O(n_blk)`     | 118,024,704 / 58,950,912      | 1.55 / 0.77% |

Overhead is `b·|W|` flops/token against the base linear's `2·out·in` — **+3.6%**
for `iso` at b=128, **+0.9%** for the `isobtt*` modes.

### 13.2 `isobtt` in its simplest form, and the orthogonal input mixer

The ES worker builds `isobtt` from a per-block SVD as `A_j Cay(·) R_j`. For BP we
use the equivalent **right-rotation** form

```
W[:, blkⱼ] = W0[:, blkⱼ] · Cⱼ ,   Cⱼ = Cay(Ωⱼ) ∈ O(b)
```

which spans the same family — `σ(W0ⱼ Cⱼ) = σ(W0ⱼ)` either way — but needs **no SVD
at all**. That removes the frozen `A`/`R0` tensors *and* the 1.6e-3 bf16
reconstruction floor: identity init is bit-exact here, where the ES `isobtt`/`fura`
runs start at 53.2 instead of 51.6 ([§7](#7-results)).

**Input mixing** (4th arm) relaxes the block-locality of Remark 5.2 — the constraint
that block *k* of the input only ever feeds core *k*. The referenced ablation
(lora-without-regret `docs/exp_results/lift_commonsense.md`) learns a *free* `n×n`
mixer. Here `M` is constrained to `O(n_blk)`, because as a full-input operator the
mixer is `M ⊗ I_b`, which is orthogonal **iff `M` is** — so an orthogonal `M` keeps
the global spectrum of `W` exactly fixed and the arm stays inside `F(W0)`, whereas a
free `M` would leave the family and turn the arm into "isobtt + capacity" rather
than a locality ablation. `M` is identity-init, adds `n_blk(n_blk−1)/2` effective
params per layer (+0.013% of the model), and is folded into the dense export.

### 13.3 Verification — `scripts/es/test_iso_bp.py`

All PASS, on a real (tiny) Qwen2 model through the actual adapter:

* **identity init reproduces the base weights bit-exactly** (0.0e+00) for all three
  modes, fp32 and bf16 — so every arm starts from the same model.
* **the spectrum is fixed after 5 real AdamW steps**: `‖Δσ‖/‖σ‖ ≤ 9.3e-8` while the
  weights move `‖ΔW‖/‖W‖ = 0.48–0.75`. This is the claim, tested against an actual
  optimizer rather than a hand-built perturbation.
* `Cay(sΩ)` orthogonal to 2e-6 for `s ∈ {1e-4, 1e-2, 1, 10}`; the gradient w.r.t. `Ω`
  is skew (the symmetric half is provably in the kernel); `M` stays in `O(n_blk)`.
* **`materialize()` == `forward()` at random Ω** (3.5e-7 fp32) — the dense weight
  handed to vLLM is the policy that was trained.

Three real bugs this caught, all before any 7B GPU-hours:

1. **`materialize()` applied `C_Lᵀ` where `forward()` applied `C_L`.** Invisible at
   Ω=0 (every rotation is the identity), so the first version of the test passed;
   in a real run the vLLM rollout weights would have silently disagreed with the
   trained policy. The check now runs at *non-zero* Ω.
2. **`export_for_vllm` returned views into FSDP's gathered flat parameter.** The
   caller wraps it in `summon_full_params`, which frees that storage on exit, so
   vLLM read tensors of `storage size 0`. Everything returned must be cloned.
   (BlockTT never hit this — its production runs are FSDP2, which takes the
   fallback path.) Exports are cast to bf16 as well, which is what vLLM stores.
3. **Missing `enable_input_require_grads()`.** With the embeddings frozen, the
   hidden states entering each *checkpointed* decoder block carry no `grad_fn`, so
   `torch.utils.checkpoint` returns a detached output and the loss has no graph at
   all. Same fix LoRA and BlockTT apply.

Plus two smaller ones: `iso` was missing from `PEFTConfig.from_omegaconf`'s
`sub_specs` (the sub-config arrived as a plain dict), and the modules originally
declared `Ω` in fp32 while the actor is bf16/fp32-uniform, which FSDP1's
`FlatParameter` rejects.

### 13.4 Setup

Identical to §3's task, differing only where BP requires it:

| Knob               | Value                                                    | Note                                                    |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------- |
| Estimator          | GRPO, rule-based binary reward, no teacher               | same grader as ES                                       |
| Sampling           | **T = 1.0, n = 8 responses**                       | ⚠️ ES is greedy; GRPO needs sampling spread           |
| Batch / mini-batch | 64 / 64 prompts                                          | 8,890 rows ⇒**138 steps** = 1 epoch              |
| Response budget    | 3,072 train and eval                                     | ES used 1,536 train / 3,000 eval                        |
| LR                 | dense**1e-6**, ISO modes **5e-6**            | matched on per-step relative weight motion — see below |
| Precision          | module fp32, FSDP `param_dtype=bf16`                   | fp32 optimizer master; uniform dtype for FlatParameter  |
| Hardware           | 2 × H100 NVL (GPUs 6+7) FSDP, 4 arms back-to-back       | 61–65 GB/GPU for dense                                 |
| Measured           | **157.5 s/step** ⇒ ~6 h/arm, **~24–30 h total** | incl. MATH-500 eval every 10 steps                      |

**LR matching.** AdamW moves each coordinate by ≈ lr, so full FT moves
`‖ΔW‖/‖W‖ ≈ lr/0.02 = 50·lr` (5e-5 at 1e-6). For the ISO modes the step lands in
the rotation generator, where entries of size lr give `‖ΔW‖/‖W‖ ≈ √b·lr` (≈11·lr at
b=128, ≈8·lr at b=64) — so 5e-6 puts all three ISO arms within ~2× of the dense
per-step motion. **This is analytic, not swept**, and §7 showed step size dominates
everything on this task (`fura` moved −12.25 → +0.82 pp on step size alone), so it
is the first knob to revisit if an arm looks mis-scaled.

<!-- BP:RESULTS BEGIN -->

### 13.5 Results — half the arms landed

MATH-500 **mean@4 at T=1.0** (verl `val-core`), before training and every 10 steps.
**Not comparable to the greedy numbers in §7**: the same base model reads 19.4 here
and 51.6 greedy.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 138 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 19.5 | 55.9 | 66.0 | ✗ | | | | | | | | | | | |
| iso | 19.4 | **63.4** | 66.0 | ✗ | | | | | | | | | | | |
| isobtt | 19.4 | 32.7 | 56.4 | 62.3 | 63.8 | 64.3 | 66.3 | 67.7 | 50.1 | **17.6** | 28.4 | 36.2 | 42.6 | 67.8 | 69.0 |
| isobtt_mix | 19.4 | 40.0 | 61.5 | 62.3 | 65.1 | 67.2 | 66.0 | 67.0 | 69.3 | 69.0 | **72.7** | 72.0 | 72.3 | 71.7 | **72.0** |

✗ — **`dense` was SIGTERM'd at step 22 and `iso` died of a CUDA illegal memory access
at step 20.** Both need re-running; until then the BP leg answers nothing. (My
`run_bp_all.sh` reported `exit 0` for both because it evaluated `$?` after an `echo`
— fixed.)

Three provisional readings:

* **BP is far cheaper than ES for the same accuracy.** `dense` and `iso` both hit
  66.0 by step 20 — ~50 min of wall clock, against ~15 h for an ES arm to plateau.
  That is the zeroth-vs-first-order gap (ES spends 30 forward passes per step
  estimating what backprop gets exactly), and it is the reason to run the BP leg.
* **`iso` tracks `dense` here too**, and leads at step 10 (63.4 vs 55.9) — consistent
  with §7, but two evals on a dead run prove nothing.
* **The orthogonal input mixer helps**, opposite to the free-`M` ablation it is
  modelled on (which lost 0.59 Avg). `isobtt_mix` rises monotonically to 72.0 while
  `isobtt` collapses 67.7 → 17.6 between steps 70 and 90 before recovering to 69.0.
  With one seed and an **unswept LR**, "`isobtt` is unstable at 5e-6" is the safer
  statement than "mixing helps".

Train reward agrees: `isobtt` 0.23 → 0.70 (s61) → **0.14 (s91)** → 0.65 (s131);
`isobtt_mix` 0.23 → 0.70 (s61) → **0.76 (s111)** → 0.68. Timing 132–158 s/step
(`isobtt` 5 h 45 m, `isobtt_mix` 6 h 48 m for 138 steps).

<!-- BP:RESULTS END -->
