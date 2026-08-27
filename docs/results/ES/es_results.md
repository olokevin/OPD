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

**Catastrophic forgetting — a null on MATH ([§14](#14-catastrophic-forgetting--does-the-perturbation-subspace-decide-it)).**
[arXiv:2601.20861](https://arxiv.org/abs/2601.20861) reports ES loses ~10% of its prior
ability (HellaSwag) while learning a new task, blaming dense, large-norm updates. On
this task **no arm forgets — `dense` included**: all six gain **+20 to +22 pp** on
MATH-500 and move an 8-benchmark prior mean by **−0.49 to +0.20 pp** (SE ≈ 0.42), where
the paper's effect would be ≈6.5 pp on HellaSwag. The update statistics reproduce
(`dense` is 7.2% sparse, LayerNorm sparsest — their Figure 4) but do not *predict*:
drift spans 100× and sparsity 7% → 99% across arms with no ordering of the deltas. The
structured arms do land on GRPO's side of both axes at full accuracy (`insparse` 99.2%
sparse, +21.2 pp). **The paper's own task pair does not reproduce it either**
([§14.5](#145-countdown--hellaswag-the-papers-own-task-pair)): on Countdown →
HellaSwag with *their* model (Qwen2.5-1.5B-Instruct), hyperparameters and horizon,
`dense` ES learns Countdown **8.0 → 42.0%** while HellaSwag goes **59.70 → 60.20** and
the prior mean moves **−0.11 pp** — 0.3% relative against their ≈10%. `iso` reaches a level `dense` never
does (**47.5%**) and, at *matched* Countdown accuracy, has lost exactly the same prior
ability (−0.11 pp at 38%). **What does order forgetting is ‖ΔW‖_F, not sparsity and not
the subspace**: across the three finished arms, drift 3.5e-2 / 4.3e-2 / **9.5e-2**
(`dense`/`iso`/`fura`) maps monotonically onto prior-mean −0.29 / −1.34 / **−3.20** and
generative-HellaSwag +0.1 / −2.4 / **−14.3**, while sparsity is unrelated. `fura` at
σ=1.25e-2 is the only arm that forgets — and it also barely learns (Countdown 18.5% vs
42.0/47.5), the signature of an over-large step; a σ=1e-3 rerun is testing that. The
generative-probe explanation for the paper's result is **falsified** (0% unparsed,
`dense` generative accuracy unchanged); the remaining candidate is simply that their ES
run sat at a larger ‖ΔW‖ than ours, which they never report.

**BP leg — half done, not yet conclusive.** `isobtt` and `isobtt_mix` finished
138/138, but **`dense` (SIGTERM @22) and `iso` (CUDA illegal memory access @20) died
early**. On what did run, BP reaches ES-level accuracy in ~1 h vs ~15 h, `iso` again
tracks `dense`, and the orthogonal input mixer is the best and most stable arm while
`isobtt` collapsed mid-run ([§13.5](#135-results--half-the-arms-landed)).

⚠️ **ES and BP numbers are not comparable as they stand**: ES evaluates greedy (n=1),
BP evaluates mean@4 at T=1.0. The same base model reads **51.6 greedy vs 19.4 sampled**.

### Leaderboard

Best configuration of each method, ES side, on the 150-iteration fixed-batch protocol
(greedy MATH-500). Ranked by **plateau mean over steps ≥ 40** — the honest statistic,
since "best" is a max over 16 noisy evals:

| # | Method | Best config | Trainable | Base | **Plateau (≥40)** | Best @ step |
|---|---|---|---|---|---|---|
| 1 | **fura** — BTT small core | σ=1.25e-2 (σ+α matched) | 97.8 M · 1.28% | 53.2 | **72.68 ± 0.90** | 74.0 @ 30 |
| 2 | **iso** — fixed spectrum | σ=5e-2 | 141.1 M† · 1.85% | 51.6 | **72.42 ± 0.78** | 74.0 @ 60 |
| 3 | insparse d=1% | σ=1e-3 | 65.4 M · 0.86% | 51.6 | 72.07 ± 0.70 | 73.4 @ 80 |
| 4 | **isobtt** — fixed per-block spectrum | σ=5e-2 | 48.5 M† · 0.64% | 53.2 | 71.95 ± 0.94 | 73.4 @ 120 |
| 5 | dense (paper ES) | σ=1e-3 | 7.62 B · 100% | 51.6 | 71.82 ± 1.19 | 73.4 @ 40 |
| 6 | zoact r=1 | σ=1e-3 | 1.39 M · 0.018% | 51.6 | 70.50 ± 0.94 | 72.2 @ 130 |

† manifold dimension searched per step, not a coefficient count ([§2](#2-the-six-runs)).
`fura`/`isobtt` start from 53.2 rather than 51.6 (bf16 BTT reconstruction, [§6](#6-numerical-health)),
so read their deltas against their own base.

**Ranks 1–5 span 0.86 pp against a per-eval SE of 2.24 pp — they are one tie.** The only
separable result is that rank-1 `zoact` is genuinely behind. Everything else says the
same thing as [§7](#7-results): at matched weight-space footprint the subspace barely
matters, and *every* structured method reaches full-weight ES from ≤1.3% of the
parameters.

Off-protocol but the strongest operating point found so far — dense ES with the
official **resampled** batch ([§12](#12-alignment-with-the-official-implementation)),
20 iterations only, so no plateau statistic:

| dense, aligned/resampled | σ=1e-3 | **σ=2e-3** | **σ=4e-3** | σ=8e-3 |
|---|---|---|---|---|
| best @ step | 71.2 @ 10 | **73.6 @ 10** | **74.4 @ 15** | 72.0 @ 10 |

BP side (GRPO), **mean@4 at T=1.0 — a different metric**; the same base model reads
19.4 here and 51.6 greedy, so these numbers do not belong in the table above:

| Method | Steps | Base | Best @ step | Status |
|---|---|---|---|---|
| **isobtt_mix** — + orthogonal input mixer | 138/138 | 19.4 | **72.7 @ 100** | complete, stable |
| isobtt | 138/138 | 19.4 | 69.0 @ 138 | complete, collapsed mid-run |
| dense | 22 ✗ | 19.5 | 66.0 @ 20 | SIGTERM — re-run needed |
| iso | 20 ✗ | 19.4 | 66.0 @ 20 | CUDA illegal memory access — re-run needed |

### Curves — best variant of each method

Greedy MATH-500, eval every 10 iterations. One column per method, each showing the
config that tops the leaderboard above.

| step | dense | zoact | insparse | fura | iso | isobtt |
|---|---|---|---|---|---|---|
| 0 | 51.6 | 51.6 | 51.6 | 53.2 | 51.6 | 53.2 |
| 10 | 70.8 | 58.8 | 66.4 | 70.4 | 70.2 | 68.2 |
| 20 | 71.4 | 66.0 | 68.4 | **73.4** | 72.4 | 68.0 |
| 30 | 72.4 | 66.4 | 71.0 | **74.0** | 71.4 | 71.2 |
| 40 | **73.4** | 70.2 | 71.4 | 72.0 | 71.8 | 70.8 |
| 50 | 69.6 | 69.8 | 71.8 | 73.8 | 72.2 | 71.0 |
| 60 | 71.6 | 70.4 | 72.2 | 73.8 | **74.0** | 72.8 |
| 70 | 73.0 | 70.6 | 72.4 | 73.4 | 72.2 | 71.0 |
| 80 | 72.0 | 70.0 | **73.4** | 73.0 | 72.6 | 72.6 |
| 90 | 73.2 | 69.0 | 73.0 | 71.0 | 73.2 | 70.6 |
| 100 | 71.4 | 69.4 | 72.2 | 71.8 | 71.8 | 72.2 |
| 110 | 70.6 | 70.4 | 72.2 | 71.8 | 71.0 | 71.4 |
| 120 | 72.2 | 71.8 | 72.2 | 73.6 | 72.2 | **73.4** |
| 130 | 70.4 | **72.2** | 71.6 | 72.8 | 72.4 | 72.6 |
| 140 | 72.8 | 70.8 | 71.6 | 72.8 | 73.2 | 72.2 |
| 150 | 71.6 | 71.4 | 70.8 | 72.4 | 72.4 | 72.8 |

Shape, not level, is what separates them: `fura` and `iso` are **fastest** (73.4 / 72.4
by step 20, where dense is at 71.4 and needs 40 to reach 73.4); `zoact` is the clear
laggard early (58.8 @ 10) and never fully closes; `insparse` and `isobtt` rise slowly
but land in the same band. After ~step 40 every column is flat inside ±1.5 pp — which
is [§11.1](#111-the-64-problem-batch-is-the-ceiling-not-the-method)'s point that the
fixed 64-problem batch, not the method, is the ceiling.

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
7. ~~**Test the generative-probe hypothesis.**~~ **Done — falsified**
   ([§14.5.2](#1452-the-generative-probe-hypothesis-is-falsified)).
8. **`fura` at σ=1e-3 on Countdown** (running, GPU 1) — the disambiguating run for
   [§14.5.1](#1451-drift-is-the-axis--a-clean-dose-response): if `fura` then learns like
   `dense` and stops forgetting, the whole `fura` effect is step size and the subspace is
   exonerated. `isobtt` still running on GPU 2.
9. **Re-run the MATH leg's retention with the generative probe.** It is 4–6× more
   sensitive than log-likelihood ranking ([§14.5.2](#1452-the-generative-probe-hypothesis-is-falsified)),
   so the ±0.5 pp nulls in [§14.4](#144-result--nothing-forgets-dense-included) may be
   understating real (small) movements. Cheap: the six checkpoints are already
   materialised under `/data/yequan/es/materialized/`.
10. **Sweep σ against retention directly.** [§14.5.1](#1451-drift-is-the-axis--a-clean-dose-response)
    makes ‖ΔW‖_F the control variable; a single arm at σ ∈ {1e-3, 5e-3, 1.25e-2, 5e-2}
    would turn the three-point dose-response into a curve and give an operating-point
    recommendation for edge training.

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

## 14. Catastrophic forgetting — does the perturbation subspace decide it?

> **Motivation: training on edge devices.** ES is attractive there because it needs no
> gradients, no optimizer state and no activation memory — but an on-device learner is
> only useful if it can keep learning without destroying what the model already knows.
> *Evolutionary Strategies lead to Catastrophic Forgetting in LLMs*
> ([arXiv:2601.20861](https://arxiv.org/abs/2601.20861),
> `docs/papers/26_Evolutionary Strategies lead to Catastrophic Forgetting in LLMs.pdf`)
> says ES fails exactly that test. This section asks whether that verdict survives when
> ES is restricted to a structured subspace — the six arms of [§7](#7-results) and
> [§10](#10-iso-fixed-spectrum-es) already sit at a matched MATH-500 plateau while
> differing by two orders of magnitude in exactly the update statistics the paper
> blames.

### 14.1 What the paper claims, and the handle it gives us

Setup: Qwen2.5-1.5B-Instruct / Llama-3.2-1B-Instruct, 200 training examples from
Countdown / GSM8K / MATH / OlympiadBench, ES (the Qiu et al. implementation this repo
reproduces) vs verl GRPO, population 30. Findings:

1. **Parity on the new task.** ES lands within 3–4 pp of GRPO on every task — but GRPO
   wins almost all of them, contradicting the ES-at-Scale claim of an ES advantage.
2. **Forgetting.** With Countdown as the new task and **HellaSwag** as the prior-ability
   probe, ES's prior accuracy falls monotonically with iteration — **≈10% below its own
   best** — and keeps falling *after* Countdown has converged (~200 iterations). GRPO's
   prior accuracy is flat. Their Figure 1 is a convex Pareto front for ES and a cluster
   in the top-right corner for GRPO.
3. **Proposed mechanism.** ΔW = W_finetuned − W_base is measured two ways:
   **Frobenius norm** (ES drifts ~10³× further than GRPO after 500 iterations, growing
   monotonically) and **sparsity** — the fraction of entries with |Δ| < τ = 10⁻⁶, which
   is ≈95% for GRPO across every layer and parameter group but near zero for ES.
   Dense, large-norm updates ⇒ global interference ⇒ forgetting.

**How they actually measure it** (§3.2, Limitations, A.2, A.3 — read in full 2026-08-25):

* **One probe, one setting.** HellaSwag is the *only* prior-task benchmark, on the single
  pair Qwen2.5-1.5B-Instruct + Countdown. Table 1's four tasks × two models are new-task
  accuracy only and play no part in the forgetting analysis. They concede the point in
  Limitations: tracking "performance on one task … does not fully capture multi-facetted
  loss of performance".
* **Per-checkpoint curves.** "We evaluate task performance across each checkpoint of our
  trained models" → Fig. 1 (Countdown-vs-HellaSwag Pareto scatter, coloured by iteration)
  and Fig. 2 (HellaSwag vs iteration). The headline is a **≈10% drop relative to the best
  observed** prior score, continuing after Countdown converges at ~200 iterations.
* **Hyperparameters (A.3):** population 30, σ=1e-3, α=5e-4, **max_tokens 1024**.
  A.2.2 also notes they use **fp16** (not bf16) plus the Qwen chat template.

**What the paper never specifies: the HellaSwag scoring protocol.** There is no mention of
lm-eval-harness, of log-likelihood ranking vs generative answering, of `acc` vs `acc_norm`,
of few-shot count, subset size, or normalisation — nor any absolute HellaSwag number
(figures only), eval frequency, or seed repeats. This is why
[§14.5.2](#1452-the-generative-probe-hypothesis-is-falsified) had to *guess* at the
protocol in order to test it.

**A confound the paper names in its own appendix.** Their GRPO runs with an explicit
**KL coefficient β = 0.001** (A.2.1), and A.4.1 attributes GRPO's flat KL to exactly that:
"the explicit KL-regularization factor in GRPO, preventing continuous drifts from the base
model". So the comparison sets an **anchored** optimiser against an **unanchored** one and
reads the difference as a property of being gradient-free. That is the same variable
[§14.5.1](#1451-drift-is-the-axis--a-clean-dose-response) isolates — a KL penalty is a
direct bound on ‖ΔW‖ — and it is why the companion paper's fix (Anchored Weight Decay, a
pull toward θ₀) works: it gives ES the anchor GRPO already had. On this reading their
result is not "ES forgets" but "**an unanchored optimiser at a large enough step forgets**",
which our dose-response supports and which is a tuning statement, not an indictment of ES.

A companion paper, *Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies*
([arXiv:2605.30148](https://arxiv.org/abs/2605.30148)), argues the loss is *drift*
rather than irreversible forgetting, attributes it to "random-walk behaviour in weakly
constrained directions of the weight space", and fixes it with **Anchored Weight Decay**
(a pull toward θ₀).

**Why this repo can test the mechanism directly.** The paper's two axes are properties
of ΔW, and [§7](#7-results)'s six arms were built to differ in exactly that while
reaching the *same* MATH-500 plateau (71.8–72.7, one statistical tie). So the confound
that usually blocks this question — "the arm that forgets less also learned less" — is
already controlled. If forgetting tracked ‖ΔW‖ and density, `insparse` and `zoact`
should be safe and `dense`/`fura`/`iso`/`isobtt` should not.

### 14.2 Harness

The ES trainer only stores coefficients (`es_coef_best.pt`), so measuring anything
outside its own MATH-500 loop needs the weights back.

| What | Path |
| --- | --- |
| Coefficients → HF checkpoint + ΔW statistics | `scripts/es/materialize_es_ckpt.py` |
| …for all six arms | `scripts/es/materialize_all.sh` |
| Prior-knowledge + MATH-500 eval of one model | `scripts/es/eval_forgetting.sh` |
| …for a list of arms | `scripts/es/run_forgetting_sweep.sh` |
| Tables in this section | `scripts/es/collect_forgetting.py` |
| **In-loop** forgetting probe (curves) | `verl/verl/trainer/es/forget_eval.py`, `es.forget_tasks` |
| Gate for the in-loop probe | `scripts/es/test_forget_eval.py` |
| Generative (format-sensitive) HellaSwag probe | `scripts/es/eval_hellaswag_gen.py` |
| Countdown data / launcher / chain | `scripts/es/prepare_countdown_data.py`, `run_countdown_es.sh`, `chain_countdown.sh` |
| Curve runs | `scripts/es/run_forget_curves.sh` |
| LoRA-ES chain / gate | `scripts/es/chain_lora_math.sh`, `scripts/es/test_lora_es.py` |
| Materialized checkpoints / eval JSON | `/data/yequan/es/materialized/<arm>`, `/data/yequan/es/forgetting/<arm>` |

Reconstruction runs the **trainer's own** `StructuredESMixin.init_es_state` +
`es_restore` against a stub that presents the base model in vLLM's *fused* layout
(`qkv_proj`, `gate_up_proj`), so the block sizes, the frozen `A`/`R0` factors and the
SVD convention cannot drift from what the run used; the fused weights are then un-fused
back to HF names. `dense`/`iso` store a full master and need no reconstruction, which
makes them a free end-to-end check on the path.

**Prior-ability suite.** HellaSwag (the paper's probe) plus the standard commonsense
battery — PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ — and MMLU for
world knowledge. All 0-shot log-likelihood ranking through lm-eval 0.4.12
(`acc_norm` where the task defines it, else `acc`), so nothing depends on the model
still being able to *generate*. New-task accuracy is re-measured with the trainer's own
prompt processor and grader (greedy, 3,000 tokens, `ttrl_math` `fast=True`), so it is
comparable to the `eval/accuracy` curves in [§7](#7-results).

**A caveat on the τ = 10⁻⁶ sparsity metric.** bf16 has ~1.6e-4 ULP at |w| ≈ 0.02, so
for a bf16 checkpoint "sparsity at 10⁻⁶" largely counts coordinates that moved *less
than one ULP* rather than coordinates the algorithm left alone. We therefore report the
paper's number *and* the exact-zero fraction (`Δ = 0`, i.e. bit-identical to base); on
these checkpoints the two agree to <0.3 pp, so the metric is measuring what it claims.

### 14.3 Update geometry of the six arms

ΔW = W_arm − W_base at each arm's best-MATH-500 checkpoint, in the paper's own terms.
The six arms span **100× in drift** and **7% → 99% in sparsity** while landing within
1.8 pp of each other on the new task:

<!-- FORGET:GEOM BEGIN -->

| Arm | ‖ΔW‖_F/‖W‖_F | sparsity(τ=1e-6) | untouched (Δ=0) | MATH-500 Δ | prior Δ |
|---|---|---|---|---|---|
| dense (paper ES) | 2.03e-02 | 7.2% | 7.2% | +22.0 | +0.00 |
| zoact r=1 | 4.70e-04 | 95.9% | 95.7% | +20.2 | +0.20 |
| insparse d=1% | 2.65e-03 | 99.2% | 99.2% | +21.2 | -0.07 |
| fura small-core | 4.78e-02 | 17.1% | 17.1% | +21.6 | -0.49 |
| iso fixed-spectrum | 2.87e-02 | 18.5% | 18.5% | +21.8 | -0.23 |
| isobtt fixed-spec | 4.04e-02 | 17.4% | 17.4% | +21.6 | -0.19 |

Layerwise sparsity by parameter group — the analogue of the paper's Figure 4:

| Arm | Q | K | V | WO | MLP | LayerNorm | Embed |
|---|---|---|---|---|---|---|---|
| dense (paper ES) | 37% | 34% | 15% | 8% | 8% | 92% | 5% |
| zoact r=1 | 97% | 97% | 98% | 90% | 96% | 100% | 100% |
| insparse d=1% | 100% | 100% | 100% | 99% | 99% | 100% | 100% |
| fura small-core | 52% | 52% | 52% | 4% | 3% | 100% | 100% |
| iso fixed-spectrum | 52% | 52% | 52% | 5% | 5% | 100% | 100% |
| isobtt fixed-spec | 52% | 52% | 52% | 4% | 4% | 100% | 100% |

<!-- FORGET:GEOM END -->

**This reproduces the paper's characterisation of `dense` ES exactly.** Our `dense`
arm's update is 7.2% sparse — the paper's "ES updates have very low sparsity" — against
the ~95% they measure for GRPO, and the per-group profile matches too: LayerNorm is the
sparsest group (92%) and everything else is dense, which is their Figure-4 finding.

**And two of our arms sit on GRPO's side of both axes while learning just as much.**
`insparse d=1%` is **99.2% sparse** with 7.7× less drift than `dense`, `zoact r=1` is
**95.9% sparse** with 43× less drift — both squarely in the range the paper reports for
GRPO — yet they gain +21.2 and +20.2 pp on MATH-500 against `dense`'s +22.0. So a
structured subspace does deliver the update geometry the paper says is protective,
without giving up the new task.

### 14.4 Result — nothing forgets, `dense` included

<!-- FORGET:MATH BEGIN -->

| Arm | σ | best @ | MATH-500 | HellaSw | PIQA | WinoG | ARC-e | ARC-c | OBQA | BoolQ | MMLU | **Prior mean** | Δ prior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base Qwen2.5-Math-7B | - | - | 52.0 | 65.4 | 74.2 | 65.0 | 74.0 | 50.3 | 39.0 | 74.8 | 57.9 | **62.57** | +0.00 |
| dense (paper ES) | 1e-3 | 40 | 74.0 | 65.3 | 74.5 | 65.2 | 74.3 | 50.6 | 38.8 | 74.2 | 57.7 | **62.57** | +0.00 |
| zoact r=1 | 1e-3 | 130 | 72.2 | 65.4 | 74.4 | 64.4 | 74.5 | 50.5 | 40.2 | 74.9 | 57.8 | **62.76** | +0.20 |
| insparse d=1% | 1e-3 | 80 | 73.2 | 65.4 | 74.1 | 64.5 | 74.0 | 50.4 | 39.2 | 74.6 | 57.7 | **62.50** | -0.07 |
| fura small-core | 1.25e-2 | 30 | 73.6 | 65.1 | 74.0 | 64.5 | 73.2 | 49.2 | 40.4 | 73.3 | 56.9 | **62.08** | -0.49 |
| iso fixed-spectrum | 5e-2 | 60 | 73.8 | 64.9 | 73.8 | 64.5 | 74.3 | 50.0 | 38.4 | 74.8 | 57.9 | **62.33** | -0.23 |
| isobtt fixed-spec | 5e-2 | 120 | 73.6 | 65.0 | 74.0 | 64.9 | 73.9 | 49.7 | 38.8 | 75.2 | 57.5 | **62.38** | -0.19 |

<!-- FORGET:MATH END -->

Per-eval standard errors on the base model are HellaSwag 0.47, PIQA 1.02, WinoGrande
1.34, ARC-e 0.90, ARC-c 1.46, OBQA 2.18, BoolQ 0.76, MMLU 0.40 pp; the 8-task mean has
SE ≈ **0.42 pp** treating the tasks as independent (a paired analysis would be tighter,
so this is the conservative bar).

**The headline is a null, and it is a null on the paper's own arm.** Every arm gains
**+20 to +22 pp** on MATH-500 and moves the prior-knowledge mean by **−0.49 to +0.20
pp** — every one inside ±1.2 SE. `dense` — the exact algorithm the paper indicts —
lands at **+0.00** (HellaSwag 65.40 → 65.31, −0.09 pp). For scale, the paper reports a
**≈10% relative** HellaSwag drop, which here would be **≈6.5 pp**; the largest movement
we see on HellaSwag in any direction is 0.54 pp.

Three consequences:

1. **We cannot confirm "ES ⇒ catastrophic forgetting" as a property of the algorithm.**
   In this setting it is not one. Whatever drives the paper's curves, it is not present
   when a 7B math-specialised base is trained with ES on in-domain MATH.
2. **Half the proposed mechanism survives; the sparsity half does not.**
   **Sparsity does not order retention at all**: the *densest* update (`dense`, 7.2%
   sparse) is the single best retainer at +0.00, while the *sparsest* (`insparse`,
   99.2%) is −0.07 — indistinguishable, across a 14× span in density. **Drift is
   weakly consistent**: the smallest-drift arm (`zoact`, 4.7e-4) is the most positive
   at +0.20 and the largest (`fura`, 4.8e-2) the most negative at −0.49. But every
   delta here is inside ±1.2 SE, so this leg can only say the ordering is *not
   contradicted*, not that it holds. [§14.5](#145-countdown--hellaswag-the-papers-own-task-pair)
   pushes drift 2–3× higher and there it becomes a clean, significant dose-response —
   so the honest reading is that **‖ΔW‖ matters and sparsity does not**, and MATH
   simply does not move the weights far enough for the ‖ΔW‖ effect to clear the noise.
3. **The question "can iso-ES or fura-ES avoid forgetting?" is not answerable on this
   task**, because there is no forgetting to avoid. What the table *does* establish is
   the other half of the requirement: the structured arms reach full-weight ES accuracy
   from ≤1.3% of the parameters **and** cost nothing in prior ability — which is the
   property an edge learner actually needs, whether or not dense ES would have been
   safe too.

**Why the disagreement with the paper is plausible.** Its setting differs on four axes
at once, each of which plausibly matters more than the optimiser: model **scale**
(1.5B/1B vs 7B — less redundancy to spare), model **type** (instruct-tuned, whose
prior abilities live in a thin post-training layer, vs a base model), **task distance**
(Countdown is a format-heavy puzzle far outside the pretraining distribution; MATH lvl
3–5 is *in-domain* for Qwen2.5-**Math**-7B), and **horizon** (500 iterations vs our
best checkpoints at 30–130). Task distance is the one we can test directly, which is
what §14.5 does.

### 14.5 Countdown → HellaSwag: the paper's own task pair

Since the MATH leg produced no forgetting to compare against, the arms were re-run on
the exact pair the paper uses, on the exact model it uses:

| Knob | Value |
| --- | --- |
| Model | **Qwen2.5-1.5B-Instruct** (the paper's model) |
| New task | **Countdown-3to4**, 200 training problems (the paper's count), 500-problem held-out split |
| Prior probe | HellaSwag + PIQA/WinoGrande/ARC-e/ARC-c/OBQA/BoolQ, 1000 docs each, **every 10 iterations** |
| ES | σ = 1e-3, α = σ/2, N = 30, greedy, 512-token responses |
| Iterations | 300 — past the ~200 where the paper says Countdown has converged but prior ability keeps falling |
| Arms | `dense` → `fura` (GPU 1), `iso` → `isobtt` (GPU 2) |
| Cost | ~2 min/iteration ⇒ ~11 h/arm; the probe adds 66 s per eval |

Step-0 baseline (Qwen2.5-1.5B-Instruct, identical for every arm): Countdown **8.0%**;
HellaSwag 59.70, PIQA 76.30, WinoGrande 63.20, ARC-e 76.50, ARC-c 46.00, OBQA 40.40,
BoolQ 77.10, prior mean **62.74**.

<!-- FORGET:COUNTDOWN BEGIN -->

#### Result — the paper's effect does not reproduce on its own task pair

`dense` and `iso` both finished all 300 iterations (2026-08-25; 8 h 45 m and 12 h 12 m,
104 and 145 s/iteration). Countdown accuracy is the 200-problem held-out split; HellaSwag and the prior
mean are the in-loop probe at 1000 docs/task.

| step | **dense** Countdown | HellaSwag | prior mean | | **iso** Countdown | HellaSwag | prior mean |
|---|---|---|---|---|---|---|---|
| 0 | 8.0 | 59.70 | 62.74 | | 8.0 | 59.70 | 62.74 |
| 20 | 20.5 | 59.30 | 62.40 | | 21.0 | 59.00 | 62.33 |
| 40 | 27.5 | 59.10 | 62.44 | | 29.0 | 59.00 | 62.41 |
| 60 | 32.0 | 59.00 | 62.09 | | 34.5 | 59.50 | 62.81 |
| 80 | 35.0 | 59.50 | 62.43 | | 38.0 | 59.20 | 62.63 |
| 100 | 36.5 | 60.10 | 62.57 | | **42.0** | 59.60 | 62.43 |
| 120 | 31.5 | 59.50 | 62.29 | | 42.5 | 59.50 | 62.81 |
| 140 | 35.5 | 59.60 | 62.39 | | 45.5 | 59.20 | 62.56 |
| 160 | 37.0 | 59.60 | 62.31 | | 43.0 | 59.30 | 62.76 |
| 180 | 40.0 | 59.60 | 62.41 | | 43.5 | 59.30 | 62.67 |
| 200 | 37.5 | 59.40 | 62.24 | | 47.0 | 59.90 | 62.60 |
| 220 | 40.0 | 60.10 | 62.80 | | 45.0 | 59.50 | 61.99 |
| 250 | **42.0** | 58.90 | 62.63 | | 47.0 | 59.70 | 61.56 |
| 280 | 39.5 | 59.30 | 62.34 | | 47.0 | 58.80 | 61.39 |
| 300 | 38.5 | **60.20** | 62.63 | | **47.5** | 59.30 | 61.53 |

Each probe is a 1000-doc subsample, so a single task carries SE ≈ 1.5 pp and the 7-task
mean SE ≈ 0.6 pp (unpaired) — about 1.4× looser than the full-suite numbers in
[§14.4](#144-result--nothing-forgets-dense-included).

**`dense` ES learns Countdown and does not forget.** Countdown **8.0 → 42.0%** (peak, and
+30.5 pp over base — the paper reports ES reaching 53.0 with a longer run, so the
learning reproduces). Over the same 300 iterations HellaSwag goes **59.70 → 60.20**
(**+0.50 pp**, and its whole-run range is 58.5–60.2) and the 7-task prior mean goes
**62.74 → 62.63** (**−0.11 pp**). Measured the paper's way — drop relative to the *best*
observed prior score — that is **0.3% relative**, against the **≈10%** they report.

This is a **negative replication on every axis they specify**: their model
(Qwen2.5-1.5B-Instruct), their new task (Countdown, 200 problems), their prior probe
(HellaSwag), their hyperparameters (σ=1e-3, α=σ/2, N=30, greedy), and past their own
convergence point — the paper's Figure 2 has prior accuracy still falling from ~200 to
500 iterations, whereas ours is flat from 0 to 300 with no trend. Combined with
[§14.4](#144-result--nothing-forgets-dense-included), **we cannot reproduce ES-induced
catastrophic forgetting in either of two settings, and one of them is the paper's own.**

**`iso` learns substantially more, and pays for it only in proportion.** It leads `dense`
at every step and ends at **47.5%** where `dense` peaks at 42.0 and ends at 38.5 — a task
level `dense` never reaches. Its prior mean does decline late (62.74 → 61.53; averaged
over the last three evals, −1.34 pp vs `dense`'s −0.29), which is ~2 SE and the only
non-null retention signal anywhere in this section.

**But that decline is bought, not leaked**, and
[§14.5.1](#1451-drift-is-the-axis--a-clean-dose-response) shows why: `iso` at σ=5e-2 moves
the weights 1.2× further than `dense`. Compared at *matched new-task accuracy*
rather than matched step count:

| | Countdown | prior mean | Δ prior |
|---|---|---|---|
| base | 8.0 | 62.74 | — |
| `dense` @ step 300 | 38.5 | 62.63 | −0.11 |
| `iso` @ step 80 | 38.0 | 62.63 | −0.11 |
| `iso` @ step 300 | **47.5** | 61.53 | −1.21 |

At the same Countdown score the two arms have lost **exactly the same** prior ability.
`iso` ends lower only because it kept going and reached a level `dense` never did. So
prior-ability loss here tracks **how far along the new task you are**, not which subspace
the perturbation lives in — the same conclusion [§14.4](#144-result--nothing-forgets-dense-included)
reached from the geometry. And on the paper's *own* probe, HellaSwag, both arms are flat
(`dense` +0.5, `iso` −0.4 pp).

Per-task at step 300 (base → arm): `dense` WinoGrande −1.8, ARC-e −1.7, BoolQ +1.9,
PIQA +0.9; `iso` ARC-e −4.4, ARC-c −2.8, BoolQ −2.8, WinoGrande +1.2, PIQA +1.1.

#### 14.5.1 Drift is the axis — a clean dose-response

Materialising the three finished Countdown arms (`scripts/es/materialize_es_ckpt.py
--base <Qwen2.5-1.5B-Instruct>`) gives ΔW against the same base, so the arms can be
ordered by how far they moved the weights:

| arm | σ | ‖ΔW‖_F/‖W‖_F | sparsity | Countdown (peak) | HellaSwag **LL** Δ | HellaSwag **gen** Δ | prior mean Δ |
|---|---|---|---|---|---|---|---|
| `dense` | 1e-3 | 3.47e-02 | 3.4% | 42.0 | **+0.5** | **+0.1** | −0.29 |
| `iso` | 5e-2 | 4.32e-02 | 17.3% | **47.5** | −0.4 | −2.4 | −1.34 |
| `fura` | 1.25e-2 | **9.50e-02** | 16.2% | 18.5 | **−3.8** | **−14.3** | **−3.20** |

(prior mean Δ = last-three-eval average vs step 0; LL = the in-loop log-likelihood probe,
gen = §14.5.2's generative probe.)

**Every retention column is monotone in ‖ΔW‖_F, and sparsity is unrelated to any of
them.** `dense` moves least and loses nothing; `iso` moves 1.2× further and loses a
little; `fura` moves 2.7× further than `dense` and loses a lot. Note this is the *same
ordering* the MATH leg hinted at ([§14.4](#144-result--nothing-forgets-dense-included)
claim 2) but could not resolve, because there the largest drift was 4.8e-2 — half of
`fura`'s here — and every delta sat inside the noise floor.

So the paper's ℓ2-norm intuition is **right about the axis and wrong about the
attribution**: drift predicts forgetting, but drift is a function of *step size*, not of
being ES rather than GRPO, and not of update density. Our `dense` Countdown arm is
**denser** (3.4% sparse) and **larger-drift** (3.47e-2) than our `dense` MATH arm and
still loses nothing — what separates `fura` is that σ=1.25e-2 moved it 2.7× further for
*less* new-task progress.

#### 14.5.2 The generative-probe hypothesis is falsified

The obvious way to reconcile our null with the paper was that our log-likelihood ranking
is format-immune while theirs might not be. Tested directly on the finished checkpoints
(`scripts/es/eval_hellaswag_gen.py`, 1000 items, A–D multiple choice, greedy, parse the
letter):

| model | generative acc | **unparsed** | log-likelihood acc |
|---|---|---|---|
| base Qwen2.5-1.5B-Instruct | 57.3 | **0.0%** | 59.7 |
| `dense` @ 300 | **57.4** | **0.0%** | 60.2 |
| `iso` @ 300 | 54.9 | 0.0% | 59.3 |
| `fura` @ 300 | 43.0 | 0.0% | 56.1 |

**Format drift is not the explanation.** After 300 ES iterations on a task whose reward
is gated on `<think>…</think><answer>…</answer>`, every arm still answers a multiple-choice
prompt with a bare letter — **0% unparsed everywhere** — and `dense`'s generative accuracy
is *unchanged* (57.3 → 57.4). So the paper's drop cannot be recovered by switching our
probe to their (possible) protocol.

What the generative probe *does* buy is **sensitivity**: where a model has genuinely
degraded, it registers 4–6× more than log-likelihood ranking (`fura` −14.3 vs −3.8;
`iso` −2.4 vs −0.4). That is worth carrying forward — it is the better instrument for
this question — but it does not manufacture degradation where there is none.

#### What is left to explain

Since the effect does not survive a faithful re-implementation, the cause is in what we
did *not* copy. Two candidates, in order of how much we think they matter:

1. ~~**The prior-task evaluation protocol.**~~ **Tested and ruled out**
   ([§14.5.2](#1452-the-generative-probe-hypothesis-is-falsified)): the generative probe
   shows 0% unparsed and unchanged accuracy for `dense`.
2. **The step size their run actually took.** [§14.5.1](#1451-drift-is-the-axis--a-clean-dose-response)
   shows forgetting is monotone in ‖ΔW‖_F, and we only ever measure ‖ΔW‖ on *our* runs.
   The paper reports its drift as a ratio to GRPO, never in absolute or relative-to-‖W‖
   terms, so we cannot tell whether their ES run sat at our `dense` operating point
   (3.5e-2, harmless) or our `fura` one (9.5e-2, damaging). If theirs is nearer the
   latter, our results and theirs are **not in conflict at all** — they would simply have
   run ES at a step size that costs prior ability, which is a tuning statement rather
   than a property of ES. Reporting ‖ΔW‖_F/‖W‖_F would settle it immediately.
3. **The KL anchor on their GRPO baseline.** Their ES-vs-GRPO gap is partly (perhaps
   wholly) a comparison of unanchored vs β=1e-3-KL-anchored training — see §14.1. This
   does not explain why *our* ES does not forget, but it does mean the paper's own
   contrast cannot separate "gradient-free" from "unanchored".
4. **Two setup deltas we did not match:** their **max_tokens = 1024** (ours 512) and
   **fp16** (ours bf16). Neither is an obvious route to a 6 pp HellaSwag swing, but the
   token budget is the cheaper of the two to align if this is pushed further.
5. **The ES implementation.** Ours is aligned with the official repo on reward shaping,
   seeds, α=σ/2 and greedy decoding ([§12](#12-alignment-with-the-official-implementation)),
   with one known deviation: `_es_noise` reseeds per layer with the bare seed, so
   same-shaped layers draw identical noise ([§10.9](#109-one-deviation-worth-flagging)).
   That shrinks the effective search dimension; it is not an obvious route to *less*
   drift, but it has never been ablated.

Not candidates: model, task, prior benchmark, population size, nominal σ (for `dense`),
or horizon — all matched.

#### `fura` — the one arm that does forget, and it also fails to learn

`fura` finished 300 iterations (6 h 19 m) at the σ that tops the MATH leaderboard,
σ = 1.25e-2 / α = 6.25e-3 ([§11.3](#113-answer-yes--but-scale-σ-not-α)):

| step | 0 | 40 | 80 | 120 | 160 | 200 | 240 | 280 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| Countdown | 9.0 | 15.0 | 12.0 | 13.5 | 14.0 | 16.0 | 16.5 | 18.0 | 18.0 |
| HellaSwag | 59.90 | 59.20 | 58.20 | 58.40 | 57.90 | 57.80 | 56.50 | 55.20 | **56.10** |
| prior mean | 62.70 | 61.63 | 61.80 | 61.34 | 60.33 | 60.47 | 59.99 | 59.30 | **59.61** |

This is the only arm anywhere in [§14](#14-catastrophic-forgetting--does-the-perturbation-subspace-decide-it)
that shows a real prior-ability decline — HellaSwag **−3.8 pp**, prior mean **−3.20 pp**
(last-three-eval average) — and unlike `iso`'s it starts at the very first evals and
trends down all the way (62.70 → 61.6 by step 40 → 60.3 by 160 → 59.6 at 300) rather than
appearing only after the new task has been learned. It is also the only arm that fails at the new task: Countdown peaks at
**18.5%** where `dense` reaches 42.0 and `iso` 47.5.

**Both failures point at step size, not at the BTT subspace.** σ = 1.25e-2 was selected
on 7B/MATH, where it was footprint-matched to dense ES; nothing re-derived it for a 1.5B
model on Countdown, and `fura` already had the largest weight-space footprint of any arm
in [§14.3](#143-update-geometry-of-the-six-arms) (4.78e-2). An arm that is simultaneously
*worse at learning* and *worse at retaining* is the signature of an over-large step —
[§11](#11-fura-learning-rate-search) has now shown three times that σ dominates this
family — not of a subspace that leaks knowledge. **The disambiguating run is `fura` at
σ = 1e-3 on the same task**: if it then learns like `dense` and stops forgetting, the
whole effect is step size.

So the arms do not order by subspace; they order by **how hard they are pushed**. `iso`
at σ=5e-2 pushes furthest and learns most, losing prior ability only in proportion to the
task progress it buys; `fura` at σ=1.25e-2 is pushed past the point where the step still
buys anything, and pays without being paid.

`isobtt` (GPU 2, started 06:54) is still running.

<!-- FORGET:COUNTDOWN END -->

## 15. LoRA-ES — a trained random projection, at fura's footprint and at rank 1

> Two arms on the [§7](#7-results) MATH protocol (Qwen2.5-Math-7B, MATH lvl 3–5 →
> MATH-500, fixed 64-problem batch, N=30, 150 iterations), started 2026-08-26 on GPU 7.
> wandb `ES-q2p5-7b`, runs `lora-r44_…` and `lora-r1_…`.

### 15.1 Parameterisation

`PERTURB_MODE=lora` adds a standard LoRA adapter and lets ES train **both** factors:

```
W = W_base + s · B A ,    A ∈ R^{r×in} ,  B ∈ R^{out×r} ,  s = lora_scale
```

Both factors live in **one flat fp32 coefficient tensor** per layer (`A` first, then `B`),
so every existing piece of machinery — `_es_noise`, `es_update`, `es_save_coef`,
`_es_target` — works unchanged; the mode is ~20 lines in
`es_worker_extension.py`. `B` is **zero-init** (standard LoRA) and `A ~ N(0, 1/in)` from a
name-derived CRC seed, so step 0 reproduces the base model *bit-exactly* and the curve
starts from the published 51.6, not from "after one update".

This makes `lora` the natural control for [`zoact`](#2-the-six-runs): both write
`ΔW = C V` with `C` the trained `out`-side coefficient, but `zoact` freezes `V` to the
top-r **calibrated activation** directions while `lora` starts from a **random** `V` and
*also trains it*. The pair therefore separates "does the projection have to be informed?"
from "does it have to be learned?".

### 15.2 Choosing the ranks

The structured modes cover the 112 fused 2-D linear weights (28 layers ×
`qkv_proj`/`o_proj`/`gate_up_proj`/`down_proj`), so LoRA costs

```
r · Σ(out + in) = r · 28 · 79,360 = r · 2,222,080  coefficients
```

| rank | trainable coeffs | % of 7.6 B | matched to |
|---|---|---|---|
| **44** | **97,771,520** | 1.28% | **`fura` exactly** (97,771,520) |
| **1** | 2,222,080 | 0.029% | minimal adapter |

Rank 44 is an *exact* match, not a rounding — verified against the live model, where
`init_es_state` reports 97,771,520. (For scale, `zoact r=1`'s 1,390,592 = Σ(out) alone,
since it trains only the `out` side.)

### 15.3 Numerical gate

`scripts/es/test_lora_es.py` on real Qwen2.5-Math-7B weights — all PASS:

| check | rank 1 | rank 44 |
|---|---|---|
| identity at init (`B=0`): max\|W − W_base\| | **0.0** | **0.0** |
| perturb → restore | **0.0** | **0.0** |
| (W⁺+W⁻)/2 − W | 2.4e-4 | 4.9e-4 (bf16 ULP ~1.6e-4) |
| ‖ΔW‖_F/‖W‖_F at σ=1e-3 | 3.84e-04 | 3.25e-03 |
| `es_update` moves coefficients | ✓ | ✓ |

⚠️ **Rank 1 sits below the bf16 rollout floor.** [§6](#6-numerical-health) put that floor
at 1.6e-3 relative; `lora r=44` at 3.25e-3 is ~2× above it (the same regime as `zoact`
4.2e-3 and `fura` 4.0e-3), but **`lora r=1` at 3.84e-4 is ~4× *below*** — most of its
perturbation is quantisation noise in the weights vLLM actually runs. The fp32
coefficient masters still accumulate updates, but the reward differences driving them may
not clear the floor. `train/reward_std` is the tell (the six §7 arms sat at 0.020–0.030);
if it collapses, the fix is **σ, not α** ([§11.3](#113-answer-yes--but-scale-σ-not-α)).

### 15.4 Setup and status

σ = **1e-3** (the paper / dense-ES value), α = **5e-3** = **10× dense ES's** 5e-4 — so
α/σ = 5 where the paper convention is 0.5. Per-iteration coefficient motion is
α/√N ≈ 9.1e-4, about one perturbation's worth per step.

`lora r=44` completed 150/150 in 15 h 06 m at 362 s/iteration (§7 arms ~363 s) —
see [§15.5](#155-result--rank-44-learns-but-slowly-and-never-plateaus). `lora r=1`
started 2026-08-27 05:46.

**One failure worth recording.** The first launch enabled the
[§14.2](#142-harness) prior-task probe alongside training and **both ranks were killed
during the first training iteration** — a hard Ray-worker death (no Python exception, no
CUDA error). The probe's 512-prompt `prompt_logprobs` batch is a large transient on a
7B model with a 152k vocab: it completes at step 0, takes the GPU from 64.4 → 82.8 GB,
and the next `generate` dies. It is fine on the 1.5B Countdown runs
([§14.5](#145-countdown--hellaswag-the-papers-own-task-pair)), which is why it had not
surfaced. Running the MATH arms with the probe off — which is also what makes them
comparable to [§7](#7-results) — fixed it. If retention numbers are wanted for these
arms, materialise the checkpoints and score them offline instead
([§14.2](#142-harness)), or drop `es.forget_batch_size` well below 512.

<!-- LORA:RESULTS BEGIN -->

### 15.5 Result — rank 44 learns, but slowly, and never plateaus

`lora r=44` finished 150/150 (15 h 06 m, 362 s/iteration). `reward_std` averaged **0.0226**
over the whole run (last-10 mean 0.0227), squarely inside the §7 band — the 10× α caused
no instability and the perturbation cleared the bf16 floor comfortably.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MATH-500 | 51.6 | 62.4 | 62.6 | 64.2 | 65.0 | 66.0 | 66.2 | 68.0 | 69.0 | 70.6 | 67.8 | 66.4 | 68.0 | 67.6 | 69.2 | **71.6** |

Against the [leaderboard](#leaderboard) (plateau = mean over steps ≥ 40):

| Method | Plateau (≥40) | Best @ step |
|---|---|---|
| fura | 72.68 ± 0.90 | 74.0 @ 30 |
| iso | 72.42 ± 0.78 | 74.0 @ 60 |
| insparse | 72.07 ± 0.70 | 73.4 @ 80 |
| isobtt | 71.95 ± 0.94 | 73.4 @ 120 |
| dense | 71.82 ± 1.19 | 73.4 @ 40 |
| zoact r=1 | 70.50 ± 0.94 | 72.2 @ 130 |
| **lora r=44** | **67.95 ± 0.56** | **71.6 @ 150** |

**The plateau statistic understates it, because `lora` never plateaus.** Every other arm is
flat by step ~40; `lora` is still climbing at 150, where it posts its best score of the
run (71.6) — within ~1–2 pp of the others' bests. What separates it is **shape, not
level**: it takes ~90 iterations to reach what `dense` reaches in 10, and the ≥40 mean
punishes that ramp.

**The likely cause is footprint, not the subspace.** At σ=1e-3 `lora r=44` moves the
weights **3.25e-3** — the same scale as `zoact` (4.2e-3) and `fura` at the paper σ
(4.0e-3), all of which are the *slow* configurations. `fura`'s winning entry runs at
σ=1.25e-2 (footprint ~5e-2, **15× larger**), and [§11.3](#113-answer-yes--but-scale-σ-not-α)
already showed that moving `fura` from −12.25 pp to +0.82 pp was a **σ** change, not an α
change. The 10× α here did not substitute: it kept the update stable but could not make a
small perturbation informative. So the honest reading is that **`lora r=44` has not yet
been given its operating point**, and the follow-up is a σ sweep (σ ∈ {4e-3, 1.25e-2}),
not a verdict on random-vs-structured projections.

What *can* be said at matched footprint (~3–4e-3) and matched trainable count
(97,771,520): a **random, trained** projection (`lora`, 67.95) is behind a **structured
block-SVD** one (`fura` at the same σ was even further behind, −12.25 pp, before its σ
fix) and behind a **calibrated, frozen** rank-1 one (`zoact`, 70.50) that has 44× fewer
coefficients. Rank has not bought quality here.

<!-- LORA:RESULTS END -->
