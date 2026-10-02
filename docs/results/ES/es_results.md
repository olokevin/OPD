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

**LoRA-ES ([§15](#15-lora-es--a-trained-random-projection-at-furas-footprint-and-at-rank-1)).**
A new `lora` mode trains **both** LoRA factors with ES; cost is `r · 2,222,080` coefficients,
so **rank 44 reproduces `fura`'s 97,771,520 exactly**. At σ=1e-3 / α=5e-3 (10× dense),
r=44 gains **+20.0 pp** (best 71.6 @ 150) and r=1 **+7.0 pp** (58.6) — both **still rising
at 150** where every §7 arm is flat by 40. The designed control is the headline: against
`zoact r=1`, whose projection is **calibrated and frozen**, `lora r=1`'s **random and
trained** projection loses **15 pp** (55.07 vs 70.50) while holding the strictly larger
hypothesis class — one calibration forward pass beats 150 × 30 ES probes at finding the
input direction. Both LoRA arms are under-scaled (footprints 3.25e-3 / 3.84e-4 vs `fura`'s
winning 5e-2), so a σ sweep is a prerequisite before reading any of this as a verdict on
subspaces.

**Population size ([§16](#16-population-size--n10-vs-n30)).** At **N=10** instead of 30 —
3× cheaper per iteration, and the N=10 population is a *nested subset* of the N=30 seeds —
`dense` loses **nothing** (−0.49 ± 0.46, t=−1.06; 70% reached in 0.33 vs 0.98 GPU-h),
`iso` loses a little (−1.52 ± 0.49), and `fura` appears to **break** (−5.44 ± 0.60).
**That break was the step size, not the probe count** ([§16.4](#164-it-was-the-step-size-fura-at-n10-fully-recovers)):
holding α fixed makes an N=10 step √3 larger, and restoring the motion
(α × √(10/30) = 3.61e-3) takes `fura` from **−5.44 to +0.41 ± 0.43 (ns)** — matching its
N=30 twin, at 5.0 GPU-h instead of 15.3. **The rule is to hold α/√N fixed when changing N.**
So the 3× compute saving is general, and `iso`'s −1.52 is now suspect for the same reason
(its motion-matched control, α=1.443e-2, is not yet run).

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

### 13.x [2026-10-02] Paper-faithful ISO-Optimizer: new `iso` / `isobtt`; the Cayley arms renamed

> The BP `iso` / `isobtt` above were **not** the paper's optimizer (Cayley rotations on a
> random block basis / right-only block rotation). They are now `iso_cayley` /
> `isobtt_cayley` (`isobtt_mix` unchanged), so the results in this section stay
> reproducible. The names `iso` / `isobtt` now mean the paper's ISO-AdamW (§4.3 of the paper).

**What the new modes do** (`verl/workers/peft/iso.py`, `IsoFrameLinear` / `IsoFrameAdapter`):

| mode | parameterisation | trained | after every optimizer step | spectrum fixed |
|---|---|---|---|---|
| `iso` | `W = U S0 Vᵀ`, thin SVD of the whole matrix | `U`, `V` (AdamW, grads = paper Eq. 34) | `U ← polar(U)`, `V ← polar(V)` in fp64 | global `σ(W)` |
| `isobtt` | the same per input block: `W[:, blk_j] = U_j S_j V_jᵀ` (fura's `_closest_factor_pair` blocks) | `U_j`, `V_j` | per-block polar | per block |

Implementation choices that don't change the maths (details in the module docstring):
- **Delta storage**: `U = U0 + dU`; forward `W0 + (U S0 Vᵀ − U0 S0 V0ᵀ)`. Same trajectory as stepping `U` when weight decay is 0, so `retract_iso` refuses any other value. Under bf16 FSDP the actor sees `W0` exactly, so step 0 is bit-identical to the base and actor weights match vLLM's.
- **Polar via fp64 Newton–Schulz**: matches the SVD polar to 4e-14, 0.018 s vs 0.7 s per 9728×2560 frame. Falls back to SVD if it doesn't converge.
- **FSDP1 only**, `use_orig_params=True`. Retraction and vLLM export gather **one FSDP unit at a time**, because the full frame model is about 3.5× the dense size.
- **Non-converted tensors** (embeddings / tied head, norms) are trained by plain AdamW (`peft.iso.train_others=True`). The paper doesn't say how it handles them.

**Gates** (`scripts/es/test_iso_frame_bp.py`; `--fsdp` under torchrun ×2): all pass.

| check | iso | isobtt |
|---|---|---|
| step 0 == base, bit-for-bit (tiny Qwen3 and **Qwen3-4B-Base**) | ✅ | ✅ |
| frame grads == Eq. 34 | 2.4e-7 | 2.5e-7 |
| delta storage == literal AdamW-on-(U,V) + SVD polar, 5 steps (rel err of update) | 1.6e-6 | 7.1e-6 |
| `σ(W) == S0` after steps (rel), fp32 / FSDP bf16-MP | 5.9e-8 / 5.9e-8 | 8.0e-8 / 7.9e-8 |
| FSDP×2 step + retraction == single process (fp32) | exact | exact |
| unit-by-unit export == dense weights | exact | exact |

**Cost on Qwen3-4B-Base** (1×H100 NVL, unsharded):

| | iso | isobtt |
|---|---|---|
| init SVD (once per rank) | 143 s | 33 s |
| stored params (fp32) | 13.8 B (51.7 GiB) | 11.5 B (42.8 GiB) |
| trainable (frame deltas + others) | 5.28 B | 4.11 B |
| retraction, whole model | 5.6 s/step | 0.4 s/step |
| dense export for vLLM | 1.5 s | 0.2 s |

Launch: `PEFT_MODE=iso|isobtt bash grpo.sh` (sets `weight_decay=0`, `use_orig_params=True`), or `BP_MODE=iso scripts/es/run_bp_math.sh` (LR default 7.5e-7 = the paper's). The ES-side `iso` / `isobtt` in `es_worker_extension.py` are unchanged.

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

**Rank 1 sits below the bf16 rollout floor — and it does not matter.**
[§6](#6-numerical-health) put that floor at 1.6e-3 relative; `lora r=44` at 3.25e-3 is ~2×
above it (the same regime as `zoact` 4.2e-3 and `fura` 4.0e-3), but **`lora r=1` at
3.84e-4 is ~4× *below***, so most of its per-entry displacement is lost to quantisation in
the weights vLLM actually runs. We predicted this might collapse the reward signal and
starve the ES estimator. **Measured: it does not.** `lora r=1`'s first four iterations give
`train/reward_std` = 0.0211 / 0.0197 / 0.0241 / 0.0294 (mean 0.0236) — inside the same
0.020–0.030 band the six §7 arms occupy — with train accuracy already climbing
50.1 → 52.0.

The prediction was wrong because it reasoned about *average per-entry* displacement. A
rank-1 update is a **single coherent direction** applied across the whole matrix, and the
ES reward is **binary per problem**: rounding destroys magnitude, not coherence, and only
a modest logit shift on borderline problems is needed to spread rewards across the 30
population members. **The bf16 floor bounds how finely a perturbation can be represented,
not whether a low-rank one is visible in the reward.** Worth carrying forward — the same
caution in §6 about `zoact`/`fura` being "only 2.6× the bf16 floor" is likewise weaker
than it reads.

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

### 15.5 Result — both ranks learn, neither plateaus, and rank matters enormously

Both arms finished 150/150 (r=44: 15 h 06 m / 362 s per iteration; r=1: 15 h 25 m / 370 s).
`reward_std` averaged **0.0226** and **0.0246** respectively — both inside the §7 band, so
the 10× α caused no instability at either rank.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lora r=44** | 51.6 | 62.4 | 62.6 | 64.2 | 65.0 | 66.0 | 66.2 | 68.0 | 69.0 | 70.6 | 67.8 | 66.4 | 68.0 | 67.6 | 69.2 | **71.6** |
| **lora r=1** | 51.6 | 53.8 | 51.8 | 53.4 | 51.6 | 54.2 | 53.2 | 54.0 | 52.2 | 53.4 | 56.4 | 56.2 | 56.6 | 57.4 | 57.0 | **58.6** |

`lora r=1` gains **+7.0 pp** (plateau 55.07 ± 0.65) against `lora r=44`'s **+20.0 pp**.
Both are still rising at step 150.

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
| **lora r=1** | **55.07 ± 0.65** | **58.6 @ 150** |

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

### 15.6 The `lora`-vs-`zoact` control: a calibrated direction beats a learned one

This is what the mode was built for. Both write `ΔW = C·V` with `C` the trained `out`-side
coefficient; they differ only in where `V` comes from:

| | `V` | trainable coeffs | plateau (≥40) | best |
|---|---|---|---|---|
| `zoact r=1` | **calibrated**, frozen (top-1 activation singular direction) | 1,390,592 | **70.50 ± 0.94** | 72.2 |
| `lora r=1` | **random**, and *also trained* | 2,222,080 | **55.07 ± 0.65** | 58.6 |

**A 15 pp gap, with `lora` holding the strictly larger hypothesis class.** `lora r=1` can
in principle rotate `A` onto the calibrated direction — it has 1.6× more coefficients and
strictly more freedom — and over 150 iterations it does not get close. One forward pass of
calibration ([§5](#5-calibration-runs-2--3)) hands ES a direction that 150 iterations ×
30 probes cannot find on its own. With a population of 30 in a 2.2 M-dimensional
coefficient space, ES simply has too few probes per step to *discover* the input subspace;
it can only exploit one it is given.

⚠️ **Confounded by footprint, and not by a little.** At σ=1e-3 the two move the weights by
very different amounts: `zoact r=1` **4.2e-3** vs `lora r=1` **3.84e-4** — an **11×** gap
(§15.3). So this comparison mixes *projection quality* with *step size*, and the same
applies to `lora r=44` (3.25e-3) against `fura`'s winning 5e-2. **The σ sweep is a
prerequisite, not a refinement**, before any of these are read as statements about
subspaces: [§11.3](#113-answer-yes--but-scale-σ-not-α) moved `fura` 13 pp on σ alone.

What is *not* confounded is the comparison **within** `lora`, where σ, α, protocol and
seed are identical: **rank 44 gains +20.0 pp, rank 1 gains +7.0 pp**. Rank buys a great
deal for a random projection — which is the mirror image of `zoact`, where rank 1 on the
*right* direction already reaches 70.50.

<!-- LORA:RESULTS END -->

## 16. Population size — N=10 vs N=30

> `dense`, `fura` and `iso` re-run on the [§7](#7-results) MATH protocol with
> **population 10 instead of 30**, everything else identical. Sequential on GPU 5,
> 2026-08-28/29. Launcher `scripts/es/chain_pop10.sh`.

### 16.1 Design

Only `population_size` changes; each arm keeps the σ/α that tops the
[leaderboard](#leaderboard) (`dense` 1e-3/5e-4, `fura` 1.25e-2/6.25e-3, `iso` 5e-2/2.5e-2),
so every N=10 curve has a directly comparable N=30 twin.

Two properties make this a tighter ablation than it looks:

* **The populations are nested.** Seeds come from
  `default_rng(global_seed + iteration).integers(..., size=N)`, so N=10 draws exactly the
  **first 10 of the same 30 seeds**. At iteration 1 the N=10 run evaluates a strict subset
  of the very same perturbed models — confirmed: `dense` reports `train/accuracy`
  **58.59375** at iteration 1 under both N, bit-identical. The runs diverge only once the
  updates differ.
* **All three arms are footprint-matched.** By construction ([§6](#6-numerical-health),
  [§10.4](#104-scale-convention--σ-is-a-relative-footprint-not-a-noise-std)) each
  perturbation moves ‖ΔW‖/‖W‖ ≈ 5.0e-2, and the per-iteration *update* motion α/√N works
  out to **4.56e-3 at N=30 and 7.91e-3 at N=10 for all three arms alike**.

⚠️ **The confound, stated up front.** The ES update moves coefficients by ~α/√N per
iteration, so holding α fixed makes an N=10 step **√3 ≈ 1.73× larger**. This ablation
therefore varies *two* things at once — estimator quality (10 vs 30 probes) and step size.
Because all three arms take the *same* larger step, differences *between* arms are
informative; the absolute size of each arm's drop is not cleanly attributable. The
separating run is α scaled by √(10/30) — **not yet run**.

### 16.2 Results

<!-- POP10:RESULTS BEGIN -->

| Arm | N | Plateau (≥40) | Best @ step | **Paired Δ (N10−N30)** | t |
|---|---|---|---|---|---|
| `dense` | 30 | 71.82 ± 0.34 | 73.4 @ 40 | | |
| `dense` | **10** | **71.07 ± 0.42** | 73.4 @ 20 | **−0.49 ± 0.46** | −1.06 (ns) |
| `iso` | 30 | 72.42 ± 0.22 | 74.0 @ 60 | | |
| `iso` | **10** | **71.05 ± 0.38** | 73.2 @ 50 | **−1.52 ± 0.49** | −3.12 |
| `fura` | 30 | 72.68 ± 0.26 | 74.0 @ 30 | | |
| `fura` | **10** | **66.63 ± 0.54** | 70.6 @ 10 | **−5.44 ± 0.60** | −9.07 |

Paired over the 15 shared eval steps (10…150). MATH-500 curves at N=10:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | 51.6 | 71.6 | 73.4 | 71.2 | 71.8 | 72.8 | 73.2 | 69.8 | 71.6 | 72.4 | 70.8 | 69.4 | 69.0 | 70.2 | 72.2 | 69.6 |
| `iso` | 51.6 | 64.8 | 70.0 | 72.8 | 72.8 | 73.2 | 71.4 | 70.8 | 71.8 | 68.6 | 69.6 | 70.0 | 70.2 | 71.6 | 71.6 | 71.0 |
| `fura` | 53.2 | **70.6** | 69.6 | 68.6 | 69.4 | 68.2 | 70.4 | 65.8 | 65.8 | 66.0 | 66.0 | 66.6 | 67.2 | 64.0 | 65.4 | 64.8 |

Cost, from median iteration times:

| Arm | s/iter N=30 → N=10 | 150-iter GPU-h | GPU-h to first reach 70% |
|---|---|---|---|
| `dense` | 354 → **118** | 14.8 → **4.9** | 0.98 → **0.33** |
| `iso` | 390 → **129** | 16.3 → **5.4** | 1.08 → **0.72** |
| `fura` | 367 → **119** | 15.3 → **5.0** | 1.02 → **0.33** |

<!-- POP10:RESULTS END -->

### 16.3 Reading

**Three arms, the same 3× cheaper iteration and the same √3 larger step, three different
outcomes — and the ordering is by *step-size tolerance*, not by subspace.**

* **`dense` pays nothing** (−0.49 ± 0.46, t = −1.06). Tripling the population buys no
  measurable accuracy, while costing 3× per iteration. It reaches 70% in **0.33 GPU-h vs
  0.98** — a **3× compute saving to target** and 14.8 → 4.9 GPU-h for a full run. This is
  what [§11.1](#111-the-64-problem-batch-is-the-ceiling-not-the-method) predicts: the
  fixed 64-problem batch is the ceiling, so a 30-sample gradient estimate is already deep
  into diminishing returns and the extra 20 probes refine a direction the batch cannot
  justify.
* **`iso` pays a little** (−1.52 ± 0.49, t = −3.12) — real but small, and it *converges
  more slowly* (64.8 at step 10 where N=30 was at 70.2) before catching up to 73.2 by
  step 50. At a third of the cost this is still a good trade.
* **`fura` breaks** (−5.44 ± 0.60, t = −9.07), and **qualitatively, not quantitatively**:
  it peaks at step 10 (70.6) and then declines monotonically to 64.8. That is slow
  divergence, not a noisier plateau — the same failure mode
  [§11.2](#112-stability-edge-125-in-40-out) measured for `fura` when its step is pushed
  past the stability edge. σ=1.25e-2 already sits near that edge at N=30; the √3 step
  increase tips it over.

~~So the honest attribution is that `fura`'s −5.44 pp is consistent with either cause.~~
**Resolved in [§16.4](#164-it-was-the-step-size-fura-at-n10-fully-recovers): it was
entirely the step size.**

### 16.4 It was the step size — `fura` at N=10 fully recovers

`fura` re-run at N=10 with α scaled to restore the N=30 per-iteration motion
(α × √(10/30) = 3.6084e-3), plus a half-motion point. σ stays 1.25e-2 throughout, so this
varies **α only** — unlike [§11.3](#113-answer-yes--but-scale-σ-not-α), which moved σ.
Launcher `scripts/es/chain_fura_pop10_lr.sh`.

<!-- POP10:FURA-LR BEGIN -->

| `fura` | α | motion α/√N | Plateau (≥40) | Best @ step | Paired Δ vs N=30 | t |
|---|---|---|---|---|---|---|
| N=30 (leaderboard) | 6.25e-3 | 1.141e-3 | 72.68 ± 0.26 | 74.0 @ 30 | — | — |
| N=10, α unchanged | 6.25e-3 | **1.976e-3** (1.73×) | 66.63 ± 0.54 | 70.6 @ 10 | **−5.44 ± 0.60** | −9.07 |
| **N=10, motion-matched** | **3.6084e-3** | **1.141e-3** (1.00×) | **73.17 ± 0.33** | **77.4 @ 30** | **+0.41 ± 0.43** | +0.96 (ns) |
| N=10, half motion | 1.8042e-3 | 5.705e-4 (0.50×) | 72.77 ± 0.23 | 73.8 @ 150 | −1.03 ± 0.66 | −1.57 (ns) |

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| α=6.25e-3 | 53.2 | 70.6 | 69.6 | 68.6 | 69.4 | 68.2 | 70.4 | 65.8 | 65.8 | 66.0 | 66.0 | 66.6 | 67.2 | 64.0 | 65.4 | 64.8 |
| α=3.61e-3 | 53.2 | 70.2 | 70.6 | **77.4** | 73.4 | 74.6 | 74.8 | 74.0 | 72.8 | 73.6 | 73.4 | 73.6 | 72.2 | 73.0 | 71.6 | 71.0 |
| α=1.80e-3 | 53.2 | 64.0 | 67.0 | 70.4 | 71.4 | 73.4 | 73.4 | 71.8 | 72.4 | 73.0 | 71.6 | 72.8 | 73.2 | 73.2 | 73.2 | 73.8 |

<!-- POP10:FURA-LR END -->

**The collapse was the learning rate, and nothing else.** Restoring the per-iteration
motion moves `fura` from **−5.44 pp to +0.41 pp** — statistically indistinguishable from
its N=30 twin (slightly above, ns). The *shape* recovers too: the monotone slide to 64.8
is gone, replaced by a normal plateau at 73–75 from step 30 on.

**The response in α is unimodal and the basin is wide on the low side.** At matched motion
(1.00×) `fura` is at its best; at half motion it loses only 1.03 pp (ns) and merely starts
slower (64.0 at step 10 against 70.2); at 1.73× it diverges. So the usable range runs from
about 0.5× to 1× the N=30 motion, and the failure is one-sided — too big a step, not too
small.

**Consequences.**

1. **[§16.3](#163-reading)'s attribution was wrong and is retracted.** N=10 does not cost
   `fura` accuracy; 1.73× its step does. The rule is to hold **α/√N** fixed when changing
   N, not α.
2. **The 3× compute saving is general, not `dense`-only.** With α set correctly, `fura` at
   N=10 matches N=30 at 119 s/iteration against 367 — 5.0 GPU-h per run instead of 15.3.
3. **`iso`'s −1.52 ± 0.49 is now suspect for the same reason** — it was also measured at
   fixed α, so it took the same 1.73× overshoot. The matching control is
   α = 2.5e-2 × √(10/30) = **1.443e-2**. Not yet run.

⚠️ **77.4 @ step 30 is the highest single MATH-500 eval on this page** (previous high 74.0,
and 74.4 in the off-protocol aligned sweep). Treat it as a max over 16 noisy evals with
~2.2 pp per-eval SE, not as a headline: the honest statistic is the plateau, 73.17 ± 0.33,
which is nominally the best plateau anywhere here but **not** significantly above `fura`
N=30's 72.68 (+0.41 ± 0.43).

## 17. N=10 as the default: step-size search, two calibrated hybrids, and what actually sets σ

> Everything here runs the [§7](#7-results) MATH protocol at **N=10** — the population
> [§16.4](#164-it-was-the-step-size--fura-at-n10-fully-recovers) showed is free once α/√N is
> held fixed — on GPUs 6 and 7, 2026-08-31. Two new perturbation modes (`fura_zoact`,
> `lora_zoact`) plus α/σ searches for `dense` and `lora`.
> Launchers `scripts/es/chain_{dense,lora}_pop10_lr.sh`,
> `chain_{furazoact,lorazoact}_pop10.sh`, `probe_reward_std.sh`.

### 17.1 σ is set by `train/reward_std`, not by weight-space footprint

Picking σ for a new subspace has been the recurring cost of this study — [§11.3](#113-answer-yes--but-scale-σ-not-α)
spent a 4-config sweep on `fura`, [§15.5](#155-result--both-ranks-learn-neither-plateaus-and-rank-matters-enormously)
left `lora` unresolved. The rule that actually predicts the outcome is **the reward spread
in the first few iterations**:

| run | σ | real ‖ΔW‖/‖W‖ | **`reward_std`** | outcome |
|---|---|---|---|---|
| `dense` | 1e-3 | 3.46e-2 | **0.0528** | 71.82 plateau (reference) |
| `fura` | 1.25e-2 | 1.18e-1 | **0.0524** | **72.68 — best on the page** |
| `insparse d=1%` | 1e-3 | 4.05e-3 | 0.0419 | 72.07 |
| `zoact r=1` | 1e-3 | 5.9e-4 | 0.0401 | 70.50 |
| `fura` | 1e-3 | 9.4e-3 | 0.0343 | 60.2 @ 20 — crawls |
| `lora r=44` | 1e-3 | 4.04e-3 | 0.0236 | never plateaus |
| `zoact r=1` | 1.2e-2 | 7.1e-3 | **0.1124** | 68.85 — *worse than its own paper σ* |

**Everything that works sits in 0.040–0.055; below ~0.035 the arm learns but crawls, above
~0.09 it degrades.** The footprint column spans **200×** across the working configs
(`zoact` 5.9e-4 to `fura` 1.18e-1) and orders nothing. Three iterations measure
`reward_std`; a screen costs 2.7 h. `scripts/es/probe_reward_std.sh` does the probe and
`pick_sigma.py` log-interpolates to the target.

⚠️ **[§6](#6-numerical-health)'s footprint table is from the *fake* model.** The
`dense 5.0e-2 / zoact 4.2e-3 / insparse 1.6e-2 / fura 4.0e-3` row comes from
`test_es_perturb_modes.py`, which runs on a 192×144 / 144×256 random matrix with entry std
0.02 — all four reproduce analytically from that (`dense` = σ/0.02, `zoact` = σ/(0.02·√in),
`fura` = σ·√b, `insparse` = σ·√d/0.02). `zoact` scales as 1/√in and `fura` as √b, so **none
of it transfers to a 7 B model**, while [§15.3](#153-numerical-gate)'s LoRA numbers *were*
measured on real weights. Measured consistently on Qwen2.5-Math-7B
(`scripts/es/measure_es_footprint.py`, at σ=0.05 so the bf16 write does not contaminate —
see [§17.4](#174-reference)):

| mode | coeffs | ‖ΔW‖/‖W‖ per unit σ | σ for `dense`'s 3.46e-2 |
|---|---|---|---|
| `dense` | 7,615,616,512 | 34.61 | 1.00e-3 |
| `fura` | 97,771,520 | 9.43 | 3.67e-3 |
| `insparse d=1%` | 65,415,168 | 4.04 | 8.56e-3 |
| `lora r=44` (linear part) | 97,771,520 | 4.04 | 8.56e-3 |
| `zoact r=1` | 1,390,592 | 0.590 | 5.87e-2 |
| `lora_zoact r=1` (linear) | 2,222,080 | 0.49 | 7.1e-2 |
| `lora r=1` (linear) | 2,222,080 | 0.485 | 7.13e-2 |
| `fura_zoact r=1` | 831,488 | 0.123 | 0.281 |

**Two claims elsewhere on this page rest on the fake numbers and are corrected:**

1. [§15.6](#156-the-lora-vs-zoact-control-a-calibrated-direction-beats-a-learned-one)'s
   "⚠️ confounded by footprint, and not by a little — an **11×** gap" compared §6 (fake)
   against §15.3 (real). On one convention `zoact r=1` is **5.9e-4** and `lora r=1`
   **4.85e-4** — **1.2×**. The 15 pp calibrated-beats-learned result is essentially
   **unconfounded**, and the caveat is withdrawn.
2. [§11.4](#114-at-matched-footprint-the-structured-subspace-beats-dense)'s "at *matched*
   footprint the structured subspace beats dense" — `fura`'s winning σ=1.25e-2 is
   **1.18e-1, or 3.4× `dense`'s**, not matched. The *empirical* finding (`fura` ≥ `dense`
   from 1.3% of the parameters) stands; the footprint-matching explanation does not.

### 17.2 LoRA-ES is not linear in σ, and that caps rank 1 structurally

Training **both** LoRA factors makes the perturbation quadratic:

```
ΔW = s·σ·ε_B A₀    +    s·σ²·ε_B ε_A
     linear, informative   cross term, pure noise
```

`‖ε_B ε_A‖_F = σ²√(out·in·r)` against the linear `σ√(out·r)`, so

> **quad / lin = σ·√in — independent of rank.**

Measured at σ=0.05: 4.27 (r=1) and 3.26 (r=44) against 4.12 predicted; the gate at σ=0.13
reads ‖ΔW‖/‖W‖ = **0.5602** where the linear term alone gives 0.063 and the two-term model
gives 0.588 (5%).

**The cross term does not cancel under antithetic sampling.** `W(+ε)` and `W(−ε)` carry the
*same* `+σ²ε_Bε_A` offset, so mirrored sampling removes the odd part and leaves it: the
pair is evaluated at a randomly displaced point, not at `W`. The gate sees this directly as
`(W⁺+W⁻)/2 − W = 2.7e-1` at σ=0.13, against 1.6e-4 for every linear mode.

Rank enters through the σ needed to reach a target footprint, `σ = F·rms·√(in/r)`, giving
**quad/lin = F·rms·in/√r at matched footprint** — a property of the parameterisation, not a
tuning choice:

| arm | quad/lin at `dense`'s footprint | largest clean σ (≤0.3) | clean footprint reachable |
|---|---|---|---|
| `lora r=1` | 3.58 (qkv) / **18.9** (`down_proj`) | 2.2e-3 | 1.1e-3 = **3% of `dense`'s** |
| `lora r=44` | 0.54 / **2.86** | 2.2e-3 | 8.8e-3 = **25% of `dense`'s** |

**So neither rank can be run at `dense`'s footprint cleanly, and rank 1 is hopeless** —
`down_proj` (in=18944) is the binding layer. `zoact r=1` has no cap at all: it perturbs only
the out-side coefficient, so it is exactly linear. This is a **third, mechanical reason**
for §15.6's 15 pp gap that has nothing to do with the projection being uninformed — at rank
1, LoRA-ES cannot take a useful step size at all. It also predicts §15.5's shape: at σ=1e-3
`lora` is clean but 8× under-footprinted, and every σ that fixes the footprint buys noise
instead of signal.

### 17.3 The two calibrated hybrids

Both compose a **structured output side** with zoact's **calibrated input side** — "the
fura/lora update with an activation-aware perturbation".

| mode | ΔW | trained | coeffs | % of 7.6 B |
|---|---|---|---|---|
| `fura_zoact` | `A_j C_j V_j` per input block *j* | `C_j` (b×r) only — **both frames frozen** | **831,488** | **0.011%** |
| `lora_zoact` | `s·B A`, `A` init = top-r calibrated directions | `A` **and** `B`, exactly as `lora` | 2,222,080 (r=1) | 0.029% |

`A_j` is `fura`'s frozen block-SVD frame (`U_j diag S_j`); `V_j` is the calibrated
activation basis restricted to block *j*'s input columns (column *i* of W lives at block
`i//b`, position `i%b`, the same layout as `fura`'s reshape). `V_j` is a *slice* of a
unit-norm row, not itself unit-norm, so blocks carrying more activation energy get a
proportionally larger perturbation — that is the point. `fura_zoact` is the **smallest arm
on the page**, below `zoact r=1`'s 1,390,592.

`lora_zoact` changes **nothing** but `A`'s initialisation, so it is an exactly matched
control — and the footprints make the rank-1 comparison three-way clean:

| arm | projection `V` | trained? | ‖ΔW‖/‖W‖ per unit σ |
|---|---|---|---|
| `zoact r=1` | calibrated | frozen | 0.590 |
| **`lora_zoact r=1`** | **calibrated** | **also trained** | **0.49** |
| `lora r=1` | random | trained | 0.485 |

All three within 1.2×, isolating *informed* from *trainable* with no footprint confound.

**Numerical gate** (`scripts/es/test_zoact_hybrid_es.py`, real weights) — all PASS:

| check | `fura_zoact` r=1 | `lora_zoact` r=1 |
|---|---|---|
| step 0 vs base | 1.291e-3 (bf16 BTT floor, same as `fura`) | **0.0** (B=0) |
| perturb → restore | **0.0** | **0.0** |
| `(W⁺+W⁻)/2 − W` | 3.9e-3 @ σ=0.2 | 2.7e-1 @ σ=0.13 — the §17.2 cross term |
| off-subspace mass of ΔW | **10.2%** @ σ=0.2 | 99.9% — expected, `A` is trained |
| `es_update` moves coefficients | ✓ | ✓ |

⚠️ **The first gate run read 94.5% off-subspace for `fura_zoact` and it was the bf16 write,
not a bug.** At σ=1e-3 its intended footprint is 1.2e-4, ~3× *below* the rounding noise of
the bf16 parameter vLLM runs, so the measured ΔW was mostly quantisation. At σ=0.2 it falls
to 10.2%. The same effect inflates every small-footprint row of a σ=1e-3 footprint table
(`fura_zoact` reads 0.295 per unit σ at σ=1e-3 against a true 0.123) — **measure footprints
at σ well above the floor and divide**, the map is exactly linear.

### 17.4 `dense`: the α search confirms the α/√N rule on a second mode

σ = 1e-3 throughout, 80 iterations, N=10. [§16.4](#164-it-was-the-step-size--fura-at-n10-fully-recovers)
derived "hold α/√N fixed, not α" from `fura` alone; `dense` reproduces it exactly.

| α | motion vs N=30 | **plateau (≥40)** | best @ step | `reward_std` | shape |
|---|---|---|---|---|---|
| 1.5e-4 | 0.52× | 71.76 ± 0.31 | 72.6 @ 80 | 0.0454 | slow start (57.2 @ 10), still rising |
| **2.887e-4** | **1.00×** | **72.68 ± 0.43** | 73.2 @ 50 | 0.0383 | **flat plateau from 50** |
| 5e-4 † | 1.73× | 71.07 ± 0.42 | 73.4 @ 20 | 0.0318 | flat |
| 1e-3 | 3.46× | 69.84 ± 0.84 | 73.0 @ **10** | 0.0340 | rise, then monotone slide to 67.2 |

† the [§16.2](#162-results) run, 150 iterations.

**Unimodal, peak exactly at motion-matched α, one-sided failure** — the same shape as `fura`.
And **72.68 is the best `dense` result anywhere on this page**, at **2.8 GPU-h against 14.8**:

| `dense` | N | iters | plateau | GPU-h |
|---|---|---|---|---|
| paper ([§7](#7-results)) | 30 | 150 | 71.82 ± 0.34 | 14.8 |
| fixed α ([§16.2](#162-results)) | 10 | 150 | 71.07 ± 0.42 | 4.9 |
| **α motion-matched** | **10** | **80** | **72.68 ± 0.43** | **2.8** |

So [§16.3](#163-reading)'s "`dense` pays nothing for N=10 (−0.49, ns)" understates it: with α
corrected, N=10 is **+0.86 pp over N=30 at 5.3× less compute**. The −0.49 was the 1.73×
overshoot, exactly as it was for `fura`.

⚠️ **If the budget is ~10 iterations, use the *larger* α.** α=1e-3 posts 73.0 by step 10
(0.34 GPU-h) — the fastest climb of any `dense` config — before walking back downhill. The
best *plateau* and the best *transient* are at different α.

### 17.5 `lora`: the σ search, and rank 1 rescued by 8.7 pp

α = σ/2·√(10/30) throughout (except the r=1 rows, which keep [§15.5](#155-result--both-ranks-learn-neither-plateaus-and-rank-matters-enormously)'s
α/σ = 5), 80 iterations, N=10.

| rank | σ | quad/lin ([§17.2](#172-lora-es-is-not-linear-in-σ-and-that-caps-rank-1-structurally)) | **plateau (≥40)** | best @ step | `reward_std` |
|---|---|---|---|---|---|
| 44 | 4e-3 | 0.24 / 0.55 | 64.68 ± 0.44 | 66.0 @ 80 | 0.0318 |
| **44** | **1.5385e-2** | 0.92 / 2.1 | **70.04 ± 0.48** | 71.2 @ 80 | 0.0454 |
| 44 | 3e-2 | 1.8 / 4.1 | 68.28 ± 0.43 | 69.4 @ 40 | 0.0473 |
| 44 | 1.3e-1 | 7.8 / 17.9 | *(not run — r=1 below settles it)* | | |
| 1 | **2.2e-3** | 0.13 / 0.30 | **63.76 ± 1.31** | 68.4 @ 80 | 0.0230 |
| 1 | 1.3e-1 | 7.8 / 17.9 | **DEAD** | 51.6 @ 0 | **0.0000** |

**α sweep at the winning σ=1.5385e-2** (added after the σ ladder above, which had α *tied*
to σ throughout and so never tested α independently):

| α/σ | α | **plateau (≥40)** | best @ step | `reward_std` |
|---|---|---|---|---|
| 0.144 | 2.2206e-3 | 68.56 ± 1.02 | 71.0 @ 70 | 0.0552 |
| 0.289 | 4.4412e-3 | 70.04 ± 0.48 | 71.2 @ 80 | 0.0454 |
| **0.500** | **7.6925e-3** | **70.68 ± 0.60** | **72.2 @ 40** | 0.0482 |
| 1.000 | 1.5385e-2 | 22.16 ± 10.76 | 68.2 @ 10 → **1.0 @ 80** | 0.0327 |

**`lora`'s α optimum is one rung higher than `dense`'s and `fura`'s.** Both of those peak at
α/σ = 0.289 and lose ground at 0.5 (`dense` 72.68 → 71.07, `fura` 73.17 → 66.63); `lora`
r=44 is **flat over [0.289, 0.5]** (+0.64 ± 0.77, ns) and only then falls off a cliff — and
the cliff is far steeper, a collapse to 1.0 where `dense` merely sags to 69.84. Take α/σ =
0.5: same plateau within noise, but it peaks at step 40 instead of still climbing at 80.

**Two results from the σ ladder.**

1. **Rank 44 is unimodal in σ with the peak at 1.5385e-2**, and 70.04 beats §15.5's
   67.95 **at 1/5 the compute** (2.8 vs 14.7 GPU-h) — still rising at 80.
2. **Rank 1 gains +8.7 pp from σ alone.** §15.5 read 55.07 at σ=1e-3/N=30/150 it; at
   σ=2.2e-3 with motion-corrected α it reaches **63.76 (best 68.4, still rising) in 2.8
   GPU-h**. σ=2.2e-3 is not arbitrary — it is the largest σ keeping the §17.2 cross term
   under 0.3, set by `down_proj`.

**σ=0.13 is the §17.2 prediction realised.** At quad/lin ≈ 8–18 the perturbation is almost
entirely the uninformative cross term: `reward_mean = reward_std = reward_max = 0.0` and
`train/accuracy = 0.0` on **every** population member from iteration 1 — the model is
destroyed and ES's estimator is identically zero. Stopped after 2 iterations.

**This substantially retracts [§15.6](#156-the-lora-vs-zoact-control-a-calibrated-direction-beats-a-learned-one)'s
15 pp.** That number compared `zoact r=1` against a `lora r=1` that was **8.7 pp
below its own best σ** *and* footprint-mismeasured ([§17.1](#171-σ-is-set-by-trainreward_std-not-by-weight-space-footprint)).
Against `zoact r=1`'s 70.50 the honest gap is now **≤6.7 pp and shrinking** — see
[§17.6](#176-the-two-hybrids) for the like-for-like N=10 comparison.

### 17.9 Reference

* Real-model footprints: `scripts/es/measure_es_footprint.py` (log `logs/es/es_footprint_pass2.log`).
* σ probe: `scripts/es/probe_reward_std.sh` + `pick_sigma.py`.
* Curves/plateaus from chain logs: `scripts/es/collect_es_curves.py` (reproduces §16's
  tables exactly), plotted by `plot_es_curves.py`.
* Hybrid gate: `scripts/es/test_zoact_hybrid_es.py`.
* Mode kernels: `_es_write` / `init_es_state` in
  `verl/verl/workers/rollout/vllm_rollout/es_worker_extension.py`.

## 18. SGD-mask ES — perturb only where plain SGD moves the weights

> *"Do We Need Adam?"* (arXiv:2602.07729) trains RLVR with **vanilla SGD** (lr 1e-1, no
> momentum, bf16) and finds it matches AdamW while touching **< 0.02 % of the parameters**
> (`|θ₁−θ₀| > 1e-5` on bf16 weights; §5, Table 4). That is a *data-driven, coordinate-sparse*
> subspace, found by the gradient rather than by activation statistics — the natural
> control for `insparse` (top activation channels) and `zoact`/`fura_zoact` (calibrated
> directions). This section runs the paper's recipe for **10 steps** on the ES thread's own
> task, takes the set of entries SGD moved as a mask, and runs `PERTURB_MODE=sgdmask` at
> N=10 on GPU 0. Started 2026-09-01.

### 18.1 Why the SGD update is sparse, and what the mask is

With bf16 parameters the update `−η·g` is only *applied* when it exceeds half a ULP of the
weight it lands on (≈6e-5 at |w|=0.02, the median). SGD with η=0.1 needs `|g| > ~6e-4` for
that; AdamW's per-coordinate normalisation lifts every coordinate to ≈η regardless of |g|,
which is why AdamW touches ~10 % and SGD ~0.01 % (the paper's §7). The mask is therefore
**"large-gradient-relative-to-weight-ULP" entries**: big |g| or small |w|. It is a property
of the (task, model, precision) triple, not of the optimiser's trajectory — which is what
makes it usable as an ES subspace.

### 18.2 Setup

| stage | knob | value | note |
|---|---|---|---|
| SGD-GRPO | model / data | Qwen2.5-Math-7B, the **same 64 MATH lvl 3–5 problems** as every ES arm (`head(64)` of the ES parquet, Qwen-Math template) | `datasets/es_math/math_lv3to5_qwenmath_train_b64.parquet` |
| | optimiser | `torch.optim.SGD`, lr **0.1**, momentum 0, weight decay 0, clip 1.0 | the paper's setting (App. A.1/B) |
| | precision | **bf16 module params** (`fsdp_config.model_dtype=bfloat16`) | the sparsity *is* bf16 rounding, so SGD must write bf16 weights — unlike §13's fp32 master |
| | rollout | GRPO, n=8, T=1.0, 1536 response tokens, batch = mini-batch = 64 → **1 optimiser step per batch**, 10 steps | ES train budget; KL off, IS-correction off |
| | eval | greedy MATH-500 at 3000 tokens, verl's `ttrl_math` grader, steps 0/5/10 | comparable to the ES evals (base reads 52.4 here vs 51.6 with the OatZero grader) |
| | hardware | 1 × H100 NVL (GPU 0), `save_contents=[hf_model]` | /data had 87 GB free; one bf16 dump is 15 GB |
| mask | rule | `|W₁₀ − W₀| > 1e-5` entrywise on bf16 weights, 2-D linear weights only | paper threshold; embed/lm_head/norm/bias reported, not perturbed (structured-mode convention) |
| | layout | q/k/v and gate/up concatenated along dim 0 into vLLM's `qkv_proj` / `gate_up_proj`; flat int64 indices | `scripts/es/build_sgd_mask.py` |
| ES | mode | `sgdmask`: `W[idx] = W₀[idx] + C`, `C` fp32, everything else frozen bit-exactly | `es_worker_extension.py` |
| | N / iters / batch | 10 / 80 / fixed 64 | the §17 protocol |
| | σ | by `train/reward_std` probe, target 0.050 (§17.1), α = σ/2·√(10/30) | `probe_reward_std.sh` + `pick_sigma.py` |

Scripts: `scripts/es/run_sgd_mask.sh` (SGD run), `build_sgd_mask.py` (mask + stats JSON),
`chain_sgdmask.sh` (wait → masks → probe → pick → ES). Unit gate: `test_es_perturb_modes.py`
now includes `sgdmask` (init exact, off-mask entries bit-identical after a perturbation,
restore exact, update = α/N·Σzε) — all PASS.

### 18.3 The SGD run and its mask

**SGD-GRPO, 10 steps, 40 min wall-clock on one GPU** (≈190 s/step after the first; three
greedy MATH-500 evals included).

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train score (T=1, n=8) | – | 0.32 | 0.56 | 0.55 | 0.55 | 0.61 | 0.60 | 0.59 | 0.59 | 0.60 | 0.61 |
| grad norm (clip 1.0) | – | 0.66 | 0.18 | 0.16 | 0.15 | 0.15 | 0.33 | 0.35 | 0.15 | 0.13 | 0.13 |
| entropy | – | 0.45 | 0.22 | 0.21 | 0.20 | 0.17 | 0.18 | 0.20 | 0.19 | 0.20 | 0.18 |
| **greedy MATH-500** | **52.4** | | | | | **72.4** | | | | | **72.2** |

The first step does almost everything (grad norm 0.66 at lr 0.1, entropy halves, train score
+24 pp); by step 5 the model sits at **72.4 greedy** — the same band every ES arm plateaus in
after 40–150 iterations (§16–17). That is the §13 zeroth-vs-first-order gap again, now on the
exact ES protocol and metric: **5 SGD steps ≈ 17 min vs 2.8–15 GPU-h for ES.**

**The mask** (`|W₁₀ − W₀| > 1e-5`, bf16):

| | step 5 | step 10 |
|---|---|---|
| entries moved by >1e-5 (all params) | 173,547 = **0.0023 %** | 288,036 = **0.0038 %** |
| entries with *any* bf16 change | 10.3 M = 0.135 % | 12.1 M = 0.159 % |
| **ES mask** (2-D linear weights only) | 112,355 over 85 weights = 0.0015 % | **200,789 over 97 of 112 fused weights = 0.0026 %** |
| overlap step 5 → 10 | 82 % of the step-5 entries persist (Jaccard 0.42) | |

The paper reports 0.01–0.06 % after 270 steps (Table 4); ten steps give 0.004 %, and the
threshold matters — 12 M entries changed by *less* than 1e-5, i.e. weights so small that one
bf16 ULP is below the cut (the `thr=0` arm of §18.5).

Where it lives (step 10):

| module | density | touched rows | touched cols | reading |
|---|---|---|---|---|
| `v_proj` | **0.117 %** | 82 % of rows | 3.3 % of cols | the densest weight by 30× — value projections in almost every head |
| `k_proj` | 0.039 % | 15 % | 11 % | |
| `lm_head` | 0.013 % (71,746) | | | excluded from the ES mask by convention |
| `o_proj` | 0.0046 % | 3.2 % | **0.29 %** | column-concentrated: a few *input channels* |
| `down_proj` | 0.0019 % | 14.7 % | **0.12 %** (639 of 530k) | same — the input-channel picture `insparse` assumes |
| `up_proj` / `gate_proj` | 0.0023 / 0.0009 % | 5.2 / 1.5 % | 3.5 / 3.8 % | |
| `q_proj` | 0.0016 % | | | |
| per layer | L0 0.013 %, L1 0.009 %, L27 0.011 %, middle ≈0.001 % | | | first two and last layers ≈10× the rest |

Two facts pin down *why* these entries and not others. The median changed weight has
**|w| = 1.8e-3 against 1.6e-2 for the model** (9× smaller), and the median change is
**exactly 2⁻¹⁶ = 1.53e-5 — one bf16 ULP at |w| ∈ [2e-3, 4e-3)**. So the mask is mostly "the
smallest weights that still clear the paper's threshold, flipped by one ULP": the update
sparsity is the bf16 mechanism the paper's §7 describes, filtered by a threshold that happens
to sit one ULP above the model's small-weight tail. Its structure is nevertheless not random —
`down_proj`/`o_proj` hits are packed into <0.3 % of their input columns (the activation-outlier
channels), while `v_proj`/`k_proj` hits are spread over most rows.

Stats: `docs/results/ES/sgdmask/sgd_mask_qwen2p5_math_7b_st{5,10}.json`; masks (gitignored)
`datasets/es_math/sgd_mask_qwen2p5_math_7b_st{5,10}.pt`; step-10 bf16 weights kept at
`/data/yequan/bp/BP-q2p5-7b/sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10/global_step_10/actor/huggingface` (15 GB).

### 18.4 σ for a 200k-coordinate subspace: the probe hits the band at the paper's σ, then falls off a cliff

`probe_reward_std.sh` (N=10, 3 iterations per point, log `logs/es/probe_sgdmask_gpu0.log`):

| σ | `train/reward_std` (3 iters) | mean | eval @ 3 | reading |
|---|---|---|---|---|
| **1e-3** | 0.048 / 0.056 / 0.051 | **0.051** | 54.8 | in the 0.040–0.055 band |
| 3e-3 | 0.089 / 0.054 / 0.057 | 0.067 | 56.0 | above |
| 1e-2 | 0.035 / 0.099 / 0.072 | 0.069 | 54.8 | erratic |
| 3e-2 … 1.0 | 0.0 / 0.0 / 0.0 | 0 | 51.6 | **dead** — every member 0/64 from iteration 1 |

Three things are new relative to every other mode on this page.

1. **Perturbing 0.0026 % of the weights at σ=1e-3 gives the same reward spread as perturbing
   100 % of them at σ=1e-3** (`dense`: 0.053, §17.1). Weight-space footprint is
   `σ·√200,789 ≈ 0.45` against `dense`'s `σ·√7.6e9 ≈ 87` — a **190× smaller footprint for the
   same functional spread**. This is the strongest instance yet of §17.1's point that footprint
   does not set σ; *which* coordinates are hit does. (Why these coordinates bite so hard: they
   are small weights — median |w| 1.8e-3 — so σ=1e-3 is a ~50 % relative kick, and they sit in
   `v_proj` and in the outlier input channels of `down_proj`/`o_proj`.)
2. **The response is flat, then binary.** From 1e-3 to 1e-2 `reward_std` barely moves
   (0.051 → 0.069, log-log slope ≈0.13, vs ≈1 for the power-law modes); at 3e-2 the model is
   destroyed outright. That is the signature of a few *load-bearing* coordinates: past the
   point where the ±σ kick exceeds the weights themselves, one sign of every member's noise
   breaks the model and both signs stop producing `\boxed{}` at all.
3. **`pick_sigma.py` would extrapolate to 5.9e-4** — the log-log fit, built for power-law
   profiles, walks *below* the measured in-band point when the slope is ~0. The rule of §17.1
   is the band itself, and σ=1e-3 measured 0.051, so the arm was launched by hand at
   **σ=1e-3, α = σ/2·√(10/30) = 2.887e-4**, the motion-matched N=10 step. The picker needs a
   "prefer a measured in-band point over extrapolation" guard before it is trusted on
   flat profiles.

### 18.5 Result — the SGD-moved coordinates are learnable, and ~6 pp short of `dense`

N=10, 80 iterations, fixed 64 batch, greedy MATH-500; 133 s/iteration, **3.1 GPU-h per arm**.
Log `logs/es/sgdmask_es_gpu0.log`; wandb `ES-q2p5-7b`.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | **plateau (≥40)** |
|---|---|---|---|---|---|---|---|---|---|---|
| `sgdmask thr=1e-5` (200,789 coefs, 0.0026 %) | 51.6 | 58.0 | 62.8 | 63.2 | 65.2 | 66.0 | **68.6** | 66.6 | 65.4 | **66.36 ± 0.61** |
| `sgdmask thr=0` (11,474,522 coefs, 0.15 %) | 51.6 | 58.0 | 63.8 | 65.8 | 67.2 | 66.0 | 67.0 | 68.0 | **68.8** | **67.40 ± 0.47** ‡ |

‡ still rising at 80. `train/reward_std` had fallen to 0.023–0.025 by iteration 80 on both
(train reward 0.72 on the fixed batch).

Against the N=10 leaderboard (§17, same protocol):

| arm | trainable | plateau (≥40) | Δ vs base |
|---|---|---|---|
| `dense` | 100 % | 72.68 ± 0.43 | +21.1 |
| `fura_zoact r=1` | 0.011 % | 71.36 ± 0.64 | +18.2 † |
| `lora r=44` | 1.28 % | 70.68 ± 0.60 | +19.1 |
| `zoact r=1` | 0.018 % | 67.68 ± 0.72 ‡ | +16.1 |
| **`sgdmask thr=0`** | **0.15 %** | **67.40 ± 0.47 ‡** | **+15.8** |
| **`sgdmask thr=1e-5`** | **0.0026 %** | **66.36 ± 0.61** | **+14.8** |
| `lora r=1` | 0.029 % | 63.76 ± 1.31 ‡ | +12.2 |

† from its own 53.2 base. Per-eval SE 2.24 pp.

![N=10 curves: SGD-mask arms against the leaderboard](figs/n10_sgdmask.png)

**Reading.**

1. **It learns — +15 pp from 0.0026 % of the weights**, tracking `zoact r=1`/`lora r=1`
   through step 40 (58.0 at 10, 65.2 at 40) — but it **plateaus ≈6 pp under `dense` and
   ≈5 pp under `fura_zoact`**, which has 4× the coefficients and lands in the `dense` tie.
   "Where SGD moves first" is a usable ES subspace, not a privileged one.
2. **Widening the mask 57× (threshold 0) buys +1 pp and a slope, not a different answer.**
   `thr=0` is 67.40 and still rising at 80, i.e. `zoact r=1`'s curve at 8× its coefficient
   count. So the paper's 1e-5 cut is *not* what is holding the 200k arm back; both masks
   are drawn from the same population — the small-|w| tail (§18.3: median |w| 1.8e-3 for
   `thr=1e-5`, **3.8e-5** for `thr=0`, against 1.6e-2 model-wide).
3. **The subspace has the capacity; the zeroth-order search is what is slow.** The SGD
   model at step 10 *is* a point of the `thr=0` subspace by construction — every one of its
   12.1 M changed entries is in the mask (11.47 M of them in the linear weights) — and it
   scores **72.2 greedy**. ES confined to the same coordinates reaches 67.4 in 80 iterations
   (3.1 GPU-h) and is still climbing; SGD got there in 5 steps (0.28 GPU-h). This is the
   §13 zeroth-vs-first-order gap on the ES thread's own metric for the first time:
   **≈10× fewer GPU-hours for +0.3 pp** (SGD 72.4 @ step 5 vs `dense` ES 72.68 @ 2.8 GPU-h).
4. **σ is set by the coordinates, not the count.** Both masks, 57× apart in size, land
   in the `reward_std` band at the *same* σ=1e-3 (§18.4, and the `thr=0` probe: 0.026 @ 1e-4,
   0.028 @ 3e-4, **0.044 @ 1e-3**, 0.069 @ 3e-3, dead @ 3e-2) — because in both the entries
   hit are small weights for which σ=1e-3 is a ≥50 % relative kick. Footprint ‖σε‖_F differs
   by 7.6× between them and by 190× against `dense`; none of that shows in `reward_std`.

### 18.6 What the paper's sparsity is, seen from here

The "SGD updates < 0.02 % of the parameters" headline reproduces (0.0038 % after 10 steps,
99.996 % sparse) and its mechanism is exactly the paper's §7 conjecture, now measured: the
entries that register are the ones whose bf16 ULP is smaller than `η·g`. That makes the
mask **a precision artefact with structure** — ~1-ULP flips of the weights in [2e-3, 4e-3)
(`thr=1e-5`), or of the tiniest 0.15 % of the weights (`thr=0`, median |w| 3.8e-5, spread
uniformly over all rows/columns/layers). Two consequences for this thread:

* As an ES *subspace* it is worth about what its size predicts on this page's
  coefficient-count axis (between `lora r=1` and `zoact r=1`), and less than the calibrated
  low-rank frames at the same or smaller size. The activation-outlier column structure it
  does have (`down_proj`/`o_proj` hits in < 0.3 % of input channels, §18.3) is what
  `insparse` already targets on purpose, and `insparse d=1 %` sits in the `dense` tie.
* As evidence for "RL lives in a low-dimensional subspace" it is weaker than it reads: the
  bf16 model's *visible* change is 0.16 % of entries, but the fp32 update that produced it
  was dense, and ES cannot recover SGD's result from the visible coordinates in 3 GPU-h.
  The right control — not run — is SGD with an fp32 master on the same 10 steps, ranking
  entries by |ΔW| at a *fixed* density, which separates "where the gradient is large" from
  "where bf16 happens to round".

### 18.7 Takeaways

1. **A 10-step bf16 SGD-GRPO run gives a 0.0026 %-of-the-model coordinate mask; ES on it
   gains +14.8 pp** (66.36 plateau) — learnable, ranked with the small-count arms, ≈6 pp under
   `dense`. The threshold-0 mask (0.15 %) reaches 67.40 and is still rising.
2. **SGD-GRPO on the ES protocol: 72.4 greedy MATH-500 in 5 steps / 17 min / 0.28 GPU-h**,
   vs 2.8 GPU-h for the best ES arm — the cleanest BP-vs-ES number on this page.
3. **The update sparsity is bf16 rounding of a dense update**, concentrated in the
   small-weight tail (median |w| 9× to 400× below the model's). Perturbing exactly those
   coordinates does not recover full-space ES.
4. **Footprint is now 190× decoupled from `reward_std`** (§17.1 extended): 200k small
   coordinates at σ=1e-3 spread rewards like 7.6 B coordinates at σ=1e-3.
5. `pick_sigma.py` now prefers a measured in-band point to its log-log extrapolation — the
   flat-then-cliff profile of a coordinate mask sent the fit *below* the measured hit.

### 18.8 Reference

* SGD run: `scripts/es/run_sgd_mask.sh` → wandb `BP-q2p5-7b`
  `sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10`; log `logs/es/sgdmask_sgd_gpu0.log`;
  step-10 bf16 HF weights at
  `/data/yequan/bp/BP-q2p5-7b/sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10/global_step_10/actor/huggingface`
  (15 GB — delete when the masks are no longer needed; the step-5 dump was removed).
* Masks: `scripts/es/build_sgd_mask.py` → `datasets/es_math/sgd_mask_qwen2p5_math_7b_st{5,10,10_nz}.pt`
  (gitignored) + stats JSON copied to `docs/results/ES/sgdmask/`.
* ES mode: `sgdmask` in `es_worker_extension.py` (`init_es_state` / `_es_write`),
  `es.mask_path`, `MASK_PATH` in `run_es_math.sh`; gate `scripts/es/test_es_perturb_modes.py`.
* Chains: `scripts/es/chain_sgdmask.sh` (thr 1e-5; its automatic σ pick was overridden by hand,
  §18.4), `scripts/es/chain_sgdmask_nz.sh` (thr 0, automatic pick with the new guard → 1e-3).
  Probe logs `logs/es/probe_sgdmask{,_nz}_gpu0.log`; ES logs `logs/es/sgdmask_es_gpu0.log`,
  `logs/es/sgdmask_nz_chain_gpu0.log`; figure `figs/n10_sgdmask.png` via
  `collect_es_curves.py` + `plot_es_curves.py`.

## 19. The leaderboard on the paper-aligned data protocol

> The §17 ranking, re-measured with the **official data protocol** instead of the fixed
> 64-problem batch. Each arm keeps *its own* best (σ, α) from §17 — only the data changes.
> GPUs 6/7, 2026-09-02/03. Launcher `scripts/es/chain_aligned_pop10.sh`.

### 19.1 Why this had to be run

Every number in §7–§18 was measured on **one fixed 64-problem batch, never refreshed** —
16× smaller than the official `--batch-size 1024` and resampled never instead of every
iteration ([§12](#12-alignment-with-the-official-implementation)). That protocol lets ES
memorise: [§11.1](#111-the-64-problem-batch-is-the-ceiling-not-the-method) measured `dense`'s
train-minus-held-out gap swinging **+3.9 → −6.1 pp**, and every arm flattening by step ~40.
So the whole leaderboard was open to the charge that it ranked *which method memorises 64
problems best*.

**Design.** Only the data protocol moves: `train_batch_size=1024`, resampled every iteration
from the full **8,890**-problem pool, 100 iterations, N=10. ⚠️ `TRAIN_MAX_SAMPLES=-1` is
required — the default truncates the pool to 64, and the resample guard
`train_batch_size < len(train_data)` then **silently falls back to the fixed batch**.
Both logs confirm `Training batch: 1024 problems resampled per iteration from a pool of 8890`.

**Remaining deviation:** train token budget stays **1,536** (paper 3,000). [§3](#3-setup-actually-used-and-deviations)
measured that 1,536 keeps 98.4% of the base model's correct answers (p99 = 2,030); at batch
1024 the paper's 3,000 would roughly double an already 15 GPU-h arm. Held-out eval keeps 3,000.

### 19.2 Results

<!-- ALIGNED:RESULTS BEGIN -->

| # | Method | σ / α | Base | **Plateau (≥40)** | Best @ step | trainable | GPU-h |
|---|---|---|---|---|---|---|---|
| 1 | `fura` | 1.25e-2 / 3.61e-3 | 53.2 | **73.89 ± 0.25** | 75.0 @ 90 | 1.28% | 14.9 |
| 2 | `dense` | 1e-3 / 2.89e-4 | 51.6 | **72.83 ± 0.55** | 74.2 @ 90 | 100% | 15.9 |
| 3 | `isobtt` (fura + ISO) | 5e-2 / 1.44e-2 | 53.2 | **72.80 ± 0.40** | 74.4 @ 70 | **0.64%** | 15.1 |
| 4 | `fura_zoact` r=1 | 5e-2 / 1.44e-2 | 53.2 | **72.46 ± 0.46** | 73.8 @ **100** | **0.011%** | 15.2 |
| 5 | `lora r=44` | 1.538e-2 / 7.69e-3 | 51.6 | **71.23 ± 0.52** | 73.0 @ 50 | 1.28% | 14.8 |
| 6 | `lora r=1` | 2.2e-3 / 2.54e-2 | 51.6 | **60.91 ± 1.76** ✗ | 69.6 @ 30 | 0.029% | 15.2 |

✗ diverging — 69.6 @ 30 → 57.8 @ 70 → 54.6 @ 100; see reading 3 below.

MATH-500 curves:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fura` | 53.2 | 69.6 | 73.4 | 73.2 | 72.8 | 73.8 | 73.8 | 73.8 | 74.2 | **75.0** | 73.8 |
| `dense` | 51.6 | 70.6 | 72.4 | 69.2 | 70.4 | 71.2 | 73.6 | 73.0 | 73.8 | **74.2** | 73.6 |
| `fura_zoact` | 53.2 | 65.0 | 69.0 | 70.4 | 73.0 | 70.8 | 72.4 | 70.8 | 72.8 | 73.6 | **73.8** |
| `isobtt` | 53.2 | 67.6 | 69.4 | 71.4 | 71.2 | 73.4 | 72.2 | **74.4** | 72.0 | 73.0 | 73.4 |
| `lora r=44` | 51.6 | 68.8 | 71.2 | 70.6 | 71.0 | **73.0** | 71.8 | 72.6 | 70.4 | 70.8 | 69.0 |
| `lora r=1` | 51.6 | 63.6 | 64.0 | **69.6** | 68.4 | 64.0 | 63.6 | 57.8 | 58.4 | 59.6 | 54.6 |

<!-- ALIGNED:RESULTS END -->

### 19.3 Reading

**1. The ranking survives, unchanged.** `fura` > `dense` > `fura_zoact` > `lora r=44` on
both protocols, and every arm *gains*:

| arm | fixed 64 | aligned 1024 | Δ |
|---|---|---|---|
| `fura` | 73.17 ± 0.33 | **73.89 ± 0.25** | +0.72 |
| `dense` | 72.68 ± 0.43 | **72.83 ± 0.55** | +0.15 |
| `fura_zoact` | 71.36 ± 0.64 | **72.46 ± 0.46** | +1.10 |
| `lora r=44` | 70.68 ± 0.60 | **71.23 ± 0.52** | +0.55 |

So §17's conclusions are **not** artefacts of the memorising batch. ⚠️ But the adjacent gaps
are small — `fura`−`dense` = 1.06 ± 0.60, `dense`−`fura_zoact` = 0.37 ± 0.72,
`fura_zoact`−`lora` = 1.23 ± 0.69. Only `fura` vs `lora` (2.66 ± 0.58) separates cleanly.
What is solid is that the *order* reproduces across two protocols, not any single gap.

**2. The memorisation ceiling is gone.** Three of four arms post their best at step 90–100
and are still rising, where every fixed-batch arm was flat by 40. The §11.1 ceiling was the
batch, exactly as claimed — and it means these plateaus are still lower bounds at 100
iterations (the paper runs 500).

**2b. Freezing the spectrum exactly still costs nothing.** `isobtt` — fura's block-wise SVD
with `A_j` frozen and the small core `R_j` held in `O(b)` by a Cayley step — lands at
**72.80 ± 0.40 from 0.64% of the weights**, statistically tied with full `dense`'s 72.83
(−0.03 ± 0.68) and above `fura_zoact`. The constraint held exactly: `max|RᵀR − I|` stayed at
**1.0e-6** (fp32 round-off) for all 100 iterations. So [§10](#10-iso-fixed-spectrum-es)'s
result reproduces on the aligned protocol — all the gain is frame rotation, none of it needs
the singular values to move.

**3. Both LoRA arms fail to transfer, and they are the only ones that do.** `lora r=44`
*declines*: 73.0 @ 50 → 69.0 @ 100. It is the
only arm that peaks early and walks back down, which is the α-too-large signature — its
α/σ = 0.5 was chosen on the fixed batch ([§17.5](#175-lora-the-σ-search-and-rank-1-rescued-by-87-pp))
where it was the best of four. `lora r=1` is worse — **69.6 @ 30 → 54.6 @ 100**, a clear
divergence, at `reward_std` **0.0075**, less than half the aligned healthy band: it takes large
steps on weak signal. **Four of four structured arms transfer their fixed-batch σ/α cleanly;
two of two LoRA arms do not.** That is consistent with [§17.2](#172-lora-es-is-not-linear-in-σ-and-that-caps-rank-1-structurally):
the bilinear parameterisation's effective step is `σ·ε_B A₀ + σ²·ε_B ε_A`, so it depends on
the reward landscape in a way the linear modes' does not, and a step calibrated on one data
distribution does not carry to another. Both LoRA arms need their own σ/α search here.

**4. ⚠️ The `reward_std` band of [§17.1](#171-σ-is-set-by-trainreward_std-not-by-weight-space-footprint)
is batch-size-dependent and does not transfer.** At batch 1024 the same σ gives:

| arm | `reward_std` @ batch 64 | @ batch 1024 | ratio |
|---|---|---|---|
| `dense` | 0.0383 | 0.0156 | 0.41 |
| `fura` | 0.0394 | 0.0174 | 0.44 |
| `fura_zoact` | 0.0519 | 0.0237 | 0.46 |
| `lora r=44` | 0.0482 | 0.0210 | 0.44 |

A strikingly consistent **≈0.44×** — larger than the 0.25× that pure 1/√B sampling would
predict, so part of the spread is genuine perturbation signal that does *not* average away.
**The healthy band at batch 1024 is ≈0.016–0.024**, and anyone tuning a new arm here with
§17.1's 0.040–0.055 would set σ far too high. Quote the band with its batch size.

## 20. BP `fura` — the small-core subspace under true gradients, on the dense-SGD protocol

> The BP counterpart of the ES `fura` arm, run on **exactly the §18.3 dense SGD-GRPO protocol**
> (Qwen2.5-Math-7B, the fixed 64-problem batch, GRPO n=8 T=1.0, 1536-token rollouts, vanilla SGD,
> 10 steps, greedy MATH-500 at 3,000 tokens) so that fura-vs-dense is a comparison of the
> *subspace* under the same first-order optimiser. Dense reference: lr 0.1 → 52.4 / **72.4 @5** /
> 72.2 @10. LR searched from **10× the dense LR** in both directions (0.3 / 1.0 / 2.0 / 3.0), plus a per-step-eval
> rerun of the dense reference. GPU 2, 2026-09-05/06.
> Launcher `scripts/es/run_bp_fura_math.sh`, chain `scripts/es/chain_bp_fura_lr.sh`, table
> `scripts/es/collect_bp_fura.py`. wandb `BP-q2p5-7b`, runs `sgd-fura_math-lv3to5-b64_lr*_n8_fp32_st10`.

### 20.1 Setup — what differs from the dense reference, and why

| knob | dense reference (§18.3) | BP `fura` | why |
|---|---|---|---|
| adapter | none (all 7.62 B) | `blocktt`, `output_one_block`, rank full, small core trains, `s_merged_to=frozen`, `factorize_by_head=False`, `train_bias=False` | per input block j, `W[:, blk_j] = A_j R_j`, `A_j = U_j S_j` frozen, `R_j = Vh_j` (b×b) trained — the ES `fura` factorisation (3584 → 56×64, 18944 → 128×148) |
| trainable | 100 % | **117.0 M = 1.54 %** | HF keeps q/k/v and gate/up as separate Linears, so 6+1 cores per layer vs ES fura's 3+1 on vLLM's fused weights (97.8 M); same discrepancy §13.1 accepted for `isobtt` |
| params dtype | bf16 (the §18 object of study) | **fp32 master**, bf16 compute (FSDP2 mixed precision) | the cores have entries ~0.1 whose bf16 half-ULP (~5e-4) is ~40× the per-entry SGD step at lr 1, so bf16 cores would silently drop the update |
| FSDP | FSDP1 | FSDP2 | blocktt's mixed trainable/frozen params break FSDP1's writeback |
| eval cadence | steps 0/5/10 | **every step** | convergence-speed question |
| optimiser | SGD lr 0.1, no momentum, wd 0, clip 1.0 | same, **lr ∈ {1.0, 0.3, 3.0, 0.1}** | see below |
| base reads | 52.4 | 51.4 | bf16 export of the fp32 `A_j R_j` product; within the ±2.2 pp eval SE of 51.6/52.4 |

**Why fura needs a larger LR under SGD.** The update to the small core is `ΔR_j = −η A_jᵀ G_j`, so
in weight space `ΔW_j = −η U_j S_j² U_jᵀ G_j`: the gradient is projected onto each block's b-dimensional
column space (≈ √(b/out) ≈ 0.13 of its norm survives) and re-weighted by S². Measured: **grad norm
0.033 at step 1 vs 0.66 for dense** (20× smaller, clip 1.0 never engages), so at lr 1.0 fura's
per-step weight motion is roughly 2× dense's at 0.1. That is the analytic reason "start at 10×"
is the right bracket, not just a convention.

### 20.2 Results

<!-- BPFURA:RESULTS BEGIN -->

Greedy MATH-500 (verl `ttrl_math` grader, 3,000 tokens) every step. `train` = GRPO rollout score of
the batch *before* that step's update. Every run is one seed; the T=1 rollouts differ per run.

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **dense SGD lr 0.1**, §18.3 run (evals @5/10) | 52.4 | | | | | 72.4 | | | | | 72.2 |
| **dense SGD lr 0.1**, rerun with per-step evals | 52.4 | **73.6** | 75.0 | 73.0 | 75.0 | 75.2 | 77.6 | 77.4 | **78.2** | 76.6 | 76.2 |
| dense rerun `train` | | 0.32 | 0.55 | 0.62 | 0.60 | 0.64 | 0.62 | 0.61 | 0.64 | 0.65 | 0.66 |
| dense rerun grad norm | | 0.66 | 0.19 | 0.17 | 0.13 | 0.13 | 0.14 | 0.14 | 0.12 | 0.11 | 0.12 |
| fura SGD lr 0.3 (3×) | 51.4 | 55.0 | 59.8 | 59.4 | 62.6 | 66.0 | 64.4 | 69.2 | 69.0 | 69.4 | 71.2 |
| **fura SGD lr 1.0** (10×) | 51.4 | 58.8 | 62.8 | 68.4 | 72.0 | 73.0 | 73.2 | 73.0 | 73.4 | 72.6 | 73.4 |
| fura lr 1.0 `train` | | 0.31 | 0.43 | 0.39 | 0.49 | 0.53 | 0.51 | 0.57 | 0.54 | 0.56 | 0.56 |
| fura lr 1.0 grad norm | | 0.033 | 0.037 | 0.035 | 0.034 | 0.025 | 0.028 | 0.028 | 0.026 | 0.021 | 0.025 |
| **fura SGD lr 2.0** (20×) | 51.4 | 63.8 | 69.6 | 73.8 | **76.0** | 74.2 | 73.4 | 75.6 | 74.8 | 73.2 | 72.6 |
| fura lr 2.0 `train` | | 0.31 | 0.46 | 0.47 | 0.53 | 0.54 | 0.59 | 0.58 | 0.59 | 0.62 | 0.63 |
| fura SGD lr 3.0 (30×) | 51.4 | 62.6 | 72.4 | 73.8 | 74.2 | **9.6** ✗ | 0.0 ✗ | – | – | – | – |
| fura lr 3.0 `train` | | 0.31 | 0.41 | 0.56 | 0.57 | 0.56 | 0.03 | | | | |

Plateau = mean ± sd of the evals from the first step ≥ 72 onwards. Timing is per step on GPU 2
(gen / update / greedy eval), all runs on the same GPU except the §18.3 dense run (GPU 0).

| arm | trainable | first ≥ 72 | best | plateau | final @10 | train @10 | s/step (gen / update / eval) | wall |
|---|---|---|---|---|---|---|---|---|
| dense lr 0.1, §18.3 run | 100 % | ≤ 5 | 72.4 @5 | 72.3 (5, 10) | 72.2 | 0.61 | 203 (48 / 119 / 53) | 40 min (3 evals) |
| **dense lr 0.1, rerun** | 100 % | **1** | **78.2 @8** | **75.8 ± 1.6** (1–10) | **76.2** | 0.66 | **133 (38 / 73 / 46)** | 36 min (11 evals) |
| fura lr 0.3 | 1.54 % | – (71.2 @10, rising) | 71.2 @10 | – | 71.2 | 0.47 | 149 | 39 min |
| fura lr 1.0 | 1.54 % | 4 | 73.4 @8 | 72.9 ± 0.5 (4–10) | 73.4 | 0.56 | 152 (49 / 75 / 56) | 38 min |
| **fura lr 2.0** | 1.54 % | **3** | **76.0 @4** | **74.3 ± 1.2** (4–10) | 72.6 | 0.63 | 149 (47 / 75 / 66) | 40 min |
| fura lr 3.0 | 1.54 % | 2 | 74.2 @4 | ✗ destroyed @5 | 0.0 @6 | – | 150 | killed @6 |

<!-- BPFURA:RESULTS END -->

### 20.3 Reading

1. **Final score: fura lands inside the dense run-to-run spread, not above it.** Two dense runs
   at identical settings read **72.2 and 76.2 @10** (plateaus 72.3 and 75.8 ± 1.6) — the T=1
   rollouts differ per run and that alone moves every eval by 3–4 pp. fura's best LR gives
   74.3 ± 1.2 (lr 2.0) and 72.9 ± 0.5 (lr 1.0), i.e. between the two dense runs and ~1.5 pp under
   the better one. The honest statement is **fura ≈ dense within single-seed noise, with no
   evidence it is better** — the ES-side "fura ≥ dense" (§17/§19) does not carry over to BP on
   this evidence, and a second seed of each arm is needed before any ±2 pp claim.
2. **Convergence: dense is faster in steps and in wall-clock.** With per-step evals dense goes
   **52.4 → 73.6 in one SGD step** (train 0.32 → 0.55, grad norm 0.66) and is at plateau by
   step 2; fura needs **3 steps at lr 2.0** (73.8) and **4 at lr 1.0** (72.0) to cross 72, with
   a 20× smaller gradient (0.033) that the higher LR only partly compensates. Per-step cost is the
   same (update 75 s vs 73 s — the frozen 7.6 B cores save the weight-grad GEMMs but the fp32
   master and the BTT forward give it back), so time-to-72 is **≈ 2.5 min vs ≈ 7.5 min**. The
   §13.5 "BP is far cheaper than ES" point stands for both; the subspace does not make BP faster.
3. **The LR window is 10–20× the dense LR, and it is a cliff on the high side.** 0.3 (3×) is
   clearly starved (71.2 @10, still rising); 3.0 (30×) climbs fastest of all (72.4 @2, 74.2 @4,
   train 0.57 @4 — the dense train curve exactly) and then **one update takes it 74.2 → 9.6 →
   0.0** with entropy 0.20 → 1.9. That is the §11.2 shape of the ES `fura` sweep (12.5× in, 40×
   out, "fastest early climb before degrading") reproduced under true gradients: the small-core
   update `ΔW = −η U S² Uᵀ G` is re-weighted by S², so the largest-singular-value directions take
   steps ~2× the mean and overshoot first. Why the window sits at 10–20× rather than the analytic
   ~5× (§13.4's √b rule): the gradient's projection onto the block column spaces keeps only
   ~13 % of its norm, so a larger η is needed for the same weight-space motion (§20.1).
4. **lr 2.0 drifts after its peak; lr 1.0 does not.** At 2.0 train climbs to 0.63 while held-out
   slides 76.0 @4 → 72.6 @10; at 1.0 held-out holds 73 ± 0.5 through step 10 at train 0.56. The
   dense rerun keeps rising to step 8 at train 0.66, so this is not simply "fixed-batch
   overfitting" (§11.1) — the 20× step is at the edge of the window and walks the weights back
   downhill after the first few updates. **Recipe on this protocol: lr 2.0 for ≤ 5 steps, lr 1.0
   for longer.** Both should be re-checked on the resampled protocol (§19), where the ES `fura`
   arm keeps improving to step 90.
5. **Against ES, the gap is the optimiser, not the subspace.** ES `fura` at N=10 needs ~20
   iterations / 0.7 GPU-h to cross 72 (§17); BP `fura` does it in 3–4 steps / ≈ 8 min / 0.13
   GPU-h, and dense BP in one step. The §13/§18 zeroth-vs-first-order gap holds on fura's own
   subspace: ≈ 5× in GPU-hours, 5–7× in steps.
6. **Method caveat for every number on this page: one seed.** The two dense runs are the first
   same-config repeat in the BP leg and differ by more than the per-eval SE (2.2 pp) at every
   step. §17–§19's ±0.5 pp "plateau SE" is the within-run eval spread, not the seed-to-seed
   spread; ranking claims at the 1–2 pp level need a second seed.

### 20.4 Reference

* Logs `logs/es/bp_fura_lr{1.0,0.3,3.0,0.1}_gpu2.log`, chain log `logs/es/chain_bp_fura_gpu2.out`.
  The dense reference is `logs/es/sgdmask_sgd_gpu0.log` (§18.8); its per-step-eval rerun is
  `logs/es/bp_dense_sgd_tf1_gpu2.log` (`TEST_FREQ=1 SAVE_FREQ=0 DEVICES=2 bash scripts/es/run_sgd_mask.sh`,
  wandb `sgd-dense_math-lv3to5-b64_lr0.1_n8_bf16_st10_tf1`). The lr 0.1 fura run was dropped from
  the chain once 0.3 had shown under-stepping; lr 2.0 was added once 3.0 had diverged.
* **GPU choice.** The run was asked for on GPU 4, but a 15-minute memory sample showed the colleague
  job idling there at 8.6 GB with ~20 s spikes to **62 GB** — a 7B run (78 GB peak here) would OOM one
  of the two. GPU 2's job stayed at 2.7 GB over 17 minutes of sampling, so the chain runs there.
* **verl memory leak fixed on the way** (`verl/workers/sharding_manager/fsdp_vllm.py`): with a PEFT
  adapter that overrides `export_for_vllm`, the dense export dict stayed referenced through the
  rollout, so `del params` freed nothing — 30 GB of fp32 export sat next to vLLM's KV cache and the
  GPU read **92 GB** at a 0.40 vLLM budget. The refs are now dropped before `del params` and the
  blocktt export is cast to bf16; the same run now peaks at **78 GB** with a 0.35 budget. This also
  affected the §13 `iso*` arms (bf16 export, ~15 GB).
* Two launch gotchas are in memory (`bp-fura-launch-gotchas`): a transient `ray start` timeout when
  another session's Ray holds the default dashboard ports, and `kill $!` after `nohup setsid sh -c`
  killing only the exited parent (a queued follow-up survived and briefly launched a second chain
  onto the same GPU; both were killed and the chain relaunched once, 23:53).
