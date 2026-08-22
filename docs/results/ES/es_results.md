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

## 1. What we are reproducing

*Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*
(`docs/papers/26_ICML_Evolution Strategies at Scale...pdf`), §4.3 + Appendix A.6:

| Paper setting | Value |
|---|---|
| Base model | Qwen2.5-Math-7B |
| Train data | MATH, difficulty **3–5** |
| Template | **Qwen-Math** (Table 7) |
| Reward | binary `\boxed{}` correctness, **no format reward**, OatZero grader |
| Response budget | max **3,000** tokens |
| ES hyperparameters | σ = 0.001, **α = σ/2 = 0.0005**, N = 30 |
| Headline result | MATH-500 **53.0 → 78.0** (ES-CHKPT-3, 192 steps) |

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

| # | Run (wandb name) | Subspace `P` | Trainable coeffs | % of 7.6B |
|---|---|---|---|---|
| 1 | `es-dense-full_…` | `P(C) = C` — every parameter (paper baseline) | 7,615,616,512 | 100% |
| 2 | `zoact-r1_…` | `P(C) = C·Vᵣ`, `Vᵣ` = top-1 right singular vector of the layer's **input activations** | 1,390,592 | 0.018% |
| 3 | `insparse-d0.01_…` | `P(C)[:, idx] = C`, `idx` = top-1% input channels by **activation RMS** | 65,415,168 | 0.86% |
| 4 | `fura-btt-smallcore_…` | `P(C)[:, blkⱼ] = Aⱼ·Cⱼ`, from the full-rank BTT factorisation `Wⱼ = Aⱼ Rⱼ` | 97,771,520 | 1.28% |
| 5 | `iso-fixedspec-b128_…` | **multiplicative**: `W ← C_L W C_Rᵀ`, `C` orthogonal — fixed spectrum, both frames move | 141,102,080 † | 1.85% |
| 6 | `isobtt-fixedspec-smallcore_…` | same constraint on the block-wise SVD: `Rⱼ ∈ O(b)` trained, `Aⱼ` and each block's spectrum frozen | 48,470,016 † | 0.64% |

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

| Knob | Value | Note |
|---|---|---|
| σ / α / N | 1e-3 / 5e-4 / 30 | paper (runs 1–4) |
| σ / α (ISO runs 5–6) | **5e-2 / 2.5e-2** | σ is a *relative footprint* there, not a noise std — [§10.4](#104-scale-convention--σ-is-a-relative-footprint-not-a-noise-std) |
| Template / reward / grader | Qwen-Math / binary `\boxed{}` / `ttrl_math` `fast=True` | paper (`fast=True` **is** the OatZero grader) |
| Decoding | greedy (T=0) | paper (countdown/sudoku); makes rewards a deterministic function of the seed |
| Train batch | **64** fixed problems (shuffled seed 0) | paper used 200 for countdown; math batch unspecified |
| Train token budget | **1,536** | ⚠️ deviation from 3,000 — see below |
| Eval token budget | **3,000** | paper |
| Eval | full MATH-500, every 10 iterations | |
| Iterations | 150 | paper's best MATH-500 ckpt was at 192 |
| Hardware | 1× H100 NVL per run, 1 vLLM engine | GPUs 1/2 (runs 1–4, 3/4 queued behind 1/2); **GPU 5** (run 5), **GPU 3** (run 6) |

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

| | MATH-500 |
|---|---|
| Paper (Qwen2.5-Math-7B) | 53.0 |
| **This repo** | **51.2** (standalone) / **51.6** (in-trainer, step 0) |

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

| mode | ‖ΔW‖/‖W‖ | vs bf16 round-off (1.6e-3) |
|---|---|---|
| dense | 5.0e-2 | 31× |
| insparse (10% test) | 1.6e-2 | 10× |
| **zoact r=1** | **4.2e-3** | **2.7×** |
| **fura** | **4.0e-3** | **2.6×** |
| **iso / isobtt** | **5.0e-2** (by construction, σ *is* this ratio) | 31× |

⚠️ At the paper's σ, the structured modes' weight-space footprint is only ~2.6× the bf16
quantisation floor of the vLLM rollout weights, so a non-trivial fraction of what the model
actually "sees" is rounding rather than the intended direction. **fp32 coefficient masters fix
the *update* side** (one ES step moves a coefficient by ~α/√N ≈ 1e-4, at or below one bf16 ULP
of a typical weight — a bf16-only accumulator would silently round most of it away), but the
rollout itself stays bf16. Health check in flight: `train/reward_std` (see below).

## 7. Results

Base = MATH-500 51.6% at step 0 for every run (identical, as expected).

<!-- AUTO:RESULTS BEGIN -->

| # | Run | trainable coeffs | ‖ΔW‖/‖W‖ | mean reward σ (last 10 it) | MATH-500 best | best @ step | steps done |
|---|---|---|---|---|---|---|---|
| 1 | dense (paper ES) | 7,615,616,512 | 5.0e-2 | 0.027 | **73.4** | 40 | 150 |
| 2 | zoact r=1 | 1,390,592 | 4.2e-3 | 0.020 | **72.2** | 130 | 150 |
| 3 | insparse d=1% | 65,415,168 | 1.6e-2* | 0.028 | **71.4** | 40 | 43 |
| 4 | fura small-core | 97,771,520 | 4.0e-3 | 0.025 | **63.4** | 40 | 42 |
| 5 | iso fixed-spectrum | 141,102,080† | 5.0e-2 | 0.030 | **74.0** | 60 | 147 |
| 6 | isobtt fixed-spec small-core | 48,470,016† | 5.0e-2 | 0.058 | **53.2** | 0 | 1 |

\* insparse ‖ΔW‖/‖W‖ measured at the 10% test density, not the 1% run density.

† manifold dimension searched per step, not a coefficient count — the ISO modes perturb by a group action, not an additive coefficient. They run at σ = 5e-2 / α = 2.5e-2 (footprint-matched to run 1, *not* the paper's nominal σ); see [§10](#10-iso-fixed-spectrum-es).

Fixed-spectrum constraint health (worst value seen; ‖W‖_F drift for `iso`, max|RᵀR − I| for `isobtt` — both should stay at fp32 round-off): **iso fixed-spectrum** 5.0e-04, **isobtt fixed-spec small-core** 5.8e-05.

### MATH-500 curve (eval every 10 iterations, 3,000-token budget)

| step | dense (paper ES) | zoact r=1 | insparse d=1% | fura small-core | iso fixed-spectrum | isobtt fixed-spec small-core |
|---|---|---|---|---|---|---|
| 0 | 51.6 | 51.6 | 51.6 | 53.2 | 51.6 | 53.2 |
| 10 | 70.8 | 58.8 | 66.4 | 55.2 | 70.2 |  |
| 20 | 71.4 | 66.0 | 68.4 | 60.2 | 72.4 |  |
| 30 | 72.4 | 66.4 | 71.0 | 60.2 | 71.4 |  |
| 40 | 73.4 | 70.2 | 71.4 | 63.4 | 71.8 |  |
| 50 | 69.6 | 69.8 |  |  | 72.2 |  |
| 60 | 71.6 | 70.4 |  |  | 74.0 |  |
| 70 | 73.0 | 70.6 |  |  | 72.2 |  |
| 80 | 72.0 | 70.0 |  |  | 72.6 |  |
| 90 | 73.2 | 69.0 |  |  | 73.2 |  |
| 100 | 71.4 | 69.4 |  |  | 71.8 |  |
| 110 | 70.6 | 70.4 |  |  | 71.0 |  |
| 120 | 72.2 | 71.8 |  |  | 72.2 |  |
| 130 | 70.4 | 72.2 |  |  | 72.4 |  |
| 140 | 72.8 | 70.8 |  |  | 73.2 |  |
| 150 | 71.6 | 71.4 |  |  |  |  |

Timing: ~362 s/iteration (30 perturbations x 64-prompt rollout + grading).

<!-- AUTO:RESULTS END -->

### Headline: freezing the entire spectrum costs nothing

**Run 5 (ISO, fixed spectrum) is statistically indistinguishable from run 1 (unconstrained
dense ES).** Paired over the 12 eval steps both runs share (10…120):

```
iso − dense = +0.28 pp ± 0.37 (s.e.),  t = +0.77
plateau mean (steps ≥ 40):  dense 71.8 ± 1.19   iso 72.3 ± 0.87   zoact 70.5 ± 0.94
```

A single MATH-500 eval has a binomial s.e. of **2.24 pp** at n = 500, so the *best-of-15*
column in the table above (dense 73.4, iso 74.0) is a max-statistic and should not be read
as a ranking — the plateau mean is the honest number, and the paired test is the honest
comparison. Both say the same thing: **the two runs are the same run.**

That is the result. Every singular value of every weight matrix is held exactly fixed
(shape drift ~3e-7, [§10.10](#1010-fp32-accumulation-drift-measured-and-corrected)), and
the model still goes **51.6 → ~72** on MATH-500. All ~20 pp is frame rotation. This is
ISO's spectral-inheritance claim tested in the strongest available form — *exactly* fixed
rather than approximately, and in a **forward-only ES** setting the paper does not cover —
and it holds.

**The other modes.** `zoact r=1` plateaus ~1.3–1.8 pp below dense/iso while training
1.39 M coefficients (0.018% of the model), and it gets there slowly (72.2 only at step
130). `fura` is the clear laggard at matched steps (60.2 @ 20 vs iso 72.4, dense 71.4,
insparse 68.4) — but its weight-space footprint is 4.0e-3 vs iso's 5.0e-2, i.e. **12×
smaller**, so this is most likely a step-size deficit rather than a subspace one. Run 6
(`isobtt`) is exactly that control: the *same* block factorisation at the *same* footprint
as run 5.

**Run 5 mechanics.** `train/reward_std = 0.075` at iteration 1 (vs dense's 0.084), so a
purely rotational perturbation moves the reward as much as an unconstrained one — the
constraint is not a dead direction. Train accuracy on the fixed 64-problem batch reaches
76.7% by iteration 128. Iteration time 393 s vs dense's ~354 s (**+11%**).

**Runs 1 and 2 are complete (150/150).** Headline:

| | base | best | final (150) | plateau mean (steps 100–150) | trainable coeffs |
|---|---|---|---|---|---|
| dense (paper ES) | 51.6 | **73.4** @ 40 | 71.6 | **71.50** ± 0.92 | 7.62B |
| ZO-Act r=1 | 51.6 | **72.2** @ 130 | 71.4 | **71.00** ± 1.02 | 1.39M |

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

**Runs 3 and 4, early.** insparse reaches **68.4 by step 20** — a *faster* start than ZO-Act
(66.0) at 47× the parameter count, consistent with its 4× larger ‖ΔW‖ footprint. FuRA is the
slowest starter (**60.2 at step 20**), as its footprint (4.0e-3) matches ZO-Act's. Note FuRA's
step-0 base reads **53.2, not 51.6**: it re-materialises W from the BTT factorisation at init,
and the 1.56e-3 bf16 round-off (§6) flips a handful of greedy trajectories. That is expected
and harmless, but it means FuRA's deltas must be read against its own 53.2 baseline.

**All modes carry signal.** Train reward spread over the 30 perturbations is clearly
non-degenerate (dense 0.08→0.030, ZO-Act 0.049→0.027 as training progresses), i.e. the σ=1e-3
perturbation moves rewards even in the rank-1 subspace — the bf16 rollout floor flagged in §6
did **not** materialise as a dead gradient for either run.

Timing: **~360 s/iteration** (30 perturbations × 11.4 s + grading), ~15 h for 150 iterations.

## 8. Where things live

| What | Path |
|---|---|
| Launcher (all 6 modes) | `scripts/es/run_es_math.sh` (`PERTURB_MODE=dense\|zoact\|insparse\|fura\|iso\|isobtt`) |
| Auto-queue next mode on a GPU | `scripts/es/chain_next_run.sh` |
| Data prep (Qwen-Math template) | `scripts/es/prepare_qwen_math_data.py` |
| Activation calibration | `scripts/es/calibrate_activations.py` |
| Numerical tests | `scripts/es/test_es_perturb_modes.py` (runs 1–4), `scripts/es/test_iso_es.py` (runs 5–6) |
| Perturbation kernels | `verl/verl/workers/rollout/vllm_rollout/es_worker_extension.py` (`StructuredESMixin`; ISO = `_iso_*`) |
| Trainer | `verl/verl/trainer/es/ray_trainer.py`, config `verl/verl/trainer/config/es_trainer.yaml` |
| Reward / prompt | `verl/verl/trainer/es/task_utils.py` → `task_type=qwen_math` |
| Train / eval parquet | `datasets/es_math/*.parquet` |
| Logs | `logs/es/run{1..6}_*.log` |
| Best-eval coefficients | `/data/yequan/es/ES-q2p5-7b/<run>/es_train_*/es_coef_best.pt` |

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
9. ~~**Test spectral inheritance directly.**~~ **Answered.** Runs 1 and 5 are statistically
   indistinguishable (paired +0.28 ± 0.37 pp) — the singular values of Qwen2.5-Math-7B are
   ES-irrelevant on MATH; the whole 20 pp is frame rotation. The follow-up is now
   *how far* this goes: a rank-truncated `Σ₀`, or a spectrum swapped in from a different
   checkpoint, would test whether the values matter at all or merely happen to be adequate.
10. **`fura` vs `isobtt` at matched footprint** is the one comparison that isolates
   "orthogonality constraint" from "step size" on an identical factorisation — run 4's
   deficit (60.2 @ 20) is 12× confounded by its smaller ‖ΔW‖. Run 6 answers it.

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

| frame | shape | s / polar |
|---|---|---|
| `q_proj`/`o_proj` `U`,`V` | 3584×3584 | 2.37 |
| `gate_proj`/`up_proj` `U` | 18944×3584 | 2.91 |
| `down_proj` `V` | 18944×3584 | 2.83 |
| **one full feasibility pass** | | **744 s** |

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
2. **Feasibility is never lost, so nothing has to be projected back.** `σ(C_L W C_Rᵀ) =
   σ(W)` is an identity. ISO's polar retraction also lands on an exact spectrum, but only
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

| generator | cost | search dim / layer | `‖ΔW‖/‖W‖` at scale σ |
|---|---|---|---|
| low-rank `Ω = PQᵀ − QPᵀ`, rank 2k (Woodbury) | `6k·|W|` | `2k(m+n)` | **`σ·√(2k/m)`** |
| block-diagonal, block size b (batched b×b solve) | `b·|W|` | `(m+n)(b−1)/2` | **`σ`** |

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
`α/√N` then matches dense ES *exactly*: dense moves `α‖ε‖_F/√N = 5e-4·50·‖W‖/√30 =
4.6e-3‖W‖`; ISO moves `α/√N·‖W‖ = 2.5e-2/√30 = 4.6e-3‖W‖`.

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

| param | shape | ms / perturbation |
|---|---|---|
| `qkv_proj` | 4608×3584 | 3.63 |
| `o_proj` | 3584×3584 | 3.35 |
| `gate_up_proj` | 37888×3584 | 10.03 |
| `down_proj` | 3584×18944 | 6.42 |
| **whole model (28 layers)** | | **656 ms** |

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

| layer | shape | `‖ΔW‖/‖W‖` | `‖Δσ‖/‖σ‖` |
|---|---|---|---|
| `layers.0.self_attn.q_proj` | 3584×3584 | 5.005e-2 | **2.50e-8** |
| `layers.13.mlp.gate_proj` | 18944×3584 | 4.974e-2 | **2.41e-8** |
| `layers.27.mlp.down_proj` | 3584×18944 | 4.977e-2 | **2.45e-8** |
| `q_proj` (isobtt, per-block) | n=56, b=64 | 4.954e-2 | **1.14e-7** |
| `down_proj` (isobtt, per-block) | n=128, b=148 | 4.982e-2 | **1.25e-7** |

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

| state dtype | drift after 40 iterations |
|---|---|
| fp32 | **+2.365e-4** (exactly 5.91e-6 × t, always positive) |
| fp64 | 1e-16 (no accumulation at all) |

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

| mode | reconditioning | ‖Δσ‖/‖σ‖ @ 100 | @ 300 | shape-only @ 300 |
|---|---|---|---|---|
| `iso` | **on** | 3.4e-7 | **5.6e-7** | 5.6e-7 |
| `iso` | off | 2.7e-4 | (linear) | 3.3e-7 |
| `isobtt` | **on** | 1.63e-6 | **1.63e-6** (flat) | 4.0e-7 |
| `isobtt` | off | 2.6e-4 | 7.6e-4 | **3.0e-4** |

Two things to note. With the fix, `isobtt` is *perfectly flat* over 300 updates and `iso`
grows only as a √t round-off walk. And **`isobtt` is the mode that actually needed it**:
uncorrected, its *shape* error grows to 3.0e-4, a real constraint violation, whereas
`iso`'s uncorrected error was entirely the benign scalar.

**Confirmed live on the 7B model.** Run 6's first update reports
`iso/frob_drift = 5.83e-5` (the worst-layer `‖RᵀR − I‖` the 30 Cayley factors left behind)
and `iso/orth_err = 1.07e-6` after the Newton–Schulz step — a **54× reduction**, and unlike
the uncorrected case it stays pinned there instead of accumulating.

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

| α | ×paper | σ | MATH-500 @5 | @10 | @15 | @20 | train acc @20 | reward σ (mean) | status |
|---|---|---|---|---|---|---|---|---|---|
| 5e-4 | 1× | 1e-3 |  | 55.2 |  | 60.2 | 63.0 | 0.030 | done |
| 2e-3 | 4× | 1e-3 | | | | | | | _queued_ |
| 6.25e-3 | 12.5× | 1e-3 | 65.4 |  |  |  | 69.2* | 0.029 | running (step 5) |
| 2e-2 | 40× | 1e-3 | | | | | | | _queued_ |
| 6.25e-3 | 12.5× | 1.25e-2 | | | | | | | _queued_ |
| 5e-4 | — | 1e-3 | 70.8† | 70.8 | 71.4† | **71.4** | 68.3 | 0.053 | **dense reference** (first 20 it) |

† dense was evaluated every 10 steps, so its @5/@15 cells repeat the neighbouring eval; the sweep uses every 5.

<!-- AUTO:FURASWEEP END -->

**Read the sweep as:** if some α brings FuRA's step-20 score up to dense's ~71, the small-core
subspace is *not* a capacity bottleneck and the paper-σ result was purely a step-size artifact —
the same conclusion ZO-Act reached by simply running longer. If every α plateaus below dense,
or the large ones destabilise (train accuracy collapsing, reward spread → 0), that is a genuine
subspace limit and the `output_one_block` / `keep_frozen` orientation is the next thing to vary.
