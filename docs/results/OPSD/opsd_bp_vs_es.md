# OPSD: BP reproduction, and es-prefill at equal step time

> On-Policy **Self**-Distillation (OPSD, [arXiv:2601.18734](https://arxiv.org/abs/2601.18734),
> `docs/papers/26_Self-Distilled Reasoner-*.pdf`, code `github.com/siyan-zhao/OPSD`): one model
> is both teacher and student, separated only by **context** — the teacher sees the reference
> solution, the student sees the problem. This page reproduces it with BP in this repo's verl,
> then runs the **es-prefill** forward-only estimator on the identical pipeline at matched
> wall-clock per step. Started 2026-09-07. Companion: [../ZO_OPD/es_prefill_paper_framing.md](../ZO_OPD/es_prefill_paper_framing.md)
> (which predicted this setting as the OPSD extension), [../ZO_OPD/es_rails_formulation.md](../ZO_OPD/es_rails_formulation.md).

## 1. Why this setting

The ds15b es-prefill study closed with **ES ≈ BP − 3 pp, then flat**, limited not by signal
but by the **random-walk displacement budget**: coherent motion `sqrt(2 δ_max C / tr(H))`
against a noise walk that grows as `ρ√S` and turns the run over at `S* = (B/ρ)²`.
OPSD changes two things at once:

1. **LoRA r=64** (the paper's own recipe) cuts the perturbed dimension from 1.72 B to **69.7 M
   (4.0%)**. The displacement budget at equal coherent motion scales as `1/√D`, so this is a
   **≈5× larger budget** — a different regime from every ds15b arm.
2. **No external teacher.** The advantage comes from a *context* difference on frozen initial
   weights, so the teacher is an anchor that cannot drift (the framing doc's caveat #3 is
   satisfied by construction).

## 2. Recipe (paper-faithful unless noted)

| knob | paper (`run_opsd_1b.sh`) | here |
|---|---|---|
| model | `Qwen/Qwen3-1.7B` instruct | same |
| teacher | frozen init policy, privileged context, thinking **ON** | same (frozen copy in verl's `reward_model` slot) |
| student | problem only, thinking **OFF** | same |
| data | `siyanzhao/Openthoughts_math_30k_opsd` | same, 29,433 rows → `datasets/opsd_openthoughts_math_30k.parquet` |
| loss | full-vocab forward KL, pointwise clip τ=0.05 | **truncated to teacher top-64** (see §5.1) |
| batch / rollouts | 32 prompts × n=1, 1 update/step | same (⇒ PPO ratio ≡ 1, pure PG) |
| lr / clip | 5e-6, grad-clip 0.1 | same |
| LoRA | r=64, α=128, 7 proj modules | same |
| sampling | T=1.1, top-p 0.95, top-k 20, ≤1024 tok | same |
| precision | bf16 | **fp32 optimizer master** (see §5.2) |
| steps | 100 | 100 |

Launchers: `scripts/opsd/bp_opsd.sh`, `scripts/opsd/es_opsd.sh`,
`scripts/opsd/calibrate_es_opsd.sh`; eval `scripts/opsd/eval_opsd.py`;
dataset `scripts/opsd/build_opsd_dataset.py`.

## 3. Results

*(filled in as runs land)*

### 3.1 BP-OPSD (paper recipe, 100 steps, ~37 s/step)

**In-loop monitor** — MATH-500, n=1, 3072 tok, thinking **off**. *Two independent seeds:*

| step | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| run 1 (complete) | 0.740 | 0.716 | 0.726 | 0.732 | **0.686** |
| run 2 (to step 76) | 0.740 | 0.738 | 0.716 | 0.700 | — |

**BP-OPSD *declines* ~4–5 pp on short-form non-thinking MATH-500 by step 75–100, in both seeds** —
beyond the ruler's ±2.0 pp single-eval noise once two seeds agree. (The two seeds also calibrate
that noise directly: 0.716 vs 0.738 at the same step 25 of the same recipe.)

**The mechanism is in the training metrics, and it is the intended one.** The privileged teacher
is *higher*-entropy than the student, and OPSD drags the student toward it:

| step | `actor/entropy` | `teacher/entropy` | top-K overlap | clipped fwd-KL/token |
|---|---|---|---|---|
| 1 | 0.232 | 0.259 | 0.868 | −0.0009 |
| 50 | 0.537 | 0.376 | 0.851 | −0.0116 |
| 75 | 0.572 | 0.412 | 0.818 | −0.0155 |

The objective falls monotonically (it *is* being optimised), the student's entropy more than
doubles, and its top-K support drifts off the teacher's. A less peaked policy is exactly what
should cost short-form accuracy at T=0.6 while potentially helping long-form reasoning — which is
why **only the paper's ruler (AIME24/AIME25/HMMT25, Avg@12, 38912 tok, thinking ON) can decide
whether this reproduces**. §3.3.

### 3.3 Paper ruler (the actual reproduction)

`scripts/opsd/eval_ckpts.sh` → `scripts/opsd/eval_opsd.py`. Paper target for Qwen3-1.7B:
AIME24 51.5 → 57.2, AIME25 36.7 → 43.9, HMMT25 23.1 → 29.2 (Avg@12).

*(running: base → step 50 → step 100)*

### 3.2 es-prefill OPSD

**Step-time parity ⇒ N = 4.** `scripts/opsd/bench_parity.sh`; each arm run **alone** on GPU 5,
validation and checkpointing off, same data in the same order, median over warm steps (step 1 is
cold — the reward module fires for the first time inside the timed phase).

| arm | step | rails | s/rail | non-rail base |
|---|---:|---:|---:|---:|
| **BP** | **37.2 s** | — | — | (gen 20.9 / update 8.1 / teacher 3.0) |
| ES N=2 | 33.8 s | 3.7 | 1.85 | 30.1 |
| **ES N=4** | **35.2 s** | 7.2 | 1.80 | 28.0 |
| ES N=8 | 45.1 s | 14.4 | 1.80 | 30.7 |

```
ES_step(N) = 29.6 + 1.82·N        (per-rail flat at 1.80 s across a 4x range of N)
BP_step    = 37.2 s
parity     ⇒ N = 4.2  →  N = 4    (35.2 s = 95 % of BP — errs on giving ES *less* time)
```

This is a **much friendlier parity point than ds15b's N ≈ 2 at 256 seqs**, for two compounding
reasons: LoRA makes BP's backward cheap (8.1 s, not the 124 s of the ds15b BP arm) *and* makes
each rail cheap, and this batch is 32 seqs rather than 256. **LoRA collapses BP's cost advantage**:
in ds15b, equal cost bought ~2 probes against a 124 s backward; here it buys 4 probes on a 25×
smaller parameter space.

**Parity must be measured in isolation.** When another job shared the card, BP's step went
37 → 80 s with `update_actor` *tripling* (optimizer-offload host traffic) — any parity number
taken under co-tenancy is void. Two separate figures on this page were affected before the
benchmark was redone: an earlier "BP ≈ 35.7 s / rail 1.80 s ⇒ N ≈ 2.9" and a claim that the
teacher phase costs 10.8 s (that was **step 1**; warm it is **3.0 s** — the same cold-step trap
[zo_opd.md §10](../ZO_OPD/zo_opd.md) documents for the ds15b BP arm).

**α ladder** (σ=1e-3, N=4, 30 steps, val@30). `update_rms = α/√n_pairs`, so
φ = α/(√2 · 7.75e-3):

| φ | α | MATH-500 @30 | `post_update_gain` @1 / @15 / @30 | `cum_footprint` @30 |
|---|---|---|---|---|
| 6.5e-4 (= BP's per-step motion) | 7.1e-6 | 0.710 | −3.45e-4 / −3.40e-4 / −2.37e-4 | 0.0036 (= φ√30 ✓) |
| 2.0e-3 | 2.19e-5 | 0.730 | −3.74e-4 / −3.33e-4 / −3.34e-4 | 0.0110 ✓ |
| 6.0e-3 | 6.6e-5 | 0.734 | −3.98e-4 / −2.88e-4 / −3.72e-4 | 0.0330 ✓ |
| *(σ calib, α≈0)* | 1e-9 | — | ≈ −1e-5 | 0 |

### ⚠ The ladder is INCONCLUSIVE — both metrics are saturated

**`post_update_gain` cannot rank α here.** Across **9.3× in α** it moves by **1.15×**, and it is
flat across 30 steps within every arm. It is not α² (would be 87×), not α¹ (9.3×), not anything.
But at α ≈ 0 the same metric reads ≈ −1e-5 — so it **saturates as soon as the update is large
enough to perturb the forward at all**, and then stops responding. Its value ≈ −3.5e-4 is a
property of the evaluation, not of the step size.

**MATH-500 @30 cannot rank α either**: 0.710 / 0.730 / 0.734 against base 0.740, all inside the
ruler's ±2.0 pp (n=1, 500 problems).

This is the OPSD instance of a trap already on the books —
[zo_opd.md §12](../ZO_OPD/zo_opd.md): *"LRs spanning 100× give identical curves on
`train/L_clean_mean`; only the fixed heldout probe can rank LRs."* An in-band per-step scalar
saturates; ranking step sizes needs a fixed external probe. **A fixed held-out OPSD-fitness probe
(one frozen batch, scored every k steps) is the missing instrument** and should be built before
any further α work.

### What the arithmetic says, independently of those metrics

`d_std/σ` gives ‖g‖ ≈ 0.107; at N=4 on **D = 69,730,304** the cosine is `√(N/D) = 2.4e-4`, so the
expected coherent gain per step is

```
alpha * cos * ||g||  =  7.1e-6 * 2.4e-4 * 0.107  =  1.8e-10
```

i.e. **~6 orders of magnitude below anything either metric can resolve**. This is a *prediction*
from `d_std`, ‖g‖ and D — it is consistent with the flat ladder but **not evidenced by it**, and
the two must not be conflated (an earlier revision of this page did conflate them).

Through `1/(1 + r_eff/N)`, using **ds15b's r_eff ≈ 160 — BORROWED, not measured for OPSD**:

| N | % of BP per step | step time | vs BP |
|---:|---:|---:|---:|
| **4** (parity) | **2.4 %** | 36.9 s | 1.0× |
| 8 | 4.8 % | 44.2 s | 1.2× |
| 32 | 16.7 % | 87.8 s | 2.4× |
| 160 | 50 % | 320.8 s | **8.6×** |

⚠ Extrapolation. `r_eff` for the OPSD objective on this model is **not measured**
(`scripts/zo_opd/ds15b/opd_curvature.py` needs adapting for the privileged self-teacher and
LoRA-subspace perturbation).

**Measured claims from this section, and only these:** parity is N=4; the ES arms neither gain nor
collapse on MATH-500 in 30 steps at any α tried; and neither in-band metric can rank α.

## 4. ES calibration — σ is set from BELOW, by the fitness noise floor

The ds15b σ does **not** transfer. LoRA's `B` is zero-initialised, so RMS over the trainable
tensors is **7.75e-3** (measured; matches the kaiming-`A`-only prediction `1/sqrt(3(in+out))`),
and at step 0 the induced dense perturbation is one-sided, `dW ≈ 2σ ε_B A`.

The binding constraint turned out to be different from ds15b's. `es/post_update_gain` at
α≈1e-9 is `F(W₀) − F(W₀)`, which is exactly 0 in exact arithmetic — it measures the **bf16
forward's own reproducibility**:

| σ | probe footprint σ/RMS(W_train) | ES signal `d_std` | signal / noise floor | even-order term | verdict |
|---|---|---|---|---|---|
| 3e-4 | 3.9 % | 2.2e-5 | **1.6×** | −3.5e-4 | noise-dominated, unusable |
| **1e-3** | 12.9 % | 1.07e-4 | **~13×** | −5.4e-4 | **chosen** |
| 3e-3 | 38.7 % | 2.0e-4 | ~20× | −2.0e-3 | `d_std` grows 1.9× for 3× σ ⇒ nonlinear onset |
| 1e-2 | 129 % | 2.0e-3 | — | **−3.8e-2** | past the cliff (70× the damage of σ=1e-3 for 10× σ) |

The **scaling test** picks σ, not any absolute footprint: 3e-4→1e-3 multiplies `d_std` by 4.9×
for 3.3× σ (super-linear — the low end is noise-inflated), while 1e-3→3e-3 multiplies it by only
1.9× for 3× σ (sub-linear — the O(σ³) term biting). σ = **1e-3** sits at the crossover with ~13×
SNR, ~3× margin to the nonlinearity onset and ~10× to the cliff. The cliff between 3e-3 and 1e-2
mirrors ds15b's between 3e-3 and 6e-3, but at a completely different probe footprint (129 % vs
5 %) — footprint-in-coefficient-space does not transfer; the σ-response does.

The even-order term `(F₊+F₋)/2` is a **real effect of the perturbation**, not a bookkeeping
artifact: `es/post_update_gain` at α≈1e-9 (i.e. `F(W₀) − F(W₀)`) reads ~1e-5, so the fitness
baseline path (`compute_distillation_reward`) and the rail path (`compute_log_probs_for_ids`)
demonstrably agree. It is roughly quadratic at the top of the ladder (1e-3→1e-2: ×70 for ×10 σ)
but has a σ-independent floor ≈ −3.4e-4 at the bottom — expected, because **every advantage here
is non-negative** (`A_v = p_T(v)·1[ℓ<τ] ≥ 0`), so `F` rewards raising log-probability on the
teacher's top-K set and *any* random perturbation generically leaks mass to the other ~151k
tokens. It cancels in `d = (F₊−F₋)/2` regardless.

*(Earlier in this session I mis-attributed this offset to a code-path mismatch; the α≈0
`post_update_gain` measurement rules that out.)*

## 5. Deviations from the paper

### 5.1 Forward KL truncated to the teacher's top-K
verl scores the teacher in a separate worker and passes tensors, so full-vocab logits
(B×T×151936) cannot be carried. The advantage uses the **un-renormalised** `p_T`, so this is the
honest truncation of the same sum, not a reweighted objective. Measured teacher top-64 mass:
**0.9997**. The paper's own `--top_k_loss` flag does the same thing (with renormalisation).

### 5.2 fp32 optimizer master
The paper trains in pure bf16. A bf16 master silently freezes ~99 % of weights at this learning
rate — the trap documented in [../ZO_OPD/opd_paper_align.md](../ZO_OPD/opd_paper_align.md).

### 5.3 In-loop validation is a cheap monitor
MATH-500, n=1, 3072 tokens, thinking **off** (matching the training distribution). The paper's
ruler — Avg@12 on AIME24/AIME25/HMMT25 at 38912 tokens, thinking **on** — runs offline on the
saved checkpoints via `scripts/opsd/eval_opsd.py`.

## 6. Implementation (verl patches)

| file | change |
|---|---|
| `workers/fsdp_workers.py` | `_build_privileged_teacher_inputs` — teacher scores the rollout under the privileged prompt. Response tokens are reused **bit-for-bit** (no decode/re-encode), placed last, so the `[-resp_len-1:-1]` logit slice stays valid and every teacher log-prob aligns with the student's token. Gated by `+reward_model.opsd_privileged=True`; prompt string travels in `extra_info["teacher_prompt"]` (one of the four non-tensor keys verl keeps through generation). |
| `workers/actor/dp_actor.py` | `reward_weight_mode=fkl_clip`: `ℓ_v = p_T(v)(log p_T(v) − log p_S(v))`, clipped entries contribute a constant ⇒ zero gradient, so `A_v = p_T(v)·1[ℓ_v<τ]` in the existing 3D-advantage PG path reproduces the clipped-forward-KL descent direction **exactly**. Elementwise identical to the reference `F.kl_div(log p_S, log p_T, log_target=True).clamp(max=τ)`. |
| `workers/fsdp_workers.py` | `_es_params()` — ES rails perturb **trainable-only** parameters. No-op under full fine-tuning; it is what makes LoRA-ES possible. |
| `trainer/ppo/es_update.py` | **Bug fix.** The 3D ES fitness gathered log-probs at `student_top_k_ids` while `only_tch`/`union` advantages are indexed by the **teacher's** ids — the rails were scoring a different objective than BP maximises. Now prefers `union_top_k_ids`/`union_top_k_log_probs`. |
| `trainer/ppo/ray_trainer.py`, `workers/config/rollout.py` | `opsd_fkl_clip` (τ) plumbing; `opsd/fwd_kl_per_token` metric logged and dropped so it never travels with the batch. |

### Gotchas found
- **vLLM memory profiling vs param offload.** `ACTOR_PARAM_OFFLOAD=True` frees ~40 GiB while
  vLLM is profiling → `AssertionError: Error in memory profiling`. Keep param_offload **False**
  (it is also what `algorithm.es_update` requires, so BP and ES share the layout).
- **Padded positions read `p_T = 1`.** `teacher_top_k_log_probs` is zero-filled outside the
  response, so `exp(0)=1` for every id and `Σ_k A_k` reads K (=64) there. The advantage is
  masked downstream, but `critic/score/*` is logged pre-mask — now zeroed at the source.
- `data.shuffle=False` is hard-coded in `on_policy_distillation.sh`; OPSD overrides it to True.
- **`<ckpt>/actor/merged_hf/` is NOT merged** — despite the name it holds the *LoRA adapter*
  (279 MB = 69.7 M params × fp32), not a standalone model. vLLM cannot load it directly; merge
  base+adapter first (`scripts/opsd/merge_lora.py`).
- **`/data` sits at 100 %.** Each checkpoint is ~8.7 GB (7.9 GB FSDP shard + 533 MB optim +
  279 MB adapter), so `save_freq=25` over 300 steps × `max_actor_ckpt_to_keep=8` would want ~70 GB
  that does not exist. Prune to every-50 while running.
- **The card runs with <10 GB of slack.** BP peaks at 48.4 GB allocated / 54.9 GB reserved
  (full-vocab logits at `ppo_max_token_len_per_gpu=6144`: 6144 × 151936 × fp32 ≈ 3.7 GB, doubled
  by log-softmax) and vLLM holds another ~33 GB of 95.8. Consequences, both hit in this session:
  vLLM's `wake_up()` re-maps its whole KV cache in one allocation and **OOMs on fragments left by
  the ES rails** (N=4 and N=8 died where N=2 survived — it is cached blocks, not resident state),
  and a cold `check_enough_kv_cache_memory` can fail outright at init. Fixed by
  `torch.cuda.empty_cache()` at the end of `es_update_actor` (BP's optimizer step already frees
  its equivalent).
- **`expandable_segments:True` is NOT available as a fragmentation fix in this repo.** The obvious
  move — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — makes every run die at init with
  `AssertionError: Expandable segments are not compatible with memory pool`
  (`vllm/device_allocator/cumem.py:150`), because verl puts vLLM in **sleep mode** and its
  `CuMemAllocator` refuses the setting outright. Tried and reverted here; reclaim memory
  explicitly instead.
- **`pipefail` + an empty `grep` kills a sweep driver.** A metric grep that legitimately matches
  nothing on a failed arm aborted the whole α ladder under `set -euo pipefail`; the per-arm
  reporting pipelines now end in `|| true`, and the launchers wait for the card to actually drain
  (the previous run's memory release races the occupancy guard).
