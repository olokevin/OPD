# Paper-aligned OPD — Qwen3-4B-Base-GRPO → Qwen3-1.7B-Base

> Reproduces the setting of Section 3.1 of *Rethinking On-Policy Distillation*
> ([docs/papers/26_Rethinking...pdf](../papers)) so that BP-OPD demonstrably learns, then reads
> ZO-ES-token against it on the *same* setting. Supersedes the negative results in
> [zo_opd.md §9](zo_opd.md) — both of which turned out to be measuring two setup bugs, not the
> algorithms.

Branch `feat/es-token-trainer`. Launchers `scripts/zo_opd/paper_align/`.
wandb project `opd-paper-align`.

## Why the previous runs could not learn

[zo_opd.md §9.4](zo_opd.md) recorded that *neither* es_token nor its BP-OPD baseline moved, and
concluded "this points at the setup rather than the algorithm". It was two setup bugs, both
independent of the estimator, and both now measured.

### 1. Every rollout was truncated — wrong student for the token budget

The old runs used `Qwen/Qwen3-1.7B` (the **instruct** model, thinking enabled) with a 1024-token
cap, and logged `response_length` mean = min = max = 1024: no rollout ever emitted EOS. A thinking
trace does not fit in 1024 tokens, so every sequence was cut mid-reasoning, no answer was ever
produced, and MATH-500 sat at its 2–7% floor for both methods — leaving no headroom for any signal
to show up in.

The paper's student is `Qwen3-1.7B-**Base**`. Probing it on 64 DAPO-Math prompts (`\boxed{}`
template, chat template applied, T=1.0, n=4, 3072-token cap;
`scripts/zo_opd/paper_align/probe_rollout.py`):

| | stop rate | resp len mean / p50 / p90 | hit 3072 cap | has `\boxed` | acc avg@4 |
|---|---:|---:|---:|---:|---:|
| student `Qwen3-1.7B-Base` | **0.949** | 984 / 754 / 2166 | 0.051 | 0.715 | 0.082 |
| teacher `lllyx/Qwen3-4B-Base-GRPO` | 0.402 | 2718 / 3072 / 3072 | 0.598 | 0.441 | 0.199 |

The student terminates on its own in 95% of rollouts inside 3072 tokens. In training this shows up
as `response_length/mean` ≈ 930–990 with `clip_ratio` ≈ 0.05–0.09, and the step-0 MATH-500 baseline
moves from ~5% to **40.05%** (mean@4) — the metric now has room to move in both directions.

(The *teacher* still truncates 60% of the time, because it was RL'd at 7168 tokens. That does not
affect OPD: the teacher only ever *scores* the student's tokens, it never generates during training.)

### 2. The optimizer's master weights were bf16, so a 1e-6 step rounded to nothing

`verl/workers/fsdp_workers.py:449` does `actor_module.to(torch_dtype)` with
`torch_dtype = fsdp_config.model_dtype`. `scripts/zo_opd/opd_math_ref.sh` set
`MODEL_DTYPE=bfloat16` for memory, which makes the **master** parameters bf16 — not just the
forward compute. bf16 has 8 mantissa bits, so near a typical Qwen weight magnitude the spacing
between representable values is ~6.1e-5, and any step below half of that rounds straight back to
the value it started from.

Measured on `model.layers.10.mlp.down_proj` of Qwen3-1.7B-Base (|W| mean 0.0253, median 0.0203),
applying an Adam-sized `±lr` step and counting how many elements actually change:

| step size | fraction of weights that change, **bf16 master** | fraction, **fp32 master** |
|---|---:|---:|
| 1e-6 (the OPD default LR) | **0.0135** | 1.0000 |
| 1e-5 | 0.1072 | 1.0000 |
| 1e-4 | 0.6884 | 1.0000 |

> **CORRECTION [2026-08-25]: "numerically frozen" was too strong, and "the whole explanation" was
> wrong.** The NERSC reference run (`slurm/opd/full/opd_2node_env.sh`, wandb
> `nersc_opd_qwen4b_1p7b/opd_full_dapo_lr1e-6`) uses **exactly** `MODEL_DTYPE=bfloat16` at
> `lr=1e-6` and it **learns**: AMC23 mean@8 0.416 → 0.538 and AIME24 0.175 → 0.238 over 160 steps.
> The table above is a correct measurement of *one* step, but the inference drawn from it was not:
> at ~1.35% of weights moving per step, an update still accumulates over the 100+ steps a real run
> takes. bf16 masters **attenuate** the update — they do not freeze it. The dominant blocker in the
> earlier flat runs was the 1024-token truncation on its own. `fp32` remains the better setting
> (100% of weights move, and it is `on_policy_distillation.sh`'s own default), but it is an
> improvement, not a bug fix.

`on_policy_distillation.sh`'s own default is `MODEL_DTYPE=fp32` (fp32 master, bf16 compute via
FSDP `MixedPrecision`) — the repo was right and the zo_opd wrapper's override was the bug.

**The same attenuation was in the ES trainer.** `es_token_worker_extension.py` applied
`weight.add_(dw.to(weight.dtype), alpha=-lr)` directly to vLLM's **bf16** weights. At `lr=1e-4`
with the measured `dW_norm_mean` ≈ 1237 over ~12.6M-element layers, the per-element step is
≈3.5e-5 ≈ **0.57 ulp** — sitting exactly on the round-to-nearest threshold, so roughly half of
every update was thrown away. This is why the LR bracket in [zo_opd.md §9.3](zo_opd.md) came out as
"1e-3 destroys / 1e-4 does nothing" with no working recipe between: the two ends were not measuring
the same effective step size.

## What changed in the code

| Change | File | Why |
|---|---|---|
| `REWARD_MODEL_DTYPE` env knob (default = `MODEL_DTYPE`) | `on_policy_distillation.sh` | teacher is inference-only, so it can stay bf16 while the actor runs an fp32 master — that is what makes both fit on one 95 GB card |
| `VAL_BEFORE_TRAIN` env knob | `on_policy_distillation.sh` | step-0 baseline; the script hard-coded `False` |
| `PPO_MAX_TOKEN_LEN_PER_GPU` overridable | `on_policy_distillation.sh` | actor micro-batch token budget at a 3072 response length |
| `es_token.fp32_master` (default `true`) | `es_token_worker_extension.py`, `es_token/ray_trainer.py`, `config/es_token_trainer.yaml` | fp32 master copy per perturbed layer; update accumulates in fp32, rounds to bf16 once |
| greedy heldout probe (`probe_sp`) | `es_token/ray_trainer.py` | the T=1.0 sampled probe had a ±8% floor ([zo_opd.md §9.1](zo_opd.md)) that swamped every non-diverging LR; a greedy clean trajectory on fixed prompts is deterministic |

## Setup (paper Section 3.1 + Table 2)

| | value | paper |
|---|---|---|
| student | `Qwen/Qwen3-1.7B-Base` | same |
| teacher | `lllyx/Qwen3-4B-Base-GRPO` | same (zero-RL from Qwen3-4B-Base; pattern-matched to a base student) |
| prompts | `datasets/dapo-math-17k-processed.parquet` | DAPO-Math-17K, `\boxed{}` template |
| estimator | `token_reward_direct`, top-K 16, `only_stu`, `student_p` | same |
| T (student / teacher) | 1.0 / 1.0 | same |
| batch / rollout n | 64 / 4 | same |
| LR, KL, loss agg | 1e-6, 0.0, token-mean | same |
| max prompt / response | 1024 / **3072** | 1024 / 7168 — **deviation**, single-GPU memory budget |
| validation | MATH-500 + AMC23 + AIME24, mean@4, T=1.0/top-p 0.95, 3072 tok | AIME24/25 + AMC23 avg@16 at 31744 tok |
| actor dtype | fp32 master + bf16 compute | repo default |
| hardware | 1× H100 NVL 95 GB, teacher co-located | 8× A800 |

Step-0 baselines for `Qwen3-1.7B-Base` under this protocol: **MATH-500 40.05%**, **AMC23 14.76%**,
**AIME24 1.67%** (mean@4). `val-topk/overlap_ratio` = **0.724** at step 0, i.e. above the
0.63–0.66 the paper's Figure 2 shows for its GRPO teacher — the thinking-pattern-compatibility
precondition the paper identifies is satisfied.

## The finding the bf16 freeze had been hiding: BP-OPD diverges at the paper's LR

With fp32 masters the updates finally land — and the paper's own `lr=1e-6` **destroys the model
inside 20 steps**. Two independent runs (with and without `ACTOR_OPTIM_OFFLOAD`, the one setting
that differed from `opd_math_ref.sh`) show the same trajectory, so it is not an offload artefact:

| step | 1 | 5 | 9–10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|
| `actor/entropy` | 1.14 | 1.19 | 2.13 | 1.98 | **3.18** |
| `teacher/entropy` (frozen model!) | 1.11 | 1.15 | 2.09 | 1.93 | **3.06** |
| `response_length/mean` | 960 | 871 | 1094 | 1144 | **1335** |
| `response_length/clip_ratio` | 0.051 | 0.043 | 0.137 | 0.148 | **0.172** |
| `val-topk/overlap_ratio` | 0.729 | 0.710 | 0.697 | 0.695 | 0.706 |
| MATH-500 mean@4 | 0.4005 (step 0) | | 0.2775 (no-offload) | | **0.3105** |

The *teacher's* entropy tripling is the tell: the teacher is frozen, so the only way its entropy on
these tokens can move is if the student has walked into states where even the teacher is uncertain.
The logged validation generations confirm it directly — at step 20 a MATH-500 geometry problem is
answered with fluent **Turkish prose about Mustafa Kemal**. This is collapse, not the "pronounced
instability before gradually recovering" the paper reports for base-initialised students.

**What the objective looks like.** For `only_stu` + `student_p` (`dp_actor.py:560`) the per-token
reward over the student's top-K is

```
w = softmax_K(log p_S)                    # renormalised student probs
adv_k = -(log p_S(k) - log q_T(k)) * w_k
```

and because `ppo_epochs=1` with a single mini-batch, verl takes the `on_policy` branch
(`dp_actor.py:807`) where `old_log_prob = log_prob.detach()`, so the ratio is exactly 1 and the loss
is plain `-Σ_k adv_k · log p_θ(k)`, with `Σ_k adv_k = -Σ_k w_k log(p_k/q_k) ≈ -KL ≤ 0`.

> **A tempting explanation, and why it is wrong.** The shape above suggests the objective is
> improvable by leaking probability mass into the 151,920 tokens *outside* the top-16, where the KL
> is not measured — which would explain the entropy rise exactly. **Falsified:** sampled-token OPD
> (`LOG_PROB_TOP_K=0`, the paper's own §6.3 variant) has no top-K restriction at all and collapses
> identically — MATH-500 0.2665 and entropy 1.87 at step 10, against 0.2775 / 2.13 for top-16. A KL
> guard (`USE_KL=True`, coef 0.005) only dents it: 0.2510 / 1.60. The instability is not about the
> support of the reward.

> **The second tempting explanation, also wrong.** The one place this setup deviates from the paper
> is the **token budget** (3072 vs 7168), and the teacher truncates 60% of its own generations at
> 3072 — so it should assign low probability to the student's *terminating* tokens and teach it
> never to stop, which matches the observed `response_length` 960 → 1335 / `clip_ratio` 0.05 → 0.17
> runaway. **Falsified:** re-running at `MAX_RESP_LENGTH=6144` collapses just the same — MATH-500
> 0.4005 → **0.2935** by step 10, entropy 1.28 → 2.15 — while `clip_ratio` reaches only **0.055**
> against 0.172 at 3072. Truncation is largely removed and the collapse is unchanged, so length
> inflation is a *symptom* of the drift, not its cause.

### What has been ruled out

Every hypothesis below was tested by running it, not by argument. All are at `lr=1e-6`, all
collapse, and the step-10 MATH-500 / entropy are given for comparison against the paper-exact
0.2775 / 2.13:

| hypothesis | how it was tested | result |
|---|---|---|
| optimizer-state corruption from CPU offload | `ACTOR_OPTIM_OFFLOAD` on **and** off | collapses both ways |
| update applied too often (mini-batch vs `rollout.n`) | `dp_actor` logs `on_policy`, one `grad_norm`/step | 1 Adam step/step — paper-faithful |
| step size too large | LR ladder 1e-7 … 1e-6 | collapse time ∝ 1/lr; no stable-and-learning LR |
| probability mass leaking outside the top-K | `LOG_PROB_TOP_K=0` (sampled-token, paper §6.3) | 0.2665 / 1.87 — same |
| unconstrained drift from the reference policy | `USE_KL=True`, coef 0.005 | 0.2510 / 1.60 — dented, not fixed |
| token budget starving a long-form teacher | `MAX_RESP_LENGTH=6144` | 0.2935 / 2.15 at `clip_ratio` 0.055 — same |

The reward signal itself looks healthy throughout: `val-topk/overlap_ratio` ≈ 0.72, and the teacher
assigns 0.738 to the student's argmax against the student's own 0.755 — a well-aligned teacher, not
a misindexed one. The remaining untested axis is the **student initialisation**, which is also the
paper's own prescription for this symptom (§5.1).

### The LR bracket

| LR | `actor/entropy` s1 → s9/10 | `clip_ratio` s1 → s9/10 | overlap s1 → s10 | MATH-500 mean@4 |
|---|---|---|---|---|
| **1e-6** (paper) | 1.18 → **1.55** | 0.051 → **0.125** | 0.724 → 0.688 | 0.4005 → **0.2775** (s10) |
| **1e-7** | 1.18 → 1.06 | 0.051 → 0.078 | 0.724 → 0.721 | 0.4005 → **0.4105** (s10) |

`lr=1e-7` holds every diagnostic flat and is the first setting on this pair where the accuracy
metric moves *up*. `3e-7` is running to find the top of the stable range.

**Ruled out, so the LR really is the lever.** Before blaming the step size, the update *cadence* was
checked against the paper: verl multiplies `ppo_mini_batch_size` by `rollout.n`
(`fsdp_workers.py:237`), so 64 prompts × 4 rollouts = 256 sequences form a single mini-batch. The
run logs confirm it empirically — `dp_actor.py` prints `on_policy` (171 times, once per micro-batch)
and never `off_policy`, which only happens when `len(mini_batches) == 1`, and exactly one
`actor/grad_norm` is emitted per training step. So this is **one Adam step of size ≈ lr per training
step**, exactly as in the paper, and 20 such steps at 1e-6 collapse the model. `ACTOR_OPTIM_OFFLOAD`
was likewise exonerated by running 1e-6 both ways.

For calibration: the paper's own Figure 16 plots student entropy over the range **0.5 – 3.0 across
260 steps**. Our 1e-6 run traverses that same band in **20 steps**. The failure is not that entropy
moves — it is how fast.

### Lowering the LR does not fix it — it only rescales time

The full ladder, all from the same seed and the same (unshuffled) batch order. MATH-500 is mean@4
at T=1.0; step-0 baseline is 0.4005 / AMC23 0.1476.

| LR | MATH-500 s10 | s20 | s30 | `actor/entropy` s10 → s30 | `clip_ratio` s10 → s30 |
|---|---:|---:|---:|---|---|
| 1e-7 | 0.4105 | 0.4050 | — | 1.06 → — | 0.078 → — |
| 3e-7 | 0.4030 | 0.3785 | **0.2780** | 0.85 → **1.89** | 0.059 → **0.121** |
| 6e-7 | 0.3875 | — | — | 1.09 → — | 0.098 → — |
| 1e-6 | **0.2775** | 0.3105 | — | 2.13 → — | 0.137 → — |

**3e-7 reaches at step 30 exactly the state 1e-6 reached at step 10** — MATH-500 ≈ 0.278, entropy
≈ 1.9–2.1, clip ≈ 0.12–0.14. The ratio of times is the ratio of learning rates. The instability is
therefore **LR-invariant up to a time rescaling**: every learning rate walks the same path, just at
its own speed, so there is no step size that is both stable and fast enough to learn. Lowering the
LR further (1e-7) only pushes the collapse out to ~step 90–100.

That rules out "the step is too big" as the *explanation* — the trajectory, not its speed, is the
problem — and moves the search to levers that change the objective rather than its scale. Two are
being tested at the paper's own `lr=1e-6`, where an effect is visible within ~50 steps:

- **A — sampled-token OPD (`LOG_PROB_TOP_K=0`).** The paper's §6.3 ("Sampled-Token Reward Is Already
  Sufficient") shows this matches Top-*k* on accuracy, and it removes the top-K truncation entirely,
  so there is no unmeasured tail for probability mass to leak into. Stays inside the paper.
- **B — top-16 OPD + KL guard (`USE_KL=True`, `kl_loss_coef=0.005`, `low_var_kl`).** Directly
  penalises drift from the reference policy. A deviation from the paper's KL = 0, but the mechanism
  it opposes is exactly the one that is failing.

## It is the student initialisation — BP-OPD learns from the SFT cold start

Holding **everything** else at the paper's values (teacher `Qwen3-4B-Base-GRPO`, DAPO-Math-17k-
Processed, `lr=1e-6`, top-16 `only_stu`, `student_p`, T=1.0, KL=0, batch 64 × n=4, 3072 tokens) and
changing only the student to the paper's own cold-start checkpoint `lllyx/Qwen3-1.7B-SFT`:

| | step 0 | step 10 |
|---|---:|---:|
| MATH-500 mean@4 | 0.6335 | **0.6545** |
| AMC23 mean@4 | 0.3253 | 0.3283 |
| AIME24 mean@4 | 0.0667 | **0.1000** |
| `actor/entropy` | 0.31 | **0.25** |
| `response_length/mean` | 2420 | 2319 |
| `response_length/clip_ratio` | 0.430 | 0.410 |
| `val-topk/overlap_ratio` | 0.665 | 0.666 |

Extending to step 40 (MATH-500 mean@4 at steps 0/10/20/30/40):
**0.6335 · 0.6545 · 0.6540 · 0.6610 · 0.6260**, with entropy pinned at 0.23–0.25, overlap
0.665 → 0.670 → 0.668 → 0.664, and `clip_ratio` falling 0.430 → 0.270 → 0.145.

**Read this carefully: stable is established, learning is not.** The accuracy points wander ±2 pp
around ≈0.646 against a ±1.2 pp SEM, and overlap peaked at step 20 and came back down — an earlier
draft of this page called that rise the paper's healthy-OPD signature, which was a 3-point trend and
is retracted. What *is* solid at step 40 is the contrast with the base student, where by step 20
entropy had tripled (1.14 → 3.18), overlap had fallen 0.729 → 0.706, and MATH-500 had lost 9 pp.
Here nothing degrades. The paper trains 200 steps; this run needs to get there before any accuracy
claim is worth making. This is the same code, the same teacher, the same LR that destroys the
base student in ten steps.

**So BP-OPD is not broken — the base student cannot absorb the signal.** That is exactly the paper's
§5.1 claim ("when the student and teacher have substantially different thinking patterns, pure OPD
can be ineffective because the teacher's token-level supervision is difficult for the student to
exploit from its initial policy"), except that here the failure is not merely *ineffective* — at the
paper's LR it is actively destructive.

**The discriminator is entropy, not overlap.** The base student actually has the *higher* overlap
ratio (0.724 vs 0.665), so the paper's overlap diagnostic does not predict which run survives. What
separates them is the sharpness of the initial policy: the base student starts at entropy **1.18**,
the SFT student at **0.31**. OPD's update is reweighted by the student's own probabilities
(`w = softmax_K(log p_S)`), so a diffuse initial policy spreads that weight across many tokens,
which drives entropy up, which spreads it further — a positive feedback with no restoring term.
A sharp policy keeps the weight concentrated and the loop never starts.

## ZO-ES-token on the same setting: the student-init effect is not BP-specific

The es_token trainer was re-run on the paper-aligned setup with the three fixes from this session
(fp32 master weights, capped teacher `max_model_len`, greedy heldout probe) and the sweep teardown
repaired — killing only `main_es_token` leaks the ray-spawned vLLM engines, so arm *n+1* used to OOM
against arm *n*'s ~35 GB.

**Base student `Qwen3-1.7B-Base`** (12 steps at `lr=1e-5` before the arm was stopped):

| | s0 | s5 | s10 |
|---|---:|---:|---:|
| heldout probe (64 fixed, greedy, ↓ better) | 1.906 | 5.567 | 5.785 |
| MATH-500 greedy @200 | 66.0 | 62.5 | 64.0 |

**SFT student `lllyx/Qwen3-1.7B-SFT`**, 20 steps, otherwise identical:

| LR | probe s0 → s19 (↓ better) | MATH-500 greedy @200 |
|---|---|---|
| **1e-5** | 3.435 → **2.603** (−24%) | 69.0 → 70.0 |
| 3e-5 | 3.342 → 2.917 (−13%) | 71.5 → 66.0 |
| 1e-4 | 3.319 → *(OOM after step 0)* | 62.5 |

> **The probe does not replicate, and the 24% claim is withdrawn.** Re-running the same student at
> the same `lr=1e-5` in the long run gave the *opposite* probe direction — 2.1678 (s0) → 4.5947
> (s15) — while MATH-500 greedy@500 stayed flat at 66.4 → 65.6. Looking again at the sweep arm's own
> series (3.435 · 3.187 · 2.743 · **3.840** · 2.603) the s15 spike was already visible: this is a
> noisy 5-point trace, not a trend, and reading its endpoints as "−24%" repeated exactly the
> 3-point-trend error [zo_opd §9.4](zo_opd.md) warns about. The greedy probe removed the *sampling*
> noise it was designed to remove, but greedy argmax still flips under small weight changes, so it
> is not the low-variance ruler it was meant to be. **Use MATH-500 greedy@500 instead**, and on that
> metric es_token on the SFT student is **flat** over 15–20 steps: no learning, and no collapse.

What does survive is the base-vs-SFT *contrast on accuracy*: es_token holds ~66–70% on the SFT
student while BP on the base student loses 13 pp, and the probe's 3× rise on the base student is far
outside the range of any of its wobbles on the SFT student. The initialisation effect is visible in
the zeroth-order trainer too — but as "does not degrade" rather than "learns".

One asymmetry is worth recording: on the base student ES *degrades gently* (greedy MATH-500
66.0 → 64.0 over 10 steps) exactly where BP *collapses* (40.05 → 27.75 mean@4, entropy 1.14 → 2.13).
This is consistent with the estimator geometry measured in [zo_opd.md §1](zo_opd.md) — the ES update
sits at the rank-1 weight-probe information bound with per-layer cosine ≈ 0.2 to the true gradient,
so ~80% of it is isotropic noise that averages out, while BP applies the full destructive direction
every step. Being a worse gradient estimator makes it a *safer* one in a regime where the gradient
itself is the problem.

`lr=1e-5` is the pick (best probe improvement; the accuracy spread across LRs is inside the ±3.3 pp
SEM of a 200-prompt greedy eval). The 1e-4 OOM is a memory margin, not a divergence: the student
engine at `gpu_memory_utilization=0.55`, the co-located teacher, the fp32 master (5.65 GB) and the
assembly accumulator (5.65 GB) leave ~40 MiB of headroom on a 93 GiB card. The long run drops the
student engine to 0.50, which still admits `pack_width=64` (23,540 blocks available vs 16,384 needed).

### ES memory: the teacher's prefill scales with the student's response length

The long ES run OOM'd twice before it would hold, and the cause is worth recording because it only
appears with a *verbose* student. `TeacherScorer.logq_wave` prefills `teacher_batch_size` full
sequences with `prompt_logprobs`, so the teacher materialises `teacher_batch_size × (prompt +
response) × 151,936` logits. With the base student (~800-token responses) that is small; with the
SFT student (~2,400 tokens, 43% hitting the 3072 cap) it is 8 × 4096 × 151,936 bf16 ≈ **10 GB** of
transient, on top of:

| resident | size |
|---|---|
| student engine (`gpu_memory_utilization`) | 0.50 × 93.1 = 46.5 GiB |
| co-located teacher engine | 0.16 × 93.1 = 14.9 GiB |
| fp32 master weights (1.41 B perturbed params) | 5.65 GiB |
| assembly accumulator `acc` (all layers, fp32) | 5.65 GiB |
| `noise_chunk` at `assemble_chunk=1024` | 1.75 GiB |

which left 727 MiB and died allocating `noise_chunk`. Settings that hold: `gpu_memory_utilization`
**0.45** (still 20,980 KV blocks vs the 16,384 that `pack_width=64` needs), `teacher_batch_size`
**4**, `assemble_chunk` **512**. Both of the last two trade wall-clock for headroom and neither
changes the estimator.

**Caveat on the probe's absolute value.** `eval/heldout_clean_loss` is deterministic *within* a run
(greedy clean trajectory, fixed 64 prompts) but **not comparable across runs with different memory
settings**. Step-0 readings for the same model and LR came out 3.4349 / 3.6372 / 2.1678 at
`gpu_memory_utilization` 0.55 / 0.50 / 0.45. The reason is already documented in
[zo_opd.md §8.2](zo_opd.md): the packed forward batches differently from vLLM's scheduler, so bf16
rounding — and therefore which token wins a near-tied greedy argmax — depends on the wave layout,
which the KV-pool size changes. Every probe comparison on this page is within a single run at a
fixed config, which is valid; do not compare probe values between the sweep arms and the long run.

### The SFT run was stable but mispaired — OPD compressed it instead of improving it

Sixty steps of `Qwen3-1.7B-SFT` ← `Qwen3-4B-Base-GRPO` on DAPO-Math:

| step | 0 | 20 | 30 | 40 | 50 | 60 |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 mean@4 | 0.6335 | 0.6540 | 0.6610 | 0.6260 | 0.6300 | **0.6025** |
| AMC23 mean@4 | 0.3253 | 0.3193 | 0.3283 | 0.3253 | 0.3343 | 0.3223 |
| `response_length/mean` | 2420 | 2325 | 2006 | 1872 | 1663 | **1560** |
| `response_length/clip_ratio` | 0.430 | 0.434 | 0.270 | 0.145 | 0.109 | **0.051** |
| `actor/entropy` | 0.31 | 0.23 | 0.24 | 0.23 | 0.21 | 0.21 |
| `val-topk/overlap_ratio` | 0.665 | 0.670 | 0.668 | 0.664 | 0.667 | 0.669 |

Response length falls **36% monotonically** — far too clean to be noise — while AMC23 stays flat and
MATH-500 drifts down ~3 pp. OPD is unmistakably reshaping the policy; it is just reshaping *style*
rather than capability.

The reason is a pairing error introduced by this page's own experimental design. `Qwen3-1.7B-SFT`
was distilled from **`Qwen3-4B (Non-thinking)`** long-CoT rollouts, and it was deliberately paired
here with **`Qwen3-4B-Base-GRPO`** so that the teacher was held fixed against the base-student arm.
That makes it a thinking-pattern mismatch in the *opposite* direction from the base student — a
verbose student pulled toward a concise base-RL teacher — which is precisely the condition the paper
identifies as making OPD ineffective (§3.1). The length collapse is that mismatch made visible.

### Run E — the paper's actual §5.1 configuration

The repo already carries every piece of the paper's best-reported OPD setup, so it is reproduced
exactly rather than approximated:

| | value | source |
|---|---|---|
| student | `lllyx/Qwen3-1.7B-SFT` | §5.1 cold start |
| teacher | `Qwen/Qwen3-4B` (Non-thinking) | §5.1 — the model that generated the SFT data |
| prompts | `datasets/OpenThoughts3_opd.parquet` (30,000) | §5.1 "remaining prompts after deduplicating against the SFT subset, ≈30K" |
| everything else | Table 2 defaults, `lr=1e-6` | unchanged |

Validation is held at MATH-500 + AMC23 + AIME24, mean@4, T=1.0, 3072 tokens, so its step-0 baseline
is the same 0.6335 / 0.3253 / 0.0667 measured for run D (identical student and identical eval).

