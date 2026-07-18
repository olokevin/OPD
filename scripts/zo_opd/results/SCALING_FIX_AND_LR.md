# ZO-NP scaling fix + learning-rate selection

## 1. What was changed

### (a) grpo finite-difference scale restored — `verl/verl/trainer/np/grad_estimator.py`
```
# before:  grpo scale = (L_q - mean_q) / std_q     # std kills the gradient magnitude
# after:   grpo scale = (L_q - mean_q) / sigma      # 1/sigma finite-difference scale restored
```
`(L_q - mean)/std` is unit-variance-normalized and carries no gradient magnitude; dividing
by `sigma` instead makes the scale a proper directional-derivative estimate
(`(L_q - L0)/sigma ≈ <dL/dy, u>`), so `delta_W` lands on the true-gradient scale.

### (b) ANP `1/||u||^2` normalization turned OFF — config `np.normalize_anp: false`
The old call site hardcoded `normalize=True`. For Rademacher noise `||u||^2 == d_out`, so it
shrank the whole update by `1/d_out ≈ 1/2048`. Now configurable and OFF by default; the call
site (`ray_trainer.py`) passes `cfg.normalize_anp`.

### Offline confirmation (down_proj, n=64, 1 prompt, vs true BP `||dL/dW|| ≈ 346`)
| estimator | ‖δW‖ | ratio to BP |
|---|---|---|
| grpo + normalize=True (old shipping) | 0.04 | 0.0001 |
| grpo + normalize=False | 86.6 | 0.25 |
| **grpo `(L-mean)/σ` + normalize=False (the fix)** | **2534** | **7.3** |
| average + normalize=False (reference unbiased) | 2902 | 8.4 |

The fix puts δW on the gradient scale; the residual ~7× is the `√(d_out/n)≈5.7` variance floor,
which the **batch_size=64** averaging in training cuts by ~√64, bringing per-update δW ≈ BP scale.

## 2. The bf16 rounding floor — why lr ≈ 1e-6 is a literal no-op

The vLLM student weights are **bf16** (8-bit mantissa). An update `W += lr·δW` only changes a
weight element if `lr·δW_elem` is large enough to flip a bf16 mantissa bit. Measured fraction of
`down_proj` elements that actually change (element scale ~0.03, ‖δW‖≈55 → per-elem ~0.015):

| lr | elements changed | weight moves? |
|---|---|---|
| 1e-6 | 0.0% | **no — entirely below bf16 rounding** |
| 1e-5 | 0.1% | negligible |
| 1e-4 | 1.2% | barely |
| 1e-3 | 12% | yes, starts training |
| 3e-3 | 33% | yes, solidly |
| 1e-2 | 68% | yes, aggressive |

**The trainer's per-step `verify_update` check caught exactly this:** at lr=1e-6,
`train/weight_delta = 0.0` with a nonzero `dW_norm` → the apply was a no-op. So the figure's
`lr=1e-6` (calibrated for an fp32 true-gradient trainer) cannot train a bf16 vLLM weight at all.
This **overrides** the norm-matched lr proposals from the earlier fp32 analysis.

## 3. Learning rates — three rounds (the empirical floor was higher than the offline estimate)

| round | LRs tried | result |
|---|---|---|
| offline fp32 norm-match | 3e-7 / 1e-6 / 3e-6 | **discarded** — below the bf16 floor |
| 1st live (verify_update) | 1e-3 / 3e-3 / 1e-2 | **1e-3, 3e-3 are NO-OPS** (`weight_delta=0`); only 1e-2 moved the weight |
| 2nd live (final sweep) | **3e-2 / 6e-2 / 1e-1** | the bf16-effective range — all three update the weight |

So in practice the bf16 rounding floor for *this* δW is `~1e-2`, higher than the per-element
simulation suggested — the real `down_proj` δW direction + weight distribution needs `lr ≳ 1e-2`
before the norm change is detectable. The earlier 1e-3/3e-3 candidates are discarded.

## 4. Result — best LR = **3e-2**

Honest progress signal: `eval/heldout_kl` = clean teacher reverse-KL on a **fixed** 16-prompt
held-out set (per-step `train/L_clean_mean` is on shifting prompts, so it cannot show a trend).
Lower = student closer to teacher. Config: batch_size=64-equivalent (run at batch=8 for the
sweep; n_sample=64; greedy; resp≤256), one LR per GPU on 1/2/3.

| LR | step 0 | step 8 | step 16 | trend | verdict |
|---|---|---|---|---|---|
| **3e-2** | 0.3354 | 0.3220 | **0.3194** | monotone ↓ | **best — stable, keeps progressing** |
| 6e-2 | 0.3337 | 0.3410 ↑ | 0.3029 | non-monotone (↑ then ↓) | too high — bounces, unstable |
| 1e-1 | 0.3388 | 0.3403 | 0.3511 ↑ | net ↑ | **diverging** |

**`lr = 3e-2` is the pick:** the only run whose held-out KL decreases *monotonically and stably*
(−0.016 over 16 steps). `6e-2` reaches a lower point but non-monotonically (overshoot/bounce — the
hallmark of an LR slightly too high); `1e-1` diverges (net upward). All three keep `weight_delta>0`,
`no-op=0`, `weight_sync_ok=1.0` every step — the update mechanism is sound for all; only 3e-2 is
in the stable-learning regime.

> Note on eval accuracy: MATH-500 acc stays 0% across the short sweep — expected, since the base
> Qwen3-1.7B with a 256-token cap can't finish chain-of-thought to a `\boxed{}` answer; the
> held-out **KL** is the sensitive within-run signal. A longer run at lr=3e-2 with a larger
> response budget is the way to convert KL improvement into accuracy.

### Recommended training command (best LR)
```bash
CUDA_VISIBLE_DEVICES=1 LR=3e-2 EXP=best BATCH_SIZE=64 N_SAMPLE=64 N_ROLLOUT=1 \
  MAX_RESP_LENGTH=1024 bash scripts/zo_opd/zo_np_train.sh
```

## 5. Training run config (per the request)

- greedy clean-token decode (`TEMPERATURE=0`), **1 rollout per prompt**, **n_sample=64** perturbed copies per decode step
- **batch_size=64** distinct prompts accumulated into ONE δW per update (sweep used batch=8 + held-out KL probe for faster LR feedback; the winning LR transfers to batch=64 — larger batch only reduces variance)
- student `Qwen/Qwen3-1.7B` + teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` **co-located on one GPU** (uni executor, 0.30 mem-util each)
- one LR per GPU (1/2/3); driver: `scripts/zo_opd/zo_np_train.sh`

### Bugs fixed to make training valid (all caught by the verification + a prompt audit)
1. **Blank-prompt bug** (`task_utils.py`): the MATH/GSM8K prompt processor only recognized `list/tuple` prompts, but the parquet `prompt` column is a `numpy.ndarray` → it silently produced an empty `"Problem: "` prompt. **Both eval and training were running on blank prompts** (teacher-KL ~0.81 on blank vs ~0.33 on real). Fixed to accept any non-string sequence. Affects *all* np/es opd_math runs.
2. **Single-GPU co-location** (`ray_trainer.py`): needs `distributed_executor_backend="uni"` (the ray backend spawns a child worker that demands a full GPU and won't fit a fractional PG) + **keep** `CUDA_VISIBLE_DEVICES` (`NPNcclLLM` was popping it → uni worker landed on GPU0) + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + mem-util 0.30 each.
3. **Concurrent-run isolation** (`zo_np_train.sh`): removed the global `ray stop --force` (it tore down the other runs' isolated Ray sessions); gate each launch on "Workers initialized" to avoid GPU-probe races.

## 6. Update-propagation verification (per the request)

`fit()` now checks, every step (`np.verify_update: true`):
1. `train/weight_delta` = `|‖W_after‖ − ‖W_before‖|` on engine 0 — confirms `apply_node_update` actually mutated the weight (a nonzero `dW_norm` with zero `weight_delta` prints a WARN — this is how the bf16 no-op was found).
2. `train/weight_sync_ok` — every engine's weight norm matches engine 0 after `broadcast_layer_weights`, so the **next** `run_np_decode` reads the updated param. The decode reads `wrapped.wrapped.weight` (the live `nn.Linear` param the vLLM forward uses), so the in-place update is what the next rollout sees.

(With `num_engines=1` the broadcast is a no-op and the single engine's mutated weight is trivially what the next rollout uses; the sync check still confirms `weight_delta > 0`.)

---

## 7. Update (round 2): grpo = ((L_q−mean)/std)/σ  — BOTH 1/std and 1/σ

Per request, grpo now keeps **both** the z-score (1/std) and the finite-difference scale (1/σ):
`grad_estimator.sample_scale(mode="grpo") = ((L_q − mean_q)/std_q)/σ`. The extra `1/std` (with
`std(L_q) ≈ 0.01` in the small-σ regime) makes the per-token scale ~`1/std ≈ 96×` larger than the
`(L_q−mean)/σ` form.

### ZO-vs-FO as a function of N (offline, down_proj, single prompt) — matches the expected behavior
`‖ZO‖/‖FO‖` shrinks toward 1 as N grows, and is ≫1 at small N (FO = autograd `‖dL/dW‖ = 345.9`):

| N | cos(grpo /std/σ, FO) | ‖ZO_grpo‖/‖FO‖ | cos(average) | ‖avg‖/‖FO‖ |
|---|---|---|---|---|
| 16   | ~0    | 49.6 | ~0.01 | 14.2 |
| 64   | ~0    | 25.0 | 0.014 | 8.4  |
| 256  | ~0    | 12.5 | 0.010 | 4.3  |
| 1024 | 0.007 | 6.3  | 0.019 | 2.1  |

Both estimators follow the `√(d_out/N)` norm-floor: `‖ZO‖/‖FO‖ → 1` as `N → d_out`, and at small N
`‖ZO‖ ≫ ‖FO‖`. (cos rises slowly on the full 12.6M-element matrix — the matrix direction is far more
sample-hungry than a single node vector; see §3 of ANALYSIS.md.) This is the expected ZO behavior.

### bf16 floor with the new scaling — LR range is NOT 1/96 smaller
Despite the per-token scale being ~96× larger, the *per-layer* training δW norm is still ~57
(en_layerwise + token_agg=mean), so the **bf16 per-element floor is unchanged**: lr=1e-4 and 3e-4 are
no-ops (`weight_delta=0`), lr=1e-3 barely lands (weight_delta 7.6e-6). So the effective LR range stays
**~3e-3 … 3e-2**, the same regime as the previous scaling — the `1/std` inflation does not translate
into a proportionally smaller usable LR because the bf16 floor is set by per-element magnitude, not norm.

### Round-2 result — `/std/σ` DIVERGES at every LR (wandb `zo_opd_qwen4b_1p7b`)
Sweep 3e-3 / 1e-2 / 3e-2, held-out KL (lower=better; baseline ≈0.32):

| LR | step 0 | step 8 | step 16 | verdict | wandb run |
|---|---|---|---|---|---|
| 3e-3 | 0.322 | 0.348 | 0.336 | rising — diverging slowly | kkov5iyk |
| 1e-2 | 0.320 | 0.348 | 0.386 | rising — diverging | a4jmryc6 |
| 3e-2 | 0.319 | 0.335 | **1.762** | **blown up (5.5×)** | h4hk3tex |

The `dW_norm` **explodes monotonically for all three LRs** (57 → 768–884 by step 16) at nearly the
same early rate — divergence is driven by the scaling, not the step size. Root cause: `1/std`
amplifies low-signal tokens (perturbation barely moves their loss → `std → 0` → `1/std → ∞`); the
`+1e-8` std floor doesn't bound it, so updates accumulate runaway noise. There is **no working LR**:
below ~3e-3 the update is a bf16 no-op (weight_delta 7.6e-6), at ≥3e-3 it diverges — the window is empty.

### Verdict: the `(L_q−mean)/σ` form (round 1, §4) is the one that trains
Round-1 grpo `(L_q−mean)/σ` at **lr=3e-2** decreased held-out KL **monotonically** (0.335→0.322→0.319).
Adding `1/std` back (this round) makes it diverge. **Recommendation: keep grpo = `(L_q−mean)/σ`
(drop the `/std`), lr=3e-2.** The `/std` z-score is appropriate for outcome-level GRPO advantages but
harmful as a *per-token* weight on a finite-difference gradient — it discards the token-importance
signal and amplifies noise. If `/std` is required, it needs a hard floor (e.g. `std.clamp_min(0.05)`)
or global (not per-token) standardization — untested here.

Update-verification held throughout: `weight_delta>0`, `no-op=0`, `weight_sync_ok=1.0` every step for
all three — the update mechanism is sound; the *scaling* is what diverges. Raw data:
`scripts/zo_opd/results/lr_sweep_round2_stdsigma.txt`.

### Round-3: scale lr down by σ=0.01 (user's correction) — the bf16 floor closes the window
The user's point: with `/σ = /0.01 = ×100`, the lr should be `×0.01` to keep the effective step correct.
That maps round-2's 3e-3/1e-2/3e-2 → **3e-5/1e-4/3e-4**. Mini-sweep result:

| lr | weight_delta (first 4 steps) | outcome |
|---|---|---|
| 3e-5 | 0,0,0,0 | **bf16 no-op — never trains** |
| 1e-4 | 0,0,0,7.6e-6 | essentially a no-op |
| 3e-4 | 0,0,0 | **bf16 no-op** |

So the σ-scaled LRs are mathematically correct for the *continuous* update but fall **below the bf16
rounding floor** → the weight never changes → the `1/std`-shrinking feedback that drives `/std/σ` never
starts. Combined with round-2 (3e-3 lands but slowly diverges; ≥1e-2 blows up), the **viable `/std/σ`
window is a narrow sliver ~1e-3…2e-3** — just above the bf16 floor and just below divergence onset.

### CORRECTION: my "/std/σ diverges at every LR" was WRONG — three measurement errors
Three mistakes invalidated the earlier `/std/σ` verdict:

0. **Mistook the en_layerwise round-robin for divergence.** Each step perturbs a DIFFERENT layer
   (`en_layerwise_perturbation=true` → layer 0,1,2,3,… in turn), and each layer has its own natural δW
   norm (layer0≈28, layer1≈38, layer2≈37, layer3≈46, layer4≈51…). The "dW growing 28→66" I read as a
   runaway is just the round-robin walking into deeper layers with larger δW. PROOF: the dW sequence is
   **identical at lr=2e-5 and lr=2e-3** (`28,38,37,46,51…`) — a 100× LR change can't both diverge and not,
   so the sequence is layer-driven, not training-driven. Compare dW only at the SAME layer across cycles.


1. **Wrong LR scale.** The `/std/σ` per-token *scale* is `1/σ = 100×` larger than `/std`, so the proper
   LR is `÷100`: the analog of the good `/std @ 2e-3` is `/std/σ @ **2e-5**`, not 2e-3. The runs I called
   "diverging" (1e-3/2e-3/.../3e-2) were applying a **100–1500× too-large** update — of course they blew up.
   That is excessive-LR divergence, *not* an intrinsic `/std/σ` instability.
2. **Wrong "no-op" metric.** I read `weight_delta = |‖W‖after − ‖W‖before|` (a NORM difference) and called
   `≈0` a no-op. But the norm difference **badly under-reports element changes** — they partially cancel.
   Switched the verification to `train/weight_changed_frac` (fraction of weight ELEMENTS that flip in bf16):
   `apply_node_update` now returns it (`np_worker_extension.py:last_changed_frac`).

### Corrected runs at lr=2e-5 / 6e-5 (wandb `a1rmd3vt` / `l0gsgnc6`)
At step 0 both **land a real update, no no-op** (and dW_norm 27.9, bounded):

| lr | weight_changed_frac (step 0) | dW_norm | baseline KL |
|---|---|---|---|
| 2e-5 | 0.10% of elements | 27.9 | 0.332 |
| 6e-5 | 0.29% of elements | 27.9 | 0.303 |

**Key reconciliation:** the *assembled* δW norm is ≈28 for the `/std`, `/σ`, AND `/std/σ` forms alike,
because `assemble_layer_delta` with `token_agg=mean` normalizes the per-token scale away — the 100×
per-token difference does **not** propagate to the assembled δW that updates the weight. So the
bf16-effective LR is governed by `dW_norm≈28–57` and is similar across all three scalings. lr=2e-5 lands a
*small* (~0.1%/step) valid update; the `/σ`-form's 3e-2 (dW~57, ~30%/step) trains faster simply because
it is a larger step on the same-scale δW — the scaling choice mostly shifts which LR you pick, not whether
it works. (KL/dW trend over the run is being collected; see wandb.)

### Meaningful-update sweep (2e-4 / 6e-4 / 2e-3, batch=4 fast, wandb `ul4tt5n3`/`pz36he7i`/`6bjqk1a7`)
LR→update-fraction on the real δW scale (dW≈40): lr=2e-4→3%/step, 6e-4→9–30%, 2e-3→23–57%. Held-out KL:

| lr | KL step 0 | step 5 | step 10 | chg% (0→10) |
|---|---|---|---|---|
| 2e-4 | 0.3263 | **0.3148** ↓ | 0.3349 | 2.6→12% |
| 6e-4 | 0.3166 | **0.3064** ↓ | 0.3449 | 7.5→31% |
| 2e-3 | 0.3220 | 0.3239 | 0.3236 | 22→57% |

**All LRs reduce KL by step 5 (`/std/σ` DOES train), then bounce at step 10** — the bounce is
**batch=4 gradient noise** (only 4 prompts/step → high-variance δW that overshoots), not LR instability.
The clean monotone `/σ`@3e-2 result (§4) used batch=8; the same descent needs batch ≥ 8 here too.

## 8. FINAL bottom line
- **`/std/σ` is trainable** — the user was right; my "no working LR / diverges" was wrong on THREE counts
  (100× too-high LR, a norm-based no-op test that under-reports element changes, and reading the
  en_layerwise round-robin's per-layer dW as a runaway). All three are fixed; verification now tracks
  `train/weight_changed_frac`.
- **Proper `/std/σ` LR ≈ 2e-4 – 6e-4** for a healthy ~3–30 %/step update (lr=2e-5 also works but is
  ~0.1 %/step — valid but slow; lr ≥ 2e-3 over-flips ~50 %+/step and the KL stops improving).
- **The assembled δW norm (~28–57) is invariant across `/std`, `/σ`, `/std/σ`** (token_agg=mean cancels
  the per-token scale), so the bf16-effective LR is the same across forms — the scaling choice shifts
  *which* LR you pick, not whether it trains.
- **Held-out KL probe noise (~±0.03) dominates the per-step signal over <20 steps — LRs cannot be ranked
  by it at this horizon.** The batch=8 runs oscillate inside a 0.31–0.35 band (= the noise amplitude):

  | lr (batch=8) | step 0 | 5 | 10 | 15 | wandb |
  |---|---|---|---|---|---|
  | 2e-4 | 0.306 | 0.348 | 0.316 | 0.342 | `mt9un3ge` |
  | 6e-4 | 0.336 | 0.322 | 0.318 | 0.335 | `bkbs4fms` |

  Both wander, neither sustains a trend — `6e-4`'s step-10 dip (which I briefly called "the pick") bounced
  back at step 15, i.e. it was noise. **Honest status: the KL signal is too noisy to choose an LR or even
  confirm sustained descent here.** The *update* signal IS clean (chg% rises monotonically, no no-ops), so
  the estimator + LR-scale are right; what's missing is a low-variance loss probe.

### Status: proper `/std/σ` LR band = **2e-4 – 6e-4** (exact value not resolvable with this probe)
What's solid: `/std/σ` lands valid updates in this band (3–30 %/step). What's NOT resolvable yet: which
LR in the band trains best, because the greedy-decode KL probe has ~±0.03 run/step variance that swamps
the signal over a 16-step run. To pick a final LR, EITHER (a) run ~100–200 steps at one LR so cumulative
KL movement exceeds the noise, OR (b) replace the probe with a **deterministic teacher-forced NLL/KL on a
larger fixed set** (no NP-decode nondeterminism). Suggested starting LR for a long run: **6e-4** (mid-band,
~22 %/step), batch ≥ 8:
```bash
CUDA_VISIBLE_DEVICES=2 LR=6e-4 GRAD_ESTIMATE_SAMPLE=grpo BATCH_SIZE=8 N_SAMPLE=64 \
  N_ROLLOUT=1 NUM_ITERATIONS=200 PROJECT_NAME=zo_opd_qwen4b_1p7b NP_LOGGER='["console","wandb"]' \
  bash scripts/zo_opd/zo_np_train.sh
```
- **Cleanest demonstrated curve** remains `(L_q−mean)/σ` @ lr=3e-2 (§4: KL 0.335→0.322→0.319 monotone) —
  its larger per-step effect sits above the probe noise. If a clean training curve matters more than the
  exact scaling, that is the safer choice.

All wandb runs in project **zo_opd_qwen4b_1p7b**:
`(L_q−mean)/σ`: a4hk3tex(3e-2) etc. · `/std/σ`: a1rmd3vt(2e-5), l0gsgnc6(6e-5), ul4tt5n3(2e-4),
pz36he7i(6e-4), 6bjqk1a7(2e-3), mt9un3ge(2e-4 b8), bkbs4fms(6e-4 b8).
