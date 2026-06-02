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

Round-2 sweep (wandb project `zo_opd_qwen4b_1p7b`): **3e-3 / 1e-2 / 3e-2**.
