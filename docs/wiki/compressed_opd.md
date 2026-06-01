# compressed_opd: BTT-compressed Qwen3-4B → ~1.7B for OPD math

> Goal: train a Qwen3-4B model BTT-compressed to ~1.7B params with the
> on-policy distillation (OPD) trainer, using the same teacher (Qwen3-4B
> Non-Thinking RL Math Step 500) for both the distillation signal and the
> student initialisation.

This page captures the full state of the `compressed_opd` workflow:
launchers, calibration cache, FSDP2 + BlockTT fixes, training stability,
and the C4 perplexity audit that exposed a fundamental gap in BTT-LLM-V2.

---

## Quick map

| artifact | purpose |
|---|---|
| `scripts/opd/math/compressed_opd/_common.sh` | shared env: paths, BTT/calib knobs, FSDP2 + memory-fit overrides |
| `scripts/opd/math/compressed_opd/btt_v2.sh` | `CALIB_MODE=v2`, GPU 4 default |
| `scripts/opd/math/compressed_opd/btt_v2_combined.sh` | `CALIB_MODE=v2_combined`, GPU 5 default |
| `scripts/opd/math/compressed_opd/_smoke_btt_v2.sh` | 1-step smoke (val_before_train + AMC23) |
| `scripts/opd/math/compressed_opd/eval_c4_ppl.py` | C4 PPL of the *cached* compressed checkpoints (HIT path) |
| `scripts/opd/math/compressed_opd/eval_c4_calib_ppl.py` | C4 PPL of an *ad-hoc* C4-calibrated BTT compression |
| `scripts/opd/math/compressed_opd/eval_c4_calib_ppl_svd.py` | same, SVD-LLM-V2 baseline |
| `verl/verl/workers/peft/blocktt.py` | adapter: untie, cache, FSDP2 plumbing, `enable_input_require_grads` |
| `verl/verl/workers/peft/calib_loader.py` | `training_data` calib source via `$TRAIN_DATASET` parquet |
| `on_policy_distillation.sh` | wrapper; `EXTRA_HYDRA_ARGS` pass-through for per-script overrides |
| `/data/yequan/huggingface/opd_calib_cache/<sha>/` | persistent calibrated BTT factors + topology + signature |

---

## Design

### Teacher and student

- **Teacher**: `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` (4.022 B, bf16).
- **Student init**: same checkpoint as the teacher.
- **Compression**: each `nn.Linear` (except `lm_head`) is replaced by a
  `compress.btt.btt_linear.BTTLinear` with `output_one_block` decomposition
  at per-linear ratio 0.36. With tied embed/lm_head this gives
  0.389 B (embed) + 0.36 × 3.633 B (linears) ≈ **1.66 B** materialized params.
- **Training**: same SimpleRL-Zoo math setting as `scripts/opd/math/full.sh`
  (MATH train, MATH-500 eval, train temp 0.6, eval temp 1.0 / top_p 0.95,
  3072 max response, 8 rollouts/prompt).
- **Calibration**: MATH training prompts (no rollouts — the worker only sees
  prompt text) + OPD-faithful surrogate loss
  (`compress/calibration_opd_loss.py`) to drive the backward covariance for
  the `v2_combined` mode.

### Two calibration modes

| script | `CALIB_MODE` | covariances used | `CALIB_LOSS=opd` actually drives |
|---|---|---|---|
| `btt_v2.sh` | `v2` | forward only | unused (CE/OPD irrelevant) |
| `btt_v2_combined.sh` | `v2_combined` | forward + backward | backward whitening uses OPD policy gradient |

Only `v2_combined` actually consumes the OPD loss; `v2` ignores `CALIB_LOSS`
and the teacher model is loaded but the backward pass is never run.

### Per-run knobs (`_common.sh`)

```bash
ACTOR_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
REWARD_MODEL_PATH=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
TRAIN_DATASET_NAME=MATH

LR=${LR:-1e-6}                      # 1e-5 was unstable; see "Training stability"
ACTOR_PARAM_OFFLOAD=False           # BTT init requires weights on CUDA
ACTOR_OPTIM_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.40

PEFT_MODE=blocktt
BTT_DECOMP_MODE=output_one_block
BTT_RANK=0.36
BTT_TRAIN_POSITION=both
BTT_S_MERGED_TO=split
BTT_CONVERT_MODE=svd
BTT_FACTORIZE_BY_HEAD=True

CALIB_SOURCE=training_data
CALIB_LOSS=opd
CALIB_TOP_K=16
CALIB_TOP_K_STRATEGY=only_stu
CALIB_REWARD_WEIGHT_MODE=student_p
CALIB_TEMPERATURE=$TEMPERATURE
CALIB_TEACHER_TEMPERATURE=1.0

EXTRA_HYDRA_ARGS="
  +data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
"
```

`+data.apply_chat_template_kwargs.enable_thinking=False` is required (the
teacher is "Non-Thinking"); the same flag propagates into the in-worker
calib loader via the `ENABLE_THINKING` env var that `_common.sh` also sets.

---

## What was fixed to make compressed_opd run end-to-end

### 1. `training_data` calibration source (in `verl/workers/peft/calib_loader.py`)

`compress.integration.build_calib_loader` for `calib_source=training_data`
expects either an in-process `training_dataset` or an `rl_rollout_fn`. The
FSDP worker has neither (it only sees `actor_rollout_ref`, not the trainer's
data). The new path reads `$TRAIN_DATASET` (already exported by
`on_policy_distillation.sh`), loads the parquet directly, applies the
trainer's chat template (with `ENABLE_THINKING` propagated), and builds a
per-prompt loader (variable-length, right-padded). The original
`_pack_token_windows` requires every text to be ≥ `max_length=3072` tokens,
which math prompts never satisfy; the per-prompt loader sidesteps that.

### 2. Calibrated model not on CUDA (in `blocktt.py`)

`compress.calibration.collect_covariances_from_loader` assumes the model is
already on the target device — it only moves the *inputs*. Inside the
adapter's `apply()` we now `model = model.to(device)` immediately before the
calibration pass, and free the lazily-loaded teacher after
`apply_calibrated_btt` returns.

### 3. Tied embed/lm_head + FSDP1 = writeback shape mismatch

FSDP1 with `use_orig_params=True` (required by BlockTT's mixed
trainable/frozen params) fails the per-step `_writeback_orig_params` check
on the frozen embedding:

```
RuntimeError: Cannot writeback when the parameter shape changes
Expects torch.Size([388956160]) but got torch.Size([151936, 2560])
```

The fix is two-fold (commit `7a8a61c`, refined in `15f9d45`):
- Always `_untie_embeddings()` BEFORE FSDP wrap (replaces lm_head's tied
  `weight` with a fresh `nn.Parameter` and clears
  `model.config.tie_word_embeddings`).
- Force **FSDP2** for actor + ref (`actor_rollout_ref.{actor,ref}.strategy=fsdp2`).
  FSDP2's per-parameter DTensor sharding has no such writeback constraint.

### 4. vLLM weight load (`export_for_vllm` in `blocktt.py`)

`materialize_calibrated_btt_weights` walks `named_modules()` on the FSDP
wrapped module, which leaves `_fsdp_wrapped_module.` segments in the keys.
vLLM's `Qwen3ForCausalLM.load_weights` walks its own module tree by
qualname and KeyErrors on the FSDP-internal segment. The adapter now
strips the segment and **also emits every non-BTT parameter** (embeddings,
norms, lm_head, any untouched linears) — otherwise vLLM keeps stale
base-model weights at those positions.

### 5. Gradient flow into frozen embedding

With BTT's `train_position=both` plus `s_merged_to=split`, the BTT cores
have `requires_grad=True`, the embedding stays frozen, and gradient
checkpointing's `use_reentrant=False` warns
`None of the inputs have requires_grad=True. Gradients will be None`, then
backward fails with
`element 0 of tensors does not require grad and does not have a grad_fn`.
The adapter calls `model.enable_input_require_grads()` before FSDP wrap
(same fix the LoRA adapter uses).

### 6. Persistent calibration cache

Calibration takes ~50 min (forward + backward covariance over 128 prompts
× 3072 tokens with 4 B student + 4 B teacher). The adapter now:

- Computes a SHA256 of every weight-affecting knob (actor path, BTT
  knobs, calib knobs, training_data path, teacher path / enable_thinking
  when `CALIB_LOSS=opd`).
- Saves `model.safetensors` + `btt_topology.json` + `signature.json` under
  `$CALIB_CACHE_DIR / $HF_HOME/opd_calib_cache / /tmp/opd_calib_cache`.
- On the next run with matching settings prints `[BlockTT calib] cache HIT`,
  rebuilds BTT topology, loads the cached factors, and skips the
  ~50-minute pass.
- `CALIB_CACHE_FORCE_REBUILD=1` overrides.

`train_position` is intentionally excluded from the signature — it only
sets `requires_grad` flags, never the tensor values.

---

## Training stability findings (LR sweep at ratio 0.36, output_one_block)

`combined v8` (LR=1e-6) is the first config that stays on a stable
trajectory. The earlier sweep is summarised below; `btt_v2 v8` and earlier
all diverged at step 2.

### `btt_v2` (forward-only calib), LR=1e-6, ratio 0.36

| step | pearson_corr | overlap_ratio | pg_loss | grad_norm |
|---:|---:|---:|---:|---:|
| 1 | 0.945 | 0.423 | 0.117 | 151 |
| 2 | 0.162 | 0.357 | 0.988 | 1028 |
| 3 | 0.674 | 0.243 | 0.400 | 1170 |
| 4 | 0.133 | 0.204 | 0.099 | 1005 |
| 5 | **−0.124** | 0.118 | — | — |

Oscillating then anti-correlated → divergent. Killed.

### `btt_v2_combined` (OPD-loss backward calib), LR=1e-6, ratio 0.36

| step | actor/entropy | pearson_corr | overlap_ratio | pg_loss | grad_norm | true_reward |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.924 | 0.969 | 0.456 | 0.005 | 554 | 0.0 |
| 2 | 0.923 | 0.990 | 0.455 | 0.007 | 702 | 0.0 |
| 3 | 0.910 | 0.984 | 0.452 | 0.008 | 818 | 0.0 |
| 4 | 0.904 | 0.974 | 0.450 | — | — | 0.0 |
| 5 | 0.890 | 0.974 | 0.446 | 0.011 | 1116 | 0.0 |
| 6 | 0.889 | 0.969 | 0.442 | — | — | 0.0 |
| 7 | 0.881 | 0.964 | 0.434 | — | — | 0.0 |

PPO stable, but MATH-500 acc=0.0 at step 5 with `format_score=0.045`. Every
response truncates at the max length of 3072 — the compressed model can't
emit a `\boxed{…}` terminator.

### LR sweep summary at ratio 0.36

| LR | btt_v2 step 2 | btt_v2 step 5 | combined step 2 |
|---|---|---|---|
| 1e-5 | pearson 0.16, grad 3626 | (KIA before step 5) | pearson 0.97 (first attempt) |
| 3e-6 | pearson 0.34, grad 1605 | — | — |
| 1e-6 | pearson 0.16, grad 1028 | pearson −0.12 | pearson 0.99, stable |

LR=1e-6 is the only setting in this sweep where any compressed_opd
configuration survives past step 2. The user's original LR=1e-5 request
gives gradient norms above 1000 even at step 1 (cause: per-token OPD
advantage extrema; see C4 PPL section below).

---

## Memory budget (single H100, shared)

`update_actor` peaks at **97 GiB allocated, 108 GiB reserved** on a 95 GiB
H100 with the v6 config (`ACTOR_OPTIM_OFFLOAD=True`,
`GPU_MEMORY_UTILIZATION=0.40`, `ppo_max_token_len_per_gpu=16384`). It
survives only because PyTorch's fragmented reservation overlaps with the
~15 GiB phantom allocation we routinely see from other users on the same
node.

If `update_actor` OOMs again, the next knob to drop is
`ppo_max_token_len_per_gpu` → `8192` (halves activations again at the
cost of slower steps).

---

## The C4 perplexity audit (the real story)

After step 5 MATH-500 returned 0.0 acc on `combined v8`, we audited the
quality of the compressed checkpoints by measuring C4 perplexity on:

- the original Qwen3-4B teacher (4.022 B);
- the cached `btt_v2` BTT factors (1.656 B, MATH+OPD calibration);
- the cached `btt_v2_combined` BTT factors (1.656 B, MATH+OPD calibration);
- ad-hoc BTT and SVD compressions calibrated on C4 directly.

### Results

| method | calib | ratio | params | C4 PPL |
|---|---|---:|---:|---:|
| Qwen3-4B-Non-Thinking-RL-Math-Step500 | – | 1.000 | 4.022 B | **19.85** |
| SVD-LLM-V2 | C4 | 0.700 | 2.931 B | **57.10** |
| SVD-LLM-V2 | C4 | 0.360 | 1.696 B | 4 339 |
| BTT-LLM-V2 (`output_one_block`) | C4 | 0.700 | 2.910 B | **1 667** |
| BTT-LLM-V2 (`output_one_block`) | C4 | 0.360 | 1.656 B | 25 918 |
| BTT-LLM-V2 (`output_one_block`) | training_data + OPD (cache HIT) | 0.360 | 1.656 B | 1 524 928 |
| BTT-LLM-V2-COMBINED (`output_one_block`) | training_data + OPD (cache HIT) | 0.360 | 1.656 B | 8 750 778 |

PPL is measured by `compress.ppl_eval.evaluate_model_ppl` (sliding window,
seqlen 2048, seed 0).

### What the numbers say

1. **`btt_llm_v2` is broken**: at ratio 0.70 it gives PPL 1 667, **29× worse
   than SVD-LLM-V2's 57.10 at the same ratio**. The SVD-LLM-V2 PPL is in
   the published-paper range; BTT is not.

2. **`output_one_block` slices the input dim**: for Qwen3-4B with
   `d_in=2560`, `_closest_factor_pair(2560)=(40, 64)` → 40 chunks of 64.
   The per-block whitening (`_per_block_input_whitening` in
   `src/compress/btt/btt_llm_v2.py`) takes only the BLOCK-DIAGONAL of the
   `(d_in, d_in)` input covariance:

   ```python
   for j in range(n):
       X_j = C_x_gpu[j * b : (j + 1) * b, j * b : (j + 1) * b].contiguous()
       out[j] = compute_whitening(X_j, device=device)
   ```

   For Qwen3-4B's `d_in=2560`, that's 40 × (64 × 64) = 164 K covariance
   entries used out of 6.5 M (2 560²) — **97.5 % of the activation
   second-order statistics are silently discarded**.

3. **Several knobs are silently ignored** by `btt_llm_v2_compress_model`:

   ```python
   if s_merged_to is not None:
       logger.warning(f"btt_llm_v2: s_merged_to={s_merged_to!r} … ignoring.")
   if not factorize_by_head:
       logger.warning("btt_llm_v2: factorize_by_head=False … ignoring.")
   ```

   So `_common.sh`'s `BTT_S_MERGED_TO=split` and
   `BTT_FACTORIZE_BY_HEAD=True` are no-ops; the only thing that controls
   the singular-value distribution is the implicit "split via √S in
   `L = U·√S`, `R = √S·Vᵀ`" inside `btt_llm_v2_decompose_layer`.

4. **MATH + OPD calibration drifts further from C4 than C4 calibration
   does** (1.5 M vs 25 K PPL at the same ratio). Expected: OPD on math
   prompts specialises the BTT factors away from generic-text token
   distributions. But the absolute floor is already too high to recover
   via OPD training: at step 5 every val rollout truncates at max length
   without ever producing a `\boxed{…}`.

5. **Training stability is consistent with the PPL findings**:

   - `btt_v2` (forward-only) calibration → PPL 1 524 928 → policy
     diverges in 5 steps (pearson_corr → −0.12).
   - `btt_v2_combined` (OPD-loss backward) calibration → PPL 8 750 778
     → policy stable (pearson 0.97 – 0.99) but starting point so far
     from a coherent LM that 138 train steps won't lift acc off zero.

### Why SVD-LLM-V2 succeeds where BTT-LLM-V2 fails

SVD-LLM-V2 does ONE global SVD on the whitened weight:

```python
# svd/svd_llm_v2.py:44
U, S, Vh = torch.linalg.svd(W @ Phi, full_matrices=False)
```

`Phi` here is computed from the FULL `(d_in, d_in)` covariance, so all
activation correlations are captured. BTT-LLM-V2 does `n` independent
SVDs on `(d_out, b)` slabs, each whitened by only the diagonal block of
the same covariance.

This is not a fixable bug in the sense of "one line is wrong"; it's a
structural limitation of doing per-input-block SVD with per-block
whitening. Two reasonable fix paths:

- **Option A** — switch the default `decomp_mode` to `input_one_block`
  for compressed_opd. With `n=1` the per-block whitening reduces to a
  single global whitening over the full input dim, mathematically
  equivalent to SVD-LLM-V2's preconditioning. The output side is then
  sliced into `m = closest_factor(d_out)` blocks instead, which still
  loses cross-output correlations but is far less damaging for LLM
  layers where outputs project to coherent head/feature groups.
- **Option B** — implement true activation-aware BTT via joint ALS
  across blocks (already partly present as `btt_twosteps` and
  `btt_als`-style code). Higher quality, several days of work.

A quick A/B for Option A (`btt_llm_v2 + input_one_block + ratio 0.7`) was
launched after the audit; result captured in
`logs/compressed_opd/ppl_eval/`.

---

## Run history (this session)

| run dir | mode | LR | outcome |
|---|---|---|---|
| `smoke1…smoke6` | btt_v2 + various WIP fixes | 1e-5 | each smoke uncovered one bug (calib loader, model.to(cuda), FSDP1 untie, vLLM key strip, gradient checkpointing) |
| `smoke7…smoke9` | btt_v2, full calibration | 1e-5 | FSDP1 writeback failure on tied embedding → motivated FSDP2 + untie |
| `full_btt_v2`, `full_btt_v2_combined` | first full launches, FSDP1 | 1e-5 | both decomposed 252 layers, then SIGKILL'd externally before training |
| `full_btt_v2_v2`, `..._combined_v2` | post-cache feature, FSDP1 | 1e-5 | hit `Cannot writeback when the parameter shape changes` at first `update_actor` |
| `full_btt_v2_v3, v4, v5` | FSDP2 enabled, ratio 0.36 | 1e-5 | CUDA OOM at `update_actor` (cumem_allocator) |
| `full_btt_v2_v6` (`combined v5`) | FSDP2 + lower mem | 1e-5 | step 1 succeeded; step 2 grad_norm 3626, pearson_corr 0.37 |
| `full_btt_v2_v7` | FSDP2 + lower mem | 3e-6 | step 2 pearson 0.33, grad 1605 — still unstable |
| `full_btt_v2_v8` | FSDP2 + lower mem | 1e-6 | step 5 pearson −0.12 → killed |
| `full_btt_v2_combined_v8` | FSDP2 + lower mem | 1e-6 | pearson 0.97 – 0.99 across 7 steps, val acc 0.0 at step 5 → killed by user for PPL audit |

---

## Open work

1. **Pick a fix path for `btt_llm_v2`** (Option A / B above). Until that
   lands, the compressed-1.7B starting point is too noisy for OPD to
   recover MATH-500 accuracy within a normal training horizon.
2. **Consider a less aggressive ratio**. At 0.7, SVD-LLM-V2 gives PPL 57
   and would be a reasonable warm start; at 0.36 even SVD-LLM-V2 needs
   PPL 4 339. The user-requested 4B → 1.7B target corresponds to ratio
   0.36 on the linear-layer footprint.
3. **Re-evaluate `BTT_TRAIN_POSITION` and `s_merged_to`**. `train_position=both`
   doubles the effective sensitivity of small updates; if the BTT path is
   kept, `train_position=small` + `s_merged_to=keep_trainable` (fura's
   verified-working configuration) is the conservative first try.
4. **Caches are still on disk** at
   `/data/yequan/huggingface/opd_calib_cache/{6fe781d2b58145a5,0a443da3981d6f24}`
   (v2 and v2_combined respectively). Delete or `CALIB_CACHE_FORCE_REBUILD=1`
   once the implementation fix lands.

---

## Commits laid down in this session

- `15f9d45` — feat(compressed_opd): calibrated BTT compression of 4 B → ~1.7 B
  teacher for OPD math; adds `scripts/opd/math/compressed_opd/`, the
  `training_data` calib loader path, the FSDP1 untie workaround, the
  vLLM key-strip + non-BTT param emission, and the persistent
  calibration cache.
- `8240edc` — compressed_opd: force FSDP2 (match fura's verified-working
  config from `7a8a61c`).
- `af170f5` — compressed_opd: lower LR to 1e-6 + memory-fit knobs for
  shared GPU.
