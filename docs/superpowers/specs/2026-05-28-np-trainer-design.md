# Node-Perturbation (NP) Trainer — Design

**Status:** approved design (supersedes the rough sketch in `docs/plans/np_trainer.md`)
**Date:** 2026-05-28
**Scope:** v1 = the per-step `n_sample`-wide perturbed-decode trainer with a teacher scorer, on top of vLLM 0.11.0 in the verl fork.

This design replaces the control flow described in `docs/plans/np_trainer.md`. That sketch assumed an antithetic, three-independent-`generate()` flow with a "recover `x` via a second clean forward" trick; investigation against the actual vLLM 0.11.0 source showed that flow is mathematically unsound (the three rollouts diverge token-for-token once a perturbation flips a sampled token) and structurally impossible (a worker RPC cannot call `generate()`). The real algorithm — clarified with the author — is a **single clean trajectory whose every decode step is evaluated `n_sample`-wide with ephemeral, non-cache-polluting perturbations**, scored per step by a teacher. This document is the authoritative spec.

---

## 1. Algorithm

Per training step, over a batch of prompts, the student decodes **one clean trajectory** per prompt. At **each decode step `t`**:

- The committed prefix `0..t-1` lives in **one shared KV cache**.
- Step `t` is evaluated `1 + n_sample` wide: row 0 = the **clean** copy (no perturbation), rows `1..n_sample` = **perturbed** copies. Each perturbed copy `q` gets an independent perturbation `u_q` injected at the **target linear layer's output** (the layer selected by `perturb_rules`). The perturbation propagates up through the remaining student layers + LM head, giving `n_sample` perturbed next-token distributions.
- The **teacher scores each** copy → `L_t^(q)` is the loss at token `t` under perturbation `u_q`. The clean copy gives the baseline `L_t`.
- The **per-token gradient estimate** is

  ```
  g_t = (1 / n_sample) * Σ_q  s(L_t^(q)) · u_q
  ```

  where `s(·)` is set by `grad_estimate_sample`:
  - `average`:  `s(L_t^(q)) = (L_t^(q) − L_t) / σ`     ← one-sided forward difference vs the clean baseline `L_t` (NOT `/2σ`; `2σ` is the antithetic central-difference form, which this one-sided design does not use)
  - `grpo`:     `s(L_t^(q)) = (L_t^(q) − mean_q) / std_q`

- We commit **only the clean token `t`** (sampled from row 0) to the sequence — exactly one KV slot grows. Advance to `t+1`, conditioned on the same clean prefix `0..t`. **Perturbations are ephemeral: never written to KV, never compounding across steps.**
- Across the rollout, per-token `g_t` accumulate into the layer's weight update as a rank-1 outer product with the captured layer input `x_t`:

  ```
  δW = Σ_t  g_t ⊗ x_t        (masked by response_mask; token_agg ∈ {sum, mean})
  ```

  with ANP-style normalization (`δy/‖δy‖²`, `‖δy‖² ≥ ε` clamp) and combined across `n_rollout` rollouts per `grad_estimate_sequence` (`average` / `grpo` over rollout reward `R_t`). Final update `W ← W + lr · δW`, broadcast to all engines.

**Why it's cheap:** decode is memory-bound and the per-step GEMM is under-utilized, so widening one decode step from 1 → `1 + n_sample` rows adds little wall-clock. Committing only the clean token keeps the KV cache single and the trajectory non-divergent — no duplicated KV, no exponential branching.

**Granularity knobs:**
- `perturb_granularity = token` → independent `u_q` per decode step (the default, richest credit assignment).
- `perturb_granularity = rollout` → one `u_q` per rollout, broadcast across all its steps.
- `en_layerwise_perturbation = true` → perturb one matched layer per step (round-robin over `perturb_rules` matches); `false` → all matched layers at once with independent noise.

---

## 2. vLLM mechanism (verified against vLLM 0.11.0 v1)

The per-step `1+n_sample` expansion is implemented by reusing vLLM v1's existing **multi-query-per-step** machinery (the same path speculative decoding uses), plus a custom linear layer and a custom decode driver. No edits to the installed `vllm` package.

**Confirmed facts:**

1. **Variable query-count per step is native.** `query_start_loc` allows `>1` query token per sequence per step — exactly how spec-decode packs `1+k` candidates against shared prefix KV (`vllm/v1/worker/gpu_model_runner.py:1413-1479`; `vllm/v1/attention/backends/flash_attn.py:516-548` `flash_attn_varlen_func` with `cu_seqlens_q`). Spec-decode (`vllm/v1/spec_decode/`) is the blueprint.

2. **Perturbed rows write no KV — `slot_mapping = −1` sentinel.** `reshape_and_cache` skips any row whose slot is `< 0` (`vllm/attention/ops/triton_reshape_and_cache_flash.py:33-37`; `PAD_SLOT_ID = -1` at `vllm/v1/attention/backends/utils.py:37`). Set perturbed rows' slots to `−1`: they compute but write nothing. Only row 0 writes its KV; step `t+1` grows the cache by exactly one token.

3. **Row→sample mapping is deterministic & contiguous.** Rows are laid out contiguously per sequence by `query_start_loc` with no in-forward reordering (`vllm/v1/worker/gpu_model_runner.py:950-958`), so the custom layer can slice its rows by sample index `q`.

4. **`enforce_eager=True` is mandatory.** Fresh `torch.randn` noise is fundamentally incompatible with CUDA-graph capture-once/replay-many (RNG state is not part of graph semantics). `enforce_eager=True` fully disables both torch.compile and cudagraph in 0.11.0 (`vllm/config/__init__.py:336-382`), so the custom forward runs in pure eager mode and per-step RNG is fine. Throughput cost ~1.5–3× vs cudagraph — accepted.

**Attention isolation — decision: prefix-sharing as separate sequences.** The `1+n_sample` rows must not attend to each other. Rather than inject a block-diagonal mask (which the default FlashAttention v1 backend does not expose), we model the `1+n_sample` rows as **separate decode sequences that share the prefix's KV blocks** (prefix sharing). Each row is then a vanilla single-query causal decode against the shared prefix — **no custom attention mask needed**, stays on the default FlashAttention path. The perturbation is the only thing that differs per row, applied by the custom linear layer.

```
step t:
  prefix KV [0..t-1]  (one physical copy, shared blocks)
   │
   ├─ seq_clean : query(tok_t, pos=t)        → row0 logits   [writes KV]
   ├─ seq_pert1 : query(tok_t, pos=t) + u1   → row1 logits   [slot=-1, no KV write]
   ├─ seq_pert2 : query(tok_t, pos=t) + u2   → row2 logits   [slot=-1]
   └─ ... n_sample
  no row attends to another row (they're separate seqs sharing prefix KV)
  commit only row0's sampled token → advance to t+1
```

**Never store `u_q` — store the seed.** Every `u_q` is regenerated on demand from a deterministic seed `noise_seed(step, layer, rollout, q)` via `torch.Generator(device).manual_seed(seed)` + a draw whose distribution follows `sample_method` (`gaussian`/`bernoulli`/`uniform`). No `u_q` tensor is persisted, crosses a Ray boundary, or outlives the forward that consumes it. The gradient estimator regenerates the same `u_q` from the same seed when forming `g_t ⊗ x_t`. (Mirrors `es_worker_extension.py:47-119` in spirit; the seed namespace now includes `(layer, rollout, q)`.)

---

## 3. Component architecture

Self-contained in the verl fork, mirroring the ES sibling layout. No edits to existing files (`main_ppo.py`, `core_algos.py`, `dp_actor.py`, `fsdp_workers.py`, the ES `main_es.py`/`es/`/`es_worker_extension.py`, or the installed `vllm`).

```
verl/verl/trainer/main_np.py                                  Hydra entry (mirror main_es.py)
verl/verl/trainer/np/__init__.py
verl/verl/trainer/np/ray_trainer.py                           RayNPTrainer + NPNcclLLM + fit()
verl/verl/trainer/np/np_decode.py                             ★ custom n_sample-wide decode driver + PerturbedLinear
verl/verl/trainer/np/grad_estimator.py                        g_t from {L_t^(q)}, accumulation, ANP-normalized δW
verl/verl/trainer/np/teacher_scorer.py                        per-step teacher scoring → L_t^(q)
verl/verl/trainer/np/task_utils.py                            re-export ES get_task_components (no copy)
verl/verl/trainer/config/np_trainer.yaml                      Hydra config (the np.* interface)
verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py install layer + drive decode + apply update + NCCL broadcast
scripts/zo_opd/opd_np.sh                                      top-level launcher (sibling to scripts/zo_opd/es.sh)
```

### Units & boundaries

1. **`PerturbedLinear` shim** (`np_decode.py`) — installed onto each `perturb_rules`-matched module after `load_model` (wraps/replaces the module's `forward`, keeping its weights & `quant_method` intact — no module reconstruction). Reads worker-local `self.np_state`:
   - `mode == "perturb"`: add `u_q` (regenerated from seed; never stored) to the perturbed rows, leave row 0 (clean) untouched.
   - `mode == "capture"`: record `x_t` (the layer's input, row 0) for the rank-1 update.
   - `mode == "off"`: pass-through.
   *Depends on:* `self.np_state` only. Handles the `(tensor)` or `(tensor, bias)` output form (bias is typically `None` for these layers — repack identically, do not assert non-None bias).

2. **`np_decode` driver** (`np_decode.py`) — the heart. Worker-side manual decode loop. Per step `t`: build the `1+n_sample`-wide step inputs (prefix-sharing sequences), set perturbed-row `slot_mapping = −1`, run one `model_runner.model(...)` forward under `set_forward_context`, harvest `1+n_sample` logits, call `teacher_scorer`, ask `grad_estimator` for `g_t`, sample+commit **only row 0's** token, advance. Reuses vLLM's `_prepare_inputs`/attention-metadata builders rather than reimplementing them; spec-decode metadata layout is the reference.
   *Interface:* `np_decode(prompts, sampling_params, target_layer, seed, sigma, n_sample) → (clean_tokens, grad_state)`.

3. **`teacher_scorer`** (`teacher_scorer.py`) — given the `1+n_sample` candidate next-token distributions at step `t`, returns `L_t^(q)`: per-token reverse-KL vs teacher over the OPD top-k set. Reads `log_prob_top_k`, `top_k_strategy`, `teacher_temperature`, `reward_weight_mode` from config (defaults from `verl/verl/workers/config/rollout.py:142-145`, read-only — do not modify that class). Teacher resides as a second vLLM engine on dedicated GPUs.
   **Sign convention:** `L_t^(q)` is minimization-oriented (lower = student closer to teacher); positive reverse-KL is already minimization-oriented. If ever sourcing from `dp_actor.compute_distillation_reward` (which returns `rm_scores = −kl·w`, maximization-oriented), **negate before use**.
   *Interface:* `score(candidate_logits, context) → L_t[q]`.

4. **`grad_estimator`** (`grad_estimator.py`) — pure math, no vLLM. `g_t = (1/n_sample) Σ_q s(L_t^(q))·u_q` (`s` per `grad_estimate_sample`); accumulate `Σ_t g_t ⊗ x_t` (masked by `response_mask`; `token_agg` sum/mean); combine across `n_rollout` per `grad_estimate_sequence`; ANP-normalize `δy/‖δy‖²` with `‖δy‖² ≥ ε` clamp + `update_clip`.
   *Interface:* `estimate(seeds, L_signal, x_t, cfg) → δW`.

5. **`RayNPTrainer.fit()`** (`ray_trainer.py`) — reuse ES's `_launch_engines` / `_init_inter_engine_group` / `_evaluate_model` / `_evaluate_with_engine` verbatim. `NPNcclLLM(LLM)` mirrors `ESNcclLLM` but forces `enforce_eager=True`, `enable_prefix_caching=True` (needed for the shared-prefix decode), `VLLM_ENABLE_V1_MULTIPROCESSING=0`. The only material rewrite is per-step orchestration: resolve `active_layers` from `perturb_rules` + `en_layerwise_perturbation`; drive `np_decode` per engine; `apply_node_update` (worker RPC, touches weights only); NCCL-broadcast the updated layer (reuse `es_worker_extension.py:132-137` pattern, single layer).

**Worker-extension RPC surface** (`np_worker_extension.py`): `install_perturb_layers(perturb_rules)`, `run_np_decode(...)`, `apply_node_update(layer, seeds, L_signal, lr, cfg)`, `init_inter_engine_group(...)`, `broadcast_layer_weights(layer, src_rank)`. Generation/decode is driven worker-side here (it's a hand-rolled model-runner loop, not `engine.generate()`), which is legitimate because it never queues engine requests — it calls `model_runner.model(...)` directly under `set_forward_context`, the same pattern `es_worker_extension.py:295-298` already uses for a manual forward.

---

## 4. Config interface (`np_trainer.yaml`)

Faithful to `docs/plans/np_trainer.md`'s stated interface, with `perturb_rules` corrected to vLLM-real module names.

```yaml
defaults:
  - _self_

np:
  sigma: 0.01
  n_sample: 8                      # perturbed copies scored per decode step
  n_rollout: 8                     # rollouts for sequence-level grad estimation
  sample_method: bernoulli         # gaussian | bernoulli | uniform
  en_layerwise_perturbation: true  # false = perturb all matched layers at once
  perturb_method: forward          # forward (one-sided) | antithetic
  perturb_granularity: token       # token | rollout
  grad_estimate_sample: grpo       # average | grpo   (over n_sample, per-token)
  grad_estimate_sequence: grpo     # average | grpo   (over n_rollout, reward R_t)
  perturb_rules:
    - '^model\.layers\.\d+$'        # all decoder layers (vLLM-real names)
  lr: 1.0e-4
  token_agg: sum                   # sum | mean
  update_clip: null                # δW / ‖δy‖² safety clamp (null = ε-floor only)

  # teacher scorer (v1, core)
  teacher_model_path: null         # second-engine teacher; required for opd loss
  log_prob_top_k: 256
  top_k_strategy: only_stu         # only_stu | only_tch | intersection | union | union-intersection
  teacher_temperature: 1.0
  reward_weight_mode: student_p    # student_p | teacher_p | none

  # engine / eval (mirror es_trainer.yaml)
  num_engines: 4
  num_iterations: 800
  precision: bfloat16
  max_tokens: 1024
  temperature: 0.0
  eval_interval: 25
  eval_batch_size: 256
  gpu_memory_utilization: 0.7
  global_seed: 42
  verbose: false
  worker_extension_cls: "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"

model:
  path: model/Qwen3-1.7B
  trust_remote_code: false

data:
  task_type: opd_math
  train_files: datasets/dapo-math-17k.parquet
  val_files: datasets/test_data/AIME24/test.parquet
  train_max_samples: 200
  val_max_samples: -1

trainer:
  project_name: OPD-NP
  experiment_name: np-run
  logger: [console, wandb]
  default_local_dir: /tmp/${oc.env:USER}/verl/np_checkpoints
  device: cuda
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: null
  test_freq: null
  save_freq: 100
```

### `perturb_rules` correction (load-bearing)

vLLM instantiates **fused** projections as single modules; the HF-style split names never appear in `named_modules()`. Regexes MUST target vLLM-real names:

| Intent              | HF name (WRONG for vLLM)        | vLLM-real name (use this)       |
|---------------------|---------------------------------|---------------------------------|
| QKV projection      | `q_proj` / `k_proj` / `v_proj`  | `self_attn.qkv_proj` (fused)    |
| attn output         | `o_proj`                        | `self_attn.o_proj`              |
| MLP gate+up         | `gate_proj` / `up_proj`         | `mlp.gate_up_proj` (fused)      |
| MLP down            | `down_proj`                     | `mlp.down_proj`                 |
| whole decoder layer | —                               | `^model\.layers\.\d+$`          |

Resolution rule: `active_modules = {m for (m_name, m) in model.named_modules() if any(re.fullmatch(r, m_name) for r in perturb_rules)}`. Hooking `qkv_proj` perturbs the joint QKV output (not separable per-head without slicing `output_size_per_partition`). Attention internals (paged-attention fused op) cannot be perturbed via this path — only the projections around it.

---

## 5. Launcher (`scripts/zo_opd/opd_np.sh`)

Mirror `opd_es.sh` env-var style. Exposes: `SIGMA`, `N_SAMPLE`, `N_ROLLOUT`, `SAMPLE_METHOD`, `PERTURB_GRANULARITY`, `GRAD_ESTIMATE_SAMPLE`, `GRAD_ESTIMATE_SEQUENCE`, `PERTURB_RULES` (newline-separated regex list), `EN_LAYERWISE_PERTURBATION`, `LR`, `TOKEN_AGG`, `ACTOR_MODEL_PATH`, `TEACHER_MODEL_PATH`, `STUDENT_GPUS`/`TEACHER_GPUS`, `LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `TEACHER_TEMPERATURE`, `N_GPUS_PER_NODE`, `NUM_ENGINES`, train/val datasets, logging/checkpoint vars. SBATCH header + local-run tee fallback like `opd_es.sh`. Invokes `python3 -m verl.trainer.main_np --config-name np_trainer ...` with `np.*` Hydra overrides.

**Attention backend (open implementation question, resolve at first-light step in the plan).** The `1+n_sample` shared-prefix multi-query decode was verified feasible on the **FlashAttention v1 backend** (`flash_attn_varlen_func` with `cu_seqlens_q`, `vllm/v1/attention/backends/flash_attn.py:516-548`). It was **not** verified on `TORCH_SDPA` (the backend `scripts/zo_opd/es.sh` currently sets). The decode driver depends on the variable-query-length + shared-prefix-KV + `slot_mapping=−1` path; whoever implements step 1 of the plan must confirm which backend exposes that path under `enforce_eager=True` and pin `VLLM_ATTENTION_BACKEND` accordingly (default assumption: FlashAttention v1, not SDPA). The σ=0 smoke test (Verification #1) is the gate for this.

---

## 6. Risks & mitigations

| Risk | Why it matters | Mitigation / verification |
|---|---|---|
| Custom decode driver diverges from a stock `engine.generate()` rollout | NP would train on a different trajectory than eval | At `σ=0`, the clean rollout (row 0) must match a stock greedy `generate()` byte-for-byte. |
| Perturbed-row KV leaks into the cache | Cache pollution → divergent / wrong continuation | Assert KV length grows by exactly 1 per step; assert perturbed rows have `slot_mapping = −1`. |
| `enforce_eager` didn't take / regex matched nothing | Silent no-op: NP "runs" but update is zero | At step 0, assert the perturbed forward actually widened to `1+n_sample` rows AND `‖δW‖ > 0` per active layer; fail loudly otherwise. |
| Teacher/student token misalignment in `L_t^(q)` | Off-by-one → wrong gradient sign/scale | Unit-test the teacher scorer on a fixed `(prefix, candidate)` pair against a hand-computed reverse-KL. |
| ANP normalization `δy/‖δy‖²` explodes at small `σ` | Numerical blow-up | `‖δy‖² ≥ ε` clamp + optional `update_clip`. |
| `(output, bias)` tuple repack | Returning a bare tensor where vLLM expects a tuple crashes downstream | Shim detects tuple-vs-tensor and repacks identically; unit-test on `qkv_proj` end-to-end. |
| vLLM 0.11.0 vendor lock | Upgrade may change spec-decode / `slot_mapping` / fused-naming contracts | Pin 0.11.0 (already pinned in `install_vllm_sglang_mcore.sh:12`); add `# vLLM internal` comments at every reach into `model_runner` internals. |
| `enable_prefix_caching` interaction | Prefix-sharing decode relies on shared KV blocks | Verify prefix caching is on and the `1+n_sample` sequences resolve to the same prefix blocks; assert single physical prefix copy. |

---

## 7. Verification ladder

1. **σ=0 smoke (1 GPU).** `np.sigma=0` → perturbation is a no-op; row-0 rollout must equal a stock greedy `generate()` token-for-token. Assert the perturbed forward widened to `1+n_sample` rows at step 0; assert KV grows by 1/step. If the rollout differs or width is 1, the driver or regex is broken.
2. **Gradient cosine-sim (offline, 1 GPU).** One layer (`model.layers.0.mlp.gate_up_proj`), one mini-batch: compute true `∂L/∂W` via an eager HF autograd backward; compute NP's `δŴ` over many perturbations. Expect `cos(δŴ, ∇_W L) ≥ 0.1` and converging; fail if negative/stuck at 0.
3. **End-to-end OPD run (small).** `LOSS=opd`, small dataset, watch teacher reverse-KL trend down over ~100 steps and `‖δW‖ > 0` per active layer. Loss noisier than PPO — expected.
4. **ES regression.** `scripts/zo_opd/es.sh` on the prior known-good config must produce identical curves — belt-and-suspenders that nothing in the shared paths was touched (NP lives entirely in new files).

---

## 8. Explicitly NOT in v1

- **Block-diagonal masked single-query-group** attention (we use prefix-sharing separate sequences instead). Revisit only if prefix-sharing proves slower than a custom-mask backend.
- **Token-tape via `attn_metadata` reconstruction.** Not needed — the prefix-sharing decode makes per-step `u_q` natural without recovering `(seq_id, position)` from forward context.
- **`enforce_eager=False` / cudagraph.** Impossible with per-step RNG noise; documented, not attempted.
- **Editing the installed `vllm` package.** All NP code is in the verl fork; vLLM internals are imported/reused, never patched in place.
- **Checkpoint hot-swap with the ES trainer.** Separate `default_local_dir`, separate trajectory.
