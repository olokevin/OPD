# NP Trainer — Node Perturbation for OPD

Zeroth-order on-policy distillation trainer that perturbs **linear-layer outputs** during a single clean rollout's decode and reads back per-token loss deltas from a teacher LLM. No backward pass. Lives in `verl.trainer.np` + `verl.workers.rollout.vllm_rollout.np_worker_extension`.

Reference branch: `feat/np-trainer`. Acceptance: 32 unit tests, σ=0 byte-equivalence smoke, cosine-similarity check against autograd (cos ≈ +0.41), 5-iter end-to-end smoke on Qwen3-1.7B student / Keven16 4B teacher.

---

## 1. One-line summary

For one clean student rollout, at every decode step `t` we evaluate **1 clean row + n_sample perturbed rows** of the same next-token query (`1+n_sample`-wide multi-query decode against the shared prompt KV with `slot_mapping=-1` on perturbed rows so they leave no KV trail). A teacher scores each row with reverse-KL. Per-sample loss deltas drive a per-token output-space gradient `g_t`, which is accumulated as the rank-1 outer product `g_t ⊗ x_t` into the perturbed layer's `δW`. Only the clean token is committed; perturbed rows are ephemeral.

```
prompt ──prefill──► shared KV ───decode step t────►┐
                                                    │
   clean row 0:        y_t = W x_t                  │
   perturbed row q:    y_t + σ·u_q   (q = 1..n)     ├── 1+n logits, one fwd
                                                    │
                       all rows query the SAME next position;
                       perturbed rows write KV to slot −1 (discarded)
                                                    │
                       teacher scores each row's distribution
                       L_t (clean) and L_t^(q) (perturbed)
                                                    │
                       g_t = (1/n) Σ_q s_q · u_q
                       δW += g_t · x_t^T
                                                    │
                       commit row-0 token; advance one step
```

---

## 2. File map

| Path                                                                    | Responsibility                                                                                                  |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `verl/trainer/np/seeding.py`                                          | `noise_seed(seed,t,layer,rollout,q) → 63-bit int`, `draw_noise(seed,shape,dev,dtype,method)` regenerator   |
| `verl/trainer/np/layer_resolve.py`                                    | `resolve_modules(rules, module_names)` (re.fullmatch over vLLM-fused names), `active_layers_for_step`       |
| `verl/trainer/np/grad_estimator.py`                                   | `sample_scale(L_q, L_clean, σ, mode)`, `accumulate_delta_w(δW, scales, u, x_t, normalize)`                |
| `verl/trainer/np/teacher_scorer.py`                                   | `reverse_kl_topk` kernel + `TeacherScorer` (vLLM engine wrapper with 5 `top_k_strategy` branches)         |
| `verl/trainer/np/ray_trainer.py`                                      | `NPNcclLLM` (enforce_eager + prefix caching), `RayNPTrainer` (per-step fit loop + per-layer NCCL broadcast) |
| `verl/workers/rollout/vllm_rollout/np_worker_extension.py`            | `PerturbedLinear` shim + `WorkerExtension` (decode driver, capture pass, apply, broadcast)                  |
| `verl/trainer/main_np.py`                                             | Hydra entry point                                                                                               |
| `verl/trainer/config/np_trainer.yaml`                                 | Default config                                                                                                  |
| `scripts/zo_opd/opd_np.sh`                                            | Launcher                                                                                                        |
| `scripts/zo_opd/np_checks/{check_decode_sigma0,check_grad_cosine}.py` | GPU-coupled verification gates                                                                                  |

Never edits ES paths: `git diff --stat main..feat/np-trainer -- verl/verl/trainer/es/ ... main_es.py ... dp_actor.py` is empty.

---

## 3. Math (as implemented)

### 3.1 Perturbation

For step `t`, the wrapped linear's clean output is `y_t = W x_t`, `W ∈ R^{d_out × d_in}`. Row-wise additive noise on rows 1..n_sample (row 0 is clean):

$$
y_t^{(q)} = W x_t + \sigma\, u_t^{(q)},\qquad q = 1, \dots, n_\text{sample}
$$

`u_t^{(q)} ∈ R^{d_out}` is regenerated from `noise_seed(global_seed, t, layer, rollout, q)`. Never stored; integer seeds are the only state that crosses the Ray boundary.

| `sample_method`       | Distribution                             |
| ----------------------- | ---------------------------------------- |
| `gaussian`            | `u ∼ N(0, I)`                         |
| `bernoulli` (default) | Rademacher,`u ∈ {-1, +1}^{d_out}` iid |
| `uniform`             | `u ∼ U(-1, 1)^{d_out}`                |

### 3.2 Teacher loss

Reverse-KL on the OPD top-k token set, minimization-oriented (lower = closer to teacher):

$$
L_t^{(q)} = \sum_{v \in V_\text{top-k}} w_v \bigl(\log p_\text{student}^{(q)}(v) - \log p_\text{teacher}(v)\bigr)
$$

Knobs:

- `log_prob_top_k` — k (default 256; teacher engine launches with `max_logprobs=max(20,k)`)
- `top_k_strategy ∈ {only_tch, only_stu, intersection, union, union-intersection}` — selects `V_top-k`
- `reward_weight_mode ∈ {student_p, teacher_p, none}` — selects `w_v`

Teacher logp for ids missing from teacher's surface is filled with `min(t_logp)` at that position (a calibrated lower bound; a naive `-1e30` exploded L_clean to ~1e25 on `only_stu`).

### 3.3 Per-sample scale (`grad_estimate_sample`)

| Mode               | Equation                                                                                         |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| `average`        | `s_q = (L_t^(q) - L_t) / σ` (one-sided forward difference vs clean baseline)                  |
| `grpo` (default) | `s_q = (L_t^(q) - μ) / (σ_L + 1e-8)`, group-relative z-score over the n_sample perturbations |

### 3.4 Per-token output gradient

Let `ũ_q = u_t^(q)` (default) or `ũ_q = u_t^(q) / max(‖u_t^(q)‖², ε)` when `normalize=True` (ANP — adaptive node perturbation):

$$
g_t = \frac{1}{n_\text{sample}} \sum_{q=1}^{n_\text{sample}} s_q\, \tilde u_q \in \mathbb{R}^{d_\text{out}}
$$

Approximates `∂L_t / ∂y_t` in expectation.

### 3.5 Per-layer weight delta

Accumulate the rank-1 outer product over all T response tokens:

$$
\delta W = \sum_{t=1}^{T} g_t\, x_t^{\!\top} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}
$$

`token_agg="mean"` divides by T; default `sum`.

### 3.6 Update rule

Since `δW ≈ +∂L/∂W` (validated by the cosine check below), the trainer does gradient **descent**:

$$
W \leftarrow W - \mathrm{lr}\cdot \delta W
$$

Optional `update_clip` clamps `δW` element-wise.

### 3.7 Full equation (default config: average, bernoulli, no ANP, sum)

$$
\delta W = \sum_{t=1}^{T} \Biggl[ \frac{1}{n_\text{sample}\, \sigma} \sum_{q=1}^{n_\text{sample}} (L_t^{(q)} - L_t)\, u_t^{(q)} \Biggr]\, x_t^{\!\top}
$$

with `u_t^(q) ∈ {-1, +1}^{d_out}` Rademacher, regenerated from `blake2b(global_seed, t, layer, rollout, q) & (2^63 − 1)`.

---

## 4. Knob → equation map

| YAML key                         | Symbol         | Default                                  | Affects                                                                 |
| -------------------------------- | -------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| `np.sigma`                     | σ             | 0.01                                     | Perturbation magnitude in `y + σ·u`                                 |
| `np.n_sample`                  | n              | 8                                        | Width of the multi-query step (1 clean + n perturbed)                   |
| `np.n_rollout`                 | —             | 8                                        | Independent rollouts; per-step signals concatenated before δW assembly |
| `np.sample_method`             | u distribution | bernoulli                                | `gaussian`/`bernoulli`/`uniform`                                  |
| `np.grad_estimate_sample`      | mode           | grpo                                     | `average` (one-sided FD) or `grpo` (group z-score)                  |
| `np.token_agg`                 | —             | sum                                      | δW Σₜ or mean over T                                                 |
| `np.lr`                        | lr             | 1e-4                                     | Descent step W ← W − lr·δW                                          |
| `np.update_clip`               | —             | null                                     | Element-wise clamp on δW                                               |
| `np.perturb_rules`             | —             | `^model\.layers\.\d+\.mlp\.down_proj$` | Regex set (vLLM-fused names) of linear modules to wrap                  |
| `np.en_layerwise_perturbation` | —             | true                                     | Round-robin one layer/step vs all-at-once                               |
| `np.log_prob_top_k`            | k              | 256                                      | OPD top-k set size                                                      |
| `np.top_k_strategy`            | —             | only_stu                                 | Which top-k to score over                                               |
| `np.reward_weight_mode`        | w_v            | student_p                                | Reverse-KL term weighting                                               |
| `np.teacher_temperature`       | T_tch          | 1.0                                      | Temperature on teacher logits before softmax                            |
| `np.teacher_model_path`        | —             | (required)                               | Path to teacher checkpoint                                              |

---

## 5. Acceptance gates

| Gate                   | Script                                                      | Result                                                                                                                                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| σ=0 byte-equivalence  | `scripts/zo_opd/np_checks/check_decode_sigma0.py`         | NP decode with σ=0 matches stock greedy `LLM.generate` token-for-token; logits width is `[1+n_sample, vocab]` every step. Validates the multi-query + `slot_mapping=-1` + shared-prefix machinery. **PASS** (Qwen3-1.7B, FLASH_ATTN backend). |
| Cosine-sim vs autograd | `scripts/zo_opd/np_checks/check_grad_cosine.py`           | See §6                                                                                                                                                                                                                                                    |
| 5-iter e2e smoke       | `scripts/zo_opd/opd_np.sh` (5 iters, 4 prompts)           | dW_norm ∈ [2.28, 2.58], L_clean ∈ [0.49, 0.76], no crash.                                                                                                                                                                                                |
| ES regression          | `git diff --stat main..HEAD -- verl/verl/trainer/es/ ...` | empty (zero ES files touched)                                                                                                                                                                                                                              |
| Unit tests             | `pytest verl/tests/np/`                                   | 32/32 PASS                                                                                                                                                                                                                                                 |

---

## 6. Cosine-similarity check (gradient validity)

**Question:** does the NP estimate `δW` point in the same direction as the true autograd gradient on a real Transformer linear?

**Setup** (`scripts/zo_opd/np_checks/check_grad_cosine.py`):

- Model: Qwen3-1.7B, loaded once via HuggingFace (`torch_dtype=fp32`, eager, no caching) for autograd, once again the same weights for the NP path. GPU 6 (single A800).
- Target layer: `model.layers.0.mlp.down_proj` (`d_out × d_in = 1536 × 8960` after Qwen3 MLP fan-in; the canonical NP target).
- Prompt: `"Compute 7*8. Answer:"`; loss = cross-entropy on the last token's argmax target (a self-consistent label, so the gradient is well-defined without ground truth).
- Reference: register a forward hook to capture `x` at the layer's input, set `W.requires_grad=True`, run one forward + backward, read `g_true = W.grad`, also record `x_t` for the last response position.
- NP estimate (forward-only):
  - For `repeats` outer iterations: draw `n_sample` Rademacher Gaussian noise rows `u_q ∈ R^{d_out}` via `draw_noise(noise_seed(0, rep, layer, 0, q), ...)`.
  - For each `q`, install a forward hook adding `σ·u_q` to the layer's **output**, recompute the same loss, record `L_q`.
  - Compute `scales = (L_q − base) / σ` (one-sided forward difference vs the unperturbed `base` loss).
  - Accumulate `dw += outer( mean_q(scales_q · u_q), x_t )`.
  - After all `repeats`, divide `dw` by `repeats`.
- Metric: `cos(dw.flatten(), g_true.flatten())`. **PASS threshold**: > 0.05; expected ≥ 0.10 with enough samples.

**Configuration used to obtain the headline number:**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n verl python scripts/zo_opd/np_checks/check_grad_cosine.py \
    --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc... \
    --layer 'model.layers.0.mlp.down_proj' \
    --n-sample 64 --repeats 50 --sigma 1e-3
```

Total forward queries per estimate: `repeats × (1 + n_sample) = 50 × 65 = 3250` (base loss + 64 perturbed losses, 50 times).

**Result:**

```
cosine(NP_dW, true_grad) = 0.4074  (n_sample=64, repeats=50)
PASS
```

8× above the gate. This validates:

- The math chain `sample_scale → accumulate_delta_w → outer(g_t, x_t)` recovers the right **direction** of the gradient (not just non-zero).
- Bernoulli/Gaussian noise both work (we tested gaussian here for the FD math to match the cosine threshold; the trainer defaults to bernoulli for variance reasons).
- The sign convention used downstream: `dw` aligns with `+∂L/∂W`, so the trainer must **subtract** (`W ← W − lr · dw`), not add. The original `apply_node_update` did `W += lr · dw` (gradient ascent) — caught by code review, fixed in `apply_node_update`, asserted by `test_apply_node_update_does_gradient_descent_sign` in the CPU unit suite.

**Variance considerations.** Cosine grows with `√(n_sample × repeats)` for true forward-difference NP estimators. At 64 × 50 we get ≈ 0.41; halving either knob would drop cosine toward 0.10–0.20. The trainer uses fewer samples per step (`n_sample=8`, `n_rollout=8`) but accumulates over T response tokens, so the effective sample count per layer per training iteration is `n_sample × n_rollout × T ≈ 8 × 8 × 1024 ≈ 6.5×10⁴` — comfortably enough for SGD to make progress. The cosine-vs-N sweep in [§9.4](#94-initial-result-b--gradient-cosine-similarity-vs-n-gradient-quality) confirms this (N=8/16/32/64 → cos 0.205/0.276/0.356/0.407 at repeats=50).

---

## 7. Pointers

- Spec: `docs/superpowers/specs/2026-05-28-np-trainer-design.md`
- Plan: `docs/superpowers/plans/2026-05-28-np-trainer.md` (16 tasks, all complete on `feat/np-trainer`)
- ES sibling: `docs/wiki/ZO.md` §2
- Paper: `arXiv:2604.13016` (token-level OPD theory)

# V1 implementation, why it's slow, and the V2 plan (one wide forward per token)

> **TL;DR.** V1 works and trains cleanly, but the student rollout is the bottleneck:
> per decode token it runs a `(1+N)`-row **eager** forward (no CUDA graph), and it runs
> that **once per prompt, serially** across the batch. Profiling shows **~99% of the time
> is this forward**. V2 must (a) CUDA-graph the step forward and (b) batch **all prompts'
> `(1+N)` rows into ONE wide forward per token** instead of `batch_size` sequential ones.

### 8.1 What V1 actually does (per training step)

For each active layer, for each of `batch_size` prompts **serially**:

1. `run_np_decode` (one Ray RPC per *sequence*) hand-drives the decode token-by-token:
   each token = a `(1+N)`-row forward (1 clean + `n_sample` perturbed), all rows querying
   the same next position against the shared committed KV, perturbed rows write `slot=-1`.
   Logits + `captured_u` + `captured_x` are saved per step; only the clean token advances.
2. ONE teacher prefill over `prompt+clean_tokens` → per-position top-k logprobs (§3.2).
3. `assemble_and_apply` builds δW from the per-token `(L_q, L_clean, u, x)` and applies it.

Two structural facts make V1 correct but slow:

- The perturbation is injected by an **eager Python forward hook** on `PerturbedLinear`
  (`enforce_eager=True` is mandatory, so CUDA graphs are off for the whole forward).
- The `(1+N)` rows are **one sequence with N throwaway twins** sharing the parent's KV —
  not expressible to vLLM's scheduler, so the decode loop + `attn_metadata` are hand-built
  (`np_worker_extension._np_step_forward` / `_np_build_attn_metadata`), bypassing the scheduler.

### 8.2 V1.1 fix already shipped (commit `21534aa`, branch `np-fold-xcapture`)

`x_t` is now captured **inside the same perturbed forward** (`PerturbedLinear` records the
clean row's input in `perturb` mode), removing a redundant **second full-sequence re-decode**
(`run_capture_pass`/`_capture_x`, now deleted). Student decode passes/seq: **2 → 1**.
Validated end-to-end: update lands (chg≈16%, sync_ok=1), held-out KL 0.284→0.264.
Estimator also switched to `(L_q−mean)/σ` (dropped the `1/std` that self-amplified to
divergence; see `docs/results/zo_opd.md` §5–6).

### 8.3 Why it's still slow — the profile (2026-06-03)

Per-token timing inside `_np_step_forward` (`torch.cuda.synchronize` around each phase):

```
meta (attn_metadata build) = 0.31 ms/tok   (~1%)
fwd  (the 1+N=65-row forward) = 13–20 ms/tok (~99%)
```

So the cost is the **eager 65-row forward itself**, run ~1024×/sequence × `batch_size`
sequences. It is **not** the metadata build, **not** per-token Ray RPC (one RPC per
sequence), **not** OMP thread oversubscription (thread caps measured ~0% gain).
Two earlier hypotheses — caching `attn_metadata`, capping OMP threads — were measured and
**abandoned** (≤2% each). Reference: full default config (BATCH 64 × 1024 tok + 200-prompt
eval) ≈ **72 min/step**; that is dominated by this forward.

### 8.4 V2 = CUDA-graphed, ONE wide forward per token across all prompts (required)

The forward is 99% of the cost, so V2 attacks the forward directly on two axes:

1. **Batch across prompts.** Today each step does `batch_size` sequential
   `(1+N)`-row forwards. V2 packs **all active prompts** into one forward per token:
   rows = `Σ_p (1 + N)` (or a shared clean + per-prompt perturbations), each prompt
   attending its own shared-prefix KV. One wide launch replaces `batch_size` tiny ones —
   far better GPU utilization at the same total FLOPs.
2. **CUDA-graph the step forward.** Remove the per-token eager dispatch overhead by
   capturing the step forward as a CUDA graph. The blocker is the eager perturbation hook
   (`PerturbedLinear.forward` adds `σ·u` in Python) — V2 must express the perturbation as a
   **graph-capturable op**: e.g. write the regenerated `u` (or its seeds→noise) into a
   fixed pre-allocated buffer that the captured graph reads, with the rest of the forward
   static (fixed row count, fixed positions slot, ragged handled via a fixed max width + mask).

**Open design questions for the V2 session (start here):**

- **Ragged batching:** prompts have different lengths and finish at different steps. Fixed
  max-width graph + padding/masking, or bucket by length? How to retire finished prompts
  without recapturing the graph.
- **Graph capture vs. the hook:** what exactly is static (row layout, slot_mapping pattern,
  block tables) vs. per-token dynamic (the `u` buffer, `q_pos`/seq_len, input ids). Likely a
  single graph parameterized by input buffers updated in-place each token.
- **KV/slot bookkeeping for the packed batch:** per-prompt `block_ids`, clean-row slot per
  prompt, `-1` pads for all perturbed rows, built once and advanced cheaply.
- **Stop handling:** EOS per prompt with a fixed-shape graph (mask + early-drop vs. run to
  max and trim).
- **Memory:** `Σ_p (1+N)` rows × hidden must fit alongside both vLLM engines on one GPU.

**Invariants V2 must preserve (don't regress):**

- σ=0 byte-equivalence vs stock greedy decode (§5 gate).
- cos(NP δW, autograd) ≈ +0.41 at N=64 (§6).
- Per-token `(L_q, L_clean, u, x)` semantics and the `(L_q−mean)/σ` estimator unchanged —
  V2 is a **performance rewrite of the decode driver only**, not a math change.

**Entry points for V2 work:** `np_worker_extension.py` →
`run_np_decode` / `_np_step_forward` / `_np_build_attn_metadata` (the decode driver);
`ray_trainer.py` `fit()` lines ~513–542 (the serial per-prompt loop to be replaced by one
packed call). Keep `PerturbedLinear` semantics; change *how* the rows are assembled and run.

### 8.5 V2 landed — buffer-in-graph

The V2 design settled on **Path B (buffer-in-graph)**, NOT the prompt-packing plan §8.4 sketched.
The full design + GPU-validated initial results are in **[§9](#9-v2--buffer-in-graph-decode-driver-design--initial-results)**.

---

## 9. V2 — buffer-in-graph decode driver (design + initial results)

> **One line.** V2 turns V1's eager, hand-driven `(1+N)`-row decode into a **CUDA-graphed** step:
> the model forward (incl. the perturbation) is captured once and **replayed** per token, with fresh
> noise entering through a **host-refilled buffer** (`u_buf`) instead of in-forward RNG. The N
> perturbed rails ride the *same* captured graph / *same* weights / *same* shared-prefix KV as the
> clean rail. **No NP math changes** — V1 stays callable as the byte-for-byte parity oracle.
> Branch `np-v2-cudagraph-rails`; spec
> [`docs/superpowers/specs/2026-06-03-np-v2-cudagraph-rails.md`](../superpowers/specs/2026-06-03-np-v2-cudagraph-rails.md).

### 9.1 Why V1 was slow, and what V2 changes

V1 forced `enforce_eager=True` for one reason: `PerturbedLinear.forward` draws the noise *in-forward*
(`noise_seed`+`draw_noise` per sample per token), and `torch.Generator().manual_seed()` cannot be
captured into a CUDA graph. Profiling (§8.3) showed **~99% of step time is the eager `(1+N)`-row
forward** — eager dispatch + the Python perturbation loop, not metadata/RPC/threads. V2 removes both:
the noise draw moves to the host, the perturbation becomes a fixed-shape in-graph op, and the whole
step forward is captured.

### 9.2 The mechanism (Path B)

| Where | What runs | Why graph-safe |
|---|---|---|
| **Host, before each `replay()`** (`_np_fill_u_buf`) | `u_buf[q].copy_(draw_noise(noise_seed(global_seed, t, layer, rollout, q)))` — the **only** RNG | Same `seeding.py` call as V1 → **u bit-identical**; just relocated earlier in wall-clock |
| **Inside the graph** (`PerturbedLinear.perturb_graph` mode) | `y[1:1+N] += σ·u_buf` (fixed shape, no RNG, no alloc) + `x_buf.copy_(x[0])` | Elementwise op on a persistent buffer the host refilled — a graph input, like `input_ids`/`slot_mapping` |
| **Host, after `replay()`** | `compute_logits(hidden_buf)` + row-0 sampling | Sampling is data-dependent (decides the next token) → must stay eager; mirrors vLLM's own model-graphed/sampler-eager split |

The `(1+N)` row layout, `slot_mapping=[clean, −1×N]` (perturbed rows write no KV), and shared-prefix
KV are **unchanged from V1** — only *how the rows are run* changed (eager → captured). Config:
`np.decode_mode ∈ {eager, graphed}` (default `eager` = V1), `np.use_cuda_graph`
(`false` = M1 eager-with-`u_buf`; `true` = M2 captured graph). `fit()` dispatches
`run_np_decode_graphed` when `decode_mode=graphed`; the held-out KL probe stays on the eager path.

**vLLM-0.11.0 facts the M2 capture relies on (verified twice against source):**
`FlashAttentionMetadataBuilder.build()` stores `slot_mapping`/`seq_lens`/`block_table`/`query_start_loc`
**by reference** (mutate-in-place → the captured kernel reads new values on replay);
`set_forward_context` stashes `attn_metadata` in a Python global read at forward time (the graph
re-reads the same tensor storage); `compute_logits` stays outside the graph; `reshape_and_cache`
skips `slot<0`. **The one load-bearing subtlety:** `max_seqlen_k` is a frozen Python int (it sizes the
FA kernel grid and cannot be a graph input), so it is captured at the **cap** `prompt_len+max_tokens`
while the *live* per-token `seqused_k` is the mutated `seq_lens_gpu` tensor — exactly how vLLM captures
its own decode graphs (`gpu_model_runner.py:3057`). Freezing it at token-0's length silently truncates
attention to the prompt (caught by adversarial review pre-GPU; fixed via `max_seq_len_override`).

### 9.3 Initial result A — throughput scaling with N (the "N is ~free" premise)

`bench_n_scaling.py`, `graphed_cuda`, Qwen3-1.7B, `model.layers.0.mlp.down_proj`, max_tokens=64, GPU 7:

| N (perturbed rails) | s/step | ms/tok | rel(N=1) |
|---:|---:|---:|---:|
| 1 | 0.658 | 10.28 | 1.00× |
| **8** | 0.785 | 12.27 | **1.19×** |
| 16 | 0.870 | 13.60 | 1.32× |
| 32 | 1.086 | 16.97 | 1.65× |
| 64 | 1.514 | 23.66 | 2.30× |

**The premise holds in the low-N regime.** 8 rails cost **+19%** wall-time for 8× the perturbation
FLOPs — the memory-bound free lunch (the rails ride resident weights+KV, adding compute on data already
loaded). It is *not* flat to N=64: by 64 the `(1+N)` GEMM has crossed into the compute-bound regime
(2.3×). **Practical operating point N≈8–16** — which is the `n_sample=8` default.

### 9.4 Initial result B — gradient cosine-similarity vs N (gradient quality)

`check_grad_cosine.py` (offline HF-autograd reference vs the NP δW math chain), Qwen3-1.7B,
`model.layers.0.mlp.down_proj`, repeats=50, σ=1e-3, GPU 6. This is the **§6 validity check**, run
across N to expose the quality/N trade-off:

| N | cos(NP δW, autograd) |
|---:|---:|
| 8 | 0.205 |
| 16 | 0.276 |
| 32 | 0.356 |
| 64 | **0.407** |

Cosine rises with N as a forward-difference NP estimator should (more samples → better direction
estimate; ≈√(N·repeats) in the ideal limit, sub-√N here at fixed repeats=50). **Read 9.3 and 9.4
together:** more N buys a straighter gradient but costs more compute; the knee where cosine gain slows
(~32) sits just past where throughput leaves the free regime (~16). Note this is a *single token's*
gradient at high repeats; the trainer accumulates over `n_sample × n_rollout × T ≈ 6.5×10⁴` effective
samples per layer per iteration (§6), so the per-token cosine is a lower bound on training-signal
quality, not the operating SNR.

**Why this number transfers to V2 unchanged:** `check_grad_cosine.py` is **decode-driver-independent**
— it uses HF forward hooks, not the NP worker decode — so it validates the `sample_scale →
accumulate_delta_w → outer(g_t, x_t)` math, which V2 does not touch. The V2 parity gates (§9.5) prove
the graphed driver feeds that math **bit-identical `u`** and matching `x`, so the assembled δW direction
is identical to V1's → cos = 0.407 is V2's number too, by construction.

### 9.5 Verification — all gates PASS (Qwen3-1.7B, GPU 6/7, FLASH_ATTN, 2026-06-03)

| Gate | Script | Result |
|---|---|---|
| CPU unit suite (38, incl. 6 new) | `pytest verl/tests/np/` | **38/38** (`test_perturb_graph.py`: row math, `x_buf` capture, σ=0 no-op, `_np_fill_u_buf` bit-identical-to-V1) |
| σ=0 byte-equiv, eager+`u_buf` | `check_decode_sigma0.py --driver graphed_eager` | **PASS** — matches greedy, width 1+N |
| σ=0 byte-equiv, graphed (= **M0 capture spike**) | `check_decode_sigma0.py --driver graphed_cuda` | **PASS** — `torch.cuda.CUDAGraph` capture of hand-built `attn_metadata` works; the one unverified-on-GPU risk is cleared |
| Parity M1 (V1 vs eager+`u_buf`) | `check_graphed_parity.py --stage m1` | **PASS** — u bit-identical, logits/x within rtol=1e-2 (noise relocation byte-correct) |
| Parity M2 (eager+`u_buf` vs graphed) | `check_graphed_parity.py --stage m2` | **PASS** — same → by transitivity **graphed ≡ V1** |
| Cosine validity | `check_grad_cosine.py` | **0.407** @ N=64 (§9.4) |

**Two bugs surfaced + fixed on GPU** (commit `01eb6bc`), neither a math/invariant bug: (a) sharing one
`graph_pool_handle` across captures tripped `CUDACachingAllocator use_count>0` once >1 graph is alive
(only the N-sweep exposes it; single-decode gates capture once) → each graph now owns its pool and the
prior graph is released before the next capture; (b) a device mismatch in the parity *check script* (V1
returns `captured_u` on GPU, the graphed driver on CPU).

### 9.6 Status & what's next

Numerical correctness (graphed ≡ V1) and the throughput/quality trade-off are established on a real
model. **Not yet run:** a multi-iter end-to-end OPD training run on the graphed driver (decode → score →
assemble → apply → broadcast) to confirm the held-out KL trends down as it does on V1 — the natural next
gate. The graph is captured **per prompt** (block_ids are prompt-specific); caching graphs across
same-length prompts is a possible future optimization but is out of scope for the landed driver.
