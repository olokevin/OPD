# NP-HF Trainer — Node-Perturbation OPD on plain HuggingFace forwards

> A standalone, single-GPU reimplementation of the Node-Perturbation OPD trainer that replaces all vLLM worker-extension machinery with plain HuggingFace `model(...)` forwards. Scope: **on-policy distillation (OPD) only** — student rolls out, teacher prefills once, per-token reverse-KL drives a zeroth-order, backward-free weight update. Reuses the validated NP math (`seeding`, `grad_estimator`, `reverse_kl_topk`) verbatim.

- **Date:** 2026-06-03
- **Status:** approved design, pre-implementation
- **Predecessor:** `docs/superpowers/specs/2026-05-28-np-trainer-design.md` (vLLM-coupled NP trainer)
- **Wiki:** `docs/wiki/zo_np_trainer.md` (the vLLM design this ports)
- **Paper:** arXiv:2604.13016 (token-level OPD theory)

---

## 1. Motivation

The existing NP trainer (`verl/verl/trainer/np`, `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`) is correct and validated (32 unit tests, σ=0 byte-equivalence, cosine ≈ +0.41 vs autograd) but is **deeply coupled to vLLM internals**: it hand-builds `attn_metadata`, abuses `slot_mapping=-1` so perturbed rows leave no KV trail, steals scratch KV blocks from the top of the GPU pool, runs forwards under `set_forward_context`, and synchronizes weights across engines via NCCL/Ray.

That machinery is the cost of riding a production inference kernel. For research iteration on the perturbation algorithm itself, we want a **plain-HuggingFace** path that is:

- runnable anywhere with just `torch` + `transformers` (no Ray, no vLLM, no verl package tree),
- single-GPU / single-process,
- structurally a 1:1 reimplementation of the *algorithm*, differing only in the forward engine.

**The "free-lunch" hypothesis (to be tested, not assumed):** decode is memory-bound and compute-underutilized, so widening each decode step from 1 row to `1+N` rows against a shared KV cache should add compute that is nearly free. This design includes a benchmark whose explicit job is to confirm *or falsify* that hypothesis for this setup.

### Out of scope (explicit)

- **SFT.** SFT is not generation; the perturbed-rollout machinery does not apply. Dropped.
- Multi-GPU, Ray, NCCL, distributed anything, checkpoint resharding.
- Models that don't fit (with teacher) on a single GPU.
- vLLM. This is the point of the rewrite.

---

## 2. Algorithm (unchanged from the vLLM design)

For one clean student rollout, at every decode step `t` we evaluate **1 clean row + n_sample perturbed rows** of the same next-token query against the shared committed-prefix KV. Perturbed rows add independent noise to matched linear-layer *outputs*; their KV is discarded. After the full rollout, a teacher scores every row with reverse-KL. Per-sample loss deltas drive a per-token output-space gradient `g_t`, accumulated as the rank-1 outer product `g_t ⊗ x_t` into each perturbed layer's `δW`. Only the clean token is committed.

```
prompt ──prefill──► clean KV (batch 1) ───decode step t────►┐
                                                            │
   clean row 0:        y_t = W x_t                          │
   perturbed row q:    y_t + σ·u_q   (q = 1..N)             ├── 1+N logits, ONE fwd
                                                            │   (rows 1..N read shared KV,
                       all rows query the SAME next pos;    │    write only to a throwaway cache)
                       only row 0's KV is kept              │
                                                            │
   ... after full rollout ...                               │
                       teacher prefills committed seq ONCE  │
                       L_t (clean) and L_t^(q) per step     │
                                                            │
                       g_t = (1/N) Σ_q s_q · ũ_q            │
                       δW_layer += g_t · x_t^T              │
                       W ← W − lr·δW                        │
```

The math is the validated chain from the wiki: `sample_scale` (`average` | `grpo`) → `accumulate_delta_w` (optional ANP normalize) → rank-1 outer product → `W ← W − lr·δW`. See `docs/wiki/zo_np_trainer.md` §3 for the equations; they are reused without modification.

### Key deviation from the vLLM design: all-matched-layers-by-default

The vLLM trainer perturbs **one** layer per step (round-robin, `en_layerwise_perturbation`). This port makes layer scope **configurable but defaults to perturbing ALL matched layers simultaneously** in the same forward, accumulating one `δW` per layer from the same rollout.

Soundness: each layer's noise `u^layer` is drawn from an independent seed (`noise_seed` is keyed by `layer_name`) and is zero-mean, so cross-layer terms vanish in expectation and each layer's `δW` remains an unbiased estimate of `∂L/∂y^layer`. The cost is **higher variance** — every other perturbed layer's noise is extra variance in any one layer's estimate. This default therefore requires its own correctness gate (§6, gate 3, run in all-layers mode), not just the inherited single-layer cosine result.

---

## 3. Module layout

New standalone package at **`src/np_hf/`**, zero verl/Ray/vLLM imports. The validated math is **copied in** (not imported across `verl/`) so the package has no dependency on the verl tree; a drift-guard test keeps the copies honest.

```
src/np_hf/
  seeding.py          # COPIED verbatim from verl/trainer/np/seeding.py (noise_seed, draw_noise)
  grad_estimator.py   # COPIED verbatim (sample_scale, accumulate_delta_w)
  reverse_kl.py       # reverse_kl_topk kernel, lifted from teacher_scorer.py (pure fn only)

  perturb.py          # PerturbLinearHook: forward-hook factory adding σ·u to output rows 1..N,
                      #   capturing clean-row x_t per layer. Keyed by worker-local perturb state.
  rollout.py          # RolloutEngine (Approach A): batched 1+N HF decode, persistent clean KV,
                      #   one-forward-per-step with row-0 KV slicing.
  rollout_oracle.py   # RolloutEngineOracle (Approach B): two-forward / full-reprefill reference
                      #   used ONLY by tests as a correctness oracle.
  teacher.py          # TeacherScorer: HF teacher prefill -> per-token top-k logp -> reverse-KL,
                      #   honoring top_k_strategy + reward_weight_mode (port of teacher_scorer.py).
  estimator.py        # assemble_layer_delta + apply_update (port of assemble_and_apply, no Ray).
  trainer.py          # NpHfTrainer: per-iter loop (rollout -> teacher -> per-layer δW -> W -= lr·δW).
  config.py           # dataclass mirroring the vLLM np.* knobs.
  main.py             # CLI entry: load student+teacher, dataset parquet, run loop.
  bench.py            # ms/step vs N sweep harness (the free-lunch test).

tests/np_hf/
  test_sigma0_equiv.py   # σ=0: A == stock model.generate(do_sample=False), token-for-token.
  test_oracle_equiv.py   # A == B (oracle) on per-step logits, small N, σ>0, fixed seed.
  test_grad_cosine.py    # δW cosine vs autograd on one layer; run single-layer AND all-layers.
  test_math_reuse.py     # copied math files byte-match verl originals (drift guard).
```

**Boundary:** `perturb.py` + `rollout.py` are the only genuinely new mechanism. `seeding`/`grad_estimator`/`reverse_kl`/`estimator` are the validated math, unchanged.

**Copy-vs-import:** copying + `test_math_reuse.py` keeps the package standalone (the chosen constraint) without re-coupling to the verl package tree. The one accepted trade is manual sync on the rare occasions the verl math changes; the drift-guard test makes drift loud.

---

## 4. Data flow (one OPD training iteration)

```
┌─────────────── STUDENT ROLLOUT (Approach A — src/np_hf/rollout.py) ───────────────┐
prompt_token_ids ──prefill──► clean DynamicCache (batch 1)                           │
                                                                                     │
for t in range(max_tokens):                                                          │
    expand clean cache → batch (1+N) VIEW            # shallow; .contiguous() only    │
                                                     #   if attn impl demands         │
    hooks (mode=perturb): rows 1..N of EVERY matched layer's output += σ·u_{layer,q} │
        u = draw_noise(noise_seed(seed, t, layer_name, rollout_idx, q))   # not stored│
        hook captures x_t (clean row 0 input) per layer                              │
    logits[1+N, vocab] = model(last_token×(1+N), past=expanded_throwaway)  # 1 fwd    │
    save candidate_logits[t]=logits.cpu(), captured_u[t][layer], captured_x[t][layer]│
    next_tok = sample(logits[0])                     # row 0 = clean                  │
    slice row-0 new K/V out of throwaway cache → append to persistent clean cache     │
    commit next_tok; stop on EOS                                                      │
                                                                                     │
returns clean_tokens, candidate_logits (T×[1+N,vocab]), captured_x, captured_u ───────┘
                                  │
                                  ▼
┌──────────── TEACHER SCORING (src/np_hf/teacher.py — ONCE, after full rollout) ──────┐
│ prefix = prompt + clean_tokens                                                      │
│ teacher forward over prefix → per-response-position top-k logprobs (ids + logp)     │
│ per step t, per row q ∈ {clean} ∪ {1..N}:                                           │
│     select id set per top_k_strategy; align teacher logp (missing → min fallback)   │
│     L_t^(q) = reverse_kl_topk(student_logp_row_q, teacher_logp, weight_mode)        │
│ returns L_q_per_step [T×N], L_clean_per_step [T]                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────── UPDATE (src/np_hf/estimator.py) — per matched layer ────────────────────┐
│ scales_t = sample_scale(L_q_t, L_clean_t, σ, mode)         # average | grpo         │
│ g_t      = (1/N) Σ_q scales_{t,q} · ũ_{t,q}                # ũ = u or u/‖u‖² (ANP)   │
│ δW_layer = Σ_t  g_t ⊗ x_t^layer                            # rank-1 accumulation     │
│ W_layer ← W_layer − lr · δW_layer                          # in-place, no autograd   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Notes:**

1. **Per-layer captures.** With all-layers default, `captured_u` and `captured_x` are keyed `[t][layer_name]`; one `δW` per layer is assembled from the same rollout. The reverse-KL loss `L_t^(q)` is a single per-(t,q) scalar shared across layers — it measures the *joint* output shift from all layers' row-q perturbations. The estimator attributes it back per-layer (unbiased in expectation; higher variance — §2 deviation).

2. **No teacher in the generation loop** — logits are stashed during rollout and the teacher runs exactly once on the committed sequence. (Hard requirement: never generate-one-token → teacher → loss → next-token; that serialization is far slower.)

3. **Single GPU, single process.** No NCCL broadcast, no Ray. Student and teacher are two HF models. If both don't co-reside, `teacher_offload` (config) brings the teacher onto the GPU only for its single prefill.

4. **`n_rollout` rollouts per iteration.** The diagram shows one rollout for clarity; an iteration actually runs `n_rollout` independent rollouts (different prompts and/or sampling seeds). Their per-step signals (`L_q`, `u`, `x_t`) are concatenated along the token axis before δW assembly — i.e. δW is summed over all `n_rollout × T` token positions per layer, exactly as the vLLM trainer accumulates. This is the variance budget that makes the noisy all-layers default trainable (effective samples per layer per iter ≈ `n_sample × n_rollout × T`).

---

## 5. Two load-bearing mechanics, pinned

### 5a. Perturbation injection (`perturb.py`)

A forward hook on each matched `nn.Linear`. On a 1+N batched decode step the layer output is `[1+N, 1, d_out]`. The hook:
- adds `σ·u_{layer,q}` to output rows `1..N` (row 0 untouched), `u` regenerated from `noise_seed(global_seed, t, layer_name, rollout_idx, q)`,
- captures `x_t = ` clean row-0 input (detached) for the rank-1 update,
- stores the stacked `u` for the layer so the estimator reuses identical noise.

Hook behavior is keyed by an engine-held perturb-state dict (mode `off`/`perturb`, current `t`, seeds, σ, N), mirroring the vLLM `np_state` pattern so the same model object serves clean and perturbed forwards without re-installing hooks.

### 5b. KV cache: expand to 1+N, keep only row 0 (`rollout.py`)

HF `model(..., past_key_values=cache, use_cache=True)` appends the new position for **all** batch rows, so the persistent clean cache cannot be passed directly to a 1+N forward.

**Chosen mechanic — one forward per step, row-0 KV slice:**
- Persistent `clean_cache`: batch-1 KV for `[prompt + committed]`.
- Per step, build a **throwaway expanded cache**: each layer K/V `[1,H,L,D]` → `expand(1+N,H,L,D)` view, wrapped in a fresh `DynamicCache` whose `.update()` writes land in throwaway storage.
- Run the single 1+N forward → logits + captures. 
- **Slice row 0's newly-appended K/V** out of the throwaway cache and append to the persistent `clean_cache`. Discard the throwaway cache.
- This keeps exactly **one forward per decode step** (the true free-lunch path). The KV-slicing is the version-sensitive part, guarded hard by the σ=0 byte-equivalence test (§6 gate 1).

**Approach B (oracle, `rollout_oracle.py`)** exists only for tests: it re-prefills the full prefix (or does a clean batch-1 forward + a separate discarded N-row forward), trading speed for obviously-correct cache semantics. A == B is gate 2.

---

## 6. Correctness gates & error handling

No autograd safety net and version-sensitive cache code → correctness is enforced by gates, not `try/except`.

**Hard gates (block trust in the trainer):**

| # | Gate | Script | Pass condition |
|---|---|---|---|
| 1 | σ=0 byte-equivalence | `test_sigma0_equiv.py` | With σ=0, A's `clean_tokens` == stock `model.generate(do_sample=False)` token-for-token; `candidate_logits[t]` is `[1+N,vocab]` with all rows == row 0. Validates expand + row-0-slice + hook plumbing in one shot. **Load-bearing.** |
| 2 | A == B oracle | `test_oracle_equiv.py` | Fixed seed, σ>0: A's per-step `[1+N,vocab]` logits match oracle B within fp tolerance. Confirms one-forward KV-slice == re-prefill. |
| 3 | Gradient cosine | `test_grad_cosine.py` | `cos(δW, autograd ∂L/∂W) > 0.05` on one layer. Run **both** single-layer and all-layers-on; the all-layers run validates the new default's variance is survivable at default `n_sample`. Low cosine in all-layers mode is a reported finding, not a silent pass. |
| 4 | Math drift guard | `test_math_reuse.py` | Copied `seeding.py`/`grad_estimator.py` byte-match verl originals. |

**Runtime checks (cheap asserts):**
- `‖δW‖ > 0` per applied layer per iter (update landed).
- teacher prefix length == prompt + committed length (alignment).
- `teacher_offload` config controls teacher GPU residency for its single prefill; no automatic OOM retry.

**Deliberately not handled** (simplicity): multi-GPU, Ray, NCCL, distributed, resharding, models that don't fit on one GPU.

---

## 7. Benchmark — the free-lunch test (`bench.py`)

Measure **ms per decode step** (and tokens/sec) for Approach A's rollout, sweeping `N ∈ {0,1,4,8,16,32}` at fixed prompt + response length, one GPU, single model (Qwen3-1.7B). `N=0` = clean-only baseline (no hooks). Report:

- ms/step vs N curve,
- overhead ratio `t(N)/t(0)` per N,
- the **knee**: smallest N where ms/step departs from flat by >10% (memory-bound → compute-bound),
- GPU memory vs N (near-flat expected; a jump flags an accidental KV copy in the expand).

Methodology: discard warm-up iters, `torch.cuda.synchronize()` around timed regions, median over ≥20 steps.

**"Free lunch confirmed"** ≡ `t(N)/t(0) ≲ 1.2` for N up to at least 8 (the trainer default). If `t(N)/t(0) > 1.5` already at N=4, the hypothesis is **falsified for this setup** and that is the reported result — no massaging.

---

## 8. Config knobs (`config.py`)

Mirror the vLLM `np.*` schema where meaningful; drop the vLLM/Ray-only knobs.

| Key | Symbol | Default | Affects |
|---|---|---|---|
| `sigma` | σ | 0.01 | perturbation magnitude `y + σ·u` |
| `n_sample` | N | 8 | perturbed rows per step (width `1+N`) |
| `n_rollout` | — | 8 | independent rollouts; signals concatenated before δW |
| `sample_method` | u dist | bernoulli | `gaussian`/`bernoulli`/`uniform` |
| `grad_estimate_sample` | mode | grpo | `average` (one-sided FD) or `grpo` (mean-centered /σ) |
| `token_agg` | — | sum | δW Σₜ or mean over T |
| `lr` | lr | 1e-4 | `W ← W − lr·δW` |
| `update_clip` | — | null | element-wise clamp on δW |
| `perturb_rules` | — | `^model\.layers\.\d+\.mlp\.down_proj$` | regex set of HF module names to wrap |
| `en_layerwise_perturbation` | — | **false** | **false ⇒ all matched layers at once (new default); true ⇒ single-layer round-robin** |
| `log_prob_top_k` | k | 256 | OPD top-k set size |
| `top_k_strategy` | — | only_stu | which top-k to score over |
| `reward_weight_mode` | w_v | student_p | reverse-KL term weighting |
| `teacher_temperature` | T_tch | 1.0 | temperature on teacher logits |
| `teacher_model_path` | — | (required) | teacher checkpoint |
| `teacher_offload` | — | false | load teacher to GPU only for its prefill |

Note the **inverted default** vs the vLLM design: `en_layerwise_perturbation=false` (all layers) here, vs the vLLM single-layer round-robin.

---

## 9. Success criteria

| # | Criterion | Verified by |
|---|---|---|
| 1 | A reproduces stock greedy decode at σ=0 | `test_sigma0_equiv.py` PASS |
| 2 | One-forward KV-slice == re-prefill oracle | `test_oracle_equiv.py` PASS |
| 3 | δW aligns with autograd gradient | `test_grad_cosine.py` cos > 0.05 (single + all-layers) |
| 4 | End-to-end OPD iter runs, ‖δW‖>0, no crash, N iters | smoke run, Qwen3-1.7B student / 4B teacher |
| 5 | Free-lunch curve measured & reported | `bench.py` output (confirmed OR falsified — both valid) |
| 6 | Zero verl/Ray/vLLM imports in `src/np_hf/` | grep / import check |

---

## 10. Pointers

- vLLM predecessor spec: `docs/superpowers/specs/2026-05-28-np-trainer-design.md`
- vLLM design wiki (math source of truth): `docs/wiki/zo_np_trainer.md`
- Reused math originals: `verl/verl/trainer/np/{seeding,grad_estimator}.py`, `teacher_scorer.py` (`reverse_kl_topk`)
- SFT example referenced (then dropped from scope): `LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml`
