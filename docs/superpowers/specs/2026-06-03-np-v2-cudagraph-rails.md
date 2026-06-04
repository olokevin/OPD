# NP Trainer V2 — 1+N CUDA-Graphed Decode Rails (buffer-in-graph)

> **One line.** V2 turns V1's eager, hand-driven `(1+N)`-row decode into a **CUDA-graphed** step:
> the model forward (incl. the `y += σ·u_buf` perturbation) is captured **once** and **replayed** per
> token; fresh noise enters via a **host-refilled persistent buffer** (`u_buf`), never via in-forward RNG.
> The N perturbed rails ride the *same* captured graph, *same* weights, *same* shared-prefix KV as the
> clean rail — so they add compute, not memory traffic. No NP math changes.

**Status:** design approved 2026-06-03. Supersedes `docs/superpowers/specs/2026-06-03-np-v2-design.md`
(the prompt-packing-first plan). Companion to `docs/wiki/zo_np_trainer.md` §8.
**Branch context:** `np-fold-xcapture` (V1.1, commit `21534aa`; x_t-capture folded; estimator `(L_q−mean)/σ`).
**Architecture (decided):** **Path B — buffer-in-graph.** Keep V1's multi-query `1+N` row layout
(`slot_mapping=[clean, −1×N]`, perturbed rows write no KV); replace the *eager* perturbation + eager
dispatch with a **captured graph** whose only per-token dynamic input is `u_buf` (plus the usual
`input_ids`/`positions`/`slot_mapping`/`seq_lens`/`block_table` refills). **V1's eager serial driver stays
callable as the parity oracle.**

---

## 1. Goal & non-goal

**Goal (the user's, verbatim intent).** Generate the `1+N` rails — 1 clean + N independently-perturbed —
**inside vLLM's native CUDA-graphed forward**, all conditioned on the **same model parameters and the same
KV cache**. Decode is memory-bound and the per-step GEMM is under-utilized, so the N extra rails read no
new weights and no new KV: they cost extra FLOPs on data already resident, not extra memory traffic. The
win is removing (a) the **eager Python perturbation hook** and (b) the **per-token eager dispatch** that
V1's profile shows is ~99% of step time.

**Explicit non-goal (corrects the superseded plan).** This is **not** "batch many *prompts* into one
forward." Cross-prompt packing (`B_pack`) was the superseded plan's headline; it is **dropped** as the
primary axis. V2's primary axis is **CUDA-graphing the single-prompt `1+N` step**. Cross-prompt batching
may return later as a *secondary* tiling knob, but only after the graphed single-prompt step is proven and
only if measured to help — it is **out of scope here** (§8).

---

## 2. Why this is now possible (the wall V1 hit, and why it falls)

V1 set `enforce_eager=True` for one reason: it draws noise **inside** the forward —
`PerturbedLinear.forward` calls `noise_seed(...)` + `draw_noise(...)` per sample per token
(`np_worker_extension.py:91-92`), and `torch.Generator().manual_seed()` cannot be captured into a CUDA
graph. The wiki states this as "mandatory" (`zo_np_trainer.md` §2 fact #4).

The vLLM-0.11.0 source shows that wall is **self-imposed, not fundamental**:

- **vLLM keeps *all* RNG outside its own captured graphs.** The sampler's `torch.Generator` runs *after*
  the model forward (`vllm/v1/worker/gpu_model_runner.py:576-579`); no randomness lives inside a captured
  region. "No RNG in the graph" is vLLM's existing contract, not an obstacle we invent.
- **vLLM feeds per-step dynamic data via host-refilled persistent buffers**, not by re-tracing. All of
  `input_ids`/`positions`/`seq_lens`/`query_start_loc`/`slot_mapping` are `CpuGpuBuffer`s `copy_`'d to GPU
  before each step (`vllm/v1/worker/gpu_model_runner.py:1044-1094`; `vllm/v1/utils.py:105-143`
  `CpuGpuBuffer`). The captured graph reads those buffers; the host updates them in place.
- **`reshape_and_cache` honors `slot < 0`** (`vllm/attention/ops/triton_reshape_and_cache_flash.py:34-37`:
  `if slot_idx < 0: return`), so the V1 trick "perturbed rows write `slot=−1` → no KV" survives graph
  capture unchanged — `slot_mapping` is just another refilled static buffer.

**Therefore:** move the noise draw from *in-forward RNG* to a *host-side buffer fill*, capture
`y[perturbed] += σ·u_buf` as a fixed-shape elementwise op, and the forward becomes capturable. This is the
user's proposed mechanism, and it matches vLLM's own graph-input discipline exactly.

---

## 3. Invariants V2 MUST preserve (this is a decode-driver rewrite, not a math change)

Re-checked at each milestone:

- **σ=0 byte-equivalence** vs stock greedy `LLM.generate` (`np_checks/check_decode_sigma0.py`).
- **cos(NP δW, autograd) ≈ +0.41 @ N=64** (`np_checks/check_grad_cosine.py`; offline, driver-independent).
- **Per-token `(L_q, L_clean, u, x)` semantics + the `(L_q−mean)/σ` estimator unchanged.**
- **Bit-identical `u`.** The graphed path's `u_buf` is filled by the **same** `draw_noise(noise_seed(...))`
  (`seeding.py`) V1 calls in-forward — same `(global_seed, step, layer, rollout, q)` → same bytes. Only the
  *location* of the draw moves (before-replay host fill vs. in-forward), never the value.
- **Update rule / sign** (`W ← W − lr·δW`) unchanged.
- **Zero ES files touched.**

**Untouched modules:** `grad_estimator.py`, `teacher_scorer.py`, `seeding.py`, `layer_resolve.py`,
`assemble_layer_delta`, `apply_node_update`, NCCL broadcast. V2 changes *how the `1+N` step runs* (eager →
captured) and *where the noise draw lives* (in-forward → host buffer). Nothing else.

---

## 4. Architecture & milestones

| Component | Where | Responsibility |
|---|---|---|
| Graph-capturable perturb op | `PerturbedLinear` (new `perturb_graph` mode) + persistent `u_buf` | In-graph static `y[1:1+N] += σ·u_buf`; capture `x[0]` as a static-buffer view. No RNG, no Python loop, no alloc. |
| Buffer refill (host) | `np_worker_extension.py` → `_np_fill_u_buf` | Before each replay, `u_buf.copy_(draw_noise(noise_seed(...)))` for the active layer's N rows. The *only* RNG, on the *existing tested* path. |
| Graph capture / replay | `np_worker_extension.py` → `_np_capture_step`, `_np_replay_step` | Capture one `1+N` step forward at fixed width with all inputs bound to persistent buffers (`input_ids_buf`, `positions_buf`, `slot_mapping_buf`, `seq_lens_buf`, `block_table_buf`, `u_buf`, output `hidden_buf`). Per token: refill buffers, `graph.replay()`, read `hidden_buf`. |
| Graphed decode driver | `np_worker_extension.py` → `run_np_decode_graphed` | The per-token loop over `_np_replay_step`: sample row-0 token (eager), commit, advance `kv_cursor`, refill buffers, replay. Mirrors `run_np_decode` control flow; swaps eager forward → replay. |
| Trainer wiring | `ray_trainer.py` `fit()` | `np.decode_mode ∈ {eager, graphed}`; per (prompt, rollout) call `run_np_decode_graphed` instead of `run_np_decode`. Everything downstream (score → assemble → apply → broadcast) **unchanged**. |
| Config | `np_trainer.yaml` | New keys `decode_mode`, `use_cuda_graph`. |

**Milestones** (each independently verifiable & revertable; `eager`/V1 stays callable throughout):

1. **M0 — Capture-replay spike (de-risk first).** Capture **one** `1+N` step forward with hand-built
   `attn_metadata` bound to static buffers; replay it; assert `graph_hidden == eager_hidden` (bitwise /
   within capture tolerance) on a single token. **This is the one real unknown** (§7.4): our hand-built
   metadata is *not* vLLM's normal capture path. **Gate:** spike passes → proceed; fails → stop and
   reconsider (V1 eager remains the shipped trainer; we lose nothing).
2. **M1 — Buffer-fill perturbation, still eager.** Move the noise draw out of `PerturbedLinear.forward`
   into `_np_fill_u_buf`; `PerturbedLinear` gains `perturb_graph` mode that only does `y += σ·u_buf` +
   `x` view. Run the existing eager driver but reading `u_buf`. **Gate:** per-token `(L_q,L_clean,u,x)` +
   clean tokens **bit-identical** to V1 (same seeds → same `u`); σ=0 byte-equiv passes. *Isolates the
   noise-relocation from the graphing.*
3. **M2 — CUDA-graphed step.** Capture/replay the `1+N` step (M0 mechanics) as the decode driver;
   `use_cuda_graph=true`. **Gate:** graphed per-token signals == M1 eager-with-buffer signals (parity);
   cos≈0.41 still holds; report s/step vs V1 eager.
4. **M3 — N-scaling micro-benchmark (validate the premise + retire).** With the graph in hand, measure
   s/tok vs N ∈ {1,8,16,32,64} on one prompt: confirm the user's "N is ~free" hypothesis directly
   (does graphed s/tok stay ~flat as N grows, as memory-bound theory predicts?). Report the curve. Then
   optionally make `graphed` the default; keep `eager` as the parity oracle. No math change.

**Ordering rationale.** M0 first because the hand-built-metadata-in-graph assumption is the only thing that
could kill the whole approach — fail fast. M1 before M2 so any parity break is attributable to *noise
relocation* (M1) vs *graphing* (M2), not both at once. M3 is the experiment the superseded plan never ran
(it assumed packing prompts; the user's actual claim is N-is-free, which M3 measures).

---

## 5. The graph-capturable perturbation op (heart of V2)

### 5.1 The V1 blocker (precise)

`PerturbedLinear.forward` (`np_worker_extension.py:61-99`) is graph-hostile in two ways per call:
(a) regenerates `u` via `noise_seed`+`draw_noise` (data-dependent control flow + fresh allocation, lines
91-92), and (b) loops `for q in range(n_sample): y[n_clean+q] += σ·u` (Python loop, line 94). Neither is
capturable.

### 5.2 The fix — split *regeneration* (host, outside graph) from *application* (static, inside graph)

**Persistent buffer.** On capture, allocate once on device:
`u_buf : [N, d_out]` (dtype = layer output dtype). Lives as long as the captured graph. (One layer is
perturbed per step under `en_layerwise_perturbation`; all targeted layers share module type/shape
(`down_proj`), so one `u_buf` per distinct `d_out` suffices — cache graphs keyed by `(d_out,)`.)

**Inside the graph (captured, static).** `PerturbedLinear` runs new `perturb_graph` mode:
```python
# fixed shapes, fixed row offsets → graph-safe
y[1:1+N] = y[1:1+N] + sigma * u_buf          # elementwise, no RNG, no Python loop, no alloc
st["captured_x"][name] = x[0]                # view into static input buffer, no clone/alloc in-graph
```
Row offsets (`0` = clean, `1..N` = perturbed) are the fixed V1 layout (`n_clean_rows=1`), constant for the
graph's life → baked in. `sigma` is a captured scalar.

**Outside the graph (host, per token, before `replay()`)** — `_np_fill_u_buf`:
```python
for q in range(N):
    seed = noise_seed(global_seed, t, layer_name, rollout, q)     # SAME call as V1 line 91
    u_buf[q].copy_( draw_noise(seed, (d_out,), device, dtype, sample_method) )   # SAME draw as V1 line 92
```
`copy_` into a persistent buffer is a graph-input update, identical in kind to vLLM refilling
`input_ids`/`positions`/`slot_mapping`. **The RNG never moved off the path that passed the cos & σ=0
gates** — it moved *earlier in wall-clock* (before replay) but the bytes are identical → parity by
construction.

### 5.3 x_t capture under the graph

V1.1 captures `x_t` inside the perturbed forward (`np_worker_extension.py:82`, the x-capture fold). Under
the graph, `x[0]` is a **view into the static `input_ids`→hidden path**; we capture it by having
`perturb_graph` mode write `x[0]` into a persistent `x_buf` (`x_buf.copy_(x[0])` inside the captured op, or
read the view out of `hidden`/layer-input static buffer after replay). Decision: **`x_buf` is a persistent
capture buffer the graph writes into**; host reads it after `replay()` exactly as V1 reads `captured_x`.
Same value, same timing relative to the token — parity preserved.

### 5.4 What stays eager (by design)

`model.compute_logits(hidden)` and **row-0 sampling** stay outside the graph (sampling is data-dependent —
which token row 0 commits decides the next step's `input_ids`). The graph covers the **`1+N` model forward**
(the 99%) and writes `hidden` into a static `hidden_buf` the eager `compute_logits` then reads. This mirrors
vLLM's own split (model forward graphed; sampler eager, `gpu_model_runner.py:576-579`).

### 5.5 Why the invariants transfer

The *value* written by `y[1:1+N] += σ·u_buf` is bit-identical to V1's `y[n_clean+q] += σ·u` (same
`draw_noise` output, same `σ`). σ=0 ⇒ `u_buf` term adds nothing ⇒ byte-equiv to greedy decode (σ=0 gate).
cos≈0.41 is offline and decode-driver-independent (it tests the math chain, not the driver) ⇒ unaffected.

### 5.6 Capture mechanics (`_np_capture_step`)

Standard `torch.cuda.CUDAGraph` capture: warm up a few eager `1+N` steps (cuBLAS workspace/autotune), then
capture one step with **all inputs bound to persistent buffers** (`input_ids_buf`, `positions_buf`,
`slot_mapping_buf`, `seq_lens_buf`, `block_table_buf`, `u_buf`; output `hidden_buf`, capture `x_buf`).
Per token: `copy_` new values into each input buffer, `graph.replay()`, read `hidden_buf`/`x_buf`. Captured
**once per `d_out`** (one perturbed layer/step, all `down_proj`-shaped); cache keyed by `(d_out,)`.

The attn_metadata must reference the **same static buffers** across replays. V1 already builds
`CommonAttentionMetadata` from `block_ids`/`seq_lens`/`slot_mapping`
(`np_worker_extension.py:194-245`); V2 binds those to the persistent buffers and rebuilds the lightweight
metadata wrapper per token pointing at them (the heavy tensors don't reallocate). See §7.4 risk.

### 5.7 Buffer-fill cost (conditional escalation)

Refilling `u_buf` is `N·d_out` host-launched `draw_noise` ops/token. M3 **measures** fill time as a fraction
of the now-graphed forward. **Only if** non-trivial do we escalate to (a) filling `u_buf` on a side CUDA
stream overlapping the previous replay (seeds for `t+1` are known ahead), or (b) a fused counter-based RNG
(Philox) kernel — written as a *conditional* follow-up with its own RNG-parity sub-gate, **not** up front.

---

## 6. Trainer wiring & config

### 6.1 New config keys (`np_trainer.yaml`)

```yaml
np:
  decode_mode: eager        # eager (V1, parity oracle) | graphed (V2)
  use_cuda_graph: false     # false = M1 eager-with-u_buf; true = M2 captured graph
```

Defaults keep V1 behavior (`eager`) so nothing regresses until explicitly switched.

### 6.2 The fit() change (minimal)

The serial `for b … for r …` loop in `ray_trainer.fit()` (`ray_trainer.py:488-624`) is **unchanged in
structure**. The only edit: dispatch `run_np_decode_graphed` instead of `run_np_decode` when
`decode_mode=graphed`. Same RPC surface, same args `(pid, sp, layer_name, np_cfg, r)`, same returned dict
(`clean_tokens`, `candidate_logits`, `captured_x`, `captured_u`). Everything downstream —
`scorer.score_rollout`, `assemble_and_apply`, `broadcast_layer_weights` — is **untouched**.

> **Note on the dropped axis.** The superseded plan rewrote this loop into a cross-prompt *wave loop*
> (`B_pack`). V2 does **not**. Per-prompt serial RPCs remain; the win is *inside* each
> `run_np_decode_graphed` (eager dispatch → graph replay), not in batching prompts. This keeps the diff
> small and the parity oracle trivially comparable (same call shape, same seeds).

### 6.3 Key points

- **One δW per step, unchanged.** Per-token signals from each (prompt, rollout) land in the same four
  accumulators fed once to the untouched `assemble_and_apply`. Effective mini-batch
  `= batch_size × n_rollout × T` exactly as V1.
- **Teacher scoring unchanged.** `TeacherScorer.score_rollout` is called once per rollout on the clean
  tokens, exactly as V1. Graphing is student-side only.
- **`decode_mode=eager` is the parity oracle.** Same prompts/seeds through `eager` and `graphed`; diff
  per-token `(L_q, L_clean, u, x)` + clean tokens. Same seeds + `draw_noise` ⇒ differences ≤ bf16
  reduction-order noise; gate asserts within tolerance.
- **Held-out KL probe (`_heldout_kl`)** stays on the `eager` path (≤16 prompts once per eval interval —
  not a bottleneck; keeps the progress probe on the proven path).

---

## 7. Verification & milestone gates

Success criterion (decided): **numerical parity + measured speedup**, both hard gates.

### 7.1 Existing gates that must keep passing (regression guards)

| Gate | Script | When |
|---|---|---|
| σ=0 byte-equivalence | `np_checks/check_decode_sigma0.py` (extended for graphed) | M1 + M2 |
| cos(NP δW, autograd) ≈ +0.41 @ N=64 | `np_checks/check_grad_cosine.py` | M2 (offline; confirms math untouched) |
| CPU unit suite (32 tests, incl. gradient-descent-sign) | `pytest verl/tests/np/` | M1 + M2 |
| ES regression (zero ES files touched) | `git diff --stat main..HEAD -- verl/.../es/` | both — must stay empty |

### 7.2 New parity gate (heart of V2 acceptance) — `np_checks/check_graphed_parity.py`

- Same model/teacher/prompts/seeds. Run **`eager`** and **`graphed`** decode on the same prompt set.
- Assert, per token: identical clean tokens; `u` **bit-identical** (same seeds → same `draw_noise`);
  `x`, `L_q`, `L_clean` equal within **bf16 reduction-order tolerance** (the only legitimate difference is
  graph vs eager matmul reduction order — documented, bounded, e.g. `rtol ≤ 1e-2` on the bf16 path; δW
  *direction* unaffected, re-confirmed by the cos gate).
- **M1:** eager-with-`u_buf` vs V1 in-forward-RNG (isolates **noise relocation**).
- **M2:** graphed vs M1 eager-with-`u_buf` (isolates **the graph**).

### 7.3 σ=0 graphed extension

Generalize `check_decode_sigma0.py`: with `σ=0`, the `u_buf` term vanishes, every perturbed row equals the
clean row, and graphed clean tokens match stock greedy `LLM.generate` token-for-token — proving the
captured `attn_metadata` (block_ids/seq_lens/`slot_mapping=[clean,−1×N]`) routes the clean rail to its own
prefix and the perturbed rails write no KV. Single most important new correctness check.

### 7.4 Capture spike (M0 — the de-risk gate, runs FIRST)

Before any driver work: capture one `1+N` step, replay it, assert `graph_hidden == eager_hidden`
(bitwise / within capture tolerance) on a single token. **Named risk:** vLLM's FLASH_ATTN backend (which
passed V1's σ=0 gate) must be CUDA-graph-capturable with our **hand-built** `attn_metadata` bound to static
tensors. Most vLLM attn kernels are graph-safe (that's how vLLM's own graphs work), but our hand-built
metadata path is **not** vLLM's normal `build_for_cudagraph_capture` path
(`vllm/v1/worker/gpu_model_runner.py:3019-3082`). If the metadata-in-graph assumption breaks, M0 finds out
in isolation. **Fallback:** the M1 noise-relocation win (no in-forward RNG) is independent and could let us
flip `enforce_eager=False` for *vLLM's own* PIECEWISE graphs even if our hand-built FULL capture fails —
documented as the degraded path.

### 7.5 N-scaling micro-benchmark (M3 — validates the user's premise)

The user's core claim is "N rails are ~free because they share weights+KV (no extra memory traffic)." V1's
profile (65-row eager forward = 13–20 ms/tok) does **not** confirm this — it conflates the eager dispatch
overhead with the FLOPs. M3 measures **graphed** s/tok vs N ∈ {1, 8, 16, 32, 64} on one prompt and reports
the curve. Expectation if the premise holds: near-flat s/tok up to the point N rails saturate the GEMM, then
linear. This number tells us whether to push N higher (cheap variance reduction) or cap it — an experiment
the superseded plan omitted entirely.

### 7.6 Speedup measurement (reported, not a fixed target)

Reuse the per-phase `torch.cuda.synchronize` timing. Report a small table at each milestone — **V1 eager**
vs **M1 eager+u_buf** vs **M2 graphed** — on (a) the smoke config (BATCH 4, RESP 128, eval off; V1 baseline
~28–36 s/step) and (b) a documented subset of the full config. Gate: measured, reported, real improvement
from M2 (no fixed pass/fail number).

### 7.7 Smoke e2e per milestone

5-iter end-to-end on Qwen3-1.7B student / Keven16 4B teacher, `decode_mode=graphed`: assert update lands
(`changed_frac>0`), `sync_ok=1`, no crash, held-out KL finite. Confirms the whole pipeline (decode → score →
assemble → apply → broadcast) works through the captured driver.

---

## 8. Scope boundaries & risks

### 8.1 Explicitly OUT of scope for V2

- **Cross-prompt packing (`B_pack`).** The superseded plan's primary axis. Dropped here; the win is
  graphing the single-prompt `1+N` step. May return as a *secondary* tiling knob post-M2 if measured to
  help — not now.
- **Native child-request rails (`n>1` style, "Path A").** Rejected: vLLM child requests write KV and
  compound (a perturbed rail that flips a token diverges and pollutes its own cache), and forcing
  `slot=−1` on them is not exposed by the scheduler (`vllm/v1/engine/llm_engine.py:243-255`). Fights
  vLLM's design and breaks the ephemeral-perturbation invariant. We keep V1's multi-query layout instead.
- **Fused noise-gen CUDA kernel (Philox).** Only a *conditional* follow-up if §5.7 buffer-fill is
  non-trivial.
- **Teacher-side batching/graphing** (student forward is 99%).
- **Any NP math change** (`grad_estimator`, `teacher_scorer`, `seeding`, `assemble_layer_delta` untouched).
- **Retiring V1 (`eager`)** — kept as the parity oracle; making `graphed` default (M3) is optional.

### 8.2 Named risks + mitigations

| Risk | Mitigation |
|---|---|
| Hand-built `attn_metadata` not graph-capturable with FLASH_ATTN | **M0 capture spike runs first**; fail fast before any driver work. Fallback: M1 noise-relocation is independent; may enable vLLM-native PIECEWISE graphs even if our FULL capture fails. |
| Noise relocation changes `u` bytes (would break cos/σ=0) | M1 gate asserts **bit-identical** `u` (same `noise_seed`+`draw_noise` call, only moved before replay). Parity by construction. |
| graph-vs-eager bf16 reduction-order drift inflates parity diff | Parity gate uses bounded `rtol`; cos≈0.41 confirms δW *direction* unaffected. |
| `compute_logits`/sampling can't be in-graph | By design they stay eager; graph covers only the model forward (the 99%) writing static `hidden_buf`. Mirrors vLLM's own model-graphed/sampler-eager split. |
| `x_t` capture under graph allocates / desyncs | `x_buf` is a persistent capture buffer the graph writes; host reads post-replay. No in-graph alloc. M1 parity gate covers `x`. |
| Recapture on shape change (e.g. different `d_out` layer) | Cache graphs keyed by `(d_out,)`; `down_proj` layers share shape so one graph covers the round-robin. |

---

## 9. File-level change map

| File | Change |
|---|---|
| `verl/workers/rollout/vllm_rollout/np_worker_extension.py` | **New:** `run_np_decode_graphed`, `_np_capture_step`, `_np_replay_step`, `_np_fill_u_buf`; `PerturbedLinear` gains `perturb_graph` mode + `u_buf`/`x_buf` plumbing. **Untouched:** all V1 eager methods (`run_np_decode`, `_np_step_forward`, `_np_build_attn_metadata`, `_np_prefill`), `apply_node_update`, `assemble_layer_delta`, broadcast. |
| `verl/trainer/np/ray_trainer.py` | `fit()` dispatches `run_np_decode_graphed` when `decode_mode=graphed`; `NPNcclLLM` keeps `enforce_eager=True` for M0/M1 (the captured graph is built by us, not vLLM's compile path — see §7.4); `_heldout_kl` stays eager. Loop structure unchanged. |
| `verl/trainer/config/np_trainer.yaml` | Add `decode_mode`, `use_cuda_graph`. |
| `scripts/zo_opd/np_checks/check_decode_sigma0.py` | Extend with a graphed σ=0 case. |
| `scripts/zo_opd/np_checks/check_graphed_parity.py` | **New** — eager-vs-eager+u_buf (M1) and eager+u_buf-vs-graphed (M2) parity. |
| `scripts/zo_opd/np_checks/bench_n_scaling.py` | **New** (M3) — graphed s/tok vs N curve. |
| `verl/tests/np/` | Add CPU unit tests for the pure-Python helpers (u_buf indexing/fill ordering, seed/rollout-id identity vs V1). No GPU. |
| `docs/wiki/zo_np_trainer.md` | Update §8 with the landed V2 design once built. |

---

## 10. Pointers

- Superseded V2 plan (prompt-packing-first): `docs/superpowers/specs/2026-06-03-np-v2-design.md`
- V1 design: `docs/superpowers/specs/2026-05-28-np-trainer-design.md`
- V1 wiki (incl. §8 V2 motivation): `docs/wiki/zo_np_trainer.md`
- Results / estimator history: `docs/results/zo_opd.md`
- Profiling memory: `~/.claude/.../memory/zo-np-threadcap-384core.md`
- vLLM 0.11.0 ground-truth (this session's investigation):
  - RNG outside graph: `vllm/v1/worker/gpu_model_runner.py:576-579`
  - Host-refilled static buffers: `vllm/v1/worker/gpu_model_runner.py:1044-1094`, `vllm/v1/utils.py:105-143`
  - `slot < 0` skips KV write: `vllm/attention/ops/triton_reshape_and_cache_flash.py:34-37`
  - vLLM's own cudagraph-capture metadata path (NOT ours): `vllm/v1/worker/gpu_model_runner.py:3019-3082`
  - `n>1` child-request fan-out (Path A, rejected): `vllm/v1/engine/llm_engine.py:243-255`,
    `vllm/v1/engine/parallel_sampling.py:34-76`
