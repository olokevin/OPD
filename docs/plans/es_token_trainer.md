# es_token trainer — per-token weight-perturbation ES for OPD

> A verl-compatible zeroth-order OPD trainer that applies a **fresh rank-1 weight perturbation at every decode token**, evaluated on **N parallel perturbed rails riding the clean rollout's KV** inside one fully-CUDA-graphed forward. It is the weight-space sibling of the NP V3 trainer ([wiki/zo_np_trainer.md](../wiki/zo_np_trainer.md) §11): same `1+N`-rail packed-graphed decode skeleton, but the perturbation is `ΔW = σ·(s_n⊙u_t)(r_n⊙v_t)ᵀ` with **fixed Hadamard sign buffers** `(s_n, r_n)` and **one shared per-token noise pair** `(u_t, v_t)` — which by construction removes the three measured NP cost centers (per-rail noise refill = 74% of decode, per-token D2H/sync glue, the 835 s Python assembly). Target: one full generation+update step within a small factor of BP-OPD at batch 64 × 1024 tokens.

Status: **design** (2026-06-09). Companion plan for the NP baseline: [np_trainer.md](np_trainer.md); measured NP V1–V3 results: [wiki/zo_np_trainer.md](../wiki/zo_np_trainer.md) §8–11; the host-glue profile this design is built around: [superpowers/plans/2026-06-07-np-decode-host-glue-optimization.md](../superpowers/plans/2026-06-07-np-decode-host-glue-optimization.md).

---

## 0. Position in the ZO family — what "improved" means

| | ES (`trainer/es`) | NP V3 (`trainer/np`, packed_graphed) | **es_token (this plan)** |
|---|---|---|---|
| What is perturbed | full weight tensor, whole rollout | linear **outputs** `y += σ·u`, per token | **weights**, rank-1 `ΔW = σ(s_n⊙u_t)(r_n⊙v_t)ᵀ`, per token |
| Estimates / rollout-step | `population_size` (~30) scalar probes | `B·T·N` per-token probes | `B·T·N` per-token probes (≈5×10⁵ at 64×1024×8) |
| Fresh RNG per token | — (per rollout) | `N·d_out` values **per layer** (896 `draw_noise` launches/token — measured 74% of decode) | **one fused draw** of `(u_t,v_t)` for all layers, shared across the N rails; signs are fixed buffers — ~`Σ_ℓ(d_out+d_in)` ≈ 0.92 M values/prompt/token, 1 kernel |
| Per-token captures | — | `u` (N·d_out/layer) + `x` (d_in/layer) D2H | **none** — noise regenerated from seeds at assembly; only `(1+N)·k` logprobs stored (wave-end batched pass) |
| Assembly | seed-regenerated full-tensor AXPY | per-token Python reduction (measured **835 s**/step; fix planned but unlanded) | **N chunked GEMMs per layer** (~1.5 PFLOP total ≈ 10–40 s), no per-token Python |
| Needs `x_t` capture | no | yes (the rank-1 factor) | **no** (the probe direction *is* the gradient component) |
| Gradient credit | sequence-level scalar | per-token, per-layer | per-token, per-layer |

The variance motivation (the user-stated premise): generation is memory-bound, so the `N` extra rails per decode step are nearly free for `N ≤ 16` (measured: **+19% wall at N=8**, wiki §9.3). Resampling `ΔW` **every token** turns one rollout into `T` independent gradient estimations instead of 1 — the same token-tape argument NP uses, now in weight space, and a ~10⁴× larger probe count per step than the sequence-level ES trainer.

What es_token deliberately gives up vs NP: the NP probe lives in the `d_out`-dim output space and gets multiplied by the *known* input `x_t` (the true gradient provably lies in `span{x_t}` on the input side), while the es_token probe direction `(r_n⊙v_t)` is oblivious to `x_t`. Per-probe direction quality (cosine vs autograd) will therefore be **lower** at equal `N` — gate V4 (§8) quantifies this before any scaling run. The bet is that the engineering wins (no x capture, no per-rail RNG, pure-GEMM assembly, smaller in-graph state) plus the unchanged probe count buy more than the per-probe quality costs.

---

## 1. Config interface (.yaml)

Mirrors `np_trainer.yaml` structure and naming. **Default = perturb all matched linears simultaneously** (the user-locked decision; contrast NP's layerwise round-robin default).

```yaml
es_token:
  # core perturbation knobs
  sigma: 0.01                       # perturbation magnitude (absolute; see sigma_mode)
  sigma_mode: absolute              # absolute | relative (sigma * RMS(W_l) per layer)
  n_sample: 8                       # N perturbed rails per decode step (+1 clean)
  sample_method: bernoulli          # base noise (u_t, v_t): bernoulli (Rademacher; exact rail
                                    # orthogonality + bounded norm) | gaussian | uniform
  sign_kind: hadamard               # hadamard (tiled H_M rows + fixed random column flips)
                                    # | rademacher_iid (ablation: iid sign rows, only E-orthogonal)
  perturb_method: forward           # forward | antithetic (rails n and n+N/2 share (u,v,signs), opposite sigma)
  perturb_rules:                    # vLLM-real fused names; default = ALL decoder linears
    - '^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))$'
  en_layerwise_perturbation: false  # default ALL layers at once (NP V3 semantics)

  # estimator / update
  grad_estimate_sample: mean_baseline  # mean_baseline: (l_n - mean_n l)/sigma   [default; the
                                       #   zo_opd lesson: no 1/std, it self-amplifies]
                                       # | clean_baseline: (l_n - l_0)/sigma (forward FD vs rail 0)
                                       # | grpo: (l_n - mean)/(std+eps)
  assembly_mode: es                 # es (pure ES: dW from (s,u,r,v) only; no x) — see §9 Q1
                                    # | np_hybrid (v2 ablation: NP-style g_t x_t^T using realized
                                    #   output perturbation; re-adds x capture; NOT in v1)
  token_agg: sum                    # sum | mean over T
  lr: 1.0e-3                        # all-layer scale — NP lesson: ~30x below single-layer LRs
  update_clip: null

  # decode driver (inherits NP V3 packed_graphed machinery)
  decode_mode: packed_graphed       # v1 ships graphed-packed only; eager fallback exists solely
                                    # as the parity oracle (check scripts), not a training path
  pack_width: 4                     # B_pack per wave; KV-ceiling-bound (wiki §10.5/§11.3)
  b_pack_buckets: [2, 4]            # fixed CUDA-graph bucket widths
  noise_refresh: fused_main         # fused_main: ONE draw into noise_buf on the main stream
                                    # | side_stream: double-buffered draw overlapped with replay (§3.3)

  # loss / teacher scoring
  loss_type: opd                    # opd | grpo (rule reward; debug only)
  opd_loss: sampled_token           # sampled_token: l_n = -A_t * log pi_n(y_t)  (k=1; A_t from
                                    #   teacher prefill)   | topk_rkl: NP's reverse-KL top-k scorer
  log_prob_top_k: 256               # topk_rkl only
  top_k_strategy: only_stu          # topk_rkl only (5 strategies, reused from np/teacher_scorer)
  reward_weight_mode: none
  teacher_temperature: 1.0
  teacher_model_path: null          # required for loss_type=opd
  teacher_batch_size: 16
  topk_store_k: 512                 # decode-side stored top-k window (topk_rkl)

  # engine / eval — identical block to np_trainer.yaml (num_engines, num_iterations, precision,
  # max_tokens, temperature, eval_interval, gpu_memory_utilization, gpu_fraction,
  # distributed_executor_backend, verify_update, global_seed, worker_extension_cls)
  worker_extension_cls: "verl.workers.rollout.vllm_rollout.es_token_worker_extension.WorkerExtension"
```

`model:` / `data:` / `trainer:` / `ray_kwargs:` blocks are copied verbatim from `np_trainer.yaml`. Launcher `scripts/zo_opd/opd_es_token.sh` exposes the same env-var style (`SIGMA`, `N_SAMPLE`, `LR`, `PACK_WIDTH`, `TEACHER_MODEL_PATH`, `OPD_LOSS`, …).

---

## 2. Math

### 2.1 Perturbation construction

For each matched linear `W_ℓ ∈ R^{d_out×d_in}`, token `t`, rail `n ∈ {1..N}` (rail 0 is clean):

$$
\Delta W_\ell^{(n,t)} = \sigma_\ell \,(s_{\ell,n} \odot u_{\ell,t})\,(r_{\ell,n} \odot v_{\ell,t})^{\!\top}
\;=\; \sigma_\ell\,(s_{\ell,n} r_{\ell,n}^{\!\top}) \circ (u_{\ell,t} v_{\ell,t}^{\!\top})
$$

- **Signs (fixed, run-lifetime buffers).** `s_{ℓ,n} ∈ {±1}^{d_out}`, `r_{ℓ,n} ∈ {±1}^{d_in}` are rows of a tiled Hadamard matrix: `s_n[i] = H_M[n+1, i mod M] · c_ℓ[i]`, with `M` the smallest power of two `> N` (M=16 for N=8; row 0 of `H_M` is skipped so all rows are zero-sum), and `c_ℓ ∈ {±1}^{d}` a **fixed** per-layer random column flip drawn once at init from `global_seed` (decorrelates the tiling from channel order; identical flip across rails preserves orthogonality exactly). All relevant Qwen3 dims (2048/4096/6144/12288) are multiples of 16, so row orthogonality `Σ_i s_m[i]s_n[i] = 0` is **exact**. Memory: `N·Σ_ℓ(d_out+d_in)` ≈ 7.3 M entries ≈ 15 MB bf16 — negligible, allocated once.
- **Base noise (per token, shared across rails, independent per prompt and per layer slice).** One flat draw per prompt per token: `noise_t^{(p)} ∈ R^{D_tot}`, `D_tot = Σ_ℓ (d_out + d_in)` (≈ 0.92 M for Qwen3-1.7B all-linears); each layer reads its `(u_{ℓ,t}, v_{ℓ,t})` as fixed **views** into the buffer. Seeded by `token_seed = blake2b(global_seed, step, wave, slot, t)` (same `seeding.py` hash family as NP) so assembly can regenerate bit-identically — **the NP invariant: noise is never stored, only seeds** ([np_trainer.md](np_trainer.md) §"Invariant").

**Why Hadamard + Rademacher is the default.** With `u, v` Rademacher, the N rail directions at a token are *exactly* Frobenius-orthogonal: `⟨ΔW_m, ΔW_n⟩ = σ²(Σ_i s_m[i]s_n[i]u_i²)(Σ_j r_m[j]r_n[j]v_j²) = σ²·0·0` since `u_i² = v_j² = 1`. The N probes per token are therefore non-redundant by construction (an N-dim orthogonal probe subspace, re-randomized every token via fresh `(u_t, v_t)`), and `‖ΔW‖_F² = σ²·d_out·d_in` is deterministic — no ANP-style norm rescue needed. Gaussian keeps unbiasedness but only E-orthogonality.

### 2.2 Forward semantics (the rail decode)

Row layout, KV handling, bucket-pad EOS, per-row attn metadata: **unchanged from NP V3** (wiki §11.1). Per decode step, each packed prompt contributes 1 clean row + N perturbed rows; perturbed rows write KV to `slot = −1` (discarded), only the clean token commits. The perturbed linear computes, per perturbed row `(p, n)`:

$$
y_{p,n} = W_\ell\, x_{p,n} + \sigma_\ell\, \alpha_{p,n}\, (s_{\ell,n} \odot u_{\ell,t}^{(p)}),
\qquad \alpha_{p,n} = (r_{\ell,n} \odot v_{\ell,t}^{(p)})^{\!\top} x_{p,n}
$$

i.e. the rank-1 weight perturbation is applied **without ever materializing ΔW**: one batched row-dot (`α`) plus one scaled rank-1 add. Extra FLOPs per layer ≈ `2·B·N·(d_in + d_out)` vs the main GEMM's `B·(1+N)·d_in·d_out` — a ~`1/d` relative overhead, three orders below the (already ~free) rail GEMM cost. All operands are persistent buffers or fixed views; shapes are static; **no RNG in-forward** → CUDA-graph-capturable by the same argument as NP's `perturb_graph` branch.

Because all rows (clean and perturbed) run the same resident `W`, the clean rail is byte-identical to stock decode at σ=0 — the existing σ=0 gate transfers.

### 2.3 Per-token losses

The OPD objective is per-state distribution matching at the clean rollout's visited states (states detached — the same semantics as BP-OPD's `token_reward_direct` and NP). One teacher **prefill** of the clean sequence scores all rails at all positions; the rails' n-dependence enters only through the student's stored per-rail logprobs.

- **`sampled_token` (default, k=1, importance-weighted).** The paper's sampled-token OPD loss is `ℓ_t = log p_t(ŷ_t) − log q_t(ŷ_t)` with `ŷ_t ~ p_t` (single-sample unbiased estimate of the token-level reverse KL). **Pitfall (decided 2026-06-09):** evaluating this *unweighted* under each rail at the frozen clean token makes the teacher term rail-independent — it cancels in any rail baseline-difference, and the assembled direction degenerates to `∇ log π(ŷ_t)` (zero-mean over sampling, teacher-free). The implemented rail loss is therefore the **importance-weighted single-sample estimate of `KL(π_n ‖ q_t)`**:
  `l_{n,t} = (π_n(ŷ_t)/π_0(ŷ_t)) · (log π_n(ŷ_t) − log q(ŷ_t))`, with `E_{ŷ~π_0}[l_{n,t}] = KL(π_n ‖ q_t)` exactly — the rail-dependent weight keeps the teacher signal in the finite difference, and to first order in σ the estimator equals the score-surrogate gradient `E[ℓ̂_t ∇log π(ŷ_t)] = ∇KL` that BP-OPD `token_reward_direct` backprops (the `+1·∇logπ` residue is a zero-mean baseline shift). Storage per token: each rail's logprob of the clean sampled token (one gathered logit + the row's logsumexp → `[B,T,1+N]` fp32); teacher side: one prefill of the clean sequence, read `log q(ŷ_t)` at each position via `prompt_logprobs`. Weight options: `student_iw` (default, the IW form), `student_p` (`w = π_n(ŷ_t)`, the NP `reverse_kl_topk` k=1 analog), `none` (documented-degenerate, kept for the cancellation ablation).
- **`topk_rkl`.** NP's reverse-KL top-k scorer **reused verbatim** (`verl/trainer/np/teacher_scorer.py`: `reverse_kl_topk`, the 5 `top_k_strategy` branches, the missing-id `min(t_logp)` fill). Storage: per-rail top-k logprobs `[B, T, 1+N, topk_store_k]`, same shapes NP already handles.

### 2.4 Per-token scales

`s̃_{n,t} = (l_{n,t} − b_t)/σ` with `b_t = mean_n l_{n,t}` (default `mean_baseline`; the `1/std` of grpo mode self-amplified to divergence on NP — [results/zo_opd.md](../results/zo_opd.md) §5–6 — so it is available but not default). `clean_baseline` uses rail 0's `l_{0,t}` as the FD anchor. `antithetic` pairs rails `(n, n+N/2)` with negated σ on shared `(u,v,signs)` — zero extra RNG, cancels even-order terms.

### 2.5 Gradient estimate and unbiasedness

$$
\delta W_\ell = \sum_{b,t} \frac{1}{N}\sum_{n=1}^{N} \tilde s_{n,t}^{(b)}\;
(s_{\ell,n} \odot u_{\ell,t}^{(b)})\,(r_{\ell,n} \odot v_{\ell,t}^{(b)})^{\!\top}
$$

This is exactly the user-stated form `l̃_{n,t} · (s_n r_nᵀ) ∘ (u_t v_tᵀ)` summed over rails and tokens. First-order: `l_{n,t} − l_{0,t} ≈ ⟨∇_W L_t, ΔW_n⟩`, and for `e_n = (s_n⊙u)(r_n⊙v)ᵀ` with Rademacher (or Gaussian) `u, v`:
`E[e_n ⟨G, e_n⟩][i,j] = Σ_{k,l} G[k,l]·E[u_i u_k v_j v_l]·s_n[i]s_n[k]r_n[j]r_n[l] = G[i,j]` — an **unbiased identity contraction**, so `δW_ℓ → +∂L/∂W_ℓ` in expectation (to O(σ)). With all layers perturbed simultaneously, cross-layer terms vanish in expectation because each layer's `(u_ℓ, v_ℓ)` slice is independent (the same argument as NP V3 all-layer; they contribute variance only).

Update: `W ← W − lr · δW` (descent; sign asserted by the cosine gate and a CPU sign unit test, mirroring the NP sign-bug lesson).

---

## 3. Decode driver — V3 skeleton + three deltas

The worker extension forks the NP V3 `packed_graphed` path (`run_np_decode_packed_graphed`, `_np_capture_step_packed`, `_np_replay_step_packed`, `_np_prefill_packed`, per-row attn metadata, bucket-pad EOS, `_select_bucket`/`_pad_waves_to_pack_width` orchestration) and changes only the perturbation op, the per-token noise refresh, and what is captured. The vLLM-0.11.0 capture facts (mutate-in-place metadata, `max_seqlen_k` frozen at the cap, `reshape_and_cache` skipping `slot<0`, `compute_logits` outside the graph) are inherited unchanged (wiki §9.2).

### 3.1 `ESTokenLinear` — the wrapped linear

Mirrors `PerturbedLinear`'s wrap/`_unpack`/`_repack` mechanics and worker-local-state dispatch, with one in-graph branch (`perturb_es_all_layers`):

```python
# fixed buffers (set at install): S[l] [N,d_out], R[l] [N,d_in] (sign rows, bf16)
# per-token views (refilled by the host): u = noise_buf view [B_pack, d_out_l]
#                                          v = noise_buf view [B_pack, d_in_l]
# packed row indices (per bucket, fixed at capture): pri [n_pert_rows],
#   rail_idx [n_pert_rows] (rail n of each perturbed row), prompt_idx [n_pert_rows]
x_p   = x[pri]                                        # [P, d_in]
v_eff = R[rail_idx] * v[prompt_idx]                   # [P, d_in]
alpha = (x_p * v_eff).sum(-1, keepdim=True)           # [P, 1]   row-dot
u_eff = S[rail_idx] * u[prompt_idx]                   # [P, d_out]
y[pri] = y[pri] + sigma * alpha * u_eff
```

Static shapes, persistent buffers, no allocation beyond capture-time temporaries reused on replay, no RNG → graph-safe. No `x_buf` copy (assembly doesn't need x), which also removes NP's per-layer in-graph capture writes.

vLLM fused linears (`qkv_proj`, `gate_up_proj`) are treated as a single `d_out`-fused matrix: one rank-1 across the fused output with a shared input-side `v` — consistent because the update in §5 applies to the same fused parameter. (See §9 Q4 for the k/v-path caveat.)

### 3.2 What is (and isn't) stored per token

- **Stored:** the per-rail student logprob payload only. `sampled_token`: gathered clean-token logit + per-row logsumexp → `[B, T, 1+N]` fp32 (≈ 2.4 MB at 64×1024×9). `topk_rkl`: raw row-block logits stashed to a device ring and reduced in a **wave-end batched top-k pass** — NP host-glue Fix 3, built in from day 1 rather than retrofitted.
- **Not stored:** `u, v` (regenerated from `token_seed` at assembly), `x` (not needed), ΔW (never materialized).
- **No per-token full `cuda.synchronize()`** — the only host read per token is row-0 sampling's `argmax(...).item()` (NP Fix 2, also built in from day 1).

Per-token host work is therefore: 1 fused RNG launch (or a buffer swap, §3.3) + graph replay + `compute_logits` + the sampling read. This is the NP isolation harness's measured "OFF floor" (~12.4 ms/token at pack 4, 28 layers) plus microseconds.

### 3.3 Noise refresh — buffer-referenced, optionally overlapped

The captured graph reads `noise_buf [B_pack_bucket, D_tot]` at a fixed address; the host refreshes it between replays. Two modes:

- **`fused_main` (v1 default).** Before replay `t`: `gen.manual_seed(token_seed(...)); noise_buf.copy_(draw(gen))` as **one** fused draw (~3.7 M values at pack 4 ≈ 7 MB, a single µs-scale kernel). This is already 896× fewer launches than NP's per-(layer,rail) refill — the measured 74%-of-decode bottleneck does not exist here by construction.
- **`side_stream` (the user-requested overlap).** Double buffer: while replay `t` runs on the main stream, a side stream draws token `t+1`'s noise into `noise_stage`; before replay `t+1`, the main stream does one device-device `noise_buf.copy_(noise_stage)` guarded by a CUDA event. Note on granularity: the user's "refresh layer ℓ's buffer as soon as layer ℓ finishes decoding" is not expressible with a single captured graph (the host cannot interleave inside a replay) — token-granular overlap is the graph-compatible equivalent, and since the refresh volume is one small fused draw, it fully hides. Implement `fused_main` first, measure, enable `side_stream` only if the draw is visible in the per-token profile.

Rejected alternative: in-graph Philox RNG (PyTorch CUDA graphs do support captured RNG with graph-safe generator state). Rejected because assembly-time bit-identical regeneration would then depend on Philox offset bookkeeping across replays/EOS-padded steps, breaking the simple `token_seed` contract that every NP parity gate is built on.

### 3.4 KV semantics of perturbed rails

Identical to NP: perturbed rails attend the clean prefix (plus the clean row's freshly-written current-token KV, per kernel ordering); their own k/v are never written. Consequence specific to weight perturbation: the probe captures `ΔW`'s effect through the q-path and all post-attention paths at the current token, but **not** through the current token's own k/v (those come from the clean row) — a dropped-path approximation shared with NP's qkv perturbation, documented in §9 Q4.

---

## 4. Teacher scoring

Unchanged from NP: one batched teacher **prefill** per wave over `prompt + clean_tokens` (`teacher_batch_size`), per-position teacher logprobs, then the `B·T·N` losses are computed vectorized on host/GPU from the stored student payload (§3.2) — `sampled_token` is a gather + multiply; `topk_rkl` calls `reverse_kl_topk` exactly as NP does. The teacher engine launch/co-location pattern (`gpu_fraction`, dedicated-GPU option) is reused from `np/ray_trainer.py`.

---

## 5. Assembly — chunked GEMMs, no per-token Python

Per layer, with token-chunks of size `T_c` (e.g. 2048 token-slots across the batch):

1. Regenerate the chunk's `(u, v)` rows from `token_seed` (CPU-staged draw + one H2D, the proven Fix-1 pattern) → `U [T_c, d_out]`, `V [T_c, d_in]`.
2. For each rail n (or batched over n with `bmm`):
   `Ũ_n = (U · s̃[:, n, None]) ⊙ s_n[None, :]`, `Ṽ_n = V ⊙ r_n[None, :]`, `δW_ℓ += Ũ_nᵀ @ Ṽ_n`.

Total cost `B·T·N·Σ_ℓ d_out·d_in ≈ 65536·8·1.41e9 ≈ 1.5 PFLOP` for Qwen3-1.7B all-linears at 64×1024×8 — **~10–40 s of pure GEMM** on one A800, vs NP's measured 835 s Python reduction. Scales (`s̃`) are computed once, vectorized, before the chunk loop. A CPU unit test pins the chunked GEMM against the naive per-`(t,n)` outer-product loop at small shapes.

Apply + broadcast: `W ← W − lr·δW` on engine 0 per layer, then per-layer NCCL broadcast — `apply_node_update`/`broadcast_layer_weights` patterns reused.

---

## 6. File map (sibling layout; zero edits to `es/`, `np/`, or PPO paths)

| Path | Responsibility | ~LOC |
|---|---|---|
| `verl/verl/trainer/main_es_token.py` | Hydra entry (clone `main_np.py`; same task_type dispatch via `es.task_utils`) | 240 |
| `verl/verl/trainer/es_token/__init__.py` | package marker | — |
| `verl/verl/trainer/es_token/ray_trainer.py` | `RayESTokenTrainer` + `ESTokenNcclLLM` (enforce_eager — we capture our own graphs, vLLM-level graphs stay off; prefix caching off). `fit()` = wave loop → decode RPC → teacher score → assemble → apply → broadcast → eval. Engine launch / NCCL / eval reused from `np/ray_trainer.py` patterns | 550 |
| `verl/verl/trainer/es_token/signs.py` | Hadamard row construction (tiled `H_M`, zero-sum rows, fixed column flips), buffer allocation | 90 |
| `verl/verl/trainer/es_token/seeding.py` | `token_seed(...)` + fused `draw_token_noise(...)` (thin layer over `np/seeding.py` primitives) | 60 |
| `verl/verl/trainer/es_token/grad_estimator.py` | scales (`mean_baseline`/`clean_baseline`/`grpo`), chunked-GEMM assembly, naive-loop reference impl for tests | 150 |
| `verl/verl/workers/rollout/vllm_rollout/es_token_worker_extension.py` | `ESTokenLinear` + decode driver (fork of the NP V3 packed_graphed path with §3 deltas), wave-end top-k pass, apply/broadcast | 900 |
| `verl/verl/trainer/config/es_token_trainer.yaml` | §1 config | — |
| `scripts/zo_opd/opd_es_token.sh` | launcher (env-var style) | 110 |
| `scripts/zo_opd/es_token_checks/{check_sigma0,check_graphed_parity,check_grad_cosine_es_token}.py`, `bench_es_token_vs_bp.sh` | gates + headline bench (adapted from `np_checks/`) | 600 |
| `verl/tests/es_token/` | CPU unit suite (§8 V1) | 400 |

Reused via import, no copies: `np/teacher_scorer.py`, `np/layer_resolve.py`, `np/seeding.py` primitives, `es/task_utils.py`, the NCCL init/broadcast blocks.

---

## 7. Predicted one-step budget vs NP V3 and BP (the benchmark target)

Reference point: batch 64 × 1024 tokens, N=8, Qwen3-1.7B student + Qwen3-4B teacher, single A800, `pack_width=4` (16 waves) — the exact NP V3 / BP bench configuration (wiki §11.4; BP side = stock verl BP-OPD `token_reward_direct` with vLLM CUDA-graph generation).

| Phase | NP V3 measured | es_token predicted | Why |
|---|---|---|---|
| decode | 1368 s | **~200–230 s** | per-token = the NP isolation "OFF floor" (~12.4 ms/tok·wave) + µs-scale fused RNG; no per-rail refill (was 74%), no per-token D2H/sync, lighter in-graph state (no x_buf) |
| teacher score | ~250 s | ~250 s | unchanged (batched prefill); v1.1 lever: overlap with next wave's decode |
| assemble + apply | 835 s | **~15–40 s** | chunked GEMMs (§5) replace the Python token loop |
| **total** | **2472 s** | **~470–520 s** | |

vs BP-OPD 53.8 s (cold) / 27.5 s (steady): predicted ratio **~9–19×**, down from NP's 46–90×. The remaining gap is structural: the hand-driven serial token loop at pack 4 decodes ~320 tok/s where vLLM's continuous batching does 1144 tok/s, and the teacher prefill is serialized. Honest target for the v1 acceptance bench: **one-step ≤ 600 s** at this scale, with the phase breakdown logged; stretch (teacher overlap + pack 8 when the teacher is on a separate GPU): **≤ 300 s**. The benchmark deliverable (`bench_es_token_vs_bp.sh` → `scripts/zo_opd/results/es_token_vs_bp.txt`) reports the same BEFORE/AFTER-style table as `bench_np_vs_bp.sh`, with NP V3 numbers as the third column.

These are predictions from measured NP component costs, not measurements; the bench replaces this table.

---

## 8. Verification gates (in order; each blocks the next)

| # | Gate | Pass criterion |
|---|---|---|
| V1 | CPU unit suite (`verl/tests/es_token/`) | sign rows: exact pairwise orthogonality + zero-sum, column flips preserve both; seed parity: assembly-regenerated `(u,v)` bit-identical (`torch.equal`) to decode-time refill across all `sample_method`s; assembly: chunked GEMM == naive per-`(t,n)` outer loop (small shapes, fp64); estimator sign: a synthetic quadratic loss decreases under `W − lr·δW`; rail math: `ESTokenLinear` rank-1 add == dense `(W+ΔW)x` at small shapes |
| V2 | σ=0 byte-equivalence (`check_sigma0.py`) | graphed packed decode at σ=0 matches stock greedy token-for-token for every prompt; logits width `Σ_p(1+N)` |
| V3 | eager-vs-graphed parity (`check_graphed_parity.py`) | eager all-layer oracle vs captured graph: clean tokens bit-for-bit, rail logprob payload within rtol 1e-2, staggered-EOS bit-for-bit (the NP F1 gate adapted; the eager driver exists only for this) |
| V4 | cosine vs autograd (`check_grad_cosine_es_token.py`) | offline HF model, single layer then all-layer: `cos(δW, ∇_W L) > 0.05` directional gate; **record the curve vs N alongside NP's (0.205@N=8 … 0.407@N=64)** — this is the per-probe quality cost of §0, measured before any scaling run |
| V5 | e2e smoke (20 steps, batch 8, `max_tokens=128`, all layers) | every matched layer `dW_norm > 0`, `weight_sync_ok = 1`, held-out teacher-KL net-down at lr≈1e-3 (LR sweep 3e-4/1e-3/3e-3 if not) |
| V6 | the headline bench (`bench_es_token_vs_bp.sh`) | one full generation+update step at 64×1024 vs BP-OPD; phase breakdown (decode / teacher / assemble) + ratio vs the §7 targets |
| V7 | regression | `git diff --stat` empty for `verl/trainer/es/`, `verl/trainer/np/`, `es_worker_extension.py`, `np_worker_extension.py`, all PPO paths |

---

## 9. Risks & open questions

1. **~~The assembly-formula ambiguity~~ — RESOLVED (user, 2026-06-09): pure ES only.** `assembly_mode: es` is the only mode; the NP-style `g_t x_tᵀ` hybrid is dropped (no x capture anywhere). The benchmark loss is sampled-token OPD with the `student_iw` weighting of §2.3.
2. **Per-probe gradient quality.** Rank-1 weight probes are oblivious to `x_t`; expected cosine at N=8 is below NP's 0.205. If V4 lands far below (≲0.02 single-layer), the method may need larger N (rails are ~free to 16, +19%→+32% wall) or the hybrid — decide on V4 data, before V5/V6.
3. **σ scale across heterogeneous layers.** A single absolute σ over qkv/o/gate_up/down with very different weight RMS may under/over-probe layers; `sigma_mode: relative` (σ·RMS(W_ℓ)) is the knob. Default absolute to match NP precedent; revisit on V5.
4. **Dropped k/v path at the current token (§3.4).** `ΔW` on `qkv_proj` is probed only through q (current token's k/v come from the clean row). Same approximation NP shipped with; if it biases qkv updates, the fallback is restricting `perturb_rules` to `o_proj|gate_up|down_proj` (a config change, no code).
5. **All-layer LR scale.** NP needed ~30× smaller LR all-layer (1e-3 vs 3e-2); es_token's δW norms differ again (`‖ΔW‖_F = σ√(d_out·d_in)` per probe). The V5 LR sweep is mandatory, not optional.
6. **Teacher phase becomes the critical path (~250 s ≈ 50% of the predicted step).** v1 accepts it; v1.1 overlaps wave `w`'s teacher prefill with wave `w+1`'s decode (independent by construction — NP wiki §10.6 lever).
7. **`pack_width` KV ceiling unchanged** (full-ctx scratch-KV reservation per packed prompt, wiki §10.5/§11.3): pack 4 with co-located teacher, 8 with a dedicated teacher GPU. es_token does not move this; a smaller per-prompt reservation (true expected length, not full ctx) is a shared NP/es_token follow-up.
8. **Graph-capture regressions.** The two known V2 gotchas (per-graph memory pool, `max_seqlen_k` frozen at the cap) are inherited as known fixes; V3's bucket-pad EOS rules transfer. New capture risk is the rank-1 op's temporaries — they must be allocated at capture time and reused on replay (asserted in V3's parity gate by running ≥2 buckets).
