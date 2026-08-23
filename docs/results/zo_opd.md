# ZO-ES-token (per-token weight-perturbation ES) OPD — results

Student `Qwen/Qwen3-1.7B`, teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, co-located on one GPU.
Per decode token a **fresh rank-1 weight perturbation** `ΔW_l = σ_l (s_n⊙u_t)(r_n⊙v_t)ᵀ` is applied to **all
112 decoder linears** on `N` parallel rails riding the clean rollout's KV, inside one fully-CUDA-graphed
packed forward. Loss = importance-weighted sampled-token KL to the teacher; gradient assembled by chunked
GEMMs from seed-regenerated noise. Trainer `verl/verl/trainer/es_token/`, driver
`scripts/zo_opd/opd_es_token.sh`, branch `feat/es-token-trainer`.
Design: [../plans/es_token_trainer.md](../plans/es_token_trainer.md) · subsystem page:
[../wiki/es_token_trainer.md](../wiki/es_token_trainer.md).

---

## Session 2026-06-09 — build, gates, headline one-step (collected)

Full records: `scripts/zo_opd/results/es_token_gates.txt`, `scripts/zo_opd/results/es_token_vs_bp.txt`.

| Gate | Result |
|---|---|
| CPU suite (`verl/tests/es_token/`, 19 tests) | PASS — exact rail orthogonality, IW unbiasedness (enumerated `E[l] == KL`), `weight_mode=none` teacher-cancellation degeneracy, chunked GEMM == naive outer-product, `ESTokenLinear` == dense `(W+ΔW)x` |
| σ=0 routing | PASS — graphed packed clean tokens == **stock greedy** `llm.generate`, 4/4 prompts, all 112 layers wrapped |
| graphed vs eager oracle (σ=0.01) | PASS — clean tokens bit-for-bit, payload max abs diff **0.000e+00** |
| staggered-EOS (`force_stop_at=[3,6,12,12]`) | PASS — bit-for-bit; pad rows do not corrupt active bucket-mates |
| trainer e2e smoke (2 steps) | PASS — 112 layers `dW>0`, `weight_sync_ok=1.0` |

Headline one-step (2026-06-09, batch 64 × 1024 greedy, N=8, all 112 linears, co-located 4B teacher):
**145.80 s** = decode 128.31 + teacher 4.24 + assemble 13.17, peak 85,129 MiB, 65,536 token-records —
vs NP-V3 **2472 s** (17×) and BP-OPD **62.72 s** cold (ES/BP = 2.33×).
Per-wave decode was flat at 7.85–8.72 s over the 16 waves.

## Session 2026-08-21 — profiling: gradient quality, decode throughput, one-step reproduction

Re-run on **H100 NVL 95 GB** (the 2026-06-09 bench used the same class of card).
New harnesses added on the branch:
`scripts/zo_opd/es_token_checks/{sweep_grad_cosine.sh,bench_decode_throughput.py,sweep_decode_throughput.sh,sweep_stock_batch.sh,sweep_decode_isolation.sh}`.
Raw records: `scripts/zo_opd/results/es_token_{grad_cosine_sweep,decode_throughput,stock_batch,decode_isolation}.txt`.

### 1. Gradient cosine vs autograd — at the rank-1 weight-probe information bound

Offline harness (`check_es_grad_cosine.py`): Qwen3-1.7B fp32, one target linear, σ=1e-3, Rademacher
`(u,v)`, mean-baseline FD, assembled with the shipping `rail_scales` + `assemble_chunk` math; reference
is `W.grad` from autograd on a last-token cross-entropy. `K = n_sample × repeats` independent rank-1
probes. Theory for an unbiased isotropic rank-1 **weight-space** probe: `cos ≈ sqrt(K/(K + d_out·d_in))`.

| layer | shape | `d_out·d_in` | N | repeats | K | cos | bound | cos/bound |
|---|---|---|---|---|---|---|---|---|
| `layers.0.mlp.down_proj` | 2048×6144 | 12.6 M | 8 | 50 | 400 | +0.0056 | 0.0056 | **0.99** |
| `layers.0.mlp.down_proj` | | | 8 | 300 | 2400 | +0.0130 | 0.0138 | 0.94 |
| `layers.0.mlp.down_proj` | | | 16 | 150 | 2400 | +0.0136 | 0.0138 | **0.98** |
| `layers.0.mlp.down_proj` | | | 32 | 150 | 4800 | +0.0193 | 0.0195 | **0.99** |
| `layers.0.self_attn.o_proj` | 2048×2048 | 4.2 M | 8 | 50 | 400 | +0.0084 | 0.0098 | 0.86 |
| `layers.0.self_attn.o_proj` | | | 16 | 150 | 2400 | +0.0222 | 0.0239 | 0.93 |

**Findings.**
1. The estimator sits at **0.86–0.99× the information bound** across two layer shapes and a 12× span of
   K — it extracts essentially all the signal a rank-1 weight probe carries. `sqrt(K)` scaling holds
   (K 400→4800 = 12×, cos 0.0056→0.0193 = 3.45× ≈ √12).
2. **Rails and repeats are interchangeable at equal K**: K=2400 gives +0.0130 (N=8 × 300) vs +0.0136
   (N=16 × 150) — within noise. The Hadamard rails buy exact per-token orthogonality but no extra
   information *per probe*. Since a training step collects `K = B·T·N`, N is nonetheless **the cheapest
   way to buy probes** — §2 shows each extra rail costs ~0.10 ms/token-step, so N is where added K
   should come from rather than from more tokens.
3. Per-probe cosine is dominated by `d = d_out·d_in`: the smaller `o_proj` reaches 1.5–1.6× the cosine
   of `down_proj` at the same K, close to the predicted `sqrt(12.6/4.2) = 1.73`. This is the structural cost of probing
   **weight** space instead of NP's output space (NP: cos 0.205 at K=400).
4. Useful cosine only appears at training scale, where `K = B·T·N`: at 64×1024×8 = 5.2e5 the bound
   predicts per-layer cos ≈ **0.20** for `down_proj` and **0.33** for `o_proj`; at N=32, ≈ 0.38.

### 2. Decode throughput — clean decode only vs clean + N parallel perturbed rails

`bench_decode_throughput.py`: Qwen3-1.7B bf16, all 112 linears wrapped, greedy, σ=0.01, EOS disabled so
every run executes exactly T token-steps. **ms/token-step is taken from the slope** of wall-clock over
T=64 → T=320, so CUDA-graph capture, prefill and teardown cancel out. `N=0` is the *same* graphed packed
driver with zero rails = **clean decode only**; the stock rows are vLLM's own `llm.generate`.

All rows below are **CUDA-graphed on both sides** (the es driver captures its own graph; stock is
`enforce_eager=False`). *Caveat:* the es driver forces `enforce_eager=True` at the engine level, so a
stock measurement taken inside the es harness is eager-mode and ~3× pessimistic
(8.63 vs 2.83 ms at B=4) — the eager stock numbers are in `es_token_stock_batch.txt` but are **not** the
right reference and are not used here.

**Rail sweep at `pack_width=4` (the shipping setting):**

| decode path | rows/token | ms/token-step | clean tok/s | row-steps/s |
|---|---|---|---|---|
| stock vLLM, B=4, cudagraph | 4 | 2.831 | 1412.7 | — |
| **es_token N=0 (clean only)** | 4 | **2.939** | **1361.1** | 1361 |
| es_token N=1 | 8 | 6.347 | 630.2 | 1260 |
| es_token N=2 | 12 | 6.698 | 597.2 | 1792 |
| es_token N=4 | 20 | 7.264 | 550.6 | 2753 |
| **es_token N=8 (shipping)** | 36 | **7.600** | **526.3** | 4737 |
| es_token N=16 | 68 | 8.170 | 489.6 | 8323 |
| es_token N=32 | 132 | 9.429 | 424.2 | 13999 |

**Findings.**
1. **The hand-driven graphed loop is not the problem.** Clean-only decode through the es driver costs
   2.939 ms/token-step vs stock vLLM's own graphed decode at 2.831 ms at the same concurrency — a **4%
   overhead**. At `pack_width=8` it is 3.229 vs 2.918 ms (11%). The custom decode driver is essentially
   free; earlier "511 vs 1134 tok/s" framing was a *concurrency* comparison, not a driver comparison.
2. **Turning rails on at all is the step; adding rails is nearly free.** N=0→1 costs **+3.41 ms**
   (+116%); N=1→32 costs a further **+3.08 ms** for 31 more rails (**+0.10 ms/rail**). The probe rate
   (row-steps/s) rises 1.26k → 14.0k — **11× more probes for 1.49× the time** — i.e. rail rows ride the
   memory-bound floor almost for free. Going
   N=8→32 costs 24% wall-clock for 4× the probes (K), which by §1 is a **2× cosine gain for 1.24×
   the time** — the cheapest gradient-quality lever available.
3. Cost per clean token is what suffers: 1361 → 526 tok/s at N=8 (2.6×). The rails do not slow the
   forward down per row; they multiply the rows.

**`pack_width` sweep at N=8** — the concurrency lever:

| pack_width | rows/token | ms/token-step | clean tok/s |
|---|---|---|---|
| 4 | 36 | 7.600 | 526.3 |
| 8 | 72 | 8.449 | **946.9** |
| 16 | 144 | — | **fails**: `packed scratch KV does not fit: b_pack=16 × blocks_per_prompt=2560 = 40960 > num_gpu_blocks=24717` |

Doubling `pack_width` 4→8 buys **1.80× clean throughput for +11% per-step cost**. `pack_width=16` is
blocked not by compute but by the **full-context scratch-KV reservation**: each slot reserves
`ceil(max_model_len/block_size) = 2560` blocks regardless of the actual 1024-token budget. Reserving to
the real response length instead of `max_model_len` is the single highest-leverage fix on this branch.

### 3. Where the per-token-step time goes

`sweep_decode_isolation.sh`, `pack_width=4`; `ES_BENCH_SKIP_NOISE=1` removes only the fused per-token
noise draw.

| configuration | ms/token-step | delta |
|---|---|---|
| N=0, no noise draw | 2.746 | bare graphed decode floor |
| N=0, with noise draw | 2.945 | **+0.199** — fused noise draw (all layers, one draw/slot/token) |
| N=8, no noise draw | 7.391 | **+4.645** — 112-layer rank-1 rail compute (32 perturbed rows) |
| N=8, with noise draw | 7.689 | +0.298 — noise draw at N=8 |

Attribution at the shipping point: **36% bare decode, 4% noise, 60% rail compute**. The design goal of
making the noise draw negligible is met — it is 4% here versus NP's 896 `draw_noise` calls/token that
were 74% of NP decode. The remaining 60% is the rank-1 op itself in 112 wrapped linears
(gather `x[pri]`, `R[rail]*v[pidx]`, the reduction, and the scatter-add), which is where any further
decode optimisation must go (e.g. fusing the four ops per layer, or batching layers).

**Why this matters for the ES/BP ratio.** Stock vLLM's per-token-step cost is almost flat in
concurrency (2.83 ms at B=4 → 4.58 ms at B=64, all graphed), so its throughput scales nearly linearly
with batch — while the es driver is pinned at 4–8 concurrent sequences by the scratch-KV reservation:

| concurrency B | stock cudagraph ms/token-step | stock tok/s |
|---|---|---|
| 4 | 2.831 | 1,413 |
| 8 | 2.918 | 2,742 |
| 16 | 3.322 | 4,816 |
| 32 | 3.586 | 8,924 |
| **64** (the OPD batch) | **4.580** | **13,975** |

At the OPD operating point the ratio is stark: es_token delivers **526 clean tok/s** (`pack_width=4`,
N=8) against stock vLLM's **13,975 tok/s** at B=64 — **26.6×**. At `pack_width=8` it is 947 tok/s,
14.8×. (The end-to-end step in §4 shows a milder 8.9–16.1× because BP's real generation phase also pays
prefill, sampling and detokenisation, which this steady-state microbench excludes by construction.)

That is the whole residual gap: es_token's decode is **not slower per row**, it is **starved of
concurrency**. Raising `pack_width` (i.e. fixing the KV reservation) attacks the gap directly; raising
N does not cost much but does not help wall-clock either.

### 4. One full OPD step — es_token vs BP-OPD, reproduced

`bench_es_token_vs_bp.sh` (`ES_GPU=7 BP_GPU=7 PACK_WIDTH=4`). Both sides: Qwen3-1.7B student +
Keven16 Qwen3-4B teacher, **batch 64 prompts × max_tokens 1024, greedy**, one GPU. Both sides emitted
**exactly 65,536 response tokens** (`response_length` mean=min=max=1024, `clip_ratio=1.0`), so the phase
comparison is like-for-like. ES = graphed packed decode, N=8 rails, all 112 linears, `pack_width=4`
(16 waves), sampled-token teacher loss, chunked-GEMM assembly, teacher co-located.
BP = stock verl PPO `token_reward_direct` (`opd_math_ref.sh`), stock vLLM cudagraph generation,
FSDP actor + FSDP teacher reward worker.
Logs: `logs/es_vs_bp/{es,bp}_20260821_171213.log`.

| phase | **es_token** | **BP-OPD** | es_token 2026-06-09 | BP-OPD 2026-06-09 |
|---|---|---|---|---|
| **one step** | **147.54 s** | **61.86 s** | 145.80 s | 62.72 s |
| decode / generation | 129.59 | 14.58 (`gen`; 8.05 pure `generate_sequences`) | 128.31 | 11.02 (7.85) |
| teacher scoring | **4.22** | 35.61 (`reward`; `rm_score` 32.48) | 4.24 | 41.71 (35.72) |
| gradient + update | 13.65 (assemble+apply) | 13.94 (`log_prob` 2.32 + `adv` 0.14 + `update_actor` 11.48) | 13.17 | 15.16 |
| peak GPU mem | 85,129 MiB | 71,710 MiB | 85,129 MiB | 71,710 MiB |
| **ratio ES / BP** | **2.39×** | — | 2.33× | — |

**The 2026-06-09 headline reproduces.** Step time 147.54 s vs 145.80 s (+1.2%), all three phases within
4%, identical peak memory, and `L_clean_mean`, `dW_norm_max/mean` bit-identical to the June run
(the pipeline is deterministic). `weight_sync_ok = 1.0`; all 112 layers had `dW > 0`.

**The shape of the gap is unchanged and confirmed by §2–§3:**
- **teacher scoring is 8.4× faster than BP's** (4.22 s vs 35.61 s) — the sampled-token loss needs one
  `prompt_logprobs` prefill per rollout, where BP pushes the full top-K machinery through an FSDP
  reward worker.
- **assembly is at parity with BP's backward+optimizer** (13.65 vs 13.94 s). The chunked-GEMM assembly
  is no longer a cost centre (NP's was 835 s).
- **100% of the residual gap is decode**: 129.59 s vs 14.58 s (`gen`), or 8.05 s against BP's pure
  `generate_sequences`. §2 shows this is *not* driver overhead (clean-only decode is within 4% of stock
  vLLM at equal concurrency); it factorises cleanly as
  `2.59× (rails: 1361 → 526 clean tok/s at N=8) × 9.89× (concurrency: stock 1,413 tok/s at B=4 →
  13,975 at B=64) × 1.04× (driver) = 26.6×`, exactly the steady-state decode ratio of §2. End-to-end the
  measured phase ratio is milder — **8.9×** against BP's `gen` and **16.1×** against its pure
  `generate_sequences` (505.7 vs 4,494 / 8,141 tok/s) — because BP's real batch-64 generation carries
  prefill, sampling and detokenisation and does not reach the synthetic microbench's peak.

Removing decode entirely would put es_token at ~18 s/step, i.e. **below BP**. The two levers, in order:
raise `pack_width` (blocked only by the full-context scratch-KV reservation — §2), then cut the 60%
rank-1 rail cost (§3).

*(Note: BP's step-1 logs `grad_norm: nan` / `actor/entropy: nan`. This is pre-existing in the reference
config — the same warning appears in the 2026-06-09 BP log — and is an artifact of the greedy
`temperature=0.0` rollout, not of anything measured here. Timing is unaffected.)*

### 5. Caveats and open items

- **No learning-quality result yet.** Everything on this page is wall-clock + correctness. The LR sweep
  is still open (NP's lesson: all-layer needs ~30× below the single-layer LR). The bench config
  `lr=1e-3, token_agg=mean` gives an update norm ≈0.3% of ‖W‖ per step. `eval/accuracy=0.0` in the bench
  logs is a 4-sample smoke probe on an untrained student — **not** a quality measurement.
- **Teardown hang reproduced.** After the step + eval the driver hangs in cleanup with the engine actor
  spinning at ~97% GPU (metrics are already logged, so measurements are unaffected). This run needed a
  manual `SIGTERM` after 240 s before the BP side could start. Untriaged; suspect vLLM V1 in-process
  engine (`uni` executor) + `ray.kill`.
- The microbench uses short synthetic math prompts and `ignore_eos`, so it measures steady-state decode
  only — prefill, detokenisation and scheduling are excluded by construction (the T=64→320 slope).
- Stock-vLLM rows must be taken with `enforce_eager=False`. A stock measurement inside the es harness
  inherits `enforce_eager=True` and is ~3× pessimistic; both variants are recorded in
  `es_token_stock_batch.txt` to make the trap explicit.

## Session 2026-08-22 — the fixed rail overhead: diagnosis and a fused kernel

§2 left an unexplained shape: turning rails on at all cost **+3.41 ms/token-step**
(N=0 → N=1) while 31 further rails cost only **+3.08 ms**. A near-fixed cost that does not
scale with the work being done is a launch-overhead signature, not an arithmetic one — at N=1 the
rail op touches at most 4 rows of ≤6144 elements per layer, microseconds of actual math.
Harnesses: `scripts/zo_opd/es_token_checks/{bench_rail_op.py,check_rail_op_parity.py}`;
raw record `scripts/zo_opd/results/es_token_rail_op.txt`.

### 6.1 It is not the RNG

The natural suspicion is the per-token noise draw — if Rademacher noise were generated on the host
and copied to the device, 0.92 M values per (slot, token) would be fatal. It is not:
`draw_noise` builds a `torch.Generator(device=cuda)` and draws **on the GPU**, and the isolation
measurement (§3) prices the whole fused draw at **0.199 ms**, paid identically at N=0 — so it is
not in the N=0 → N=1 delta at all. It remains a real but secondary cost (§6.5).

### 6.2 It is CUDA-graph node count

Replaying the shipping rail op alone at the true Qwen3-1.7B shapes for all 112 matched linears
(`bench_rail_op.py`, CUDA-graphed) reproduces the end-to-end delta almost exactly — **3.376 ms at
N=1** against the 3.41 ms measured in the live decode. The PyTorch formulation

```python
x_p   = x[pri]                                  # gather
v_eff = R[rail] * v[pidx]                       # 2 gathers + mul
alpha = (x_p * v_eff).sum(dim=-1, keepdim=True) # mul + reduce
u_eff = S[rail] * u[pidx]                       # 2 gathers + mul
y[pri] = y[pri] + sigma * alpha * u_eff         # gather + 2 mul + add + index_put
```

issues **14 kernels per layer**. Profiler counts, one decode token, N=1:

| kernel class | count | total | per call |
|---|---|---|---|
| `vectorized_gather_kernel` | 672 | 1207.8 µs | 1.80 µs |
| `elementwise_kernel<128,4>` (non-vectorised) | 224 | 602.9 µs | 2.69 µs |
| `vectorized_elementwise_kernel<8>` | 336 | 545.3 µs | 1.62 µs |
| `index_elementwise_kernel` (the `index_put`) | 112 | 530.0 µs | 4.73 µs |
| `reduce_kernel` (the `alpha` sum) | 112 | 508.7 µs | 4.54 µs |
| remaining elementwise | 112 | 184.1 µs | 1.64 µs |
| **total** | **1568** | **3578.8 µs** | **2.28 µs** |

1568 graph nodes at ~2.3 µs each *is* the 3.4 ms. Two secondary findings fall out: over half the
launches are **gathers of operands that do not depend on the layer's activations** (`R[rail]`,
`v[pidx]`, `S[rail]`, `u[pidx]` — pure functions of the token), and the `elementwise_kernel<128,4>`
rows are the *non-vectorised* path, taken because `noise_buf[:, off:off+d]` is a strided view
(row stride `d_total` = 917 504).

This also explains the sub-linear rail scaling: extra rails only make each of those 1568 kernels
slightly wider, and they are all far below the width where the GPU notices.

### 6.3 Fixes considered, all measured

`bench_rail_op.py` replays each candidate in a CUDA graph at the real shapes;
`check_rail_op_parity.py` checks each against the shipping op in fp32 (each variant compared in
*its own* row layout, since v3+ reorder the packed rows). Rail-op time only, ms:

| variant | idea | N=1 | N=8 | N=32 | speedup @N=1 |
|---|---|---|---|---|---|
| `v0_current` | shipping PyTorch branch | 3.376 | 4.646 | 5.225 | 1.00× |
| `v1_flat` | one broadcast kernel builds `sign*noise` for **all** layers into a flat `[P, d_total]`; each layer reads views | 2.373 | 3.581 | 4.276 | 1.42× |
| `v2_fused` | v1 + `vecdot` / `addcmul_` per layer | 1.992 | 3.135 | 3.847 | 1.69× |
| `v3_contig` | v2 + `[clean \| perturbed]` row layout so `x_p`/`y_p` are slices, not gather/scatter | 1.138 | 2.110 | 2.597 | 2.97× |
| `v4_bmm` | v3 + batched GEMV for `alpha` | 0.765 | 1.147 | 1.937 | 4.41× |
| `v5_blocked` | per-layer **contiguous** noise blocks (restores the vectorised elementwise path) | 1.527 | 2.695 | 3.033 | 2.21× |
| `v6_triton` | one fused Triton kernel/layer (needs the v3 layout) | 0.491 | 0.874 | 1.225 | 6.88× |
| **`v7_triton_rowidx`** | fused Triton that **reads the row indices**, so no layout change and no `[P, d_total]` buffer | 0.478 | 0.790 | 0.870 | 7.06× |
| **`v7` tuned** | `BLOCK_IN=BLOCK_OUT=4096, num_warps=16` | **0.313** | **0.472** | **0.560** | **10.79×** |

All variants reproduce the shipping op to ≤3.0e-06 relative error.

**Why v7 wins.** It collapses the whole per-layer op into one launch: one program per perturbed
row, which reads that row's `(rail, slot)` from the index tensors and forms the sign-modulated
noise *on the fly*. That removes all six operand gathers, the strided-view penalty, the separate
reduce, and the `index_put` — and because it addresses rows through `pri` it needs **no change to
the packed row layout** (so none of the NP-inherited attention/KV metadata is touched) and never
materialises the `[P, d_total]` sign×noise buffer that v1–v6 need (235 MB at N=32).

**Why the tuning matters so much.** The grid is only `P = bucket × n_sample` programs — 4 at the
shipping N=1. The kernel is latency-bound, not occupancy-bound, so large blocks that cut the number
of reduction iterations beat the usual "more, smaller programs" instinct: 4096/4096/16 warps is
1.47× faster than the conventional 1024/1024/4.

### 6.4 Result — the fixed overhead is 7× smaller and the goal is met

Shipped as `verl/verl/trainer/es_token/rail_kernel.py`, called from
`ESTokenLinear.forward`; the PyTorch branch is retained as a fallback when Triton is unavailable or
the tensors are not row-contiguous. Same protocol as §2.

| N | before | after | speedup | rail overhead vs N=0 |
|---|---|---|---|---|
| 0 (clean only) | 2.939 | 2.943 | 1.00× | — |
| **1** | 6.347 | **3.424** | **1.85×** | 3.408 → **0.481 ms** (7.1× less) |
| 2 | 6.698 | 3.462 | 1.93× | 3.759 → 0.519 |
| 4 | 7.264 | 3.651 | 1.99× | 4.325 → 0.708 |
| **8** (shipping) | 7.600 | **3.885** | **1.96×** | 4.661 → 0.942 |
| 16 | 8.170 | 4.259 | 1.92× | 5.231 → 1.316 |
| 32 | 9.429 | 5.208 | 1.81× | 6.490 → 2.265 |
| 8 @ `pack_width=8` | 8.449 | 4.547 | 1.86× | — |

**The goal — a single rail costing close to clean-only decode, under 3.5 ms/token-step — is met at
3.424 ms**, i.e. +0.481 ms over clean decode instead of +3.408 ms. Clean throughput at the shipping
N=8 rises 526 → 1,030 tok/s.

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher, same GPU and
config as §4):

| | before | after | speedup |
|---|---|---|---|
| **step_time** | 147.54 s | **89.59 s** | **1.65×** |
| decode | 129.59 s | 72.00 s | 1.80× (16 waves, 7.85 → 4.25 s each) |
| teacher | 4.22 s | 3.95 s | — |
| assemble | 13.65 s | 13.56 s | — |
| peak GPU mem | 85,129 MiB | 85,107 MiB | — |
| **ratio vs BP-OPD** (61.86 s) | **2.39×** | **1.45×** | |

Correctness is unchanged and checked three ways: the 19 CPU tests still pass; `check_rail_op_parity.py`
matches the shipping op to ≤3.0e-06 for every variant; and on GPU all three parity gates still pass
(σ=0 ≡ stock greedy, graphed ≡ eager **bit-for-bit** with payload max|diff| 0.000e+00, staggered-EOS
bit-for-bit). The step's `L_clean_mean` is **bit-identical** to the pre-optimisation run
(0.2556177764199674) — the clean trajectory is untouched — while `dW_norm_mean` moves 239.726 →
239.514 because the rail now accumulates in fp32 rather than bf16.

### 6.5 What is left

1. **The noise draw — 0.199 ms**, now ~14% of the N=1 step and the largest remaining fixed cost.
   `draw_noise` materialises an **int64** `randint` buffer (7.34 MB per slot at `d_total` = 917,504)
   and then runs a five-kernel cast/scale/copy chain: ~42 MB of traffic and ~6 kernels per slot per
   token. Drawing straight into the bf16 buffer, or folding a Philox stream into the rail kernel
   itself, removes most of it. The same routine regenerates noise for every token during assembly,
   so this also attacks the 13.6 s assemble phase. Constraint: decode and assembly must keep
   regenerating **bit-identical** noise, so both call sites have to move together.
2. **`pack_width`** is now unambiguously the dominant wall-clock lever (§2): decode is still 80% of
   the step, and the full-context scratch-KV reservation caps concurrency at 4–8 slots while BP runs
   64.

## Session 2026-08-22b — direct Rademacher noise (the last fixed decode cost)

§6.5 left the per-token noise draw as the largest remaining fixed cost: **0.199 ms/token-step**,
and the same routine is called once per token record during assembly. Raw record:
`scripts/zo_opd/results/es_token_noise.txt`; gate
`scripts/zo_opd/es_token_checks/check_noise_parity.py`.

### 7.1 What was wrong

`draw_noise(method="bernoulli")` produced ±1 the long way round:

```python
bits = torch.randint(0, 2, shape, generator=gen, dtype=torch.int64)  # 8 bytes/elt
n    = bits.to(torch.float32) * 2.0 - 1.0
out.copy_(n.to(torch.bfloat16))
```

At `d_total` = 917,504 that is an **int64 buffer of 7.34 MB per slot** plus a five-kernel
cast/scale/copy chain — roughly **42 MB of memory traffic and ~6 kernels per slot per token** to
produce 1.83 MB of ±1 values. It also constructs a fresh `torch.Generator` per slot per token, and
derives the blake2b seed on the host inside the decode loop.

### 7.2 What replaced it

`verl/verl/trainer/es_token/noise_kernel.py` draws ±1 **directly, in the destination dtype**, in one
Triton launch for a whole batch of rows. Values come from Triton's counter-based Philox
(`tl.randint`), so they are a pure function of (seed, position) — no generator state and no host RNG,
which is exactly what the "regenerate, never store" invariant wants. Two supporting changes:

- **Seeds are hoisted out of the token loop.** `build_seed_table` derives every (token, slot) seed
  for a wave once and uploads them, so the decode loop does no blake2b and no host→device copy.
- **Assembly fills a whole chunk in one launch** instead of `m` separate draws
  (was 1024 × ~6 kernels per chunk).

A torch fallback is kept for non-Triton environments and for `sample_method != "bernoulli"`; it still
avoids the int64 buffer by drawing straight into the destination with `Tensor.random_(0, 2)`. The
implementation is selected once at import, so decode and assembly can never disagree within a run.

### 7.3 Correctness — the regeneration invariant

`check_noise_parity.py` exercises both call paths (decode's table slice vs assembly's host-derived,
freshly uploaded seeds) and the properties the estimator depends on. **ALL PASS**: decode ≡ assembly
byte-for-byte over every token; chunk row *j* equals its own (t, rollout) record over 64 records;
values are exactly {−1, +1}; |mean| < 0.02 per row; noise is distinct across both *t* and rollout;
and regeneration is bit-identical. The 19 CPU tests and all three GPU parity gates (σ=0 ≡ stock
greedy, graphed ≡ eager bit-for-bit with payload max|diff| 0.000e+00, staggered-EOS bit-for-bit)
still pass.

Note this **changes the noise values** relative to earlier runs — Philox counter mode is a different
stream from `torch.randint`. Nothing depends on the old stream (no trained checkpoint exists), and
both consumers moved together, which is the only property that matters.

### 7.4 Isolated cost

| | before | after | speedup |
|---|---|---|---|
| decode fill, one token × 4 slots | 0.203 ms | **0.015 ms** | **13.5×** |
| assembly fill, one 1024-row chunk | 38.9 ms | **2.9 ms** | **13.4×** |

### 7.5 End-to-end

Decode ms/token-step, `pack_width=4`, same protocol as §2:

| N | original | + fused rail (§6) | **+ direct noise** | total speedup |
|---|---|---|---|---|
| 0 (clean only) | 2.939 | 2.943 | **2.783** | 1.06× |
| **1** | 6.347 | 3.424 | **3.244** | **1.96×** |
| 2 | 6.698 | 3.462 | 3.329 | 2.01× |
| 4 | 7.264 | 3.651 | 3.481 | 2.09× |
| **8** (shipping) | 7.600 | 3.885 | **3.722** | **2.04×** |
| 16 | 8.170 | 4.259 | 4.107 | 1.99× |
| 32 | 9.429 | 5.208 | 4.956 | 1.90× |
| 8 @ `pack_width=8` | 8.449 | 4.547 | 4.237 | 1.99× |

Rail overhead over clean-only decode at N=1: 3.408 → 0.481 → **0.461 ms**.

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher):

| | original | + fused rail | **+ direct noise** | total |
|---|---|---|---|---|
| **step_time** | 147.54 s | 89.59 s | **83.80 s** | **1.76×** |
| decode | 129.59 s | 72.00 s | 68.44 s | 1.89× |
| teacher | 4.22 s | 3.95 s | 4.23 s | — |
| assemble | 13.65 s | 13.56 s | **11.03 s** | 1.24× |
| `n_token_records` | 65,536 | 65,536 | 65,536 | — |
| `weight_sync_ok` | 1.0 | 1.0 | 1.0 | — |
| **ratio vs BP-OPD** (61.86 s) | 2.39× | 1.45× | **1.35×** | |

`L_clean_mean` is 0.2556177764199674 in all three runs — the clean trajectory never moved.
`dW_norm_mean` shifts 239.51 → 243.78 with the new noise stream, as expected.

### 7.6 What is left

Decode is now **82% of the step** and the noise fill is down to 0.015 ms (0.5% of a token-step), so
**`pack_width` is the only lever of consequence left**: the full-context scratch-KV reservation
(2560 blocks per slot regardless of the real 1024-token budget) caps concurrency at 4–8 slots while
BP-OPD runs 64. §2 measured stock vLLM at 1,413 tok/s at B=4 against 13,975 at B=64 — that ~9.9×
concurrency deficit is the entire remaining gap.

## Session 2026-08-23 — budget-sized scratch-KV: pack_width unlocked, ES overtakes BP

§7.6 left `pack_width` as the only lever of consequence. Raw record:
`scripts/zo_opd/results/es_token_kv_reservation.txt`; gates
`scripts/zo_opd/es_token_checks/{check_kv_reservation.py,check_kv_output_neutral.sh}`.

### 8.1 The over-reservation

`_np_prefill_packed` carves a private KV region off the **top** of vLLM's block pool, one disjoint
slice per packed slot — the decode driver bypasses vLLM's scheduler and must own static KV for a
captured CUDA graph. It sized each slice at the **full `max_model_len`**:

```python
blocks_per_prompt = ceil(max_model_len / block_size) = ceil(40960/16) = 2560
assert b_pack * blocks_per_prompt <= num_gpu_blocks     # 24,717
```

A 1024-token generation from a ~90-token prompt needs ~70 blocks, so this over-reserved **~20×** and
capped the driver at **9 slots**. Sizing it to (longest prompt + `max_tokens`) instead:

| reservation basis | blocks/slot | max slots |
|---|---|---|
| `max_model_len` (old) | 2,560 | 9 |
| 1024 prompt + 1024 response | 128 | **193** |
| 1024 + 3072 response | 256 | 96 |

`_np_prefill_packed` gained `max_new_tokens=None` (None = old behaviour, so the NP trainer is
untouched); es_token passes `max_tokens`. **Safety:** the attention block table is zero-filled and
only the first `len(block_ids)` entries are written, so a slot that outgrew its slice would read
block 0 and silently corrupt *another* slot's KV rather than crash. A second assert now makes that
unreachable.

### 8.2 The gate had to be rewritten — and what it found

The obvious gate (packed clean tokens == stock greedy) **fails at pack_width ≥ 10**, and that finding
turned out to be about rounding, not KV. Evidence it is not corruption:

- **Neighbour-independence**: hold slots 0–3 fixed and swap the *content* of every other slot —
  output is byte-identical. Slices do not alias. PASS at widths 4/8/16/32/64.
- It changes with the wave **width alone** (slots 0–3 identical at width 4 and 16, different at 32).
- It appears at the shipping `pack_width=4` too, for prompts whose top-2 logits are close — and
  there the reservation change is provably byte-neutral.
- Divergent slots come in pairs (*i*, *i+8*) — the same prompt text — so it is prompt-dependent, and
  it compounds with generation length.

The hand-driven packed forward batches differently from vLLM's scheduler, so bf16 rounding differs
and greedy argmax flips on near-ties. Comparing packed output to stock measures rounding, not KV
safety. The gate therefore checks: **[A]** output-neutrality vs the old reservation (across separate
processes — flipping it in-process reuses the already-captured graph and is vacuous), **[B]**
neighbour-independence, **[C]** every slot reaches `max_tokens`. All pass; `check_es_parity`'s three
gates still pass unchanged.

### 8.3 Result

| `pack_width` | ms/token-step (N=8) | clean tok/s | waves for a 64-prompt batch |
|---|---|---|---|
| 4 | 3.734 | 1,071 | 16 |
| 8 | 4.259 | 1,878 | 8 |
| 16 | 5.435 | 2,944 | 4 |
| 32 | 8.431 | 3,795 | 2 |
| **64** | 15.036 | **4,257** | **1** |

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher):

| | `pack_width=4` | **`pack_width=64`** | speedup |
|---|---|---|---|
| **step_time** | 83.80 s | **42.67 s** | **1.96×** |
| decode | 68.44 s | 25.40 s | 2.69× (16 waves → 1) |
| teacher | 4.23 s | 3.99 s | — |
| assemble | 11.03 s | 13.18 s | — |
| **ratio vs BP-OPD** (61.86 s) | 1.35× | **0.69×** | |

**es_token is now faster than BP-OPD.** Cumulative over §6–§8: one step **147.54 → 42.67 s (3.46×)**,
decode **129.59 → 25.40 s (5.10×)**, ES/BP **2.39× → 0.69×**.

The next lever is no longer decode: assembly is now 31% of the step.

## Session 2026-08-23b — learning rate: a bound, and a measurement trap

First training runs into wandb `zo-opd-q34b-1p7b`. Raw record:
`scripts/zo_opd/results/es_token_lr.txt`; sweep harness
`scripts/zo_opd/es_token_checks/lr_probe.sh`.

### 9.1 Do not read `train/L_clean_mean` as a learning curve

`L_clean_mean` is computed on whatever 64 prompts that step drew, and on MATH lv3–5 it swings
**0.23 – 3.4 batch to batch** — far larger than any LR effect. Three LRs spanning 100× give
indistinguishable curves:

| step | LR 1e-3 | LR 1e-5 | LR 3e-5 |
|---|---|---|---|
| 0 | 0.22714 | 0.22714 | 0.22714 |
| 1 | 3.195 | 2.850 | 2.844 |
| 3 | 2.865 | 3.375 | 3.347 |
| 5 | 0.279 | 0.227 | 0.225 |
| 7 | 2.651 | 3.220 | 3.099 |

The applied update differs 100× (`lr·dW` ≈ 2.2 vs 0.02) yet the shape is the same, and the same
steps are low in every run — it is **data, not the optimizer**. `dW_norm_mean` is no divergence
signal either: it is the gradient-estimate norm *before* the LR multiplies it, so it is similar
across LRs by construction. Use **`eval/heldout_clean_loss`** — a fixed 16-prompt probe
(`ray_trainer.py:333`), logged only every `EVAL_INTERVAL` steps.

**Probe noise floor.** The probe scores *sampled* rollouts at T=1.0, so it is stochastic even for a
frozen model: three sweep runs read the same untouched step-0 model as 0.1908 / 0.2126 / 0.2242 —
a spread of 0.033, about **±8%**. Nothing smaller than that is interpretable.

### 9.2 LR 1e-3 — the shipped default — degrades the model

| | step 0 | step 25 | step 50 |
|---|---|---|---|
| probe KL (fixed 16) | 0.2244 | 0.5565 | **1.1559** |
| MATH-500 (fixed 200) | 5.0% | 1.5% | **0.0%** |

Monotonic on both, ~2× per interval, far outside the noise floor. Killed at step 50.

The cause is the temperature change: 1e-3 was calibrated on the **greedy** benchmark where
`dW_norm_mean` ≈ 240. At T=1.0 the rails ride a higher-entropy trajectory, the importance weights
spread, and `dW_norm_mean` is ~866 at step 0 and ~2,000–2,550 in steady state — **≈3.6× larger
before any LR is applied**. (T=1.0 is nonetheless required: the `student_iw` rail loss is an
unbiased estimate of `KL(π_n‖q)` only when the clean token is *sampled* from π₀.)

### 9.3 The sweep gives a bound, not a ranking

21 steps, `EVAL_INTERVAL=10`, fixed probe + MATH-500 on 50:

| LR | probe s0 | s10 | s20 | MATH-500 |
|---|---|---|---|---|
| 1e-4 | 0.2126 | 0.1886 | 0.2035 | 8 / 8 / 10% |
| 1e-5 | 0.1908 | 0.2126 | 0.2010 | 8 / 6 / 8% |
| 1e-6 | 0.2242 | 0.1988 | 0.1909 | 8 / 4 / 10% |

All flat within ±8%; MATH-500 at n=50 has σ ≈ 4pp and carries no signal either. So:
**1e-3 destroys the model, 1e-4 and below do not, and 20 steps cannot separate 1e-4/1e-5/1e-6.**

The 150-step run uses **1e-4** — the largest non-degrading LR, a principled default rather than a
measured optimum. Separating it from 1e-5 needs a horizon long enough for the signal to clear ±8%,
or a lower-variance probe (greedy probe rollouts, or many more probe prompts).

### 9.4 The 150-step run at 1e-4 — negative result

| step | 0 | 25 | 50 | 75 | 100 | 125 | 149 |
|---|---|---|---|---|---|---|---|
| probe KL (fixed 16) | 0.2126 | 0.2002 | 0.1987 | 0.2169 | 0.2049 | 0.2157 | **0.2228** |
| MATH-500 (fixed 200) | 6.0% | 7.0% | 6.5% | 5.5% | 7.0% | 2.0% | **4.0%** |

150 steps, 37.18 s/step, 92.9 min, 9.71 M token-records, `weight_sync_ok=1.0` throughout.

**es_token does not learn measurably at 1e-4 over 150 steps.** All seven probe readings lie in
0.199–0.223 with no direction; the endpoint is +4.8% vs step 0, *inside* the ±8% noise floor, so the
honest statement is "no change". MATH-500 wanders 2–7% with no trend (σ ≈ 1.7pp at n=200). The
apparent monotone decline at steps 25/50 broke at step 75 — exactly the false signal the noise floor
predicts, and the reason 3-point trends on this probe must not be reported as progress.

**The bracket, with no working recipe inside it:** 1e-3 destroys the model (+415% by step 50,
accuracy 0%); 1e-4 holds it steady. One order of magnitude between "destroys" and "does nothing".

**This is not ES-specific.** The BP-OPD baseline was equally flat (MATH-500 2.8 / 2.2 / 2.8 / 1.8%
over 138 steps at LR 1e-6). *Neither* method moved, which points at the setup rather than the
algorithm — see the truncation item below.

### 9.5 Still open

- **Whether es_token can learn** is still unanswered — the bracket is too wide to conclude no
  working step size exists between 1e-4 and 1e-3.
- The **BP-OPD baseline** (LR 1e-6, 138 steps) was also flat — MATH-500 2.8 / 2.2 / 2.8 / 1.8%
  across its four evals. It is a wall-clock reference, not a learning baseline.
- **Both runs cap responses at 1024 tokens and every rollout hits the cap without emitting EOS**
  (`response_length` mean=min=max=1024), so MATH-500 reads near its floor for both. Fixing that is a
  prerequisite for accuracy being a usable metric on this pair.

---

# ZO-NP (zeroth-order node-perturbation) OPD — results

Student `Qwen/Qwen3-1.7B`, teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`.
Loss = per-token reverse-KL to the teacher over the student top-K=16 set (`reward_weight_mode=student_p`).
Trainer: `verl/verl/trainer/np/` (custom n_sample-wide perturbed vLLM decode); driver
`scripts/zo_opd/zo_np_train.sh`. Offline gradient harness: `verl/verl/trainer/zo_np/grad_check.py`
(`scripts/zo_opd/zo_np.sh`). Full working notes: `scripts/zo_opd/results/{ANALYSIS,SCALING_FIX_AND_LR}.md`.

---

## Session 2026-06-02 — gradient scaling, LR search, and a self-amplifying divergence

### 1. NP estimate vs the true BP gradient (offline, `grad_check.py`)
For one perturb layer (`model.layers.0.mlp.down_proj`, d_out=2048) on a frozen (prompt, greedy-response),
the harness computes the NP δW (reusing the **shipping** estimator math) and the true `dL/dW` via
`loss.backward()` of the same OPD loss.

- **cos(NP δW, BP dL/dW) ≈ 0.01–0.02** at the trainer's 64 perturbations/token — the δW *matrix* direction
  is variance-starved. NOT a bug: a per-token `dL/dy` probe shows cos rising 0.03 → 0.18 as N: 16 → 4096.
- **‖NP‖/‖BP‖ tracks √(d_out/N)** exactly (≫1 at small N, → 1 as N → d_out), on both d_out=2048 and 1024.
- Binding constraint is the **rank-1-assembled weight matrix** (12.6 M elements over ~24 noisy g_t), far
  more sample-hungry than a single node-gradient vector.

### 2. Scaling fixes (`grad_estimator.py`, `ray_trainer.py`)
- ANP `1/‖u‖²` normalization made a config (`np.normalize_anp`, default **false**); it was hardcoded
  `True` and shrank the update by `1/d_out ≈ 1/2048`.
- `grad_estimate_sample=grpo` scale, two iterations:
  - `(L_q−mean)/σ` — restores the `1/σ` finite-difference scale (drops `/std`).
  - `((L_q−mean)/std)/σ` — **current code, per request** — keeps BOTH the z-score (`1/std`) and `1/σ`.
- Offline: the fixed estimators put ‖δW‖ on the true-gradient scale (vs the old `2e-4` ratio).
- **Key invariant:** the *assembled* δW norm is ≈28–57 for the `/std`, `/σ`, AND `/std/σ` forms alike,
  because `token_agg=mean` cancels the per-token scale. So the per-token 100× difference (`1/σ`) does **not**
  reach the weight update — the bf16-effective LR is similar across all three forms.

### 3. Training infrastructure built this session
- `fit()` restructured: **batch_size prompts/update** (1 rollout/prompt, n_sample=64), greedy clean decode.
- **Student + teacher co-located on one GPU** (one LR per GPU): needs `distributed_executor_backend="uni"`
  + keep `CUDA_VISIBLE_DEVICES` + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + mem-util 0.30 each.
- **Update-propagation verification** every step: `train/weight_changed_frac` (fraction of weight elements
  that flip in bf16 — the true "did it land" signal) + `train/weight_sync_ok` (all engines hold the same
  weight after broadcast → the next rollout reads the update). `apply_node_update` returns the changed frac.
- **Fixed held-out teacher-KL probe** (`eval/heldout_kl`) — per-step `train/L_clean_mean` is on shifting
  prompts so it can't show learning.
- **Bug fixes that made training valid:** (a) the MATH/GSM8K prompt processor only recognized `list/tuple`
  prompts but the parquet `prompt` is a `numpy.ndarray` → it silently fed an empty `"Problem: "` prompt;
  **both eval and training ran on blank prompts** (teacher-KL ~0.81 blank vs ~0.33 real) — fixed in
  `task_utils.py`, affects all np/es opd_math runs. (b) removed a global `ray stop --force` that killed
  concurrent runs' Ray sessions.

### 4. The bf16 reality
vLLM student weights are bf16. An update lands only if `lr·δW_elem` clears the mantissa step. Two
consequences that shaped the whole LR search:
- **`weight_delta` (the ‖W‖-norm difference) badly UNDER-reports the update** — element changes partly
  cancel in the norm, so a 20 %-of-elements update can show ~0 norm-delta and *look* like a no-op. Use
  `weight_changed_frac`, not the norm difference.
- For the production δW (norm ≈ 28–57), the LR → fraction-of-weights-changed map is roughly:
  lr 2e-5 → 0.1–0.3 %/step, 2e-4 → 3–12 %, 6e-4 → 7–31 %, 2e-3 → 22–57 %.

### 5. LR search for grpo = `((L_q−mean)/std)/σ` (wandb project `zo_opd_qwen4b_1p7b`)
The proper LR is `≈ ÷100` vs the `/std`-only form (the `1/σ=100×` per-token factor): the analog of a good
`/std @ 2e-3` is `/std/σ @ 2e-5`, etc. Swept the meaningful-update band (2e-4 / 6e-4) at batch=8.

| phase | observation |
|---|---|
| steps 0–10 | both 2e-4 & 6e-4 **dip the KL** (e.g. 6e-4: 0.336→0.322→0.318) — looks like training |
| steps 10–25 | KL **oscillates in a 0.31–0.35 band** = the probe's own ~±0.03 noise (greedy NP-decode is not bit-deterministic: two runs gave step-0 KL 0.306 vs 0.336 with identical weights) |
| **steps ~28–35** | **both runs DIVERGE**: 2e-4 KL → 0.47→0.48; 6e-4 KL → **0.93→1.14**. dW had grown 57→~2000 across the round-robin and chg% had climbed to 40–65 % before the layer-cycle reset |

**Honest verdict:** `/std/σ` *lands valid updates* in the 2e-4–6e-4 band (update signal is clean: chg%
rises monotonically, no no-ops), but the **held-out KL never sustainably decreases** — early steps are
buried in probe noise and by ~step 30 (one full 28-layer round-robin) the run **diverges**. This is the
`1/std` self-amplification (low-signal tokens, `std→0 ⇒ 1/std→∞`, +1e-8 floor insufficient) playing out
over a longer horizon than the wildly-too-high LRs did. No LR in the tested band gives stable training.

### 6. Cross-check: grpo = `(L_q−mean)/σ` (drop `/std`)
The `/σ`-only form trained **cleanly and monotonically** at **lr=3e-2** over the first ~16 steps
(held-out KL 0.335 → 0.322 → 0.319, bounded dW). It is the cleanest demonstrated training curve.
(A long run to check whether it too eventually diverges was not done this session.)

### 7. Important measurement caveats discovered (so future runs don't repeat them)
- **`weight_delta` norm-diff ≠ no-op** — use `weight_changed_frac`.
- **dW grows step-over-step from the `en_layerwise` round-robin**, not (only) from divergence — each step
  perturbs a *different* layer with its own δW norm. PROOF: the dW sequence `28,38,37,46,51…` is identical
  at lr=2e-5 and lr=2e-3. Compare dW only at the **same layer** across cycles before calling divergence.
- **The held-out KL probe is noisy (~±0.03)** because it re-runs the nondeterministic NP-decode. To rank
  LRs cleanly, either run ≥100–200 steps (cumulative signal > noise) or replace it with a **deterministic
  teacher-forced NLL/KL on a larger fixed set**.

### 8. Recommendation / open items
- **For a clean, demonstrably-training config:** grpo `(L_q−mean)/σ` at **lr=3e-2** (drop `/std`).
- **If keeping `/std/σ`** (current code): no tested LR trains stably past ~30 steps; needs either a hard
  std floor (`std.clamp_min(~0.05)`) or **global** (batch-level, not per-token) standardization to stop the
  `1/std` blow-up — untested.
- **Before any further LR pick:** add the deterministic teacher-forced loss probe; the current KL probe's
  noise was the single biggest obstacle to ranking LRs this session.

**Code touched:** `verl/verl/trainer/np/{grad_estimator.py,ray_trainer.py}`,
`verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`,
`verl/verl/trainer/config/np_trainer.yaml`, `verl/verl/trainer/es/task_utils.py`,
`verl/verl/trainer/zo_np/grad_check.py` (new), `scripts/zo_opd/{zo_np.sh,zo_np_train.sh}` (new).

**wandb** (`zo_opd_qwen4b_1p7b`): `/std/σ` — a1rmd3vt(2e-5), l0gsgnc6(6e-5), ul4tt5n3(2e-4), pz36he7i(6e-4),
6bjqk1a7(2e-3), mt9un3ge(2e-4 b8), bkbs4fms(6e-4 b8). `(L_q−mean)/σ` — h4hk3tex(3e-2) and siblings.
