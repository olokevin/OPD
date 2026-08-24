# es_token Trainer — per-token weight-perturbation ES for OPD

> Zeroth-order OPD trainer that applies a **fresh rank-1 weight perturbation at every decode token** on **N parallel rails riding the clean rollout's KV**, inside one fully-CUDA-graphed packed forward. Weight-space sibling of the NP V3 trainer ([zo_np_trainer.md](zo_np_trainer.md) §11) built to delete NP's three measured cost centers — and it does: **one-step 145.8 s vs NP V3's 2472 s (17×)** at batch 64×1024, N=8, ALL 112 decoder linears. Branch `feat/es-token-trainer`. Design/plan: [plans/es_token_trainer.md](../plans/es_token_trainer.md).

## 1. One-line mechanism

Per decode token `t`, slot `p`, rail `n` (rail 0 clean), at **every** matched linear `l`:

```
y = W_l x                                        (all 1+N rows, resident weights)
y[rail n] += sigma_l * ((r_n ⊙ v_t)ᵀ x) * (s_n ⊙ u_t)     -- rank-1 ΔW, never materialized
```

- `s_n, r_n`: **fixed Hadamard sign rails** (tiled `H_16` rows 1..N × fixed random column flip; with Rademacher `(u,v)` the N rail directions are *exactly* Frobenius-orthogonal per token).
- `(u_t, v_t)`: **one fused noise draw per (slot, token)** covering all layers (flat `noise_buf [bucket, Σ_l(d_out+d_in)]`, ≈0.92 M values for Qwen3-1.7B; layers read fixed views). Seeded by `es_token_seed(global_seed, t, rollout_id)` — never stored, regenerated bit-identically at assembly (NP invariant).
- `sigma_l`: a `[1]` **device tensor multiplied in-graph** (graph input — a baked python float would freeze capture-time σ into the graph; this also makes the σ=0 gate reuse the same graph).
- Rail rows write KV to slot −1 (discarded); only the clean token commits. Decode skeleton (bucketed capture, pad-row EOS, per-row attn metadata, scratch-KV carving) inherited from NP V3 by subclassing.

**Loss (sampled-token OPD, the benchmark variant).** `ℓ_t = log p_t(ŷ_t) − log q_t(ŷ_t)`, `ŷ_t ~ p_t`. Rail evaluation must keep the teacher rail-coupled — unweighted, the teacher term cancels in any rail baseline-difference and the update degenerates to teacher-free `∇log π(ŷ_t)` (zero-mean). Implemented rail loss = **importance-weighted single-sample KL estimate**:
`l_{n,t} = (π_n(ŷ_t)/π_0(ŷ_t)) · (log π_n(ŷ_t) − log q(ŷ_t))`, `E[l_{n,t}] = KL(π_n ‖ q_t)` exactly. Per-token storage = each rail's logprob of the clean token (`[B,T,1+N]` fp32, gather+logsumexp on device, one D2H per wave); teacher phase = **one prefill per rollout** reading `log q(ŷ_t)` via `prompt_logprobs=1`. The CPU suite demonstrates the `weight_mode=none` cancellation explicitly.

**Estimator.** `δW_l = (1/(N·σ_l)) Σ_{t,n} (l_{n,t} − mean_m l_{m,t}) (s_n⊙u_t)(r_n⊙v_t)ᵀ`, descent `W ← W − lr·δW`. Unbiased identity contraction for Rademacher/Gaussian `(u,v)`; 1/σ applied **per layer** so `sigma_mode=relative` stays unbiased. Assembly = chunked GEMMs from seed-regenerated noise (N GEMMs per chunk per layer, fp32 accumulators) — **no per-token Python**.

## 2. File map

| Path | Role |
|---|---|
| `verl/verl/trainer/es_token/{signs,seeding,grad_estimator}.py` | Hadamard rails, fused token-noise seeding/layout, losses/scales/assembly math (pure, CPU-tested) |
| `verl/verl/trainer/es_token/ray_trainer.py` | `RayESTokenTrainer(RayNPTrainer)` + `SampledTokenTeacher`; fit = decode waves → teacher prefill → scales → assemble RPC → per-layer NCCL broadcast; logs `train/{decode,teacher,assemble}_s` |
| `verl/verl/trainer/es_token/rail_kernel.py` | Fused Triton rank-1 rail op — ONE launch per layer (was ~14 PyTorch kernels); see §7 |
| `verl/verl/trainer/es_token/noise_kernel.py` | Direct Rademacher ±1 fill in the destination dtype (Philox counter mode); shared by decode and assembly; see §8 |
| `scripts/zo_opd/launch_zo_opd_q34b_1p7b.sh` | Sequential BP-OPD then ZO-ES-token training launcher (wandb `zo-opd-q34b-1p7b`) |
| `verl/verl/workers/rollout/vllm_rollout/es_token_worker_extension.py` | `ESTokenLinear` + `WorkerExtension(NPWorkerExtension)`: install/capture/replay/eager-oracle/orchestrator/`es_assemble_and_apply` |
| `verl/verl/trainer/main_es_token.py`, `config/es_token_trainer.yaml`, `scripts/zo_opd/opd_es_token.sh` | entry / config / launcher |
| `scripts/zo_opd/es_token_checks/{check_es_parity,check_es_grad_cosine}.py`, `bench_es_token_vs_bp.sh` | gates + headline bench |
| `verl/tests/es_token/` | 19 CPU tests |

Zero edits to ES/NP/PPO paths (NP is imported/subclassed, never modified).

## 3. Verification — all gates PASS (2026-06-09, GPUs 2/3, Qwen3-1.7B, 112 layers)

Full record: `scripts/zo_opd/results/es_token_gates.txt`.

| Gate | Result |
|---|---|
| CPU suite (19) | PASS — incl. exact rail orthogonality, IW unbiasedness (enumerated E == KL), teacher-cancellation degeneracy of `none`, GEMM==naive assembly, `ESTokenLinear` == dense `(W+ΔW)x` |
| σ=0 routing | PASS — graphed packed clean tokens == **stock greedy** `llm.generate`, 4/4 prompts, all 112 layers wrapped |
| graphed vs eager oracle (σ>0) | PASS — clean tokens bit-for-bit, payload max \|diff\| **0.000e+00** |
| staggered-EOS | PASS — `force_stop_at=[3,6,12,12]` bit-for-bit; pad rows don't corrupt active slots |
| cosine vs autograd | PASS at **0.98× the information bound** `sqrt(K/(K+d_out·d_in))`: cos +0.0136 measured vs 0.0138 theory @ K=2400 (N=16×150 reps, down_proj 2048×6144); √K scaling verified. The ~40× per-probe gap to NP (0.205 @ K=400) **is exactly the d_in factor** of probing weight space instead of output space — recovered at training scale: K=B·T·N≈5.2e5/step → predicted per-layer cos ≈ 0.20 |
| trainer e2e | PASS — 2-step smoke: 112 layers `dW>0`, `weight_sync_ok=1.0`, probe/eval paths exercised |

## 4. The headline benchmark — one full generation+update step

`bench_es_token_vs_bp.sh`, batch=64 prompts × max_tokens=1024, greedy, N=8 rails, Qwen3-1.7B student + Keven16 Qwen3-4B teacher co-located on ONE GPU (96 GB), `pack_width=4` (16 waves), ALL 112 decoder linears updated. Results: `scripts/zo_opd/results/es_token_vs_bp.txt`.

| Phase | NP V3 (measured, wiki §11.4) | **es_token (measured)** | BP-OPD (same-box, step-1) | why es_token wins vs NP |
|---|---|---|---|---|
| decode / generation | 1368 s | **128.3 s (10.7×)** | 11.0 s (gen) | no per-rail RNG (one fused draw/slot/token vs 896 `draw_noise`/token = 74% of NP decode), no per-token D2H captures, no per-token full sync; steady 7.9 s/wave |
| teacher score | ~250 s | **4.2 s (60×)** | 41.7 s (reward) | sampled-token = one `prompt_logprobs` prefill per rollout; **10× faster than BP's own teacher phase** |
| gradient + update | 835 s (assemble) | **13.2 s (63×)** | 15.0 s (log_prob + update_actor) | chunked GEMMs from seed-regenerated noise; no per-token Python loop; **at parity with BP's backward** |
| **one-step total** | **2472 s** | **145.8 s (17×)** | **62.7 s** | |
| peak GPU mem | 82,372 MiB | 85,129 MiB | 71,710 MiB | |

**Ratio es_token / BP = 2.33× (cold step-1; ≈5.3× vs the prior A800 steady-state 27.5 s) — down from NP V3's 46×/90×.** 65,536 token-records (64×1024, no early EOS), `weight_sync_ok = 1.0`. Full table: `scripts/zo_opd/results/es_token_vs_bp.txt`.

**Reading.** es_token closes essentially all of NP's *engineering* overhead: decode sits at the memory-bound rail floor (≈7.8 ms/token-step for 36 rows × 28 layers), teacher scoring is now *faster* than BP's, and assembly is at parity with BP's backward+optimizer. The entire remaining 2.3× lives in decode and is structural, not glue — and the [2026-08-21 profile](../results/zo_opd.md#2-decode-throughput--clean-decode-only-vs-clean--n-parallel-perturbed-rails) pins down which part: the hand-driven graphed loop is **not** the cost (clean-only decode is 2.939 ms/token-step vs stock vLLM's own graphed decode at 2.831 ms **at the same concurrency**, a 4% overhead), so the earlier "~511 vs 1134 tok/s" reading was a *concurrency* gap, not a driver gap. What costs is (a) running 1+N=9 rows per token (2.6× the clean per-token-step) and (b) being limited to 4 concurrent sequences where BP runs 64. Next levers, in order, now quantified: **(1) `pack_width`** — 4→8 buys 1.80× clean throughput for +11% per-step cost, and 16 fails only on the full-context scratch-KV reservation (2560 blocks/slot regardless of the real 1024-token budget), so shrinking that reservation is the single highest-leverage fix; **(2) the rank-1 rail op itself** — 60% of the per-token-step cost lives in the 112 wrapped linears (the fused noise draw is only 4%); **(3) larger N** — rails cost just +0.10 ms each, so N=32 gives 4× the probes (2× cosine) for 1.24× the time, though it does not help wall-clock. Per-probe gradient quality is at the rank-1 weight-probe information bound (§3; re-confirmed at 0.86–0.99× the bound across two layer shapes, and rails ≡ repeats at equal K) — learning-quality runs (LR sweep first) are the open follow-up.

## 5. Known issues / gotchas

- **`train/L_clean_mean` is NOT a learning curve.** It is scored on whatever 64 prompts that step
  drew and swings 0.23–3.4 batch to batch on MATH lv3–5 — LRs spanning 100× produce
  indistinguishable curves on it. `dW_norm_mean` is not a divergence signal either (it is the
  gradient-estimate norm *before* the LR multiplies it). Use `eval/heldout_clean_loss`, the fixed
  16-prompt probe, which is logged only every `EVAL_INTERVAL` steps. Its noise floor is ±8% —
  three reads of the same frozen model gave 0.1908 / 0.2126 / 0.2242. See
  [results/zo_opd.md §9](../results/zo_opd.md).
- **The shipped `lr: 1e-3` degrades the model at TEMPERATURE=1.0** (probe KL 0.22 → 0.56 → 1.16,
  MATH-500 5% → 1.5% → 0% over 50 steps). It was calibrated on the greedy benchmark where
  `dW_norm_mean` ≈ 240; at T=1.0 the importance weights spread and it is ≈866 at step 0, ~3.6×
  larger before any LR applies. Use **1e-4** — the largest LR measured not to degrade. T=1.0 itself
  is required: the `student_iw` loss is unbiased for `KL(π_n‖q)` only when the clean token is
  *sampled* from π₀.

- **Teardown hang (post-measurement only):** after the final step + eval, the driver can hang in cleanup with the engine actor spinning (observed on the bench run; the 2-step smoke exited cleanly). Step metrics are logged before teardown, so measurements are unaffected; kill the process tree if it lingers. Suspect: vLLM V1 in-process engine (`uni` executor) + `ray.kill` interaction. Untriaged.
- The trainer-side `scales` RPC sends **raw** rail differences; 1/σ_l is applied per layer inside `es_assemble_and_apply` — don't double-divide.
- `iw_clamp=10` caps the importance weight (σ small → ratios ≈1; the clamp is a safety rail).
- LR scale: bench config `lr=1e-3`, `token_agg=mean` gives update-norm ≈ 0.3% of ‖W‖ per step; an LR sweep (NP lesson: all-layer needs ~30× below single-layer) is REQUIRED before any learning-quality claim. Learning-rate/quality runs are follow-up — this page's claim is the wall-clock + correctness result.

## 6. Profiling record (2026-08-21)

Standalone profile of the three axes, on H100 NVL, filed in
[results/zo_opd.md §ZO-ES-token](../results/zo_opd.md). Harnesses live in
`scripts/zo_opd/es_token_checks/`; raw records in `scripts/zo_opd/results/`.

| Harness | What it measures | Headline |
|---|---|---|
| `sweep_grad_cosine.sh` | cos(es dW, autograd) vs probe count K, two layer shapes | 0.86–0.99× the `sqrt(K/(K+d_out·d_in))` bound; rails ≡ repeats at equal K |
| `bench_decode_throughput.py` + `sweep_decode_throughput.sh` | ms/token-step (T=64→320 slope) for clean-only (N=0) and N=1..32, plus `pack_width` | clean-only within 4% of stock vLLM at equal concurrency; +0.10 ms/rail; pw 4→8 = 1.80× |
| `sweep_stock_batch.sh` | stock vLLM tok/s vs concurrency, eager **and** cudagraph | 1,413 → 13,975 tok/s for B=4 → 64; eager stock is 3× pessimistic (a measurement trap) |
| `sweep_decode_isolation.sh` | `ES_BENCH_SKIP_NOISE` deltas | 36% bare decode / 4% noise / 60% rail compute |
| `bench_es_token_vs_bp.sh` | one full OPD step vs BP-OPD | 147.54 s vs 61.86 s = 2.39× (June's 2.33× reproduced) |

## 7. Fused rail kernel (2026-08-22) — the fixed rail overhead, removed

The 2026-08-21 profile found a cost shaped like launch overhead, not arithmetic: turning rails on at
all cost **+3.41 ms/token-step** (N=0 → N=1) while 31 further rails cost only +3.08 ms. Diagnosis
(full record: [results/zo_opd.md §6](../results/zo_opd.md), raw
`scripts/zo_opd/results/es_token_rail_op.txt`):

- **Not the RNG.** `draw_noise` already draws on the GPU (`torch.Generator(device=cuda)`); the fused
  per-token draw costs 0.199 ms and is paid at N=0 too, so it is not in the delta.
- **It is CUDA-graph node count.** The PyTorch rail branch issues **14 kernels per layer** — six of
  them gathers of operands (`R[rail]`, `v[pidx]`, `S[rail]`, `u[pidx]`) that depend only on the
  token, plus a strided-view penalty because `noise_buf[:, off:off+d]` has row stride `d_total`.
  At 112 layers that is **1568 graph nodes per decode token at ~2.3 µs each = the 3.4 ms**.

`rail_kernel.py` collapses the per-layer op into **one Triton launch**: one program per perturbed
row, which reads its `(rail, slot)` from the index tensors and forms the sign-modulated noise on the
fly. Because it addresses rows through `perturbed_row_idx` it needs **no change to the packed row
layout** (none of the NP-inherited attention/KV metadata is touched) and never materialises a
`[P, d_total]` sign×noise buffer. Tuned to `BLOCK_IN = BLOCK_OUT = 4096, num_warps = 16` — the grid
is only `bucket × n_sample` programs, so the kernel is latency-bound and large blocks beat occupancy.

| | before | after |
|---|---|---|
| rail op alone, N=1 | 3.376 ms | **0.313 ms** (10.8×) |
| decode ms/token-step, N=1 | 6.347 | **3.424** (rail overhead 3.408 → **0.481**) |
| decode ms/token-step, N=8 | 7.600 | **3.885** |
| **one OPD step** (64×1024, N=8) | **147.54 s** | **89.59 s** (1.65×) |
| — decode | 129.59 s | 72.00 s |
| **ratio vs BP-OPD** (61.86 s) | 2.39× | **1.45×** |

All gates re-run and still green: 19/19 CPU tests, `check_rail_op_parity.py` ≤3.0e-06 vs the shipping
op, and on GPU σ=0 ≡ stock greedy / graphed ≡ eager bit-for-bit / staggered-EOS bit-for-bit. The
step's `L_clean_mean` is bit-identical to the pre-optimisation run (the clean trajectory is
untouched); `dW_norm_mean` shifts 239.726 → 239.514 because the rail now accumulates in fp32.

Remaining: the noise draw (0.199 ms, an int64 `randint` plus a five-kernel cast chain, also paid per
token during assembly) and `pack_width`, which is now unambiguously the dominant wall-clock lever.

## 8. Direct Rademacher noise (2026-08-22b)

§7 left the per-token noise draw as the last fixed decode cost. `draw_noise("bernoulli")` built an
**int64** `randint` buffer (7.34 MB per slot at `d_total` = 917,504) and ran a five-kernel
cast/scale/copy chain — ~42 MB of traffic and ~6 kernels per slot per token for 1.83 MB of ±1 — plus
a fresh `torch.Generator` per slot and a host blake2b inside the decode loop. It was also called once
per token record during assembly (65,536 per step).

`noise_kernel.py` draws ±1 **directly, in the destination dtype**, one Triton launch per batch of
rows, from counter-based Philox (`tl.randint` → low bit → ±1): no generator state, no host RNG, no
intermediate buffer. `build_seed_table` hoists every (token, slot) seed out of the decode loop, and
assembly now fills a whole chunk in one launch. A torch fallback (`Tensor.random_(0, 2)` straight
into the destination) covers non-Triton environments and `sample_method != "bernoulli"`; the
implementation is chosen once at import so **decode and assembly can never disagree within a run**.

| | before | after |
|---|---|---|
| decode fill, one token × 4 slots | 0.203 ms | **0.015 ms** (13.5×) |
| assembly fill, one 1024-row chunk | 38.9 ms | **2.9 ms** (13.4×) |
| decode ms/token-step, N=1 | 3.424 | **3.244** |
| decode ms/token-step, N=8 | 3.885 | **3.722** |
| assemble phase | 13.56 s | **11.03 s** |
| **one OPD step** | 89.59 s | **83.80 s** |
| **ratio vs BP-OPD** | 1.45× | **1.35×** |

`check_noise_parity.py` gates the invariant the design rests on and **all pass**: the decode path
(per-wave seed table) and the assembly path (host-derived, freshly uploaded seeds) produce
byte-identical noise; chunk row *j* equals its own (t, rollout) record; values are exactly {−1, +1}
and zero-mean; noise is distinct across *t* and rollout; regeneration is bit-identical. The 19 CPU
tests and all three GPU gates still pass, and `L_clean_mean` is unchanged.

This **changes the noise stream** relative to earlier runs (Philox counter mode ≠ `torch.randint`);
nothing depended on the old stream and both consumers moved together. Visible only as
`dW_norm_mean` 239.51 → 243.78.

**Cumulative across §7 + §8**: one OPD step **147.54 → 83.80 s (1.76×)**, decode 129.59 → 68.44 s,
ES/BP 2.39× → **1.35×**. Decode is now 82% of the step and `pack_width` is the only lever of
consequence left.

## 9. Budget-sized scratch-KV (2026-08-23) — pack_width unlocked

`_np_prefill_packed` reserved each packed slot a KV slice of the full `max_model_len`
(ceil(40960/16) = 2560 blocks) against a 24,717-block pool — ~20× what a 1024-token generation needs,
and the reason `pack_width` was capped at 9 slots. It now reserves (longest prompt + `max_tokens`);
`max_new_tokens=None` keeps the old behaviour for the NP trainer. Because the attention block table is
zero-filled and only `len(block_ids)` entries are written, an undersized slice would read block 0 and
silently corrupt a *neighbour's* KV, so a second assert makes that unreachable.

Gating this needed care: comparing packed output to stock `llm.generate` fails at pack_width ≥ 10, but
that is bf16 rounding, not corruption — a slot's output is byte-identical when the *content* of every
other slot is swapped, it varies with wave width alone, and it shows up at pack_width=4 too where the
change is provably byte-neutral. The gate checks output-neutrality vs the old reservation
(across processes), neighbour-independence, and full-length generation instead.

| `pack_width` (N=8) | ms/token-step | clean tok/s | waves per 64-prompt batch |
|---|---|---|---|
| 4 | 3.734 | 1,071 | 16 |
| 16 | 5.435 | 2,944 | 4 |
| **64** | 15.036 | **4,257** | **1** |

One OPD step (64 × 1024, N=8): **83.80 → 42.67 s**, decode 68.44 → 25.40 s.
**Cumulative over §7–§9: 147.54 → 42.67 s (3.46×), decode 129.59 → 25.40 s (5.10×), and the ratio vs
BP-OPD goes 2.39× → 0.69× — es_token is now faster than BP.** Assembly (13.2 s, 31%) is the next
lever, not decode.
