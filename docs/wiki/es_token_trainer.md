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

**Reading.** es_token closes essentially all of NP's *engineering* overhead: decode sits at the memory-bound rail floor (≈7.8 ms/token-step for 36 rows × 28 layers), teacher scoring is now *faster* than BP's, and assembly is at parity with BP's backward+optimizer. The entire remaining 2.3× lives in decode and is structural, not glue: the hand-driven serial token loop at pack 4 yields ~511 tok/s vs vLLM continuous batching's 1134 tok/s, with each token running 1+N=9 rows. Next levers, in order: `pack_width` 8+ (teacher on its own GPU, or shrink the full-ctx scratch-KV reservation), overlap teacher prefill with the next wave's decode (minor now), larger N (rails ~free to ~16). Per-probe gradient quality is at the rank-1 weight-probe information bound (§3) — learning-quality runs (LR sweep first) are the open follow-up.

## 5. Known issues / gotchas

- **Teardown hang (post-measurement only):** after the final step + eval, the driver can hang in cleanup with the engine actor spinning (observed on the bench run; the 2-step smoke exited cleanly). Step metrics are logged before teardown, so measurements are unaffected; kill the process tree if it lingers. Suspect: vLLM V1 in-process engine (`uni` executor) + `ray.kill` interaction. Untriaged.
- The trainer-side `scales` RPC sends **raw** rail differences; 1/σ_l is applied per layer inside `es_assemble_and_apply` — don't double-divide.
- `iw_clamp=10` caps the importance weight (σ small → ratios ≈1; the clamp is a safety rail).
- LR scale: bench config `lr=1e-3`, `token_agg=mean` gives update-norm ≈ 0.3% of ‖W‖ per step; an LR sweep (NP lesson: all-layer needs ~30× below single-layer) is REQUIRED before any learning-quality claim. Learning-rate/quality runs are follow-up — this page's claim is the wall-clock + correctness result.
