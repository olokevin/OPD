# ZO-ES-token (per-token weight-perturbation ES) OPD — results

Student `Qwen/Qwen3-1.7B`, teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, co-located on one GPU.
Per decode token a **fresh rank-1 weight perturbation** `ΔW_l = σ_l (s_n⊙u_t)(r_n⊙v_t)ᵀ` is applied to **all
112 decoder linears** on `N` parallel rails riding the clean rollout's KV, inside one fully-CUDA-graphed
packed forward. Loss = importance-weighted sampled-token KL to the teacher; gradient assembled by chunked
GEMMs from seed-regenerated noise. Trainer `verl/verl/trainer/es_token/`, driver
`scripts/zo_opd/opd_es_token.sh`, branch `feat/es-token-trainer`.
Design: [../plans/es_token_trainer.md](../plans/es_token_trainer.md) · subsystem page:
[../wiki/es_token_trainer.md](../wiki/es_token_trainer.md).

## Session summary

Wall-clock is settled and positive. Learning is settled too as of 2026-08-25/26 (§11): **BP-OPD
learns on the fixed setting, es_token does not.** Each row is one session below.

| Session               | What it records                                                                                                                                                                                                                                                                                                                                                                   | One OPD step (batch 64 × 1024, N=8)                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **2026-06-09**  | Build + all 5 gates PASS (σ=0 ≡ stock greedy, graphed ≡ eager bit-for-bit, staggered-EOS, 19 CPU tests, e2e smoke). First headline vs NP-V3 (2472 s) and BP-OPD (62.7 s).                                                                                                                                                                                                      | **145.8 s** — ES/BP 2.33×                              |
| **2026-08-21**  | Profiling. Gradient cosine sits at**0.86–0.99× the rank-1 weight-probe bound** `sqrt(K/(K+d_out·d_in))`; rails ≡ repeats at equal K. Clean decode through the custom graphed driver is only **4% over stock vLLM** at equal concurrency — the residual gap is *concurrency*, not driver overhead.                                                            | **147.5 s** — 2.39× (reproduces June within 1.2%)      |
| **2026-08-22**  | The fixed `N=0→1` rail cost (+3.41 ms) was **CUDA-graph node count** (1568 nodes/token at ~2.3 µs), not the RNG. Fused Triton rail kernel → rail op 10.8× cheaper.                                                                                                                                                                                                    | **89.6 s** — 1.45×                                     |
| **2026-08-22b** | Rademacher noise drawn**directly in the destination dtype** via a Triton Philox kernel, shared by decode and assembly (gated byte-for-byte); fill 13.5× cheaper.                                                                                                                                                                                                           | **83.8 s** — 1.35×                                     |
| **2026-08-23**  | Scratch-KV reserved to (prompt +`max_tokens`) instead of `max_model_len` → `pack_width` 4→64, a 64-prompt batch in **one wave**. Cumulative 3.46× on the step, 5.10× on decode.                                                                                                                                                                                   | **42.7 s** — 0.69× vs BP's *cold* step (§10 corrects this) |
| **2026-08-23b** | First training runs. Shipped `lr=1e-3` **degrades** the model (probe KL 0.22→1.16, MATH-500 5%→0%). Measurement trap: `train/L_clean_mean` is data-driven noise and cannot rank LRs; only the fixed 16-prompt heldout probe can, and its floor is ±8%.                                                                                                               | —                                                             |
| **2026-08-23c** | **NEGATIVE.** 150 steps at `lr=1e-4`: probe 0.2126→0.2228, entirely inside the noise floor; MATH-500 shows no trend. Bracket is 1e-3 destroys / 1e-4 does nothing, with no recipe found between. **Not ES-specific** — the BP-OPD baseline was equally flat, implicating the setup (every rollout hits the 1024-token cap without EOS) rather than the algorithm. | 37.2 s/step, 92.9 min total                                    |
| **2026-08-25/26** | **The setting where BP-OPD learns.** Student `Qwen/Qwen3-1.7B` (non-thinking) ← `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, NERSC hyperparameters verbatim. The old "neither learns" result was `enable_thinking` defaulting **true** on a hybrid Qwen3 student, so every rollout hit the cap mid-`<think>`. Apples-to-apples n=8 eval: **BP up on 4/4** benchmarks (MATH-500 0.7250→0.7532, AMC23 0.3931→0.4172), **es_token inside noise on all four** (`RMS(dW)` only 0.18%). BP tracks the 8-GPU reference exactly to step 60 then plateaus at the 3072 cap. Ships `fp32_master`, es_token HF checkpointing, `enable_thinking`, prompt-length filter, `teacher_max_model_len`. §11 | BP 126.0 s vs ES 130.7 s/step (4.2× per sequence) |
| **2026-08-24** | **CORRECTION.** The BP reference every ratio above was divided by (61.86 s) is BP's **cold step 1**. `compute_rm_score` is 29.15 s at step 1 and **3.80 s median over the next 137 steps**; a from-scratch microbenchmark of the reward path predicts 3.03 s. So BP's teacher phase was never slow, and the honest verdict is **ES/BP = 1.48×**, not 0.69×. | ES 37.17 vs BP **25.11 s** steady |

**Bottom line (updated 2026-08-26, see §11):** the "neither method learns" verdict below was a
**setup** artefact — `enable_thinking` defaulted to true on a hybrid Qwen3 student, so every rollout
hit the token cap mid-`<think>`. On the fixed setting **BP-OPD learns** (up on 4/4 benchmarks,
MATH-500 0.7250 → 0.7532) while **es_token stays inside noise** over 200 steps, with `RMS(dW)` at
only 0.18% of typical weight magnitude. Step time is now near parity per step (126.0 vs 130.7 s)
though still 4.2× per sequence. The open question is no longer wall-clock: it is that es_token's
**training-time decode inflates response length 1397 → 3024 while the trained checkpoint generates
exactly what the base model does** — i.e. it may be estimating its gradient on off-distribution
trajectories (§11.7).

## Wall-clock: every es_token variant vs BP-OPD

One full OPD step, identical config throughout — batch 64 × 1024 tokens, N=8 rails, all 112 decoder
linears, Qwen3-4B teacher co-located with the Qwen3-1.7B student on one H100 NVL. Seconds.

### The optimisation progression (cold step-1 bench)

| variant | decode | teacher | grad + update | **step** |
|---|---:|---:|---:|---:|
| ES v0 — PyTorch rails, `randint` noise, `pack_width=4` | 129.59 | 4.22 | 13.65 | **147.54** |
| ES + fused Triton rail kernel (§6) | 72.00 | 3.95 | 13.56 | **89.59** |
| ES + direct Rademacher noise (§7) | 68.44 | 4.23 | 11.03 | **83.80** |
| ES + budget-sized scratch-KV, `pack_width` 4→64 (§8) | **25.40** | 3.99 | 13.18 | **42.67** |

Cumulative: step **147.54 → 42.67 s (3.46×)**, decode **129.59 → 25.40 s (5.10×)**. These rows are
cold-vs-cold on one harness, so they measure the optimisations against each other correctly.

### The ES-vs-BP verdict (steady state)

Earlier revisions of this page divided every row above by a **single cold BP step (61.86 s)** and
concluded es_token had overtaken BP. It has not — §10 shows 25 s of that 61.86 was one-time teacher
warm-up. Both columns below are **medians over a full run**, same launcher
(`scripts/zo_opd/launch_zo_opd_q34b_1p7b.sh`), same batch 64 × 1024, T=1.0, same GPU:

| phase | ES-token (146 steps) | BP-OPD (135 steps) | ES / BP |
|---|---:|---:|---:|
| decode / `gen` | 23.06 | 9.85 | 2.34× |
| teacher scoring | 3.10 | 3.80 (`rm_score`) | 0.82× |
| grad + update | 10.89 (assemble) | 9.63 (`log_prob` + `adv` + `update_actor`) | 1.13× |
| **one step** | **37.17** | **25.11** | **1.48×** |

**Reading it by phase:**

- **decode 23.06 vs 9.85 s (2.34×)** — the only phase where ES is materially behind, and it is a
  *concurrency* cost, not driver overhead: clean decode through the graphed packed driver is within
  4% of stock vLLM at equal concurrency (§2), and each rail after the first adds only
  ~0.10 ms/token-step (§2, §6.4).
- **teacher 3.10 vs 3.80 s** — ES is ~20% cheaper because it scores one sampled token per position
  against BP's full top-K machinery, but the two are the same order. The "8–10× cheaper" claim in
  §1/§4 came from the cold BP number and is wrong; see §10.
- **grad + update 10.89 vs 9.63 s** — chunked-GEMM assembly from seed-regenerated noise costs about
  what BP's `log_prob` + backward + optimizer step costs. It is 29% of the ES step.

Peak memory: ES 85,129 MiB vs BP 71,710 MiB. Cold-start asymmetry is the reason the two framings
disagree: ES pays a 13% cold penalty (42.67 → 37.17 s) because its teacher is the already-warm vLLM
engine, while BP pays 146% (61.86 → 25.11 s), almost all of it inside the teacher's first forward.

---

## Session 2026-06-09 — build, gates, headline one-step (collected)

Full records: `scripts/zo_opd/results/es_token_gates.txt`, `scripts/zo_opd/results/es_token_vs_bp.txt`.

| Gate                                           | Result                                                                                                                                                                                                            |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU suite (`verl/tests/es_token/`, 19 tests) | PASS — exact rail orthogonality, IW unbiasedness (enumerated `E[l] == KL`), `weight_mode=none` teacher-cancellation degeneracy, chunked GEMM == naive outer-product, `ESTokenLinear` == dense `(W+ΔW)x` |
| σ=0 routing                                   | PASS — graphed packed clean tokens ==**stock greedy** `llm.generate`, 4/4 prompts, all 112 layers wrapped                                                                                                |
| graphed vs eager oracle (σ=0.01)              | PASS — clean tokens bit-for-bit, payload max abs diff**0.000e+00**                                                                                                                                         |
| staggered-EOS (`force_stop_at=[3,6,12,12]`)  | PASS — bit-for-bit; pad rows do not corrupt active bucket-mates                                                                                                                                                  |
| trainer e2e smoke (2 steps)                    | PASS — 112 layers `dW>0`, `weight_sync_ok=1.0`                                                                                                                                                               |

Headline one-step (2026-06-09, batch 64 × 1024 greedy, N=8, all 112 linears, co-located 4B teacher):
**145.80 s** = decode 128.31 + teacher 4.24 + assemble 13.17, peak 85,129 MiB, 65,536 token-records —
vs NP-V3 **2472 s** (17×) and BP-OPD **62.72 s** cold (ES/BP = 2.33×).
Per-wave decode was flat at 7.85–8.72 s over the 16 waves.
The BP side is a **cold step-1** and the 2.33× is not the trainer-vs-trainer ratio — see §10.

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

| layer                         | shape      | `d_out·d_in` | N  | repeats | K    | cos     | bound  | cos/bound      |
| ----------------------------- | ---------- | --------------- | -- | ------- | ---- | ------- | ------ | -------------- |
| `layers.0.mlp.down_proj`    | 2048×6144 | 12.6 M          | 8  | 50      | 400  | +0.0056 | 0.0056 | **0.99** |
| `layers.0.mlp.down_proj`    |            |                 | 8  | 300     | 2400 | +0.0130 | 0.0138 | 0.94           |
| `layers.0.mlp.down_proj`    |            |                 | 16 | 150     | 2400 | +0.0136 | 0.0138 | **0.98** |
| `layers.0.mlp.down_proj`    |            |                 | 32 | 150     | 4800 | +0.0193 | 0.0195 | **0.99** |
| `layers.0.self_attn.o_proj` | 2048×2048 | 4.2 M           | 8  | 50      | 400  | +0.0084 | 0.0098 | 0.86           |
| `layers.0.self_attn.o_proj` |            |                 | 16 | 150     | 2400 | +0.0222 | 0.0239 | 0.93           |

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

| decode path                         | rows/token | ms/token-step   | clean tok/s      | row-steps/s |
| ----------------------------------- | ---------- | --------------- | ---------------- | ----------- |
| stock vLLM, B=4, cudagraph          | 4          | 2.831           | 1412.7           | —          |
| **es_token N=0 (clean only)** | 4          | **2.939** | **1361.1** | 1361        |
| es_token N=1                        | 8          | 6.347           | 630.2            | 1260        |
| es_token N=2                        | 12         | 6.698           | 597.2            | 1792        |
| es_token N=4                        | 20         | 7.264           | 550.6            | 2753        |
| **es_token N=8 (shipping)**   | 36         | **7.600** | **526.3**  | 4737        |
| es_token N=16                       | 68         | 8.170           | 489.6            | 8323        |
| es_token N=32                       | 132        | 9.429           | 424.2            | 13999       |

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

| pack_width | rows/token | ms/token-step | clean tok/s                                                                                                             |
| ---------- | ---------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 4          | 36         | 7.600         | 526.3                                                                                                                   |
| 8          | 72         | 8.449         | **946.9**                                                                                                         |
| 16         | 144        | —            | **fails**: `packed scratch KV does not fit: b_pack=16 × blocks_per_prompt=2560 = 40960 > num_gpu_blocks=24717` |

Doubling `pack_width` 4→8 buys **1.80× clean throughput for +11% per-step cost**. `pack_width=16` is
blocked not by compute but by the **full-context scratch-KV reservation**: each slot reserves
`ceil(max_model_len/block_size) = 2560` blocks regardless of the actual 1024-token budget. Reserving to
the real response length instead of `max_model_len` is the single highest-leverage fix on this branch.

### 3. Where the per-token-step time goes

`sweep_decode_isolation.sh`, `pack_width=4`; `ES_BENCH_SKIP_NOISE=1` removes only the fused per-token
noise draw.

| configuration        | ms/token-step | delta                                                                  |
| -------------------- | ------------- | ---------------------------------------------------------------------- |
| N=0, no noise draw   | 2.746         | bare graphed decode floor                                              |
| N=0, with noise draw | 2.945         | **+0.199** — fused noise draw (all layers, one draw/slot/token) |
| N=8, no noise draw   | 7.391         | **+4.645** — 112-layer rank-1 rail compute (32 perturbed rows)  |
| N=8, with noise draw | 7.689         | +0.298 — noise draw at N=8                                            |

Attribution at the shipping point: **36% bare decode, 4% noise, 60% rail compute**. The design goal of
making the noise draw negligible is met — it is 4% here versus NP's 896 `draw_noise` calls/token that
were 74% of NP decode. The remaining 60% is the rank-1 op itself in 112 wrapped linears
(gather `x[pri]`, `R[rail]*v[pidx]`, the reduction, and the scatter-add), which is where any further
decode optimisation must go (e.g. fusing the four ops per layer, or batching layers).

**Why this matters for the ES/BP ratio.** Stock vLLM's per-token-step cost is almost flat in
concurrency (2.83 ms at B=4 → 4.58 ms at B=64, all graphed), so its throughput scales nearly linearly
with batch — while the es driver is pinned at 4–8 concurrent sequences by the scratch-KV reservation:

| concurrency B                | stock cudagraph ms/token-step | stock tok/s      |
| ---------------------------- | ----------------------------- | ---------------- |
| 4                            | 2.831                         | 1,413            |
| 8                            | 2.918                         | 2,742            |
| 16                           | 3.322                         | 4,816            |
| 32                           | 3.586                         | 8,924            |
| **64** (the OPD batch) | **4.580**               | **13,975** |

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

| phase                   | **es_token**     | **BP-OPD**                                                  | es_token 2026-06-09 | BP-OPD 2026-06-09 |
| ----------------------- | ---------------------- | ----------------------------------------------------------------- | ------------------- | ----------------- |
| **one step**      | **147.54 s**     | **61.86 s**                                                 | 145.80 s            | 62.72 s           |
| decode / generation     | 129.59                 | 14.58 (`gen`; 8.05 pure `generate_sequences`)                 | 128.31              | 11.02 (7.85)      |
| teacher scoring         | **4.22**         | 35.61 (`reward`; `rm_score` 32.48)                            | 4.24                | 41.71 (35.72)     |
| gradient + update       | 13.65 (assemble+apply) | 13.94 (`log_prob` 2.32 + `adv` 0.14 + `update_actor` 11.48) | 13.17               | 15.16             |
| peak GPU mem            | 85,129 MiB             | 71,710 MiB                                                        | 85,129 MiB          | 71,710 MiB        |
| **ratio ES / BP** | **2.39×**       | —                                                                | 2.33×              | —                |

**The 2026-06-09 headline reproduces.** Step time 147.54 s vs 145.80 s (+1.2%), all three phases within
4%, identical peak memory, and `L_clean_mean`, `dW_norm_max/mean` bit-identical to the June run
(the pipeline is deterministic). `weight_sync_ok = 1.0`; all 112 layers had `dW > 0`.

**The shape of the gap is unchanged and confirmed by §2–§3:**

- **teacher scoring is 8.4× faster than BP's** (4.22 s vs 35.61 s) — the sampled-token loss needs one
  `prompt_logprobs` prefill per rollout, where BP pushes the full top-K machinery through an FSDP
  reward worker. **CORRECTED in §10:** BP's 35.61 s is its *cold* step-1; its steady-state teacher
  phase is 3.80 s, so the real advantage is ~1.2×, not 8.4×.
- **assembly is at parity with BP's backward+optimizer** (13.65 vs 13.94 s). The chunked-GEMM assembly
  is no longer a cost centre (NP's was 835 s).
- **100% of the residual gap is decode**: 129.59 s vs 14.58 s (`gen`), or 8.05 s against BP's pure
  `generate_sequences`. §2 shows this is *not* driver overhead (clean-only decode is within 4% of stock
  vLLM at equal concurrency); it factorises cleanly as
  `2.59× (rails: 1361 → 526 clean tok/s at N=8) × 9.89× (concurrency: stock 1,413 tok/s at B=4 → 13,975 at B=64) × 1.04× (driver) = 26.6×`, exactly the steady-state decode ratio of §2. End-to-end the
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

| kernel class                                     | count          | total                | per call           |
| ------------------------------------------------ | -------------- | -------------------- | ------------------ |
| `vectorized_gather_kernel`                     | 672            | 1207.8 µs           | 1.80 µs           |
| `elementwise_kernel<128,4>` (non-vectorised)   | 224            | 602.9 µs            | 2.69 µs           |
| `vectorized_elementwise_kernel<8>`             | 336            | 545.3 µs            | 1.62 µs           |
| `index_elementwise_kernel` (the `index_put`) | 112            | 530.0 µs            | 4.73 µs           |
| `reduce_kernel` (the `alpha` sum)            | 112            | 508.7 µs            | 4.54 µs           |
| remaining elementwise                            | 112            | 184.1 µs            | 1.64 µs           |
| **total**                                  | **1568** | **3578.8 µs** | **2.28 µs** |

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

| variant                        | idea                                                                                                                     | N=1             | N=8             | N=32            | speedup @N=1      |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------- | --------------- | --------------- | ----------------- |
| `v0_current`                 | shipping PyTorch branch                                                                                                  | 3.376           | 4.646           | 5.225           | 1.00×            |
| `v1_flat`                    | one broadcast kernel builds `sign*noise` for **all** layers into a flat `[P, d_total]`; each layer reads views | 2.373           | 3.581           | 4.276           | 1.42×            |
| `v2_fused`                   | v1 +`vecdot` / `addcmul_` per layer                                                                                  | 1.992           | 3.135           | 3.847           | 1.69×            |
| `v3_contig`                  | v2 +`[clean \| perturbed]` row layout so `x_p`/`y_p` are slices, not gather/scatter                                 | 1.138           | 2.110           | 2.597           | 2.97×            |
| `v4_bmm`                     | v3 + batched GEMV for `alpha`                                                                                          | 0.765           | 1.147           | 1.937           | 4.41×            |
| `v5_blocked`                 | per-layer**contiguous** noise blocks (restores the vectorised elementwise path)                                    | 1.527           | 2.695           | 3.033           | 2.21×            |
| `v6_triton`                  | one fused Triton kernel/layer (needs the v3 layout)                                                                      | 0.491           | 0.874           | 1.225           | 6.88×            |
| **`v7_triton_rowidx`** | fused Triton that**reads the row indices**, so no layout change and no `[P, d_total]` buffer                     | 0.478           | 0.790           | 0.870           | 7.06×            |
| **`v7` tuned**         | `BLOCK_IN=BLOCK_OUT=4096, num_warps=16`                                                                                | **0.313** | **0.472** | **0.560** | **10.79×** |

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

| N                      | before | after           | speedup          | rail overhead vs N=0                    |
| ---------------------- | ------ | --------------- | ---------------- | --------------------------------------- |
| 0 (clean only)         | 2.939  | 2.943           | 1.00×           | —                                      |
| **1**            | 6.347  | **3.424** | **1.85×** | 3.408 →**0.481 ms** (7.1× less) |
| 2                      | 6.698  | 3.462           | 1.93×           | 3.759 → 0.519                          |
| 4                      | 7.264  | 3.651           | 1.99×           | 4.325 → 0.708                          |
| **8** (shipping) | 7.600  | **3.885** | **1.96×** | 4.661 → 0.942                          |
| 16                     | 8.170  | 4.259           | 1.92×           | 5.231 → 1.316                          |
| 32                     | 9.429  | 5.208           | 1.81×           | 6.490 → 2.265                          |
| 8 @`pack_width=8`    | 8.449  | 4.547           | 1.86×           | —                                      |

**The goal — a single rail costing close to clean-only decode, under 3.5 ms/token-step — is met at
3.424 ms**, i.e. +0.481 ms over clean decode instead of +3.408 ms. Clean throughput at the shipping
N=8 rises 526 → 1,030 tok/s.

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher, same GPU and
config as §4):

|                                     | before           | after             | speedup                                |
| ----------------------------------- | ---------------- | ----------------- | -------------------------------------- |
| **step_time**                 | 147.54 s         | **89.59 s** | **1.65×**                       |
| decode                              | 129.59 s         | 72.00 s           | 1.80× (16 waves, 7.85 → 4.25 s each) |
| teacher                             | 4.22 s           | 3.95 s            | —                                     |
| assemble                            | 13.65 s          | 13.56 s           | —                                     |
| peak GPU mem                        | 85,129 MiB       | 85,107 MiB        | —                                     |
| **ratio vs BP-OPD** (61.86 s — cold, see §10) | **2.39×** | **1.45×**  |                                        |

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

|                                   | before   | after              | speedup          |
| --------------------------------- | -------- | ------------------ | ---------------- |
| decode fill, one token × 4 slots | 0.203 ms | **0.015 ms** | **13.5×** |
| assembly fill, one 1024-row chunk | 38.9 ms  | **2.9 ms**   | **13.4×** |

### 7.5 End-to-end

Decode ms/token-step, `pack_width=4`, same protocol as §2:

| N                      | original | + fused rail (§6) | **+ direct noise** | total speedup    |
| ---------------------- | -------- | ------------------ | ------------------------ | ---------------- |
| 0 (clean only)         | 2.939    | 2.943              | **2.783**          | 1.06×           |
| **1**            | 6.347    | 3.424              | **3.244**          | **1.96×** |
| 2                      | 6.698    | 3.462              | 3.329                    | 2.01×           |
| 4                      | 7.264    | 3.651              | 3.481                    | 2.09×           |
| **8** (shipping) | 7.600    | 3.885              | **3.722**          | **2.04×** |
| 16                     | 8.170    | 4.259              | 4.107                    | 1.99×           |
| 32                     | 9.429    | 5.208              | 4.956                    | 1.90×           |
| 8 @`pack_width=8`    | 8.449    | 4.547              | 4.237                    | 1.99×           |

Rail overhead over clean-only decode at N=1: 3.408 → 0.481 → **0.461 ms**.

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher):

|                                     | original | + fused rail | **+ direct noise** | total            |
| ----------------------------------- | -------- | ------------ | ------------------------ | ---------------- |
| **step_time**                 | 147.54 s | 89.59 s      | **83.80 s**        | **1.76×** |
| decode                              | 129.59 s | 72.00 s      | 68.44 s                  | 1.89×           |
| teacher                             | 4.22 s   | 3.95 s       | 4.23 s                   | —               |
| assemble                            | 13.65 s  | 13.56 s      | **11.03 s**        | 1.24×           |
| `n_token_records`                 | 65,536   | 65,536       | 65,536                   | —               |
| `weight_sync_ok`                  | 1.0      | 1.0          | 1.0                      | —               |
| **ratio vs BP-OPD** (61.86 s — cold, see §10) | 2.39×   | 1.45×       | **1.35×**         |                  |

`L_clean_mean` is 0.2556177764199674 in all three runs — the clean trajectory never moved.
`dW_norm_mean` shifts 239.51 → 243.78 with the new noise stream, as expected.

### 7.6 What is left

Decode is now **82% of the step** and the noise fill is down to 0.015 ms (0.5% of a token-step), so
**`pack_width` is the only lever of consequence left**: the full-context scratch-KV reservation
(2560 blocks per slot regardless of the real 1024-token budget) caps concurrency at 4–8 slots while
BP-OPD runs 64. §2 measured stock vLLM at 1,413 tok/s at B=4 against 13,975 at B=64 — that ~9.9×
concurrency deficit is the entire remaining gap.

## Session 2026-08-23 — budget-sized scratch-KV: pack_width unlocked, decode 5.1× faster

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

| reservation basis           | blocks/slot | max slots     |
| --------------------------- | ----------- | ------------- |
| `max_model_len` (old)     | 2,560       | 9             |
| 1024 prompt + 1024 response | 128         | **193** |
| 1024 + 3072 response        | 256         | 96            |

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

| `pack_width` | ms/token-step (N=8) | clean tok/s     | waves for a 64-prompt batch |
| -------------- | ------------------- | --------------- | --------------------------- |
| 4              | 3.734               | 1,071           | 16                          |
| 8              | 4.259               | 1,878           | 8                           |
| 16             | 5.435               | 2,944           | 4                           |
| 32             | 8.431               | 3,795           | 2                           |
| **64**   | 15.036              | **4,257** | **1**                 |

**One full OPD step** (batch 64 × 1024, N=8, all 112 linears, co-located 4B teacher):

|                                     | `pack_width=4` | **`pack_width=64`** | speedup                |
| ----------------------------------- | ---------------- | --------------------------- | ---------------------- |
| **step_time**                 | 83.80 s          | **42.67 s**           | **1.96×**       |
| decode                              | 68.44 s          | 25.40 s                     | 2.69× (16 waves → 1) |
| teacher                             | 4.23 s           | 3.99 s                      | —                     |
| assemble                            | 11.03 s          | 13.18 s                     | —                     |
| **ratio vs BP-OPD** (61.86 s — cold, see §10) | 1.35×           | **0.69×**            |                        |

Cumulative over §6–§8: one step **147.54 → 42.67 s (3.46×)**, decode **129.59 → 25.40 s (5.10×)**.
This session concluded "es_token is now faster than BP-OPD (ES/BP 0.69×)" — **that conclusion is
withdrawn in §10**: the 61.86 s BP denominator is a cold step-1, and against BP's steady-state
25.11 s the ratio is 1.48×.

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
| ---- | ------- | ------- | ------- |
| 0    | 0.22714 | 0.22714 | 0.22714 |
| 1    | 3.195   | 2.850   | 2.844   |
| 3    | 2.865   | 3.375   | 3.347   |
| 5    | 0.279   | 0.227   | 0.225   |
| 7    | 2.651   | 3.220   | 3.099   |

The applied update differs 100× (`lr·dW` ≈ 2.2 vs 0.02) yet the shape is the same, and the same
steps are low in every run — it is **data, not the optimizer**. `dW_norm_mean` is no divergence
signal either: it is the gradient-estimate norm *before* the LR multiplies it, so it is similar
across LRs by construction. Use **`eval/heldout_clean_loss`** — a fixed 16-prompt probe
(`ray_trainer.py:333`), logged only every `EVAL_INTERVAL` steps.

**Probe noise floor.** The probe scores *sampled* rollouts at T=1.0, so it is stochastic even for a
frozen model: three sweep runs read the same untouched step-0 model as 0.1908 / 0.2126 / 0.2242 —
a spread of 0.033, about **±8%**. Nothing smaller than that is interpretable.

### 9.2 LR 1e-3 — the shipped default — degrades the model

|                      | step 0 | step 25 | step 50          |
| -------------------- | ------ | ------- | ---------------- |
| probe KL (fixed 16)  | 0.2244 | 0.5565  | **1.1559** |
| MATH-500 (fixed 200) | 5.0%   | 1.5%    | **0.0%**   |

Monotonic on both, ~2× per interval, far outside the noise floor. Killed at step 50.

The cause is the temperature change: 1e-3 was calibrated on the **greedy** benchmark where
`dW_norm_mean` ≈ 240. At T=1.0 the rails ride a higher-entropy trajectory, the importance weights
spread, and `dW_norm_mean` is ~866 at step 0 and ~2,000–2,550 in steady state — **≈3.6× larger
before any LR is applied**. (T=1.0 is nonetheless required: the `student_iw` rail loss is an
unbiased estimate of `KL(π_n‖q)` only when the clean token is *sampled* from π₀.)

### 9.3 The sweep gives a bound, not a ranking

21 steps, `EVAL_INTERVAL=10`, fixed probe + MATH-500 on 50:

| LR   | probe s0 | s10    | s20    | MATH-500    |
| ---- | -------- | ------ | ------ | ----------- |
| 1e-4 | 0.2126   | 0.1886 | 0.2035 | 8 / 8 / 10% |
| 1e-5 | 0.1908   | 0.2126 | 0.2010 | 8 / 6 / 8%  |
| 1e-6 | 0.2242   | 0.1988 | 0.1909 | 8 / 4 / 10% |

All flat within ±8%; MATH-500 at n=50 has σ ≈ 4pp and carries no signal either. So:
**1e-3 destroys the model, 1e-4 and below do not, and 20 steps cannot separate 1e-4/1e-5/1e-6.**

The 150-step run uses **1e-4** — the largest non-degrading LR, a principled default rather than a
measured optimum. Separating it from 1e-5 needs a horizon long enough for the signal to clear ±8%,
or a lower-variance probe (greedy probe rollouts, or many more probe prompts).

### 9.4 The 150-step run at 1e-4 — negative result

| step                 | 0      | 25     | 50     | 75     | 100    | 125    | 149              |
| -------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ---------------- |
| probe KL (fixed 16)  | 0.2126 | 0.2002 | 0.1987 | 0.2169 | 0.2049 | 0.2157 | **0.2228** |
| MATH-500 (fixed 200) | 6.0%   | 7.0%   | 6.5%   | 5.5%   | 7.0%   | 2.0%   | **4.0%**   |

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

## Session 2026-08-24 — the BP reference was a cold step; the honest ratio is 1.48×

Triggered by the question "why is BP-OPD's teacher phase so slow?". It is not. Every ES/BP ratio on
this page before today divided by **one cold BP step**, and roughly 25 s of that 61.86 s was one-time
warm-up inside the teacher.

### 10.1 The reward path cannot cost 35 s

`RewardModelWorker._forward_micro_batch` (`verl/workers/fsdp_workers.py:2022`) always materialises
full-vocab logits — `use_fused_kernels=False` in this config, and `compute_entropy` is hard-coded
`True` for a logging metric, so `need_logits` is always set. That looked like the culprit. Replaying
one micro-batch (8 seqs × 1112 tok, the shipped `reward.micro_batch_size_per_gpu=8`) on the real
Qwen3-4B teacher says otherwise:

| stage | ms / micro-batch | s / step (×8) | share |
|---|---:|---:|---:|
| transformer fwd + lm_head | 324.7 | 2.60 | 85.8% |
| `logits.div_(teacher_temperature)` | 1.6 | 0.01 | 0.4% |
| `_compute_entropy_safe` (logging only) | 21.1 | 0.17 | 5.6% |
| `logprobs_from_logits` | 1.7 | 0.01 | 0.4% |
| teacher top-K + overlap masks | 29.4 | 0.24 | 7.8% |
| **total** | **378.5** | **3.03** | 100% |

The logits tensor is 8,896 × 151,936 bf16 = 2.52 GiB per micro-batch, but all the full-vocab
reductions together are **0.42 s of a step**. Wrapping the teacher in the FSDP1
`CPUOffload(offload_params=True)` the reward worker hard-wires (`fsdp_workers.py:1883`) changed
nothing either — 333 ms warm, versus 325 ms unwrapped.

### 10.2 It is cold-start, and the 138-step run shows it

`logs/train/bp_opd_20260823_010128.log`, per-step `timing_s/compute_rm_score`:

| step | 1 | 2 | 3 | 4 | … | 138 | median (steps 4+) |
|---|---:|---:|---:|---:|---|---:|---:|
| `rm_score` (s) | **29.15** | 3.82 | 3.86 | 4.82 | | 3.57 | **3.80** |

29.15 s once, then 3.80 s median (min 3.46, max 7.31) — matching the 3.03 s microbenchmark to within
the co-tenancy overhead. Step 1 pays the teacher's first CPU→GPU param fetch, kernel autotune, and
first-touch allocation of the 2.52 GiB logits and 2.49 GiB fp32 entropy buffers into an allocator
that already holds vLLM's 55% reservation. The `es_token_vs_bp.txt` record does label its BP column
"step-1, cold"; the derived headline did not carry the qualifier.

### 10.3 The corrected verdict

Steady-state medians, both runs from `scripts/zo_opd/launch_zo_opd_q34b_1p7b.sh`, batch 64 × 1024,
T=1.0, same GPU, first 3 steps dropped:

| phase | ES-token (146 steps) | BP-OPD (135 steps) | ES / BP |
|---|---:|---:|---:|
| decode / `gen` | 23.06 | 9.85 | 2.34× |
| teacher | 3.10 | 3.80 | 0.82× |
| grad + update | 10.89 | 9.63 | 1.13× |
| **step** | **37.17** | **25.11** | **1.48×** |

- **ES/BP is 1.48×, not 0.69×.** The claim "es_token is faster than BP-OPD" (§8.3) is withdrawn.
- **BP's teacher phase was never a cost centre** — 3.80 s, ~20% *more* than ES's 3.10 s, not 8–10×.
  The "teacher 10× faster" line in §1 and the "8.4×" in §4 are both artefacts of the cold number.
- **The cold ratios in §6.4, §7.5, §8.3 (2.39× / 1.45× / 1.35× / 0.69×) understate ES's cost
  uniformly.** Against BP steady-state 25.11 s the same four variants are **5.88× / 3.57× / 3.34× /
  1.70×** (or 1.48× using ES's own steady-state 37.17 s).
- **What survives unchanged:** the 3.46× step and 5.10× decode improvements from §6–§8, which were
  measured cold-vs-cold on one harness; the gradient-quality results (§1); and the negative learning
  result (§9.4), which never depended on the ratio.

### 10.4 Why the cold penalties differ so much

ES 42.67 → 37.17 s is a **13%** cold penalty; BP 61.86 → 25.11 s is **146%**. ES's teacher is the
same vLLM engine that just ran decode, so it is warm by the time it scores; BP's teacher is a
separate FSDP module whose first forward of the run happens inside the timed phase. Any future
ES-vs-BP number on this page must be a steady-state median, not a step-1 reading.

Raw records: `scripts/zo_opd/results/es_token_bp_teacher_cold.txt`.

---

## Session 2026-08-25/26 — the setting where BP-OPD learns, and es_token measured against it

> **Read this first if you are picking up es_token.** This session found the pair/config where
> BP-OPD demonstrably learns on ONE GPU, fixed five bugs that made every earlier es_token run
> uninterpretable, and produced the first apples-to-apples BP-vs-ES number. Branch
> `feat/es-token-trainer`. Launchers: `scripts/zo_opd/nersc_align/`.
> wandb: `nersc_opd_qwen4b_1p7b` (same project as the NERSC reference, so curves overlay).

### 11.1 The working setting — copy this, do not re-derive it

Hyperparameters are taken verbatim from `slurm/opd/full/opd_2node_env.sh`, the 8×A100 run behind
wandb `nersc_opd_qwen4b_1p7b/opd_full_dapo_lr1e-6`:

| | value |
|---|---|
| student | `Qwen/Qwen3-1.7B` (**non-thinking**) |
| teacher | `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` |
| data | `datasets/dapo-math-17k.parquet`; val AIME25 + AMC23 + AIME24 |
| estimator | `token_reward_direct`, top-K 16, `only_stu`, `student_p` |
| T student/teacher | 1.0 / 1.0 |
| batch / rollout n | 64 / 4 → 256 seqs, ONE optimizer step per training step |
| LR, KL, loss agg | 1e-6, 0.0, token-mean |
| **`MODEL_DTYPE`** | **bfloat16** (yes, really — see §6) |
| val | n=8, T=1.0, top-p 0.95 |

**`+data.apply_chat_template_kwargs.enable_thinking=False` is load-bearing.** Qwen3-1.7B is a hybrid
model; without it every rollout opens a `<think>` block and overruns the budget. This is the single
bug behind the old "every rollout hits the 1024-token cap" result in §9.

Single-GPU deltas from the 8-GPU original (teacher is CO-LOCATED here, not sharded):
`MAX_RESP_LENGTH` 7168→3072, `GPU_MEMORY_UTILIZATION` 0.75→0.45,
`REWARD_MICRO_BATCH_SIZE_PER_GPU` 12→4, `REWARD_PARAM_OFFLOAD` False→True,
plus `trainer.max_actor_ckpt_to_keep=1` (required at `SAVE_FREQ=10`; verl otherwise keeps every
~20 GB checkpoint).

### 11.2 Headline — BP-OPD learns, es_token does not

Offline re-scoring of all three checkpoints under an **identical** protocol
(`scripts/zo_opd/paper_align/eval_math.py`, n=8, T=1.0, top-p 0.95, 3072 tokens, non-thinking):

| benchmark | base Qwen3-1.7B | **BP-OPD** step279 | **ZO-ES-token** step169 |
|---|---:|---:|---:|
| AMC23 acc@8 | 0.3931 ±0.043 | **0.4172** | 0.3916 |
| AIME24 acc@8 | 0.0958 ±0.037 | 0.1042 | **0.1250** |
| AIME25 acc@8 | 0.0875 ±0.041 | **0.1250** | 0.0792 |
| MATH-500 acc@8 | 0.7250 ±0.016 | **0.7532** | 0.7265 |
| MATH-500 resp len | 837 | 1241 | 843 |

- **BP is up on 4/4** (+2.4 / +0.8 / +3.8 / +2.8 pp). Each gain alone is only ~1–1.5σ; the evidence
  is the *consistency* (4/4 same direction ≈ 6% by chance) plus the in-run curve below.
- **es_token is inside noise on all four** (up 2, down 2), and its response length is unchanged from
  base (843 vs 837) — the model is functionally the one it started with.

**BP in-run vs the NERSC reference** (`val-core/*/acc/mean@8`):

| step | AMC23 ours@3072 | AMC23 ref@7168 |
|---|---:|---:|
| 0 | 0.405 | — |
| 60 | **0.450** | 0.453 |
| 100 | 0.420 | 0.465 |
| 160 | 0.429 | **0.538** |
| 260 | 0.431 | — |

Ours **tracks the reference exactly through step 60**, then plateaus while the reference keeps
climbing. The split is the token budget: the reference's `response_length/mean` grows to 3583, past
our 3072 cap, so the rest of its gain is bought with length we cannot spend. **If you want the full
reproduction, you need ≥5120 tokens**, which does not fit alongside a co-located 4B teacher on one
95 GB card (`perf/max_memory_reserved` was 95.69/95.83 GB at 3072).

**es_token training internals** (200 steps total, lr=1e-5, N=8 rails, `fp32_master`):
`eval/accuracy` (MATH-500 greedy) 72.2–74.8 with no trend; `dW_norm_mean` steady 810–1050 (no
divergence); `RMS(dW)` over the 1.41 B perturbed params reached only **3.60e-5**, ~0.18% of typical
weight magnitude. The update is real but far too small/noisy to move the model in 200 steps.

### 11.3 Code changes shipped this session

| file | change | why |
|---|---|---|
| `verl/trainer/es/task_utils.py` | `template_kwargs` → `apply_chat_template` | the es prompt path had **no** `enable_thinking`, so a hybrid Qwen3 student silently ran in thinking mode |
| `es_token_worker_extension.py` | `fp32_master` (host-resident) | vLLM holds weights in bf16; a sub-ulp SGD step rounds away. Master is on the **host**: 5.65 GB of GPU is not available on this box |
| `es_token_worker_extension.py` | `es_export_weights()` | pull perturbed weights for checkpointing |
| `es_token/ray_trainer.py` | `_save_hf_checkpoint()` | es_token wrote **no checkpoints at all**; splits fused `qkv_proj`→q/k/v `[2048,1024,1024]` and `gate_up_proj`→gate/up `[6144,6144]` back to HF |
| `es_token/ray_trainer.py` | `max_prompt_length` filter | mirrors BP's `filter_overlong_prompts`; see §4 |
| `np/ray_trainer.py` | `teacher_max_model_len` | teacher sized its KV for the model's full 32k context |
| `on_policy_distillation.sh` | `REWARD_MODEL_DTYPE`, `VAL_BEFORE_TRAIN`, `PPO_MAX_TOKEN_LEN_PER_GPU` | teacher can stay bf16 while the actor runs fp32; step-0 baseline |
| `paper_align/eval_math.py` | `--enable-thinking` | otherwise the eval measures a different model than was trained |

New launchers: `scripts/zo_opd/nersc_align/{bp_opd_nersc.sh,es_opd_nersc.sh,final_eval.sh}`.

### 11.4 Operational gotchas that will bite again

1. **vLLM's `EngineCore` is a SUBPROCESS.** `pkill -f <your_script>.py` kills the parent and leaves
   the engine holding the whole card. This broke three separate things this session (ES sweep arm 2,
   the ES long run, and the final eval, which died with
   `Free memory on device (11.61/93.1 GiB) ... less than desired`). Always
   `pkill -9 -u $(id -u) -f "VLLM::EngineCore"` and **poll until the memory actually returns** before
   starting the next job. `scripts/zo_opd/nersc_align/final_eval.sh` and
   `paper_align/es_lr_sweep.sh` both do this now.
2. **Check process ownership before killing.** This box is shared (`jiayi`, `ryan` appear on GPUs
   0/3/6). Print the owner and kill only `yequan`'s.
3. **es_token has no prompt-length filter** unless you set `es_token.max_prompt_length`. DAPO-Math-17k
   has 9/17,917 prompts over 1024 tokens (max **1552**), and each one breaks two things: the teacher
   refuses `prompt+response > teacher_max_model_len`, and the packed decode reserves
   `longest_prompt + max_tokens` of scratch KV **per slot** (1552+3072 → 18,560 blocks vs 17,635
   available). Killed a run at step 34.
4. **es_token memory budget on a 93 GB card** (student + co-located teacher): student engine ~39 GB,
   teacher ~15 GB, fp32 master 5.65 GB, assembly accumulator `acc` 5.65 GB (all layers, fp32),
   `noise_chunk` 1.75 GB at `assemble_chunk=1024`. Working config: `GPU_MEMORY_UTILIZATION=0.42`,
   `TEACHER_GPU_MEMORY_UTILIZATION=0.16`, `TEACHER_BATCH_SIZE=4`, `assemble_chunk=512`, master on host.
5. **`assemble_chunk` dominates assembly time**: 1024→27.9 s, 512→41.3 s, 256→84.8 s. Do not lower it
   for memory when a bigger lever exists — it cost 40 s/step before this was noticed.
6. **Warm-restart is cheap now.** es_token is plain SGD with no optimizer state, so relaunching with
   `ACTOR_MODEL_PATH=<step_N checkpoint>` loses only the steps since the last save. Used it to move
   the run across GPUs after a crash at step 34.
7. **`eval/heldout_clean_loss` is not a usable ruler.** Greedy removed the sampling noise it was
   designed to remove, but greedy argmax still flips under small weight changes: the same LR gave
   3.435→2.603 in one run and 2.168→4.595 in a re-run, and its step-0 value depends on
   `gpu_memory_utilization` (3.4349/3.6372/2.1678 at 0.55/0.50/0.45, via the bf16-rounding path in
   §8.2). **Use MATH-500 accuracy.**

### 11.5 Step time, corrected

Both single GPU, batch 64, 3072 tokens, medians over the run (first step dropped):

| phase | BP-OPD | ZO-ES-token |
|---|---:|---:|
| generation / decode | 45.9 | 93.3 |
| teacher | 25.5 | ~9 |
| grad + update | 50.6 | 31.7 |
| **step** | **126.0** | **130.7** |
| sequences per step | 256 (64×4) | 64 (64×1) |
| **per sequence** | **0.49 s** | **2.04 s (4.2×)** |

Per *step* they are near parity; per *sequence* es_token is ~4× BP. An earlier reading of 6.6× was
inflated by `assemble_chunk=256` — that was a setting, not a property of the estimator.

### 11.6 Corrections to earlier claims on this page

- **§9's "neither method learns" is explained.** The cause was `enable_thinking` defaulting to true
  on a hybrid Qwen3 student, so every rollout hit the token cap mid-`<think>`. Not the algorithms.
- **bf16 master weights ATTENUATE, they do not freeze.** A one-step measurement (an Adam-sized 1e-6
  step changes only ~1.35% of bf16 weights vs 100% in fp32) is correct, but the inference drawn from
  it was wrong: the NERSC reference runs `MODEL_DTYPE=bfloat16` at `lr=1e-6` and **learns**
  (AMC23 0.416→0.538 over 160 steps). fp32 masters are an improvement, not a bug fix.

### 11.7 Open — start here

1. **es_token's training-time decode inflates response length, and it does not transfer.** During
   training mean length grew 1397 → 3024 (pinned at the 3072 cap by step ~120), yet the trained
   checkpoint generates 843 tokens on MATH-500 and 2266 on AIME24 — **both identical to base**. So
   the inflation lives in the packed training decode, not in the model. If the training rollouts are
   drifting off the policy's real inference distribution, es_token has been estimating its gradient
   on off-distribution trajectories. **This is the most important thing to chase.** A first check:
   decode the same prompts through the packed driver and through stock `llm.generate` at the same
   weights and compare length distributions.
2. **`_np_is_eos` only sees `hf_config.eos_token_id`.** `SamplingParams` built bare gives
   `_all_stop_token_ids = set()`, so the packed decoder falls back to config.json's single
   `151645` and **misses `151643`** (`<|endoftext|>`, present only in `generation_config.json`).
   Not proven to be the cause of (1), but it is a real gap on the same path.
3. **es_token LR is unresolved.** 1e-5 is flat over 200 steps with `RMS(dW)` at 0.18%; 1e-3 destroyed
   the model in the old setting (§9.2). The bracket between them has never been swept on a setting
   where BP is known to learn — which now exists.
4. **Full reproduction needs ≥5120 response tokens**, which does not fit with a co-located teacher.
   Either shard the teacher across 2 GPUs or serve it out-of-process.

Raw records: `logs/nersc_{bp,es}.log`, `logs/final_eval/{base,bp_step279,es_step169}.json`.
Checkpoints: BP `global_step_279` (merged to HF via `verl/scripts/legacy_model_merger.py merge
--backend fsdp`), ES `.../singlegpu_es_opd_r3072_lr1e-5_resume30/es_token_*/step_169`.

---

## Session 2026-08-28 — why es_token does not learn: it is the update footprint, not the code and not σ

> Follow-up on [§11.7](#117-open--start-here). Four candidate causes were tested and three
> were falsified. The estimator is correctly implemented, correctly checkpointed, unbiased,
> and run inside its linear regime — but its per-step *update footprint* is 1.4e-4 of the
> weight scale, where **every ES arm in this repo that learns runs at 1.6e-2 – 5e-2**
> ([ES §10.4](ES/es_results.md), [§11.3](ES/es_results.md)). Section 1's "cos ≈ 0.20 at
> training scale" is also wrong, and the correction is the reason the update is ~99.9% noise.
> New harness: `scripts/zo_opd/es_token_checks/es_grad_audit.py`.

### 12.1 Falsified — "the update is not applied / the checkpoint is not saved"

Direct diff of every tensor, trained checkpoint vs base (`Qwen/Qwen3-1.7B`):

| | RMS(ΔW) global | rel. to RMS(W) | per-linear rel. | elements actually changed |
|---|---:|---:|---:|---:|
| **BP** `step_279` | 1.56e-5 | 2.67e-4 | 3.7e-4 – 7.2e-4 | **0.9 – 2.1 %** |
| **ES** `step_169` | 3.26e-5 | 5.30e-4 | 9.0e-4 – 1.17e-3 | **20.1 – 33.3 %** |

All 196 HF keys matched on the ES side with no unmatched-key warning, so the fused
`qkv_proj`→q/k/v and `gate_up_proj`→gate/up split in `_save_hf_checkpoint` is correct.
**On the layers both methods train, ES moved FURTHER from base than BP did** — over all 196
perturbed linears the ES/BP ratio of relative displacement has median **1.86×** (IQR 1.72–2.13,
range 1.34–3.34) — and BP learned while ES did not. Whatever is wrong, it is not a lost or unsaved update.

(The low "elements changed" fractions are bf16 rounding, not a bug: a step below half a bf16
ulp rounds back. BP at ~1.4 % reproduces the §11.6 measurement; it learns anyway.
`scripts/zo_opd/es_token_checks/check_weight_displacement.py`.)

### 12.2 Falsified — "the learning rate is too small"

Per-step, per-parameter update size:

| | rule | RMS(ΔW) / step | footprint = RMS(ΔW)/RMS(W) |
|---|---|---:|---:|
| BP-OPD | AdamW, lr 1e-6 (Adam normalises, so the step ≈ lr) | 1.0e-6 | 3.0e-5 |
| ES-token | plain SGD, lr 1e-5, `dW_norm_mean` 742 → ~1400 | 4.1e-6 | **6.5e-5 → 1.3e-4** |

**ES already takes a 2–4× larger step than BP** (2.2× at step 0, ~4.3× once `dW_norm` settles).
Raising the LR to "catch up with BP" is arguing the wrong direction — see §12.5 for the reference
that *does* matter. (BP's row uses Adam's normalisation property, `|m/√v| ≲ 1`, so 1.0e-6 is an
*upper* bound on its step; the ES/BP gap is if anything larger.)

### 12.3 Falsified — σ is outside the linear regime

`es_grad_audit.py`, fp32, real DAPO-Math prompts, real teacher `log q`, 576 probe positions
× 8 rails, all 196 linears perturbed exactly as in training:

| σ | mean \|Δlogp\| per rail | max | IW clamped | IW underflow | cos(dW, g_direct) |
|---|---:|---:|---:|---:|---:|
| 1e-5 | 0.0002 | 0.006 | 0 | 0 | ≈0 |
| 1e-4 | 0.0017 | 0.062 | 0 | 0 | ≈0 |
| 1e-3 | 0.0174 | 0.582 | 0 | 0 | ≈0 |
| 3e-3 | 0.0542 | 1.453 | 0 | 0 | ≈0 |
| **1e-2 (shipping)** | 0.3216 | 4.862 | 0.001 | 0.002 | ≈0 |
| 3e-2 | 11.84 | — | 0 | **0.659** | ≈0 |

σ=1e-2 sits just inside the usable band (σ=3e-2 collapses: 66 % of importance weights
underflow). But **the cosine is flat and ≈0 across three orders of magnitude of σ** — a
linear-regime problem would show good cosine at small σ and decay at large σ. It does not.
σ is not the defect.

### 12.4 Falsified — the clean-KV "myopia"

The rails are perturbed only at the current decode step and read the clean row's KV
(`_packed_replay_row_meta`: *"n_sample perturbed rails (slot=-1, never write KV)"*), so they
can only ever see the **detached-history** gradient `g_direct`, not BP's `g_full`. Measured on
identical token positions:

| layer | cos(g_direct, g_full) |
|---|---:|
| `layers.0.mlp.down_proj` | +0.28 |
| `layers.7.self_attn.o_proj` | +0.24 |
| `layers.14.self_attn.k_proj` | +0.07 |
| `layers.21.self_attn.o_proj` | +0.92 |
| `layers.27.mlp.down_proj` | +1.00 |

Positive everywhere and near-perfect in the late layers. Myopia costs something in the early
layers but is **not** what stops the run.

### 12.5 The real defect, in two parts

**(a) §1 finding 4 is wrong: the estimator law is `cos ≈ sqrt(N/D)`, not `sqrt(K/(K+d))`
with `K = B·T·N`.**

The §1 offline gate perturbed ONE layer and probed ONE loss repeatedly, so all K probes
measured *the same* gradient and `cos → sqrt(K/(K+d))` held (0.86–0.99× bound). In training,
each token draws its own `(u_t, v_t)` and therefore probes **that token's own gradient**.
Per-token gradients in an LLM are mutually near-orthogonal (measured ρ = ‖ΣG_t‖²/Σ‖G_t‖² =
0.77 ≈ 1), so signal and noise both grow as √T and the token count **cancels**:

```
cos ≈ sqrt(ρ·N / d)          (single layer)
cos ≈ sqrt(ρ·N / D_total)    (all layers — the other layers' gradient energy is cross-talk)
```

Measured against prediction, at N=8:

| regime | predicted | measured | ratio to the §1 "K-bound" |
|---|---:|---:|---:|
| one layer (`layers.0.mlp.down_proj`, d=12.6 M) | sqrt(8/12.6e6) = **8.0e-4** | **+7.0e-4** | 0.04× |
| all 196 linears (D = 1.41 B) | 7.5e-5 | ≲2e-4 (at the 1/√d floor) | 0.00–0.02× |

The one-layer prediction lands within 13 %. **The per-token machinery buys no gradient
information over sequence-level ES at the same rail count** — it multiplies targets, not
probes. §1's extrapolation ("at 64×1024×8 the bound predicts cos ≈ 0.20") never applied.

**(b) But low cosine alone does not prevent ES from learning — a too-small step does.**

The estimator is unbiased, so `E[ΔL] = −lr·‖G‖²` regardless of how noisy the direction is; the
noise only enters at second order through curvature. That is why the sequence-level `es` trainer
learns at *comparable* per-step cosine: `dense` (full 7.6 B parameters, N=30) gains +20 pp on
MATH-500. What separates the arms there is **scale**, and [ES §11.3](ES/es_results.md) is explicit
that moving FuRA from −12.25 pp to +0.82 pp was a scale change, not a direction change.

There are **two different footprints** and es_token is mis-set on both, in opposite directions.
Both are relative to `RMS(W)` (Qwen3-1.7B linears 0.033; Qwen2.5-Math-7B 0.020):

| | *probe* footprint ‖ΔW_probe‖/‖W‖ | *update* footprint per step |
|---|---:|---:|
| `es` dense (paper ES, +20 pp) | 5.0e-2 | `α/√N` = 5e-4/√30 = **4.6e-3** |
| `es` iso (footprint-matched, +20 pp) | 5.0e-2 | 2.5e-2/√30 = **4.6e-3** |
| **`es_token` (the flat 200-step run)** | **0.30** (σ/RMS(W) = 0.01/0.033) | **6.5e-5 → 1.3e-4** |

**es_token probes 6× too far and steps 35–70× too short.** The probe number is exact and
shape-independent: a rank-1 Rademacher perturbation has ‖ΔW‖_F = σ·√(d_out·d_in) and
‖W‖_F = RMS(W)·√(d_out·d_in), so the ratio is just σ/RMS(W). The update number is the one that
decides whether anything accumulates, and it is the one the LR sweep moves.

(The `es` "footprints" quoted in [ES §6](ES/es_results.md) and [§15.5](ES/es_results.md) —
1.6e-2, 3.25e-3, 3.84e-4 — are **probe** footprints, not update footprints; they are not
comparable to es_token's per-step motion and are not used as the reference here.)

**(c) Control — sequence-level ES at the same rail count lands in the same place.**
`es_seq_audit.py` runs the *other* design (ONE fixed rank-1 direction per rail for the whole
rollout, N=8 directions shared across a 24-problem batch, scored by teacher-forced forwards) on
the same prompts. At σ=1e-3 the per-layer cosines are +1.4e-4 … +1.6e-3 against a bound of
8.0e-4 … 2.0e-3 — the **same order of magnitude as token-level's +7.0e-4**, obtained from 8
sequence forwards instead of 576 probe positions × 9 packed rows. K=8 is too small for a precise
cosine (the 1/√d floor is 2.8e-4 – 6.9e-4 here), so this is a consistency check, not a ranking —
but it is the check the `cos ≈ sqrt(N/D)` law predicts, and it says the per-token decode is
buying nothing the cheap design does not already give. (σ=1e-2 is far outside the linear regime
for that design — mean |Δlogp| = 9.14, since a fixed perturbation compounds along the sequence.)

### 12.6 Two secondary defects found on the way

1. **The training decode uses no top-p.** `run_es_decode_packed` samples from the full
   151 k-token softmax at T=1.0, while BP's rollout and every eval use `top_p=0.95`. This is
   the mundane explanation for [§11.7](#117-open--start-here) item 1: at step 0 — identical
   weights — the training decode averages 1397 tokens and the eval 837. es_token has been
   estimating its gradient on a heavier-tailed trajectory distribution than the one it is
   scored on.
2. **[§11.7](#117-open--start-here) item 2 confirmed.** A bare `SamplingParams` leaves
   `_all_stop_token_ids` empty, so `_np_is_eos` falls back to `config.json`'s
   `eos_token_id: 151645` and misses `151643` (`<|endoftext|>`, present only in
   `generation_config.json`).

Neither is the cause of the flat run; both are real. Both are now **config knobs whose defaults
reproduce the old decode exactly**, so the LR sweep below stays interpretable and the next run is
a one-line switch:

| knob | default (= old behaviour) | fix |
|---|---|---|
| `es_token.top_p` / `ES_TOP_P` | `1.0` | `0.95` — matches BP's rollout and every eval (verified equal to HF's `TopPLogitsWarper` to 6e-8) |
| `es_token.use_generation_config_eos` / `ES_EOS_FROM_GENCFG` | `false` | `true` — also stops on `151643` |

### 12.7 Shipped, and the relaunch

**Shipped** — two per-step metrics, because the run was previously blind to exactly the
quantity that decides it (`es_token_worker_extension.py`, `es_token/ray_trainer.py`):

| metric | meaning |
|---|---|
| `train/update_footprint` | RMS(lr·dW)/RMS(W), averaged over perturbed layers — the §12.5(b) number |
| `train/dW_cos_prev_mean` | cos(dW_t, dW_{t−1}) on a fixed 100 k-coordinate sketch, averaged over the 112 perturbed layers — the coherent fraction of the estimate; ≈0 means random walk |

**First 20 steps of the sweep — and the metric's detection limit.** Over 21/20 logged steps:

| arm | n | mean | per-step std | mean / SE |
|---|---:|---:|---:|---:|
| lr 1e-4 | 21 | +3.80e-5 | 2.77e-4 | **+0.63 σ** |
| lr 1e-3 | 20 | −1.60e-5 | 3.78e-4 | **−0.19 σ** |

No detectable coherence in either arm. The metric is **well calibrated**: two independent isotropic
vectors sketched at 100 k coordinates give cos std 1/√1e5 = 3.16e-3, and averaging over 112 layers
divides that by √112 = 10.6 → **3.0e-4 predicted vs 2.8e-4 measured**. So the scatter is exactly
the pure-noise prediction.

**Read it as a bound, not a detector.** If the gradient is stable step to step,
`cos(dW_t, dW_{t−1}) ≈ cos²(dW, G)`, so §12.5's per-step `cos ≈ 2e-4` predicts **4e-8** — six
orders of magnitude below this metric's floor. The metric therefore cannot confirm the estimator
works; what it does is *exclude* the design's premise: 2σ on the mean bounds the coherence at
≲1e-4, i.e. **`cos(dW, G) ≲ 0.01`**, ruling out the `cos ≈ 0.20` §1 extrapolated and agreeing with
the offline audit's ≲2e-4.

**Relaunched** (the [§11.7](#117-open--start-here) item-3 sweep, in the §11 setting where BP
demonstrably learns), `scripts/zo_opd/nersc_align/es_opd_nersc.sh`, 200 steps, everything else
identical to the flat run so the LR is the only variable:

| GPU | LR | footprint (measured @ step 0 → est. steady) | rationale |
|---|---|---:|---|
| 6 | 1e-4 | **6.50e-4** → ~1.3e-3 | ~1/7 of `dense` ES's per-iteration motion |
| 7 | 1e-3 | **6.50e-3** → ~1.3e-2 | brackets `dense` ES's 4.6e-3 from above |

Both arms report identical `L_clean_mean` (0.30247) and `dW_norm_mean` (741.7) at step 0, so LR
is the only difference between them. Logs: `logs/lrsweep/opd_es_token_lr1e-{4,3}_*.log`,
wandb `nersc_opd_qwen4b_1p7b/singlegpu_es_opd_r3072_lr1e-{4,3}`.

§9.2's "1e-3 destroys the model" was measured in the pre-§11 setting (thinking-mode bug, every
rollout truncated at 1024, and a probe §11.4 later showed is not a usable ruler), so it does
not carry over and is being re-tested rather than assumed.

**Interim read at step 20 (not a verdict).** Both arms' in-run `eval/accuracy` fires *after* that
step's update, so there was no zero-update reference on this protocol; one was measured
separately (`eval_math.py`, same MATH-500 parquet, same `ttrl_math` grader, greedy n=1, 3072
tokens, non-thinking — so it is directly comparable):

| | base (0 updates) | step 0 (after 1 update) | step 20 |
|---|---:|---:|---:|
| **base reference** | **73.60 ± 1.97** | — | — |
| lr 1e-4 | | 73.8 | 72.4 |
| lr 1e-3 | | **70.6** | **69.4** |

lr 1e-4 sits on top of base. lr 1e-3 is 3.0–4.2 pp below it at both reads — ~1.5–2 σ each, so not
individually decisive, but consistent in sign and already 3 pp down after a *single* update.

**This is what §13.2's σ curve predicts.** With `cos ≈ 0` the update is a random walk, so after `S`
steps the model has moved `√S ×` the per-step footprint — which is directly comparable to the
probe footprints §13.2 measured on the same model:

| arm | displacement @ step 20 | @ step 200 | nearest §13.2 probe point |
|---|---:|---:|---|
| lr 1e-4 | √20 × 6.5e-4 = 2.9e-3 | 9.2e-3 | below 1.6e-2 (KL 1.10×) → expect flat |
| lr 1e-3 | √20 × 6.5e-3 = 2.9e-2 | **9.2e-2** | 3.3e-2 (KL 1.36×) → **9.2e-2 ≈ the 9.8e-2 cliff** |

So the two experiments agree quantitatively, and the standing prediction is that **lr 1e-4 ends
flat and lr 1e-3 ends degraded, possibly badly**. If that holds, the honest conclusion is not
"the LR was wrong" but **"at cos ≈ 0 there is no good step size"** — small steps do nothing, large
steps are random-walk damage, and the window between them is empty. That is the argument for
fixing the *direction* (§13) rather than the step.

Recorded here before the endpoint so it can be falsified: the endpoint may still surprise.

### 12.8 Endpoint — both LRs degrade; the random-walk model is confirmed quantitatively

Both arms ran the full 200 steps. MATH-500 greedy n=1, base reference **73.60 ± 1.97**:

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 199 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **lr 1e-4** | 73.8 | 72.4 | 73.8 | 71.2 | 71.6 | 73.0 | 71.2 | 71.2 | 72.2 | 70.4 | **70.8** |
| **lr 1e-3** | 70.6 | 69.4 | 66.8 | 67.4 | 67.0 | 63.6 | 60.8 | 59.8 | 61.0 | 59.8 | **57.4** |

| arm | OLS slope | t | end vs base |
|---|---:|---:|---:|
| lr 1e-4 | **−1.28 pp / 100 steps** | −3.2 | −2.8 |
| lr 1e-3 | **−6.52 pp / 100 steps** | −11.4 | **−16.2** |

**Correction to §12.7's prediction.** It said lr 1e-4 would end *flat*. It does not — the decline is
statistically clear (t = −3.2). The prediction was right about lr 1e-3 (severe degradation) and
right about the ordering, but wrong to call 1e-4 flat. The correct statement is that **both LRs
damage the model, at a rate that scales with step size**, and the earlier lr 1e-5 run looked flat
only because its damage rate (~10× smaller again) is below the ±2 pp resolution of this probe.

**The random-walk model is confirmed by direct measurement.** Displacement of the lr 1e-4 step-199
checkpoint from base (`check_weight_displacement.py`):

| | predicted | measured |
|---|---:|---:|
| coherent accumulation (`S × footprint`) | 0.13 – 0.20 | — |
| **random walk (`√S × footprint`)** | **9.2e-3 – 1.4e-2** | **1.05e-2** per perturbed linear |

(footprint 6.5e-4 at step 0 rising to ~1.0e-3; `√200` = 14.1. Global ratio 5.36e-3, diluted by the
untouched norms/embeddings; 79–80 % of elements moved, vs 20–33 % for the lr 1e-5 run.)

The measurement lands inside the random-walk bracket and is **5–8 % of** what coherent accumulation
would give, so the update is ≥ 92 % incoherent — independently agreeing with the `dW_cos_prev`
bound (cos ≲ 0.01, §12.7) and the offline audit (cos ≲ 2e-4, §12.5).

**Verdict on the LR question.** 1e-5 flat-within-noise, 1e-4 −1.3 pp/100 steps, 1e-3 −6.5 pp/100
steps: monotone damage with no learning anywhere in the bracket. This is what §12.7 anticipated —
**at cos ≈ 0 there is no good step size**, because the estimator supplies no direction to descend,
only a step length. Fixing the step was never going to work; the direction has to change (§13).

`lr 1e-3`'s checkpoints were deleted per the keep-the-better-arm instruction, so its displacement
was not measured — a small missed cross-check (the model predicts 9.2e-2, right at the §13.2 cliff).

---

## Session 2026-08-28b — sequence-level ES-OPD: the baseline es_token should have been measured against

> §12 showed es_token's rails read the CLEAN row's KV and are perturbed only at the current
> decode step, so the token dimension multiplies *targets* rather than probes. This section
> builds the other design — ONE fixed perturbation of EVERY parameter held for the WHOLE
> rollout, scored by the OPD loss itself — and calibrates its operating point by measurement.
> Code: `es.fitness=opd_kl` in `verl/trainer/es/ray_trainer.py`.
> Launcher: `scripts/zo_opd/nersc_align/es_seq_opd_nersc.sh`.

### 13.1 What it does

Each ES rail `n` draws `ε_n ~ N(0, I)` over **every** `named_parameter`, sets `W + σ·ε_n`, and
generates its **own** rollout `y ~ π_n`. The perturbation therefore propagates through the
trajectory exactly as a real weight change would. Fitness is the OPD objective itself:

```
fitness_n = − mean_t [ log π_n(y_t) − log q(y_t) ] ,      y ~ π_n
```

a single-sample estimate of `−KL(π_n ‖ q)`, unbiased precisely because `y` is **sampled** from
`π_n` (the trainer refuses to start at `temperature=0`). `log π_n` is free from generation
(`SamplingParams(logprobs=0)`); `log q` costs ONE teacher prefill per rollout via
`prompt_logprobs`. Update is verl's existing `p += (α/N)·Σ_n z_n ε_n`, so per-iteration motion
is exactly `α/√N`.

### 13.2 Calibration — σ measured, not assumed

`es_seq_sigma_probe.sh`, one iteration per σ, N=8, 16 prompts, 512 tokens. **σ=0 is the
unperturbed reference** (all rails share the generation seed, so at σ=0 the rollouts are
identical and the spread is exactly 0 — every bit of spread at σ>0 is perturbation-induced):

| σ | KL(π‖q) | vs σ=0 | fitness spread | resp_len | probe footprint σ/RMS(W) |
|---|---:|---:|---:|---:|---:|
| **0 (reference)** | **0.2838** | 1.00× | 0.0 | 499 | — |
| 1e-3 | 0.3116 | 1.10× | 0.0115 | 505 | 1.6e-2 |
| 2e-3 | 0.3865 | 1.36× | 0.0398 | 500 | 3.3e-2 |
| **3e-3 (chosen)** | **0.6420** | **2.26×** | **0.0879** | 472 | **4.9e-2** |
| 6e-3 | 4.945 | **17.4×** | 7.44 | 417 | 9.8e-2 |

Three things fall out.

1. **The σ=0 reference KL, 0.2838, independently reproduces es_token's `L_clean_mean` ≈ 0.30**
   on the same student/teacher/data — an end-to-end check of the new fitness path against a
   completely separate implementation.
2. **There is a cliff between 3e-3 and 6e-3.** At 6e-3 the population is destroyed (KL 17× the
   reference, spread 7.44, response length collapsing) — this is the σ that would have been
   picked by extrapolating "bigger σ is a regulariser" from [ES §11.3](ES/es_results.md).
3. **The mean KL rise grows super-quadratically past 2e-3** (rise 0.028 → 0.103 → 0.358 for
   σ 1→2→3e-3, vs 4×/2.25× for a pure `½σ²·tr(F)` curvature term), i.e. σ=3e-3 sits at the
   edge of the linear regime, not inside it.

**Chosen: σ = 3.0e-3, α = σ/2 = 1.5e-3.** It footprint-matches the `dense`/`iso` arms that gain
+20 pp ([ES §6](ES/es_results.md): probe footprint 5.0e-2; ours 4.9e-2) and reproduces their
per-iteration motion (`α/√N`/RMS(W) = 1.5e-3/√30/0.0615 = **4.5e-3** vs their **4.6e-3**), with a
measured 2× margin to the cliff. **σ=2e-3 / α=1e-3 is the documented fallback** if the KL curve
stalls or destabilises — it keeps the population at 1.36× the reference instead of 2.26×.

Contrast with es_token, which was mis-set on both axes (§12.5b): probe footprint 0.30 (6× too
far) and update footprint 6.5e-5–1.3e-4 (35–70× too short).

### 13.3 Four bugs found bringing it up

| symptom | cause | fix |
|---|---|---|
| `Cannot schedule RayWorkerWrapper ... {'GPU': 1.0} cannot fit into [{'GPU': 0.5}]` | vLLM's `ray` executor spawns a worker demanding a WHOLE GPU, so a co-located teacher's bundle can never be granted | `es.distributed_executor_backend=uni` (in-process worker) + `es.engine_gpu_fraction` |
| `Free memory on device (9.37/93.1 GiB)` — engine on the wrong card | `ESNcclLLM.__init__` popped `CUDA_VISIBLE_DEVICES` unconditionally; with `uni` there is no child worker to re-derive the device, so vLLM went to physical GPU0 | `ES_KEEP_CUDA_VISIBLE=1` (the identical gate `NPNcclLLM` already had) |
| ~48 s/iteration of CPU | the sympy task-reward grader ran on all 30 rails, though it is only a diagnostic here | graded on ONE rail per iteration; `accuracy` is NaN elsewhere and excluded from the mean |
| teacher never torn down | `_cleanup` only killed student engines | teacher engine + its placement group reaped first |

### 13.4 Configuration

| | value |
|---|---|
| student / teacher / data | identical to [§11.1](#111-the-working-setting--copy-this-do-not-re-derive-it) |
| perturbation | `dense` — every `named_parameter`, fp32 master |
| N (population) | **30** |
| σ / α | **3.0e-3 / 1.5e-3** |
| T / top-p | 1.0 / 0.95 (top-p for parity with BP's rollout and every eval, §12.6) |
| batch | 24 prompts, **resampled** per iteration from the 17 k pool |
| max_tokens | 2048 (train), 3072 (eval) |
| eval | MATH-500 greedy, every 10 iterations |

**`num_engines` must be 1.** `ES_KEEP_CUDA_VISIBLE=1` — required so the `uni` executor puts vLLM
on the pinned card instead of physical GPU0 (§13.3) — makes every engine actor inherit the *same*
device list, so two engines both land on the first GPU and `_init_inter_engine_group` dies with
`NCCL error: invalid usage`. Multi-engine needs per-actor device assignment, which is not wired up.
To use two cards, run two **separate single-engine jobs**.

**Running (2026-08-29), one arm per GPU** — the σ pair §13.2 could not separate on one iteration,
now run to 120 iterations each at N=30, batch 16, 1536 tokens:

| GPU | σ | α | rationale |
|---|---|---|---|
| 6 | 3.0e-3 | 1.5e-3 | footprint-matched to `dense`/`iso` (probe 4.9e-2, motion 4.5e-3) |
| 7 | 2.0e-3 | 1.0e-3 | the §13.2 fallback — keeps the population at 1.36× the reference instead of 2.26× |

**The §13.2 probe under-measured σ, because it ran at 512 tokens.** First production iteration
(N=30, batch 16, **1536** tokens) vs the probe (N=8, batch 16, **512** tokens), same σ:

| σ | probe KL @512 | production KL @1536 | production spread |
|---|---:|---:|---:|
| 2.0e-3 | 0.386 (1.36× ref) | **0.400** (1.41×) | 0.081 |
| 3.0e-3 | 0.642 (2.26× ref) | **0.936** (3.30×) | **1.758** |

**The σ=0 reference is length-invariant; only the perturbation compounds.** Re-measured at 1536
tokens: **0.2878** (resp_len 1142), against 0.2838 at 512 — unchanged, as expected for a per-token
mean on the clean model. So the whole length effect sits in the perturbation:

| σ | KL/ref @512 | KL/ref @1536 |
|---|---:|---:|
| 0 | 1.00 | 1.00 (0.2878) |
| 2.0e-3 | 1.36× | **1.39×** |
| 3.0e-3 | 2.26× | **3.25×** |

σ=2e-3 is stable across lengths; σ=3e-3 degrades badly, and its fitness spread is **22× larger**
than σ=2e-3's — heavy-tailed, with individual rails blown out rather than informatively perturbed.
**A fixed weight perturbation compounds along the trajectory, so a σ that is inside the linear
regime at 512 tokens can be outside it at 1536** — the probe must be run at the production
`max_tokens`, which §13.2's was not. Same class of mistake as §12.5(a): calibrating on a cheap
proxy and extrapolating. On present evidence the **fallback σ=2e-3 is the better operating point**,
and the footprint-matching argument that picked 3e-3 (probe footprint 4.9e-2 ≈ `dense`'s 5.0e-2)
does not survive contact with the real sequence length.

Measured cost: **~240 s/iteration** at N=30 / batch 16 / 1536 tokens on one H100 — 4× faster than
the 17 min/iteration estimated from the probe, so 120 iterations is ~8 h, not ~35 h.

**Do not read `train/kl_mean` as a learning curve — §9.1 repeats here.** The batch is *resampled
every iteration* (16 problems from a pool of 17,917), so `kl_mean` moves with the draw, not just
with the weights. First three iterations:

| iter | 1 | 2 | 3 |
|---|---:|---:|---:|
| σ=3e-3 | 0.9358 | 0.7093 | 0.6710 |
| σ=2e-3 | 0.4004 | 0.4634 | 0.3837 |

σ=2e-3 is non-monotone with a ±0.08 swing on a 0.40 mean — that swing *is* the batch noise floor,
and it is the same order as any plausible per-iteration learning signal. (Resampling is correct for
the *update* — every rail shares the batch within an iteration, so the z-scores stay comparable and
it prevents the fixed-batch overfitting [ES §11.1](ES/es_results.md) measured — it just makes the
metric useless as progress.) **The honest ruler is `eval/accuracy`**: MATH-500 greedy on a fixed
500-prompt set every 10 iterations, deterministic given the weights, σ ≈ 2 pp. Both arms read 73.2
there before training, matching the 73.60 ± 1.97 base reference measured through a separate harness.

New metrics: `train/kl_mean`, `train/kl_min`, `train/kl_spread`, `train/resp_len`.
`kl_spread` is the ES signal strength — at 0.0879 it is ~4× the `reward_std` ≈ 0.023 the math-
accuracy fitness gives in [ES §15.5](ES/es_results.md), which is the point of a dense objective.

### 13.5 The first ES-OPD run length-hacked its own fitness — and the fix

The first two arms (σ=3e-3 / 2e-3, N=30, 30 iterations) were **killed at iteration 30**. They were
minimising the fitness successfully and getting *worse* at the task:

| | eval @0 | @10 | @20 | @30 | KL/ref @1 → @30 | resp_len @1 → @30 |
|---|---:|---:|---:|---:|---:|---:|
| σ=3e-3 | 73.2 | 74.8 | 73.8 | **68.2** | 3.25 → 1.30 | 893 → 1438 (**+61 %**) |
| σ=2e-3 | 73.2 | 73.4 | 73.4 | **71.4** | 1.39 → 0.94 | 1069 → 1408 (**+32 %**) |

The mechanism is unambiguous:

| | length slope | t | corr(length, KL) | % of the 1536 cap at iter 30 |
|---|---:|---:|---:|---:|
| σ=3e-3 | +20.8 tok/iter | +14.9 | **−0.884** | 94 % |
| σ=2e-3 | +14.7 tok/iter | +10.9 | **−0.874** | 92 % |

**The fitness was a per-token MEAN** — `−mean_t[log π_n(y_t) − log q(y_t)]`. A rail lowers that by
appending easy, low-KL filler that dilutes the high-KL reasoning tokens, so ES selects for length:
both arms grew toward the cap and accuracy fell. The KL decline reported at iteration 20
(§13.4, t = −6.8 / −4.8) was therefore **substantially length-hacking, not distillation** — a
correction to that reading.

**Why BP-OPD does not have this failure.** Its token-mean loss is differentiated over a *fixed*
rollout; length is not a decision variable. In ES the fitness *compares rollouts*, so any
length-dependent term becomes selection pressure. **Aggregation is a free choice under BP and a
load-bearing one under ES** — the general lesson, and the reason a metric that is standard on the
BP side cannot be copied across unexamined.

**Fix (`es.opd_kl_agg`, default `sum`).** Score the per-sequence **total** log-ratio, i.e. the true
sequence-level reverse KL of the trajectory, EOS included, then average over prompts — an estimator
of `E_x[KL(π(·|x) ‖ q(·|x))]`. Padding then costs exactly what it is worth, and stopping early is
rewarded only when the teacher agrees. `mean` is kept as an ablation and is documented as hackable.

**Why this does not just invert the failure.** `sum` penalises length, so the obvious worry is a
mirror-image collapse to very short answers. It is self-correcting **only if the EOS token is
scored** — otherwise stopping early is free. Verified in the installed vLLM: the stop token is
excluded from the detokenised *text* but explicitly appended back to `token_ids`
(`v1/engine/detokenizer.py:126`, "Cleanup after skipping detokenization"), and `check_stop` fires on
`output_token_ids[-1]` (`v1/core/sched/utils.py:59`). So EOS is in the scored response, and an
early stop the teacher disagrees with carries a large positive log-ratio. **`train/resp_len` is
still the metric to watch** — a significant negative slope would mean the penalty is not biting.

Raw records: `logs/esseq/meanagg_sig{3e-3,2e-3}.log`. Relaunched at the same two σ with
`opd_kl_agg=sum` → `logs/esseq/sumagg_sig{3e-3,2e-3}.log`.

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

| phase                   | observation                                                                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| steps 0–10             | both 2e-4 & 6e-4**dip the KL** (e.g. 6e-4: 0.336→0.322→0.318) — looks like training                                                                                                     |
| steps 10–25            | KL**oscillates in a 0.31–0.35 band** = the probe's own ~±0.03 noise (greedy NP-decode is not bit-deterministic: two runs gave step-0 KL 0.306 vs 0.336 with identical weights)           |
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
