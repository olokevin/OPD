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
