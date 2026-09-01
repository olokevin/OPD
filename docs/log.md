# Knowledge-base log

Chronological, append-only record of knowledge-base activity: ingests, queries, lint passes.
Companion to `index.md` (content catalog). See `CLAUDE.md` → **Knowledge system** for conventions.

Entry format — one `##` header per event, so `grep '^## \[' log.md | tail -5` lists recent activity:

```
## [YYYY-MM-DD] <ingest|query|lint|design> | <short title>
- what changed, which pages touched
```

---

## [2026-05-26] design | verl + LlamaFactory PEFT (BlockTT/SVD/LoRA/QLoRA) specs
- Filed design specs and implementation plans for integrating BlockTT/SVD finetuning into both frameworks (seeds of the `compressed_opd` thread).

## [2026-05-27] design | ZO trainers (ES vs NP)
- `wiki/ZO.md`: documented the two zeroth-order OPD trainers — Weight Perturbation (ES) and Node Perturbation (NP) — and their variance/memory/update-shape tradeoffs.

## [2026-05-28] design | NP trainer spec + implementation plan
- Approved design spec and step-by-step plan for the node-perturbation trainer.

## [2026-05-29] design | NP trainer pages
- `wiki/zo_np_trainer.md` + `plans/np_trainer.md`: NP internals (1+n_sample-wide perturbed vLLM decode, teacher reverse-KL, rank-1 δW) and the build plan. Branch `feat/np-trainer`.

## [2026-05-31] ingest | compressed_opd session summary + C4-PPL audit
- `wiki/compressed_opd.md`: full state of BTT-compressed Qwen3-4B→1.7B OPD workflow — launchers, calib cache, FSDP2 + BlockTT fixes, C4-PPL audit exposing the BTT-LLM-V2 gap.

## [2026-06-02] ingest | reason_aware_compress ARIS thread (v1→v2, TRACER)
- `aris/reason_aware_compress/`: idea-discovery → literature survey → TRACER proposal → experiment plan. Block 0 SER probe **falsified** the central thesis (steering subspace is best-preserved, not eroded) → pivot to M1/rank-deficiency. See MANIFEST for v1/v2 provenance.

## [2026-06-02] ingest | compressed_opd + zo_opd results
- `results/compressed_opd.md`: post-train compression table (SparseGPT/SVD_V2/Nystrom × C4/OpenThought3 calib vs C4-PPL + MATH-500); SparseGPT+math-calib = 45%, structured/one-shot SVD collapse to 0%.
- `results/zo_opd.md`: ZO-NP OPD results — NP-vs-BP gradient scaling, LR search, self-amplifying divergence.

## [2026-06-02] lint | knowledge-base scaffolding
- Created `index.md` (content catalog) and `log.md` (this file); added the **Knowledge system** section to `CLAUDE.md` pointing future sessions at the wiki/results/aris layout and the ingest/query/lint workflow.

## [2026-06-02] ingest | reason_aware_compress — re-promote ideas A/B/D after TRACER falsification
- TRACER C2 (steering-subspace) falsified at Block 0 → re-promoted A (low-rank + sparse residual, M1), B (sequential re-linearized / SRC, M3), D (OPD bi-whitened SVD, M2) from ablations to first-class one-shot method candidates.
- **Added Blocks A/B/D** to `aris/reason_aware_compress/EXPERIMENT_PLAN.md` (self-contained, launch-now; run order D→A→B; gating note on the in-flight subsystem ablation). Code-grounded the design against `src/compress` (Explore inventory): D needs no new core code (`collect_both_covariances_from_loader_opd` + `objective="combined"`); A needs `hybrid/lr_sparse.py` (~100 LoC); B needs `sequential/relinearized.py` loop + per-layer cov hook. Noted Nystrom-MLP has no SVD tail (A/D "+Patch" apply to the attention path).
- Synced `IDEA_CANDIDATES.md` (v3 table + new active thesis, TRACER kept as audit trail), `EXPERIMENT_TRACKER.md` (D/A/B PENDING rows; subsystem ablation attn-only=3% partial), `index.md` (plan/results/tracker/candidates rows).

## [2026-06-02] ingest | reason_aware_compress — A/B/D operating-point + trace-diff refinements
- Set A/B/D **starting retain ratio = 0.8** (keep 80%; 0.36 is the fully-collapsed floor that hides method differences) with a sweep-down to find each method's cliff; **skip the last decoder layer's linears** (`model.layers.{N-1}.*`). Verified `skip_layers` is leaf-name-only (`name.split('.')[-1]`), so the drivers must add a `name.startswith("model.layers.{N-1}.")` filter — documented in the plan, not assumed free.
- Added **Block T (reasoning-trace diff)**: 5 fixed dense-correct MATH probes, greedy, dense-vs-compressed trace per method, first-divergence + failure-mode tagging → qualitative inspiration on *how* compression bends reasoning. Reuses `eval_math500`'s prompt-build+greedy-generate; new `trace_diff.py` returns per-example `(problem, text, correct)`, writes `trace_probe_set.json` + `TRACE_DIFF.md`.
- Updated `EXPERIMENT_PLAN.md` (operating-point section, all D/A/B cells re-anchored to 0.8-first, Block T, run-order, files-to-touch) and `EXPERIMENT_TRACKER.md` (Block-T row + 0.8 note).

## [2026-06-02] ingest | reason_aware_compress — prune plan: drop long-context + subsystem split, apply operating point to Blocks 0/1/2
- **Removed Block 3** (long-context × compression / RULER / M5) entirely. **Removed the attn-only/mlp-only subsystem split** (and its gating logic) — SA-* marked DROPPED in tracker (attn-only=3% partial kept for record).
- Applied the **0.8-first, last-layer-skipped** operating point to **Blocks 0, 1, 2**; **Block 2** retain sweep set to **{0.8, 0.7, 0.6, 0.5}**. Block 0 marked DONE (falsified), Block 1 demoted to its surviving C1/C3 cells (C2 dropped).
- Rewrote the plan header (was TRACER/steering claim) to the **M1/M2/M3 method-search** framing; reframed Block 4 headline + Block 5 robustness around the D/A/B candidates; cleaned dangling refs (`--max-context`, RULER, "2,3 run in parallel", "1–4 positive"). Synced `index.md` plan row.

## [2026-06-03] ingest | nystrom_combined: trainability-aware joint-kernel MLP compression
- Implemented `nystrom_combined` in `src/compress` (joint fwd+bwd Nystrom kernel K_joint=C̄f^½·C̄b·C̄f^½+λI per docs/plans/nystrom_combined.md): structured/nystrom.py, calibration.py collector, compress_model routing, tests/test_nystrom_combined.py (6/6 pass).
- Validation: Llama-3-8B 60% MLP retain, C4 calib (n=128) → C4 PPL dense 9.45 / nystrom 19.38 / nystrom_combined 20.94. **Similar** (+8%); joint kernel does not hurt forward PPL (expected — payoff is under FT/OPD recovery).
- Filed: results/compressed_opd.md (dated block), index.md row, src/compress/README.md "Structured MLP Compression (Nystrom)" section.

## [2026-06-03] ingest | reason_aware_compress — A/B/D + Block-T drivers implemented, GPT-5 reviewed, deploying
experiment-bridge: implemented Block D (`bi_whitened_svd.py`, M2 — no new core), Block A (`hybrid/lr_sparse.py` + `lr_sparse_residual.py`, M1 — LR+sparse residual), Block B (`sequential/relinearized.py` + `sequential_src.py`, M3 — depth-ordered re-linearization), Block T (`trace_diff.py`), and shared `compress_common.py` (last-layer skip via `drop_protected_stats`, MATH+C4 eval contract). GPT-5-high review: 1 CRITICAL (D3/B2 degenerate when teacher==student → fail-fast guard added; **user chose to skip D3/B2 this pass**) + 2 MAJOR (A2 upstream fixed via refinement Hessian pass; response-only masking documented). Deploying teacher-free cells D0/D1/D2, A0/A1/A2, B0/B1 at retain 0.8, last decoder layer dense, MATH/100 + C4 PPL. GPU 5 externally occupied (vLLM) → running on free GPU 7. See `EXPERIMENT_TRACKER.md` § "A/B/D drivers".

## [2026-06-03] ingest | zo_np_trainer §8 — V1 throughput profiled (forward=99%), V2 plan (one wide forward/token)
Profiled the NP student decode: per token meta=0.31ms (1%), the (1+N)=65-row eager forward=13-20ms (99%); the bottleneck is the eager forward run ~1024x/seq x batch_size serial sequences, NOT metadata/RPC/threads (OMP caps & metadata-caching both measured <2%, abandoned). Shipped V1.1 (commit 21534aa, branch np-fold-xcapture): folded x_t capture into the perturbed forward, deleted the redundant 2nd full-sequence re-decode (run_capture_pass/_capture_x) -> 2->1 decode passes/seq; estimator now (L_q-mean)/sigma (dropped 1/std). Wrote docs/wiki/zo_np_trainer.md §8 = concise V1->V2 guide: V2 must CUDA-graph ONE wide forward per token across ALL prompts (rows = Σ_p(1+N)), with the perturbation expressed as a graph-capturable buffer op; lists open design Qs (ragged batching, graph capture vs eager hook, packed KV/slot bookkeeping, stop handling, memory) and invariants to preserve (σ=0 byte-equiv, cos≈+0.41, math unchanged). Entry points: np_worker_extension.{run_np_decode,_np_step_forward,_np_build_attn_metadata}, ray_trainer.fit ~513-542.

## [2026-06-03] ingest | reason_aware_compress — A/B/D results + forward-only ratio sweep (experiment-bridge complete)
Ran the teacher-free A/B/D cells + a forward-only ratio sweep at the operating point (retain 0.8 first, last decoder layer dense, MATH-500/100 + C4 PPL, OpenThought3 reasoning-trace calib). **Findings**: (D, M2 objective) **null** — bilateral CE-gradient SVD (D2=OBD-LLM) ≈ plain input-whitened (D0), 70% vs 73%; backward-only (D1) collapses 0%. (A, M1 rank floor) **HEADLINE** — full-rank sparse residual (A2, compressed-upstream acts) recovers MATH **72→82%** (beats dense 80.5%) at the same budget, where attention tail-rescue had failed (0→4%). (B, M3) B0 baseline only; B1/B2 **skipped per user**. (Sweep) forward-only cliff at **r*≈0.65** (72/66/37/20/4% @ 0.8/0.7/0.6/0.5/0.4); trace-diff shows failure is **late-trace convergence loss / RAC looping** (trace len 1.1×→7.3×), not early divergence. D3/B2 OPD-teacher cells skipped per user. New code: `compress_common.py`, `bi_whitened_svd.py`, `hybrid/lr_sparse.py`+`lr_sparse_residual.py`, `sequential/relinearized.py`+`sequential_src.py`, `trace_diff.py`, `ratio_sweep_trace.py`. GPT-5 review (1 CRIT + 2 MAJOR) applied. Results: `aris/reason_aware_compress/INITIAL_RESULTS_ABD.md`, `EXPERIMENT_TRACKER.md`.

## [2026-06-03] ingest | zo_np_trainer wiki §9 — V2 design + initial results (throughput-vs-N, cosine-vs-N)
Consolidated the V2 buffer-in-graph design + GPU-validated initial results into a NEW top-level section §9 of docs/wiki/zo_np_trainer.md (the old in-§8 V2-plan prose is now a stale-but-preserved history; §8.5 trimmed to a pointer to §9). §9 covers: 9.1 why V1 was slow + what V2 changes; 9.2 the Path-B mechanism table (host u_buf fill / in-graph y+=σ·u_buf / eager compute_logits) + the verified vLLM facts incl. the max_seqlen_k-cap subtlety; 9.3 throughput-vs-N (ms/tok 10.3/12.3/13.6/17.0/23.7 @ N=1/8/16/32/64, "N is free" holds for N≤16); 9.4 NEW cosine-vs-N sweep (cos 0.205/0.276/0.356/0.407 @ N=8/16/32/64, repeats=50, σ=1e-3) + why the number transfers to V2 (check_grad_cosine is decode-driver-independent; parity proves bit-identical u → identical δW direction); 9.5 gate table (all PASS); 9.6 status (e2e training run is the next gate). Re-ran check_grad_cosine to confirm 0.407 @ N=64 on the current model. Linked §6 variance note → §9.4 (the sweep confirms its √(N·repeats) prediction). index + log updated.

## [2026-06-04] ingest | OPD-math sweep — ref OPD epoch done (MATH-500 0.508); NP graphed CORRECT at small scale but HANGS at batch 64
Final state of the sweep. **Reference OPD (GPU7): completed 1 epoch, 138 steps, ~62s/step, MATH-500 acc@1=0.508** — a full valid baseline (token_reward_direct = per-token teacher reverse-KL as advantage; the NaN PPO-outcome metrics are by-design, not a bug). **NP V2 graphed: PROVEN CORRECT end-to-end at small scale** — tiny canary (batch2, max_tok32, student 0.30 + teacher 0.45 co-located, use_cuda_graph=true) completed step 0 cleanly: step_time 14s, dW_norm 28.5, weight_changed_frac 0.58, **L_clean (KL) 0.097**, weight_sync_ok=1. So the graph capture + replay + apply + broadcast pipeline WORKS on GPU in the live trainer (earlier graphed "crashes" were the teacher-engine OOM from too-high memory fraction, now fixed). **BUT at the real config (batch_size=64, max_tokens=1024) all NP runs HANG**: confirmed GPU bursts (66% util) for the first ~minutes, then go fully idle (0% GPU, log frozen, tqdm 0/150, ~10 CPU cores spinning state=S) and never complete step 0 (>45 min). Same hang in eager and graphed → it's NOT the graph; it's a scale-triggered deadlock/stall partway through the 64-prompt serial decode loop (suspect: per-prompt graph re-capture ×64 leaking, or vLLM/Ray resource exhaustion, or teacher-scoring of 64×1024 rollouts). The single small step proves the math/pipeline; batch 64 needs a dedicated hang fix before the LR sweep (1e-2/3e-2/1e-1) can produce data. NO NP LR data produced. Scripts: scripts/zo_opd/opd_math_np.sh (NP graphed), opd_math_ref.sh (reference). Working fractions: GPU_MEMORY_UTILIZATION=0.30 + TEACHER=0.45 co-located. Logs: logs/opd_math_sweep/.

## [2026-06-04] ingest | OPD-math sweep launch — ref OPD trains (MATH-500 0.508); NP blocked by engine-memory/placement, NOT graph capture
Prepared two scripts in scripts/zo_opd/: opd_math_np.sh (NP/ZO-OPD, V2 graphed, greedy, batch_size=64, n_rollout=1, co-located teacher) and opd_math_ref.sh (standard verl OPD = on_policy_distillation.sh, adv_estimator=token_reward_direct, greedy, mbs=64, n=1; wraps full.sh with EXTRA_HYDRA_ARGS val_kwargs.do_sample=False since greedy eval + do_sample=True trips verl's temperature>0 assert). wandb project opd-qwen-math. **Reference (GPU7): VALID OPD baseline** — token_reward_direct sets advantage = per-token teacher reverse-KL reward directly (core_algos.py:854), so the PPO outcome metrics (actor/pg_loss, ppo_kl, critic/advantages) are vestigial/NaN by design and NOT a bug (my earlier "n=1 degenerate" read was WRONG — OPD has no group baseline to collapse). Ref trained 100+ steps, ~62s/step, critic/true_reward 0.4–0.5, **MATH-500 acc@1 = 0.508**. **NP runs (GPU4/5/6): never completed a step.** Root cause (decisive sep-GPU test): the TEACHER vLLM engine fails — `ValueError: Free memory on device (46.05/93.1 GiB) < desired gpu_memory_utilization 0.55` at gpu_worker.py:187 init_device — i.e. the student+teacher engine **memory fractions/placement** don't fit (two engines collide on one device, or the fraction is too high for the second engine), and the first teacher-scoring RPC then hits the dead actor (ActorDiedError at teacher_scorer.py:146). This is a LAUNCHER memory/placement issue, NOT the V2 CUDA-graph code (the standalone graph gates still pass; graphed≡V1). Fix pending: correct per-engine gpu_memory_utilization + reliably pin student/teacher to separate GPUs (gpu_fraction=1.0 needs 2 distinct visible devices placed by Ray, not a single CUDA_VISIBLE pin). NP LR sweep (1e-2/3e-2/1e-1) produced no data yet. Ref logs: logs/opd_math_sweep/ref_gpu7_*.out.

## [2026-06-03] ingest | NP V2 GPU gates ALL PASS — graph capture works, "N is free" measured
Ran the V2 GPU gates on GPU 6/7 (Qwen3-1.7B, FLASH_ATTN). ALL PASS. σ=0 byte-equiv: graphed_eager AND graphed_cuda match greedy generate (16 tok, width 1+4) — the graphed_cuda pass is the M0 capture spike: torch.cuda.CUDAGraph capture of our HAND-BUILT attn_metadata works, no "operation not permitted when stream is capturing", the one unverified-on-GPU risk is CLEARED. Parity m1 (V1 vs eager+u_buf) and m2 (eager+u_buf vs graphed): u BIT-IDENTICAL, logits/x within rtol=1e-2 → by transitivity graphed≡V1 (math unchanged, confirmed on real model). N-scaling (graphed_cuda, max_tokens=64): ms/tok = 10.3/12.3/13.6/17.0/23.7 at N=1/8/16/32/64 (1.00/1.19/1.32/1.65/2.30x) — the user's "N rails are ~free (memory-bound)" premise HOLDS for N≤16 (8 rails = +19% wall-time for 8× gradient samples) then crosses to compute-bound (2.3× at N=64); sweet spot N≈8–16 = the n_sample=8 default. Two bugs found+fixed on GPU (commit 01eb6bc), neither a math/invariant bug: (a) shared graph_pool_handle across captures tripped CUDACachingAllocator use_count>0 once >1 graph captured (bench) → each graph owns its pool + release prior graph before next capture; (b) parity script device mismatch (V1 captured_u on GPU, graphed on CPU). 38/38 CPU tests still pass. Wiki §8.5 + index updated with the gate table + bench numbers.

## [2026-06-03] ingest | NP V2 IMPLEMENTED — buffer-in-graph (branch np-v2-cudagraph-rails)
Implemented the V2 buffer-in-graph spec (additive; V1 eager kept as parity oracle). New code in np_worker_extension.py (+406): PerturbedLinear `perturb_graph` mode (y[1:1+N]+=σ·u_buf, RNG-free, x_buf.copy_(x[0])); `_np_fill_u_buf` host noise refill (only RNG, SAME noise_seed+draw_noise as V1 → u bit-identical); `run_np_decode_graphed(...,use_cuda_graph)` driver (same return contract as run_np_decode); M2 `_np_capture_step`/`_np_replay_step`/`_np_build_attn_metadata_persistent` (torch.cuda.CUDAGraph capture of the 1+N step, replay per token mutating input/metadata buffers in place; compute_logits+sampling stay eager). ray_trainer.fit() dispatches graphed when decode_mode=graphed; np_trainer.yaml adds decode_mode(eager default)/use_cuda_graph. New gates: check_graphed_parity.py (m1 eager-vs-eager+u_buf, m2 eager+u_buf-vs-graphed), check_decode_sigma0.py --driver graphed_{eager,cuda}, bench_n_scaling.py; 6 new CPU unit tests (test_perturb_graph.py). Verified vLLM-0.11.0 source twice (metadata-by-reference, set_forward_context global, slot<0 skip, compute_logits eager). 3-agent adversarial review: parity PASS(4/4), surgical PASS(5/5, +406/-0 additive, ES untouched), graph-correctness found 1 BLOCKER (max_seqlen_k frozen too low at token-0 q_pos → FA truncates context to prompt) + 2 majors (dead seq_lens_cpu write; per-prompt graph-pool leak) + 2 minors (per-token slot host-alloc; per-token sync). ALL FIXED: max_seq_len_override caps max_seqlen_k at prompt_len+max_tokens (mirrors vLLM gpu_model_runner.py:3057, re-verified by a 2nd agent that kernel bounds attention by live seqused_k not the frozen int), slot_mapping[1:]=-1 once + [0] per token, graph_pool_handle reuse, dtype-floating-point guard, dead x_buf fallback dropped. 38/38 CPU tests pass. GPU gates (M0 capture spike → M1/M2 parity → M3 bench) PENDING the user; M2's hand-built-attn_metadata-in-graph is the one unverified-on-GPU risk (spec §7.4). Wiki §8.5 + index updated.

## [2026-06-03] ingest | NP V2 spec rewritten — CUDA-graphed 1+N rails (buffer-in-graph), supersedes prompt-packing plan
Reviewed the prior V2 plan (`specs/2026-06-03-np-v2-design.md`) against the user's actual goal and found a mismatch: that plan's primary axis was packing many *prompts* into one forward (`B_pack`), but the user wants the **N perturbation rails of a single token** generated inside vLLM's **native CUDA-graphed forward** (same params + same KV → memory-bound decode makes N rails ~free). Investigated current NP code + vLLM 0.11.0 source (two Explore agents). Ground truth: (1) vLLM keeps ALL RNG outside its captured graphs (`gpu_model_runner.py:576-579`); (2) per-step dynamic data enters via host-refilled `CpuGpuBuffer`s (`gpu_model_runner.py:1044-1094`, `utils.py:105-143`); (3) `reshape_and_cache` skips `slot<0` (`triton_reshape_and_cache_flash.py:34-37`) so V1's `slot=-1` no-KV trick survives capture; (4) `n>1` child-requests (Path A) write KV + compound + can't be forced to slot=-1 (`llm_engine.py:243-255`) → rejected. **Decision: Path B (buffer-in-graph)** — move the noise draw out of `PerturbedLinear.forward` into a host `_np_fill_u_buf` that `copy_`s the SAME `draw_noise(noise_seed(...))` bytes into a persistent `u_buf`; perturbation becomes a captured `y[1:1+N] += σ·u_buf`; capture the 1+N step, replay per token; sampling/compute_logits stay eager. Keep V1 eager as the parity oracle. Milestones M0 (capture spike — de-risk hand-built attn_metadata-in-graph FIRST) → M1 (noise relocation, bit-identical u) → M2 (graph) → M3 (N-scaling bench validating the "N is free" premise the old plan never tested). Wrote `specs/2026-06-03-np-v2-cudagraph-rails.md`; marked the old spec superseded; index updated.

## [2026-06-04] ingest | reason_aware_compress — full-sequence calibration (reweighting × length) beats 2048-window, pushes the cliff lower
Tested calibration-data prep as a lever on forward-only SVD-V2+Nystrom (same method/budget): full (un-windowed) OpenThought3 sequences with **token vs sequence reweighting** × **full vs <2048 length**, mask-aware. **Stage 1 @0.7**: sequence-reweighting dominates (both seq settings 69–71% > both token 65–66%); winner **sequence·lt2048 = 71%** (+5pp over the 2048-window 66%, best PPL 92.4). **Stage 2 sweep**: vs the 2048-window cliff (37/20/4% @ 0.6/0.5/0.4), full-seq seq-reweighted gives **47/36/13%** — Δ **+10/+16/+9pp** (peak gain at r=0.5, nearly doubles 20→36%). Bonus: relaxed==strict + bounded gen_len at every ratio → better calibration also fixes the termination/looping failure (no 5–7× trace blow-up). Pure calibration change, orthogonal to the M1 sparse-residual headline. Also fixed a 20× perf bug in covariance collection (GPU-resident fp32 accum + batching). New code: `loaders.build_fullseq_calib_loader`, `calibration.collect_covariances_reweighted`, `compress_common.eval_math_capture`, driver `fullseq_calib_sweep.py`. Results: `aris/reason_aware_compress/FULLSEQ_CALIB_RESULTS.md`.

## [2026-06-04] ingest | reasoning_aware_compress_calib wiki — synthesize A/B/D + cliff + calib study
Created `docs/wiki/reasoning_aware_compress_calib.md` consolidating the durable design knowledge from `INITIAL_RESULTS_ABD.md` + `FULLSEQ_CALIB_RESULTS.md`: problem/operating-point/eval-contract; the M1/M2/M3 hypothesis space (post steering-falsification); results — **M1 rank floor = headline** (full-rank sparse residual 72→82% at fixed budget), **M2 objective = null** (D2≈D0, D1 collapses), M3 not run; the forward-only cliff (r*≈0.65) and **termination-before-reasoning** failure mode (looping cliff ~0.65 / reasoning cliff ~0.55, RAC length↑/acc↓); the **sequence-reweighted full-seq calibration** lever (now the repo default, +5→+16pp across the sweep) with the escape hatch + 4096-cap caveat. Catalogued in `docs/index.md` (wiki table).

## [2026-06-04] ingest | NP packed multi-prompt decode + throughput grid + NP-vs-BP one-step comparison
- Added `docs/wiki/zo_np_trainer.md` §10 (V2.1 packed multi-prompt decode, branch `np-v2-cudagraph-rails` commits `b91615c..7cb1f23`): `B_pack` prompts in ONE wide forward (`Σ_p(1+N)` rows, per-prompt disjoint scratch-KV + per-row attn-metadata, autoregressive loop unavoidable) via `run_np_decode_packed`/`_np_prefill_packed`/`_np_build_attn_metadata_packed`/`_np_step_forward_packed` + `PerturbedLinear` scatter-add; config `np.decode_mode=packed`/`np.pack_width`, launcher `PACK_WIDTH`. Gates PASS (`check_packed_sigma0.py` per-prompt routing @ b_pack 2/4/8; `check_packed_parity.py` serial-vs-packed u BIT-identical @ b_pack 4/8; 58 CPU tests). Throughput grid (`scripts/zo_opd/results/packed_grid_1024.txt`, pw=8, mem=0.55, max_tok=1024, decode-only): **~2.6× tok/s batch 1→8 @ rails=8** (74→194), shrinks to ~1.4× @ rails=64; tok/s plateaus by batch≈8; SM% does NOT climb (memory/CPU-bound, win = amortizing per-step overhead); peak mem ~flat ~49GB. **NEGATIVE result** — NP packed one-step (batch64, 865s = decode 584 + teacher 250 + assemble 31, ONE layer) is **~14× slower than BP-OPD first-step (62s) / ~38× steady-state (22.6s, 1144 tok/s)**; per-layer-equivalent far wider (BP backward updates ALL layers in 6.8s vs NP's 65 536 forwards for one layer). Packing is correct + delivers its decode win, but the zeroth-order forward-count tax is not closable by tiling (graphing the packed forward is out of scope and would not close the gap). Packed-scratch-KV ceiling: `B_pack × ceil(max_model_len/block_size) ≤ num_gpu_blocks`; `pack_width` caps per-wave width (batch>pack_width → multiple waves, always fits); operating point pack_width≈8 + student gpu_mem_util≈0.55. index + log updated.

## [2026-06-05] ingest | compress_sft — in-process svd_nystrom compress-then-train (Qwen3 PASS, OLMoE fused-expert blocked)
New LlamaFactory `finetuning_type: svd_nystrom` (branch `compress_sft`, worktree `OPD-compress-sft` off `np-v2-cudagraph-rails`): compress at model-init with SVD-LLM-V2 on `self_attn` + Nystrom/MoDeGPT on MLP (retain 0.7, sequence-reweighted full-seq OpenThought3 calib), keeping factors **trainable** (SVD `U_r/V_r` + Nystrom-shrunk MLP Linears), then SFT on OpenThoughts3. Two objectives: `svd_v2` (forward) / `svd_v2_combined` (fwd+bwd). Last-layer skip is **attn-only** (SVD is shape-preserving; MLP shrinks uniformly so the saved checkpoint keeps one global `intermediate_size`, updated to the Nystrom `k`). `CompressSaveCallback` writes a smaller dense HF checkpoint (SVD→dense, Nystrom passthrough). Eval = MATH-500@4096 + **MMLU-Pro** via the ttrl grader (LlamaFactory eval CLI is disabled in this fork); mid-training growth via val-loss (wandb) + post-hoc `sweep_sft_ckpts.sh`. wandb project `compress_sft_{model}`. **Qwen3-4B-Base smoke PASS for both objectives** (36 triplets, 140 trainable SVD attn, intermediate_size 9728→6810, save/reload 0 missing/0 unexpected). **OLMoE-1B-7B BLOCKED**: transformers 5.2 stores its 64 experts as fused 3D tensors (`OlmoeExperts.gate_up_proj/down_proj`), not `nn.Linear` triplets → `find_mlp_triplets` finds 0; added a fast-fail `_assert_no_fused_experts` guard; per-expert fused-tensor (or unfuse-at-load) Nystrom is the follow-up. Files: `LlamaFactory/src/llamafactory/{model/compress_setup.py,hparams/finetuning_args.py,hparams/parser.py,model/adapter.py}`, `LlamaFactory/examples/compress_train/*.yaml`, `scripts/opd/math/compressed_opd/{run_compress_sft.sh,eval_mmlu_pro.py,sweep_sft_ckpts.sh,_smoke_svd_nystrom.py}`. Results: `docs/results/compress_sft.md`.

## [2026-06-06] ingest | NP V3 all-layer fully-graphed decode — parity gates PASS, learns at LR=1e-3, NP-vs-BP goal NEGATIVE (~46× at batch64/1024)
- Added `docs/wiki/zo_np_trainer.md` §11 (V3 fully-CUDA-graphed ALL-LAYER packed decode, branch `worktree-np-alllayer-graphed`): every matched linear perturbed at once in ONE graphed packed decode (row layout 1+N shared, independent per-(layer,q) `u_buf` add `y += σ·u_buf[layer][q]`, each layer captures its own clean `x[layer]`, per-layer δW from its own (u,x); unbiased to first order). Path: `run_np_decode_packed_graphed` (capture ONE graph at fixed bucket `R = bucket_b_pack·(1+N)` via `_np_capture_step_packed`, replay per token `_np_replay_step_packed` with bucket-pad EOS, one batched teacher score, one `assemble_all_layers_and_apply`); config `DECODE_MODE=packed_graphed EN_LAYERWISE=false` + knobs `b_pack_buckets/pack_width/teacher_batch_size/topk_store_k`; fit() branch `len(active)>1` + `_pad_waves_to_pack_width` pins one captured bucket. **F1 parity ALL PASS** (`check_alllayer_graphed_parity.py`, GPU1): (a) σ=0 routing == greedy oracle; (b) graphed vs eager u BIT-IDENTICAL (torch.equal 256/256), logprobs max rtol 0.000e+00; (c) staggered-EOS [3,6,12] bit-for-bit. E2 staggered-EOS + E3 orchestrator (B=4 exact-fit AND B=3 one PAD) also bit-for-bit. **F2 e2e learns at the RIGHT LR** (`check_alllayer_e2e.sh`, 20 steps, all 28 down_proj): single-layer default LR=3e-2 DIVERGES (heldout-KL 3.30→6.24→7.25→13.59→6.44, 28 layers → ~28× aggregate step); LR=1e-3 LEARNS (0.5263→0.7417→0.8366→0.2465→0.2708, noisy but net DOWN ~half); all-layer NP needs ~30× smaller LR. pack_width KV cap: pack_width·2560 ≤ num_gpu_blocks → pw=4 fits w/ co-located teacher, 8 does not. **F3 goal proof NEGATIVE** (`bench_np_vs_bp.sh`, results `scripts/zo_opd/results/np_vs_bp_alllayer_graphed.txt`, batch64/1024/N8): NP one-step 2472.37s (peak 82372 MiB) vs BP 53.79s cold / 27.54s steady (peak 71710 MiB) → RATIO 45.97× cold / 89.78× steady, FAIL. NP breakdown: decode 1368s (55%) + assemble 835s (34%, CPU-bound per-token Python g_t/x_t reduction at np_worker_extension.py:1899-1911) + ~269s gaps; BP for contrast = gen 10.8s + teacher 30.3s + ONE FSDP backward 8.8s. **FOLLOW-UP (2026-06-07, `bench_noise_refill_isolation.py`): the decode 1368s was largely an ARTIFACT, not a fundamental floor** — the per-token noise refill (896 `draw_noise`/token = 28 layers × 4 prompts × 8 rails) is 74% of decode (48.0→12.4 ms/token, 3.9× when skipped); plus a per-token full `cuda.synchronize()` + 28 D2H captures/token + eager full-vocab topk. Fix plan: `docs/superpowers/plans/2026-06-07-np-decode-host-glue-optimization.md` (CPU-stage noise + drop per-token sync + batch captures → decode → memory-bound floor). index + log updated.

## [2026-06-07] ingest | moe_compress idea-discovery Phase 1 (literature)
5-agent survey of MoE expert compression (prune/merge/low-rank/unstructured) + the train-free-vs-finetuned gap. Filed `docs/aris/moe_compress/LITERATURE.md`. Key: the cell "which MoE expert-compression method matters AFTER short training" is empirically empty; dense analog (A Free Lunch 2510.14444) predicts gap closes, stronger at scale. **SlimQwen (2605.08738)** already asserts the hypothesis at 400B-token / 80B-model scale (whole-expert only) — our novelty = SHORT-recovery trajectory, ALL families, experts isolated, small MoE (OLMoE). User decisions: base=OLMoE-Instruct; naive control=SlimQwen expert-compression part.

## [2026-06-08] ingest | moe_compress idea-discovery Phases 2-3 (ideas + novelty)
Phase 2: GPT-5.4 generated 10 ideas → filed `IDEA_REPORT.md`. Converged to 1 paper, 3 pillars (A trajectory atlas / B family>criterion / C predictive diagnostic). De-risked: OLMoE experts = per-Linear in verl-env tfm 4.56 (existing tooling works; fused-3D is sft-env-only) → memory `olmoe-experts-per-linear-in-verl-env`. User decisions: all-3-pillars, breadth-across-families slate. Phase 3: 2-agent novelty search + GPT-5.4 verdict → filed `NOVELTY_CHECK.md`. **5.5/10 PROCEED-WITH-CAUTION**: Claim1(A+B)=paper (narrow to cross-family INVERSION as the sharp result), Claim2(C)=supporting mechanism, DROP effective-rank from headline. "Methods converge" headline is taken (SlimQwen+A-Free-Lunch); our edge = cross-family + low-rank/unstructured + experts-isolated + variance decomposition + short-horizon curve.

## [2026-06-08] ingest | moe_compress idea-discovery Phase 4 (design review)
GPT-5.4 design review → filed `RESEARCH_REVIEW.md`. **4.5/10 as specced — 3 identifiability killers** (n=1/family collinear; "retain 0.75" not a common budget across families; frozen router = family-specific confound) + 2 (thin tasks; underpowered Claim 2). MVP = 6 methods (3 families x 2) x 2 retains (0.75/0.50) x 3 seeds = 36 runs; eval {0,100,500,2k}; pre-registered inversion criterion. **User sign-off on 2 protocol changes vs original instruction:** (1) router TRAINABLE during recovery (experts-only compression preserved, but router re-adapts; frozen=ablation); (2) STANDARDIZED calibration primary (native recipes→appendix). Both reduce confounds. Next: research-refine-pipeline → EXPERIMENT_PLAN.

## [2026-06-08] ingest | moe_compress idea-discovery Phase 4.5+5 (refine + plan + final)
research-refine-pipeline: method stable (reused settled thesis, no re-refine). Wrote FINAL_PROPOSAL / EXPERIMENT_PLAN / EXPERIMENT_TRACKER / PIPELINE_SUMMARY / MANIFEST under docs/aris/moe_compress/. Cataloged all 8 pages in docs/index.md. User-confirmed: router TRAINABLE in recovery, STANDARDIZED calib primary. Plan = Phase0 smoke (gate G0) → 6 methods (3 families×2) → 36-run atlas (×2 retain ×3 seed, steps {0,100,500,2k}) → pre-registered inversion test + family>criterion variance → LOFO step-0 diagnostics. ~210-320 GPU-h, eval-dominated. **IDEA-DISCOVERY PIPELINE COMPLETE.** Next: /run-experiment (Phase 0) or /experiment-bridge to build src/moe_compress.

## [2026-06-08] experiment-bridge | moe_compress training-free leg IMPLEMENTED + LAUNCHED
Built src/moe_compress: compress_olmoe.py (driver, config-sync for num_experts+intermediate_size), methods/ (7 plugins: random_drop/reap_drop/slimqwen_merge/hcsmoe_merge/svd_llm_v2/sparsegpt/magnitude — reuse src/compress Nystrom/SparseGPT), calib.py (standard 256x2048 + coverage), budget.py (dual storage/active-capacity axes), eval_tasks.py (lm-eval 4-task). scripts/moe_compress/{run_trainfree_atlas.sh, analyze_atlas.py}. **Gate G0 PASS** (compress→reload→eval verified all families). Validated: baseline MMLU=0.542 (matches paper), budget axes (drop active=1.0 vs svd active=0.75). Gotchas → memory: OLMoE per-Linear in verl env; SparseGPT needs memory_limit_gb=2.5 (per-layer groups) else hangs. **LAUNCHED** 14-job training-free atlas (7 methods x 2 retains x seed0, eval limit 200) on GPU 1/2/3. Next: collect step-0 ranking via analyze_atlas.py; then recovery-SFT phase.

## [2026-06-08] results | moe_compress TRAINING-FREE leg complete (12/14 focal jobs)
Step-0 atlas done (eval limit 200). Filed `EXPERIMENT_RESULTS.md`. HEADLINE: the **weight-approx family (SVD+SparseGPT) — which SlimQwen omits — WINS step-0 at both retains** (means @0.75: 0.491 vs expert-removal 0.422 vs merge 0.357; @0.50 the gap explodes: 0.424 vs ~0.25). **SparseGPT is near-LOSSLESS training-free** (@0.75 MMLU 0.549 ≥ uncompressed 0.542; @0.50 0.521). Both FAMILY and CRITERION matter at step-0 (within weight-approx@0.50: SparseGPT 0.521 vs SVD 0.341). This directly motivates the paper: SlimQwen's "no method dominates" is partly an artifact of only testing whole-expert prune/merge. Sets up the inversion test with max leverage. Next: recovery-SFT harness in VERL env (per-Linear ckpts incompatible with sft-env tfm-5.2 fused-3D) — HF Trainer, freeze attn, train experts+router, ckpts {0,100,500,2000} x3 seeds.

## [2026-06-08] results | moe_compress training-free atlas FINAL (14/14, magnitude control fixed)
Magnitude control bf16-kthvalue bug fixed (fp32 quantile-on-sample). FINAL step-0 means @0.75: control(magnitude) 0.546 ≈ weight-approx 0.491 > expert-removal 0.422 > merge 0.357. **Sharpened finding: GRANULARITY (weight-level vs expert-level) dominates criterion sophistication** — naive magnitude (0.516@0.50) ≈ SparseGPT (0.521) ≫ all drop/merge (0.24-0.29). Reframes the recovery question: does training close the granularity gap? All 14 metrics in /data/yequan/moe_compress/metrics/. Training-free deliverable COMPLETE.

## [2026-06-08] results | moe_compress + Nystrom (relabel) + MoBE added
Renamed mislabeled svd_llm_v2 plugin -> nystrom (it always called nystrom_compress_model); relabeled its existing results. Implemented MoBE (src/moe_compress/mobe.py, training-free shared-basis weight reconstruction; gotchas: softplus not relu, lr 0.02, factor-budget reporting → memory mobe-impl-notes). Ran mobe @0.75/0.50. **MoBE COLLAPSES the model (MMLU 0.246@0.75, 0.231@0.50 — near-chance, worse than random drop)**: shared-basis (m=8 for 64 experts) too strong for a small MoE; published wins are at 235B/512-expert scale. Weight-approx family now SPLITS: SparseGPT near-lossless / nystrom mid / MoBE collapse → report per-method, not family-mean. EXPERIMENT_RESULTS.md + index updated. Full step-0 atlas = 8 methods × 2 retains = 16 jobs DONE.

## [2026-06-08] results | moe_compress + svd_llm_v2 (real whitening SVD) + nystrom_combined (fwd+bwd)
Added 2 more weight-approx methods (family now 5: sparsegpt/nystrom_combined/nystrom/svd_llm_v2/mobe). Results @0.75 MMLU: sparsegpt 0.549 > nystrom_combined 0.499 > nystrom 0.474 > svd_llm_v2 0.450 ≫ mobe 0.246 (same order @0.50). FINDINGS: (1) fwd+bwd Nystrom > fwd-only Nystrom (gradient covariance helps, modest); (2) structured-triplet Nystrom > per-matrix whitening SVD (joint gate/up/down factoring preserves MLP function vs independent truncation). COST: nystrom_combined (full CE backward/batch ~50min@256seqs) + svd_llm_v2 (3136 cov hooks + 3072 SVDs, CPU-bound) are 5-25x slower → use 32-64 calib seqs in recovery atlas. EXPERIMENT_RESULTS.md + index updated. Step-0 atlas now 10 methods × 2 retains = 20 jobs.

## [2026-06-08] ingest | reweighted_compress — KL-importance token reweighting (NEGATIVE)
New wiki page `wiki/reweighted_compress.md`: derive + test within-sequence token reweighting of calibration covariances. Idea: compress once (uniform seq-reweight) → run uncompressed teacher vs compressed student on calib traces → per-token forward KL `δ_t=D(p^T‖p^S)` → exp-tilt `w_t=min(exp(β·δ̃_t),w_max)` of per-seq-mean-normalized KL scales each token's activation outer product in `C_w=mean_seq[(Σ w v vᵀ)/(Σ w)]` → recompress. β=0 ≡ uniform (anchor). Code: `src/compress/calibration.py` (`_accumulate_cov(weights_2d=)` + `collect_covariances_weighted`, unit-tested w=1≡unweighted), `src/compress/kl_reweight.py`, `scripts/reasoning_aware_compress/kl_reweight_compress.py`. **RESULT @ Qwen3-4B nonthink / seq-reweight / full / retain 0.7 / MATH-500(100): NEGATIVE** — B(β=0) 67.0% / K-mid(β=1) 62.0% / K-sharp(β=2) 62.0% (−5pp). PPL flat-to-worse (96.9/95.6/100.6); relaxed==strict & n_reached tracks strict (67/62/62) → genuine reasoning loss, not looping. Interpretation: forward-KL chases *teacher-uncertain* tokens, not *task-leverage* tokens → budget pulled from what math needs. Cliff sweep (0.6/0.5) NOT run (no winner @0.7). Conclusion: within-seq axis exhausted, **uniform sequence-reweight (§5 of reasoning_aware_compress_calib) stays the recipe**; no production change. `results/reweight/kl_r0.7.json`.

## [2026-06-08] ingest | reweighted_compress_v2 — correct damage-aware lever (DESIGN, derivation)
New wiki page `wiki/reweighted_compress_v2.md` re-deriving v1's failed idea correctly (exploration, no impl/run). Diagnosis of v1's −5pp: two category errors — (1) wrong SPACE (reweighted *input* cov `C_x` in d_in; damage lives at the *output*/logits in d_out), (2) wrong SIGNAL (forward logit-KL `δ_t` ∝ teacher *entropy*, not task leverage). Correct derivation: Taylor-expand end-to-end `Σ_t D(p^T_t‖p^S_t)` → per-layer objective is `Σ_t ε_t^⊤ H^ℓ_t ε_t` with `H^ℓ_t = J^{ℓ→z⊤} F_t J^{ℓ→z}` (Fisher-curvature output-error metric), NOT `Σ‖ε_t‖²`. Aggregate `G_ℓ=Σ_t H^ℓ_t` = **backward (output-grad) covariance of a teacher–student KL loss** = task-leverage. The weighted layer fit `‖G^{1/2}(W−Ŵ)C_x^{1/2}‖_F` is the **doubly-whitened SVD already in repo** (`svd_compress_layer_combined`); MLP analogue = `collect_nystrom_combined_statistics`. Structure = **initialize→measure-realized-gap→refit→iterate** error-feedback, realized by EXISTING harnesses (`sequential_relinearized_compress` depth loop / `lr_sparse refine_passes`). Per-token "balance" = the LEFT METRIC `G_ℓ` (per-direction), not a scalar; optional `δ_t^α` curvature tilt = minimax worst-token protection (α the knob v1 morally wanted but on the wrong object). One new piece needed: plain forward-KL calib loss (~10 lines; `calibration_opd_loss` is a PG surrogate, related but ≠ Fisher). Why ≠ D-block null: D used CE backward on DENSE model, no reweight/refit; v2 = KL backward on COMPRESSED student, re-measured each refit. Proposed cells V0/C1/C2/C3 @ retain 0.7 vs 67% anchor / 62% v1; honest prior = gain (if any) below the cliff. index updated.

## [2026-06-08] recovery | moe_compress recovery-SFT LAUNCHED (nystrom/nystrom_combined/reap_drop @0.5)
Built src/moe_compress/recover_sft.py (verl-env HF Trainer: load per-Linear compressed ckpt, freeze attn / train mlp.* experts+router [87.2% trainable], completion-only SFT on OpenThoughts3, eval at steps {0,100,500,2000,final}, wandb olmoe_compress_sft). Gotcha: gradient-checkpointing + frozen embeddings → "element 0 does not require grad" → fixed with model.enable_input_require_grads(). Smoke PASS (train_loss 0.83). LAUNCHED 3 runs on GPU 1/2/3: nystrom, nystrom_combined (weight-approx, led step-0 @0.5: 0.341/0.382) vs reap_drop (expert-removal, collapsed step-0 @0.5: 0.289). 10k samples ≈1250 optim steps, ~2-4h/run. This is the first cut at the inversion test: does recovery close the weight-approx-vs-drop gap at retain 0.5? scripts/moe_compress/run_recovery.sh.

## [2026-06-08] recovery v2 | moe_compress OLMoE-native calibration + relaunch
User flagged: v1 calib used Qwen3-4B-generated traces (OpenThought3-Qwen3-4B), not OLMoE-native. (1) KILLED v1 recovery runs. (2) Regenerated 500 OLMoE-native traces (OpenThoughts3 prompts → OLMoE-Instruct, vllm_rollout.py; gotcha: OLMoE max_pos=4096 → max-model-len 4096/max-tokens 3072). Short traces (p50 525 tok, 153k total, min 21/expert, 0 dead). calib.py now defaults to native (env MOE_CALIB_JSONL override). (3) Re-compressed nystrom/nystrom_combined/reap_drop @0.5 native: **native calib slightly WORSE** (MMLU 0.338/0.349/0.276 vs Qwen 0.341/0.382/0.289) — fewer calib tokens (153k vs 428k) + terse OLMoE answers exercise reasoning experts less; **ORDERING UNCHANGED → headline robust to calib source**. (4) RELAUNCHED recovery on *_native ckpts (GPU 1/2/3, 10k OpenThoughts3 samples, wandb olmoe_compress_sft, tags *_native_sft). v1 recovery curves (Qwen-calib) preserved. Memory: openthoughts3-rollout-dataset-gen (OLMoE 4096 caveat).

## [2026-06-08] ablation | moe_compress nystrom_combined calibration-source split (recipe 1 & 2)
User Q: where should C_f (fwd) vs C_b (bwd) come from? Added 2 methods (nystrom_combined_fwdnat_bwdot3 = fwd native / bwd OT3; nystrom_combined_ot3 = both OT3). Split-source = 2 collect passes, zip C_f from one + C_b from other (cosmetic 0-pairs log bug fixed: compress consumes the dict). TRAINING-FREE @0.5: both-OT3 0.381 ≈ both-Qwen 0.382 > split(fwd-nat/bwd-OT3) 0.377 ≫ both-native 0.349. **FINDING: the BACKWARD signal source is what matters — moving only C_b native→OT3 recovers +0.028 (0.349→0.377), nearly matching all-OT3. C_f can stay on-policy/native. Split = principled (on-policy activations + target-quality gradients), ~tied with all-OT3.** EXPERIMENT_RESULTS.md §v2b. Launched recovery for both on GPU 1/2, wandb olmoe_compress_sft.

## [2026-06-09] calib v3 | moe_compress ORIGINAL OpenThoughts3 traces (QwQ-distilled)
User: use the dataset's OWN traces, not Qwen3-4B rerollout. Extracted 600 math convs (conversations field) from local OpenThoughts3-1.2M shards → ot3_original_math.jsonl (long: p50 ~17k tok). OLMoE template verified. Re-compressed nystrom/nystrom_combined @0.5. Calib-source ranking (MMLU): original-QwQ (0.365/0.384) > Qwen-rollout (0.341/0.382) > native (0.338/0.349). **Original traces best — richness/length > on-policy match.** calib.py OT3_JSONL now defaults to original (env MOE_OT3_JSONL). Launching v3 recovery.

## [2026-06-09] ingest | es_token trainer design (per-token weight-perturbation ES)
User-requested design for an improved verl ZO trainer `es_token`: per-token rank-1 weight perturbation (ΔW = σ(s_n⊙u_t)(r_n⊙v_t)ᵀ), N parallel perturbed rails on the clean KV (NP-V3 packed_graphed skeleton reused), fixed Hadamard sign buffers (exact rail orthogonality with Rademacher u,v), ONE fused per-token noise draw shared across rails (vs NP's 896 draw_noise/token = 74% of decode), no x capture, assembly = chunked GEMMs (~1.5 PFLOP ≈ 10–40s vs NP's 835s Python loop). Losses: sampled-token OPD (k=1, −Â_t·logπ_n(y_t)) + reuse of np/teacher_scorer topk-RKL. Unbiasedness shown (identity contraction for Rademacher/Gaussian). Predicted one-step ~470–520s vs NP-V3 2472s / BP 54s at batch64×1024×N8; honest risk = per-probe cosine below NP (probe oblivious to x_t) — gate V4 measures before scaling. Open Q flagged: user msg contained both pure-ES and "multiply with x" assembly forms; v1 = pure ES, np_hybrid reserved. → docs/plans/es_token_trainer.md

## [2026-06-09] build+bench | es_token trainer implemented, all gates PASS, 17x over NP-V3, 2.33x of BP
Implemented the full es_token trainer (user: pure-ES only, sampled-token OPD benchmark): verl/verl/trainer/es_token/{signs,seeding,grad_estimator,ray_trainer}.py + es_token_worker_extension.py (ESTokenLinear rank-1 rails subclassing NP-V3 WorkerExtension) + main/yaml/launcher + 19 CPU tests + GPU gates + bench. Key math decision validated in tests: unweighted sampled-token rail loss degenerates (teacher term cancels in rail differences); implemented student_iw = (pi_n/pi_0)(log pi_n - log q), unbiased single-sample KL(pi_n||q). Gates: sigma=0 == stock greedy (112 layers); graphed==eager BIT-FOR-BIT (payload diff 0); staggered-EOS clean; cosine 0.98x the sqrt(K/(K+d_out*d_in)) info bound (the ~40x per-probe gap to NP is exactly the d_in factor; training-scale K=B*T*N -> cos~0.2/layer). BENCH (64x1024, N=8, all 112 linears, co-located 4B teacher): one-step 145.8s = decode 128.3 (16 waves x 7.9s, at the rail floor) + teacher 4.2 (10x FASTER than BP's reward phase) + assemble 13.2 (parity with BP backward) vs NP-V3 2472s (17x) vs BP token_reward_direct 62.7s cold -> ES/BP 2.33x (NP was 46x). Remaining gap = serial pack-4 decode loop (511 vs 1134 tok/s). Known issue: post-measurement teardown hang (killed manually; untriaged). -> wiki/es_token_trainer.md, results: scripts/zo_opd/results/{es_token_gates,es_token_vs_bp}.txt, plan: plans/es_token_trainer.md

## [2026-07-17] ingest | Full vs FURA GRPO on Qwen2.5-7B (NERSC)
Launched 2× 2-node interactive GRPO jobs on Qwen2.5-7B base (MATH lv3–5 train, MATH-500 eval during, AMC23/AIME24/Minerva/Olympiad-Bench after), wandb `nersc_grpo_qwen2p5_7b`. New `slurm/grpo/{full,fura,eval}` infra reusing `opd_2node_inside.sh` via ENV_SCRIPT + `grpo_wandb_sync_daemon.sh`. Downloaded Qwen2.5-7B. Fixes: (1) grpo.sh RAY_EXTERNAL/RAY_ISOLATE support; (2) overridable CKPT_PATH/EXPERIMENT_NAME (were timestamped → broke auto-resume); (3) exit-code propagation (trailing `if` masked crashes as exit 0 → controller stopped instead of retrying); (4) `verl/.../fsdp_workers.py` forces real-weight init when peft.mode∈{blocktt,svd} — 7B (tie_word_embeddings=False) meta-inits → BlockTT SVD `model.to(cuda)` "Cannot copy out of meta tensor"; (5) env-knobs TRAINER_LOGGER/PPO_MAX_TOKEN_LEN_PER_GPU/VAL_ONLY. Full validated: step-5 MATH-500 acc@4=0.526, mem 76.4/80GB, synced online. FURA relaunched after fix #4. See results/fura_grpo.md.

## [2026-07-18] results | Full vs FURA GRPO on Qwen2.5-7B — COMPLETE
Both 138-step runs done (auto-resume: full 3 segments, fura 2). MATH-500 acc@4: Full 0.526→0.668 (peak 0.682), FURA 0.498→0.633. Post-training val_only on 4 benchmarks (avg@16, mean@16 Full/FURA): AMC23 0.362/0.345, AIME24 0.060/0.081, Minerva 0.276/0.247, Olympiad-Bench 0.307/0.281. FURA (only BTT factors trainable) within ~2-3pts of full, ahead on AIME24; mem 60.8 vs 76.4GB. FURA eval needed a salloc retry (transient NERSC "Connection timed out"). All on wandb nersc_grpo_qwen2p5_7b. Full writeup: results/fura_grpo.md.

## [2026-08-20] ingest | ES on math reasoning (Qwen2.5-Math-7B, MATH-500)

Built the ES math-reasoning thread: Qwen-Math-template data prep, `qwen_math` task/reward
(OatZero grader), one-shot activation calibration, and four subspace-restricted ES
perturbation modes (dense / zoact / insparse / fura) in `StructuredESMixin`, all with fp32
coefficient masters. Base-model check: MATH-500 51.2-51.6 vs paper 53.0. Runs 1 (dense) and
2 (ZO-Act r=1) launched on GPUs 1/2; 3 (insparse) and 4 (fura) queued.
-> `docs/results/ES/es_results.md`

## [2026-08-21] ingest | ISO fixed-spectrum ES (runs 5-6) — derivation, implementation, launch

Derived and implemented the ES analogue of *ISO: An RLVR-Native Optimization Stack*
(arXiv:2607.19331) for the ES-q2p5-7b thread. Key step: ISO's fixed-spectrum family
`F(W0) = {U S0 V^T}` is exactly the **bi-orthogonal orbit** `{C_L W0 C_R^T}`, so a Cayley
perturbation of a block-diagonal skew keeps both frames orthonormal and the spectrum
*exactly* fixed with no SVD and no retraction — necessary because ES needs N=30 feasible
perturbations per iteration, where ISO's own fp64-SVD polar retraction (1st-order exact
only) would need ~6e3 large SVDs. Two new modes in `StructuredESMixin`: `iso` (full-matrix,
both frames) and `isobtt` (same constraint on the block-wise SVD; `R_j in O(b)` trained,
`A_j` + per-block spectrum frozen -> trainable state 6.53B -> 97.8M, a clean one-variable
ablation vs the existing `fura` run). sigma is redefined as the *relative footprint*
||dW||/||W|| and set to 5e-2 to match dense ES (the paper's 1e-3 would sit below the
1.6e-3 bf16 rollout floor); alpha=sigma/2 then reproduces dense's per-iteration motion
alpha/sqrt(N)=4.6e-3 exactly. Verified in `scripts/es/test_iso_es.py`: kernels match a
materialised `P^T blkdiag(C) P` to 9e-17 (fp64), and fp64 spectrum drift 2.8e-14 vs fp32
2.3e-8 proves the residual is round-off not algebra; on real Qwen weights ||d(sigma)||/
||sigma|| = 2.5e-8 at ||dW||/||W||=5.0e-2. Launched run 5 (`iso`) on GPU 5 with run 6
(`isobtt`) chained; step 0 = 51.6 (matches every other run), iteration 1 reward_std 0.075
(vs dense 0.084) and frob_drift 3.5e-6, 403.7 s/it (+14%). Also flagged a pre-existing
quirk: `_es_noise` reseeds per layer with the bare seed, so same-shaped layers in runs 1-4
draw identical noise (28x search-dimension loss); the ISO modes mix in the layer id.
-> `docs/results/ES/es_results.md` §10


## [2026-08-21] ingest | ES runs 1-2 complete; ZO-Act r=1 matches full-weight ES

Runs 1 (dense, paper ES) and 2 (ZO-Act r=1) finished 150/150 iterations. MATH-500
51.6 -> 73.4 (dense best @ 40) / 72.2 (ZO-Act best @ 130); plateau means over steps
100-150 are 71.50 vs 71.00, a 0.50 pp gap against a >=0.8 pp SE floor -- statistically
indistinguishable from 5,500x fewer trainable coefficients. Both flatten by step 40 while
train reward spread decays to ~0.02, pointing at the fixed 64-problem batch (not the
iteration count) as the binding constraint on the remaining 4.6 pp vs the paper's 78.0.
Runs 3 (insparse) and 4 (fura) auto-started on GPUs 1/2.
-> `docs/results/ES/es_results.md`

## [2026-08-21] results | ISO fixed-spectrum ES matches unconstrained ES; fp32 gain-drift found + fixed

**Headline.** Run 5 (`iso`, spectrum of every weight exactly frozen) is statistically
indistinguishable from run 1 (unconstrained dense ES) on MATH-500: paired over the 12
shared eval steps, iso - dense = **+0.28 pp +- 0.37 (s.e.), t = 0.77**; plateau means
(step>=40) dense 71.8+-1.19 vs iso 72.3+-0.87 (zoact 70.5+-0.94). A single n=500 eval has
s.e. 2.24 pp, so the best-of-15 column (dense 73.4 / iso 74.0) is a max-statistic, not a
ranking. => on this task **all ~20 pp of ES gain is singular-frame rotation**; the singular
values are inert. That is ISO's spectral-inheritance claim tested in its strongest form
(exactly fixed, not approximately) and in a forward-only ES setting the paper doesn't cover.
Runs 1/2 finished 150/150; runs 3 (insparse) / 4 (fura) at ~21 steps on GPUs 1/2; run 5 at
128/150 on GPU 5 with run 6 (`isobtt`) chained.

**Bug found and fixed: fp32 Cayley accumulation drifts as an isotropic gain.**
`iso/frob_drift` grew linearly (log-log slope 1.03, +3.43e-6/iteration) to 4.3e-4 by step
126 -- systematic, not a round-off random walk. Diagnosed: reproduced offline (+5.91e-6 x t
in fp32, 1e-16 in fp64, TF32 confirmed off), and decomposed -- every singular value scales
by the *same* factor (per-mode ratio sd 7.6e-6), so the *shape* of the spectrum is
preserved to 1.7e-7 and only a scalar per matrix drifts. Fix `_iso_recondition` (called
from `_iso_commit`, cost ~= the norm already computed for the metric): `iso` renormalises
`state *= ||W0||/||W||`; `isobtt` takes one Newton-Schulz step `R <- R(1.5I - 0.5 R^T R)`.
Verified over 300 real `es_update` calls: `iso` 2.7e-4 -> 5.6e-7, `isobtt` flat at 1.63e-6
(uncorrected 7.6e-4, of which 3.0e-4 is genuine *shape* error -- so isobtt is the mode that
actually needed it). Real-shape smoke: ||R^T R - I|| 1.0e-5 -> 8.3e-7 in 0.3 ms/layer.
Run 5 ran uncorrected but is unaffected: its shape error stayed ~3e-7 and its 5e-4 gain is
3x below the 1.6e-3 bf16 quantisation the vLLM forward already applies. Matters at horizon:
uncorrected the linear growth reaches 3.4e-2 at 10k iterations.
-> `docs/results/ES/es_results.md` §7 (headline) + §10.10 (drift)

## [2026-08-21] runs | isobtt (run 6) launched on GPU 3, not chained behind run 5

Moved run 6 off the GPU-5 chain (chain killed to avoid a duplicate) and launched it on the
now-free GPU 3, so runs 3/4/5/6 train concurrently on GPUs 1/2/5/3. Init matches the
analytic prediction exactly: coef_params 97,771,520 (the *same* tensors as run 4's fura),
base_params 6,525,288,448 frozen bf16 A, manifold_dim 48,470,016. Step-0 eval **53.2%,
byte-identical to run 4's 53.2%** -- both start from the same BTT reconstruction, so run 4
vs run 6 is a clean one-variable ablation (free additive core vs core constrained to O(b))
off a shared baseline. Iteration 1: reward_std 0.0583, train acc 57.6%, 372 s/it (ETA
~15.5 h). The `_iso_recondition` fix is confirmed live at 7B: frob_drift 5.83e-5 (pre) ->
orth_err 1.07e-6 (post Newton-Schulz), a 54x reduction, pinned rather than accumulating.
-> `docs/results/ES/es_results.md` §7, §10.10


## [2026-08-21] ingest | FuRA LR search: sigma, not alpha, is the knob

FuRA at the paper's sigma was step-size-starved. Sweeping alpha at fixed sigma=1e-3 helps
(1x: 60.2@20 -> 12.5x: 70.0@15) but destabilises by 40x (train acc peaks at step 3, decays
to 58). Scaling sigma AND alpha together (sigma=1.25e-2, alpha=sigma/2) gives MATH-500
**74.2 @ step 15** -- above dense's best (73.4 @ 40) and 2.7pp above its plateau. The two
share alpha and differ only in sigma, and the train/test gap inverts (-5.4 vs +10.0):
sigma is the Gaussian-smoothing bandwidth of the ES objective, i.e. a regulariser, while
alpha only takes bigger steps on the same sharp objective. Qualifies the earlier
'64-problem batch is the ceiling' finding -- that ceiling is sigma-dependent.
-> `docs/results/ES/es_results.md` section 11.3

## [2026-08-21] profile | es_token trainer — gradient cosine, decode throughput vs N, one-step reproduction
User: profile the `feat/es-token-trainer` branch in a separate worktree (`OPD-estoken`) and file results under the ZO-ES-token section of results/zo_opd.md, collecting prior profiling too. Built 4 new harnesses (`scripts/zo_opd/es_token_checks/{sweep_grad_cosine.sh,bench_decode_throughput.py,sweep_decode_throughput.sh,sweep_stock_batch.sh,sweep_decode_isolation.sh}`), all on H100 NVL. **(1) Gradient quality**: cos(es dW, autograd) sits at **0.86–0.99× the rank-1 weight-probe bound** sqrt(K/(K+d_out·d_in)) over K=400..4800 and two layer shapes (down_proj 2048×6144, o_proj 2048×2048); √K scaling verified; **rails and repeats are interchangeable at equal K** (N=8×300 = +0.0130 vs N=16×150 = +0.0136) → N is a cheap way to buy probes, not extra quality per probe; per-probe cosine scales as 1/√(d_out·d_in), which is exactly the cost of probing weight space instead of NP's output space. **(2) Decode throughput** (ms/token-step from the T=64→320 slope, EOS disabled): clean-only (N=0) through the packed graphed driver = 2.939 ms vs **stock vLLM cudagraph at the same concurrency = 2.831 ms — only 4% driver overhead** (11% at pw=8), which corrects the earlier "511 vs 1134 tok/s" framing (that was a concurrency gap, not a driver gap). Switching rails on costs **+3.41 ms (N=0→1)**, then only **+0.10 ms/rail** (N=1..32: 6.35→9.43 ms = 11× probes for 1.49× time). `pack_width` 4→8 = **1.80× clean tok/s for +11%**; pw=16 fails on the **full-context scratch-KV reservation** (b_pack 16 × 2560 blocks > 24717) — the highest-leverage fix. **(3) Cost attribution** (ES_BENCH_SKIP_NOISE): 36% bare graphed decode / 4% fused noise draw / **60% 112-layer rank-1 rail compute**. **(4) One OPD step reproduced**: 147.54 s (decode 129.59 + teacher 4.22 + assemble 13.65, peak 85,129 MiB) vs BP-OPD 61.86 s (gen 14.58 + reward 35.61 + logprob/adv/update 13.94, peak 71,710 MiB) = **2.39×**, matching June's 2.33×, step time within 1.2%; both sides emitted exactly 65,536 response tokens so phases are like-for-like; teacher phase **8.4× faster than BP's**, assembly at parity with BP's backward, **100% of the residual gap is decode concurrency**. Known-issue **teardown hang reproduced** (240 s, manual SIGTERM needed before the BP side could start). Still open: LR sweep / learning quality — this page remains a wall-clock + correctness result. → results/zo_opd.md §ZO-ES-token, raw: scripts/zo_opd/results/es_token_{grad_cosine_sweep,decode_throughput,stock_batch,decode_isolation}.txt

## [2026-08-22] results | ES thread COMPLETE at 150/150 — fixed-spectrum matches unconstrained ES

Five of six primary runs finished 150 iterations. Paired vs run 1 (dense) over all 15
shared eval steps, in pp of MATH-500: **iso +0.44+-0.31 (t=+1.40)** and **isobtt
-0.37+-0.48 (t=-0.78)** -- both statistically indistinguishable from unconstrained ES;
insparse -0.39+-0.47 (t=-0.82); zoact r=1 **-2.61+-0.87 (t=-3.02), the only mode
significantly worse**. So with every singular value of every matrix exactly frozen the
model still goes 51.6 -> 72.4 on MATH-500: **all ~21 pp of ES gain is singular-frame
rotation**, and isobtt gets there on a 48.5M-dim manifold (0.64%) with the per-block
spectrum frozen.

**The step-size-vs-subspace confound is resolved** by the user's sigma-matched controls.
`fura` at sigma 4.0e-3 -> 1.25e-2 goes **-12.25 pp -> +0.82 pp** vs dense: same
factorisation, same tensors, only the footprint changed -- the BTT subspace was never the
problem. `zoact` goes the *other* way (-2.61 -> -4.23) and was actively collapsing (train
acc 78% -> 50%) before dying with `Aborted (core dumped)` at step 73: a rank-1 subspace
concentrates the whole footprint into one activation direction, so a 12x step destabilises
it. alpha-only sweeps on fura confirm sigma and alpha must move together (a12.5x -> 70.0,
a40x -> 68.2 then divergence to 61.0). General reading: what matters is how *widely* the
perturbation energy is spread, not which structured basis it lives in.

**Drift fix validated over a full run.** Run 6's post-Newton-Schulz `iso/orth_err` is flat
at ~1.07e-6 across all 150 updates (max/first = 1.17x, no accumulation), against a per-step
pre-fix violation of ~1.0e-5. Run 5 (uncorrected) ended at frob_drift 5.13e-4, matching the
predicted 3.43e-6 x 150 = 5.1e-4 -- a pure isotropic gain, 3x below the bf16 floor.

Still running: `fura` sigma-matched long at 117/150 (GPU 2). Crashed/stopped: zoact
sigma-matched at 73/150; original run 4 fura terminated at 42/150 to free the GPU.
-> `docs/results/ES/es_results.md` §7


## [2026-08-22] ingest | Aligned with the official es-at-scale implementation

Read github.com/VsonicV/es-at-scale. Template / z-score shaping / alpha=sigma/2 / population /
greedy decoding / seed schedule / bfloat16 all already matched. Two real gaps: the official
math run uses **batch 1024 RESAMPLED every iteration** from an 8.5k pool
(DataLoader(shuffle=True)) and max-tokens 3000, where we held ONE fixed 64-problem batch at
1536 tokens -- exactly the overfitting mechanism measured in section 11.1. Implemented
`es.train_batch_size` per-iteration resampling; launched a sigma search (1e-3/2e-3/4e-3/8e-3,
alpha=sigma/2) at aligned settings, batch 128, 3000 tokens. Also corrected section 11.3/11.4:
FuRA's plateau edge over dense is +0.92pp (72.73 vs 71.82), not the ~3pp the step-15 point
suggested.
-> `docs/results/ES/es_results.md` sections 11.5, 12

## [2026-08-22] optimize | es_token fused rail kernel — the fixed rail overhead was graph-node count, not RNG
User asked why N=0→N=1 doubles ms/token-step while N=1→N=8 barely moves, and set the goal N=1 < 3.5 ms; hypothesised CPU-side RNG. **RNG ruled out**: `draw_noise` already builds `torch.Generator(device=cuda)` and draws on-GPU, and the isolation delta prices the fused per-(slot,token) draw at 0.199 ms which is paid at N=0 too — so it is not in the N=0→N=1 delta at all. **Real cause: CUDA-graph node count.** `bench_rail_op.py` (new) replays the rail op alone, graphed, at true Qwen3-1.7B shapes for all 112 linears and reproduces the delta exactly (3.376 ms vs the 3.41 ms measured live); torch-profiler shows **1568 kernels per decode token at ~2.3 µs each** = 14 kernels/layer (672 gathers 1.80 µs, 224 non-vectorised elementwise 2.69 µs, 336 vectorised 1.62 µs, 112 index_put 4.73 µs, 112 reduce 4.54 µs). Two corollaries: >half the launches gather operands that depend only on the token (R[rail], v[pidx], S[rail], u[pidx]), and the non-vectorised rows are the strided-view penalty from `noise_buf[:, off:off+d]` (row stride d_total=917504). **Seven variants built and measured** (rail-op ms @N=1): v0 3.376 → v1 flat sign*noise 2.373 → v2 vecdot/addcmul 1.992 → v3 contiguous rows 1.138 → v4 bmm 0.765 → v5 blocked-noise 1.527 → v6 Triton (needs v3 layout) 0.491 → **v7 Triton reading row indices 0.478, tuned (BLOCK 4096/4096, 16 warps) 0.313 = 10.8×**. v7 shipped as `verl/verl/trainer/es_token/rail_kernel.py` + `ESTokenLinear.forward` (PyTorch fallback kept): one launch per layer, forms sign*noise on the fly, needs **no packed-row-layout change** (NP attention/KV metadata untouched) and never materialises the [P,d_total] buffer (235 MB at N=32). Tuning insight: grid is only bucket*n_sample programs → latency-bound, so big blocks beat occupancy. **End-to-end**: decode N=1 6.347 → **3.424 ms (goal <3.5 MET**, overhead 3.408 → 0.481, 7.1× less), N=8 7.600 → 3.885, N=32 9.429 → 5.208; **one OPD step 147.54 → 89.59 s (1.65×)**, decode 129.59 → 72.00 s (waves 7.85 → 4.25 s), **ES/BP 2.39× → 1.45×**, peak mem unchanged. **Correctness**: 19/19 CPU tests, new `check_rail_op_parity.py` ≤3.0e-06 vs the shipping op for every variant (compared in each variant's own row layout), GPU gates all green (σ=0 ≡ stock greedy, graphed ≡ eager BIT-FOR-BIT payload diff 0, staggered-EOS bit-for-bit), and step `L_clean_mean` bit-identical (0.2556177764199674) so the clean trajectory is untouched; dW_norm_mean 239.726 → 239.514 (rail now accumulates fp32 not bf16). Remaining headroom: the noise draw (int64 randint 7.34 MB/slot + 5-kernel cast chain, 0.199 ms, ALSO paid per token in the 13.6 s assemble — decode and assembly must move together to stay bit-identical) and pack_width. → results/zo_opd.md §6, wiki/es_token_trainer.md §7, raw scripts/zo_opd/results/es_token_rail_op.txt

## [2026-08-22b] optimize | es_token direct Rademacher noise — 13.5x cheaper fill, step 89.6 -> 83.8 s
User: "we can directly have bf16 noise. also, try directly draw from rademacher distribution with +-1." Both done. **Problem**: `draw_noise(method="bernoulli")` produced ±1 the long way — `torch.randint(0,2,dtype=int64)` (a 7.34 MB buffer per slot at d_total=917,504) → `.to(float32)` → `*2-1` → `.to(bf16)` → `copy_`: ~42 MB of traffic and ~6 kernels **per slot per token** for 1.83 MB of output, plus a fresh `torch.Generator` per slot and a host blake2b inside the decode loop — and the same routine ran once per token record during assembly (65,536/step). **Fix** (`verl/verl/trainer/es_token/noise_kernel.py`): draw ±1 **directly in the destination dtype**, one Triton launch per batch of rows, from counter-based Philox (`tl.randint` → low bit → ±1) — no generator state, no host RNG, no intermediate. Plus `build_seed_table()` hoists every (token, slot) seed out of the decode loop (no blake2b, no H2D in the hot path), and assembly fills a whole chunk in ONE launch instead of m × ~6 kernels. Torch fallback kept (`Tensor.random_(0,2)` straight into the destination) for non-Triton envs and non-bernoulli methods; impl chosen once at import so **decode and assembly can never disagree within a run**. **Isolated**: decode fill 0.203 → **0.015 ms**/token×4slots (13.5×); assembly 1024-row chunk 38.9 → **2.9 ms** (13.4×). **End-to-end**: decode N=0 2.943 → 2.783, N=1 3.424 → **3.244**, N=8 3.885 → **3.722**, N=32 5.208 → 4.956, pw=8 4.547 → 4.237; **one OPD step 89.59 → 83.80 s**, decode 72.00 → 68.44 s, **assemble 13.56 → 11.03 s**, ES/BP 1.45× → **1.35×**. **Cumulative with the fused rail kernel: step 147.54 → 83.80 s (1.76×), decode 129.59 → 68.44 s (1.89×), ES/BP 2.39× → 1.35×.** **Correctness**: new `check_noise_parity.py` gates the regeneration invariant — decode path (per-wave seed table, device-resident) == assembly path (host-derived, freshly uploaded) BYTE-FOR-BYTE every token; chunk row j == its own (t,rollout) record; values exactly {-1,+1}; |mean|<0.02/row; distinct across t and rollout; regeneration bit-identical — ALL PASS. 19/19 CPU tests, GPU gates all still green (σ=0 == stock greedy, graphed == eager bit-for-bit payload diff 0, staggered-EOS). `L_clean_mean` bit-identical across all three versions (0.2556177764199674) so the clean trajectory never moved; `dW_norm_mean` 239.51 → 243.78 because Philox counter mode is a different stream from torch.randint (nothing depended on the old one; both consumers moved together). Decode is now 82% of the step and the noise fill 0.5%, so **pack_width is the only lever of consequence left** (scratch-KV reserves 2560 blocks/slot regardless of the real 1024-token budget, capping concurrency at 4-8 vs BP's 64). → results/zo_opd.md §7, wiki/es_token_trainer.md §8, raw scripts/zo_opd/results/es_token_noise.txt

## [2026-08-23] ingest | BP (true-gradient) counterpart of the ISO thread — 4 GRPO arms launched

Built the first-order counterpart to the ES fixed-spectrum work: a new
`verl/workers/peft/iso.py` PEFTAdapter exposing `iso` / `isobtt` / `isobtt_mix` to
verl's FSDP actor, so ES-vs-BP is a controlled comparison of the *optimiser* on an
identical task (Qwen2.5-Math-7B, MATH lv3-5 -> MATH-500, GRPO). Key design: every
orthogonal factor is `Cay(Omega)` for a **trainable skew**, so the fixed-spectrum
constraint holds for *any* optimizer step -- plain AdamW + plain FSDP, no Riemannian
optimizer, no retraction, nothing to drift off (unlike ES, which needed
`_iso_recondition`). Omega=0 init => step 0 is the pretrained model bit-exactly. The
base weight is never materialised in training: each mode is 1-2 cheap block-diagonal
rotations around the frozen linear (+3.6% flops for iso, +0.9% for isobtt*).
`isobtt` uses the right-rotation form `W[:,blk_j] = W0[:,blk_j] C_j`, equivalent to
the ES SVD form but needing no SVD -- which also removes the 1.6e-3 bf16
reconstruction floor that makes ES fura/isobtt start at 53.2 instead of 51.6. The
input mixer is constrained to **O(n_blk)** (user-confirmed): as a full-input operator
it is `M (x) I_b`, orthogonal iff M is, so the arm stays inside F(W0) -- a free M
(as in the cited lift_commonsense ablation) would leave the family.

Verified in `scripts/es/test_iso_bp.py` (all PASS): identity init bit-exact for all
3 modes; **spectrum fixed to <=9.3e-8 after 5 real AdamW steps while ||dW||/||W|| =
0.48-0.75**; Cay orthogonal across 4 decades of scale; grad w.r.t. Omega is skew;
materialize()==forward() at random Omega. That last check caught a **real bug**:
materialize() applied C_L^T where forward applied C_L -- invisible at Omega=0, so
the first test version passed, but vLLM's rollout weights would have silently
disagreed with the trained policy. Two more integration bugs found by a 0.5B smoke
run before spending 7B GPU-hours: `export_for_vllm` returned views into FSDP's
`summon_full_params` storage (freed on exit -> vLLM read `storage size 0`; BlockTT
never hit it because its runs are FSDP2), and a missing
`enable_input_require_grads()` (frozen embeddings -> checkpointed blocks return
detached outputs -> loss has no graph).

Launched 4 arms back-to-back on GPUs 6+7 (2-GPU FSDP each, user-confirmed layout):
dense (LR 1e-6) / iso / isobtt / isobtt_mix (LR 5e-6, matched on per-step relative
weight motion -- analytic, not swept). 138 steps = 1 epoch. Measured **157.5 s/step**
=> ~6 h/arm, ~24-30 h total. Step 1: grad_norm 0.933, train reward 0.244.
Host note: `/` is 96% full (76 GB free); raylet warns object spilling may fail.
-> `docs/results/ES/es_results.md` §11

## [2026-08-23] optimize+launch | es_token budget-sized scratch-KV: pack_width unlocked, ES now FASTER than BP; training launched
User: "resolve it, we should not have excessive kv cache. do not care about NP implementation... re-profile after the fix, then launch opd and zo-es-token training... wandb project zo-opd-q34b-1p7b, gpu 1 sequentially." **Problem**: `_np_prefill_packed` carves a private KV region off the top of vLLM's block pool (the driver bypasses vLLM's scheduler and needs static KV for a captured graph) and sized each slot's slice at the FULL `max_model_len` — ceil(40960/16)=2560 blocks against a 24,717-block pool = **9 slots max**, ~20x what a 1024-token generation needs. **Fix**: reserve (longest prompt + max_tokens), capped at max_model_len; `max_new_tokens=None` preserves the old behaviour so NP is untouched; `ES_KV_FULL_RESERVE=1` A/B knob. Added a second assert because the attention block table is zero-filled and only `len(block_ids)` entries are written — an undersized slice would read block 0 and **silently corrupt a neighbour's KV rather than crash**. **The gate had to be rewritten, and that was the substantive finding**: the obvious check (packed clean tokens == stock greedy) FAILS at pack_width>=10, but it is bf16 rounding, not corruption — (i) neighbour-independence: hold slots 0-3 fixed and swap the CONTENT of every other slot → byte-identical output, so slices do not alias (PASS at 4/8/16/32/64); (ii) output changes with wave WIDTH alone (slots 0-3 identical at width 4 and 16, differ at 32); (iii) it appears at the shipping pack_width=4 too, where the reservation change is provably byte-neutral; (iv) divergent slots come in pairs (i, i+8) = same prompt text, and divergence compounds with length. The hand-driven packed forward batches differently from vLLM's scheduler → different bf16 rounding → greedy argmax flips on near-ties. Gate now checks [A] output-neutrality vs the old reservation ACROSS PROCESSES (flipping it in-process reuses the already-captured graph and is vacuous — that artifact initially looked like a pw=8 failure), [B] neighbour-independence, [C] full-length generation. All PASS; check_es_parity's 3 gates unchanged; cross-process [A] byte-identical (payload diff 0.0) at pw=4 and 8. **Throughput** (N=8): pack_width 4/8/16/32/64 → 3.734/4.259/5.435/8.431/15.036 ms per token-step = 1071/1878/2944/3795/**4257** clean tok/s; at pw=64 a 64-prompt batch is **ONE wave** instead of 16. **One OPD step 83.80 → 42.67 s** (decode 68.44 → 25.40, teacher 3.99, assemble 13.18), peak mem unchanged, weight_sync_ok 1.0. **Cumulative over the three optimisations: 147.54 → 42.67 s (3.46x), decode 129.59 → 25.40 s (5.10x), ES/BP 2.39x → 0.69x — es_token is now FASTER than BP-OPD.** Assembly is now 31% of the step and is the next lever, not decode. **Training launched** on GPU 1, sequential, wandb project `zo-opd-q34b-1p7b` via new `scripts/zo_opd/launch_zo_opd_q34b_1p7b.sh`: (1) BP-OPD `token_reward_direct` LR 1e-6, (2) ZO-ES-token N=8, pack_width=64, sigma=0.01 absolute, bernoulli, student_iw, token_agg=mean, LR=1e-3, 150 iters, heldout probe 16. Both: Qwen3-4B teacher → Qwen3-1.7B student, MATH lv3-5 train / MATH-500 eval, batch 64 x 1024, **TEMPERATURE=1.0** (on-policy sampling — the IW rail loss is an unbiased estimate of KL(pi_n||q) only when the clean token is SAMPLED from pi_0; the greedy benchmark regime would bias it). ES LR 1e-3 = the shipped all-layer default (NP lesson: ~30x below the single-layer LR); **no ES LR sweep exists yet** — heldout probe is on so divergence shows within a few steps. → results/zo_opd.md §8, wiki/es_token_trainer.md §9, raw scripts/zo_opd/results/es_token_kv_reservation.txt

## [2026-08-23b] train | first zo-opd-q34b-1p7b runs: lr=1e-3 degrades, LR bound established, and a measurement trap
Launched BP-OPD then ZO-ES-token on GPU 1 (wandb `zo-opd-q34b-1p7b`). **BP-OPD** completed 138 steps at ~25 s/step, LR 1e-6, T=1.0: MATH-500 2.8/2.2/2.8/1.8% across its four evals — flat, so it is a wall-clock reference, not a learning baseline. **ZO-ES-token at the shipped lr=1e-3 DEGRADES the model**: fixed 16-prompt probe KL 0.2244 → 0.5565 (step 25) → 1.1559 (step 50), MATH-500 (200 fixed) 5.0% → 1.5% → 0.0%, monotonic and ~2x per interval. Cause: 1e-3 was calibrated on the GREEDY benchmark where dW_norm_mean≈240; at TEMPERATURE=1.0 the rails ride a higher-entropy trajectory, the importance weights spread, and dW_norm_mean is ≈866 at step 0 / ~2000-2550 steady — ~3.6x larger before any LR applies. (T=1.0 is nonetheless required: student_iw is unbiased for KL(pi_n||q) only when the clean token is SAMPLED from pi_0.) **MEASUREMENT TRAP — I killed a healthy run on this and had to retract**: `train/L_clean_mean` is scored on whatever 64 prompts that step drew and swings **0.23–3.4 batch to batch**; LRs 1e-3/1e-5/3e-5 (100x span) produce *indistinguishable* curves on it, with the same steps low in every run — it is data, not the optimizer. `dW_norm_mean` is not a divergence signal either: it is the gradient-estimate norm BEFORE the LR multiplies it. The only usable metric is `eval/heldout_clean_loss` (fixed 16 prompts, ray_trainer.py:333, logged **only every EVAL_INTERVAL**, not per step). **Probe noise floor = ±8%**: three sweep runs read the same untouched step-0 model as 0.1908/0.2126/0.2242. **Sweep** (21 steps, EVAL_INTERVAL=10, fixed probe + MATH-500@50): 1e-4 → 0.2126/0.1886/0.2035; 1e-5 → 0.1908/0.2126/0.2010; 1e-6 → 0.2242/0.1988/0.1909 — all flat within the noise floor (and MATH-500@50 has sigma≈4pp). **This is a BOUND, not a ranking**: 1e-3 destroys, ≤1e-4 does not, 20 steps cannot separate 1e-4/1e-5/1e-6. Relaunched the 150-step run at **1e-4** (largest non-degrading LR = most movement per step; a principled default, NOT a measured optimum). Whether es_token learns at ANY LR is still unanswered — the 1e-4 run is the first horizon that could show it. Separating 1e-4 from 1e-5 needs hundreds of steps or a lower-variance probe (greedy probe rollouts / more probe prompts). Standing blocker for accuracy as a metric: in BOTH runs every rollout hits the 1024-token cap without emitting EOS (response_length mean=min=max=1024), pinning MATH-500 near its floor. → results/zo_opd.md §9, wiki/es_token_trainer.md §5, raw scripts/zo_opd/results/es_token_lr.txt

## [2026-08-23c] result | ZO-ES-token 150 steps @ lr=1e-4 — NEGATIVE, and BP-OPD is flat too
Completed the 150-step run (wandb `zo-opd-q34b-1p7b` / N8_pw64_lr1e-4): 37.18 s/step, 92.9 min, 9,713,323 token-records, weight_sync_ok=1.0. Fixed 16-prompt probe across the seven evals: 0.2126 / 0.2002 / 0.1987 / 0.2169 / 0.2049 / 0.2157 / **0.2228**; MATH-500 (fixed 200): 6.0 / 7.0 / 6.5 / 5.5 / 7.0 / 2.0 / 4.0%. **es_token does NOT learn measurably at 1e-4 over 150 steps** — every probe reading lies in 0.199–0.223 with no direction, the endpoint is +4.8% vs step 0 which is INSIDE the ±8% noise floor (so "no change", not "slightly worse"), and MATH-500 shows no trend (sigma≈1.7pp at n=200). The monotone decline at steps 25/50 that looked promising broke at step 75 — precisely the false signal the noise floor predicts, and a second reminder not to read short trends on this probe. **Bracket established, no recipe inside it**: lr 1e-3 destroys the model (+415% probe by step 50, accuracy 0%), lr 1e-4 holds it steady — one order of magnitude between "destroys" and "does nothing", and the gap is too wide to conclude no working step size exists. **Crucially this is NOT ES-specific**: the BP-OPD baseline was equally flat (MATH-500 2.8/2.2/2.8/1.8% over 138 steps at lr 1e-6). Neither method moved, which implicates the SETUP rather than the algorithm. **Leading suspect: truncation** — in both runs every rollout hits the 1024-token cap without emitting EOS (response_length mean=min=max=1024, clip_ratio=1.0), so the student never produces a terminated answer and the teacher reward is computed entirely on truncated continuations; MATH-500 reads a floor for both. **Prerequisites before judging es_token as a method**: (1) fix truncation — longer budget or a terminating template; (2) a probe that can resolve the effect — 16 prompts sampled at T=1.0 gives ±8%, greedy rollouts over ~100 prompts would cut that several-fold for far less than a 93-min run; (3) only then sweep LR between 1e-4 and 1e-3. Wall-clock side is settled and positive: one OPD step 147.5→42.7 s (3.46x) and es_token is now faster than BP-OPD. → results/zo_opd.md §9.4, raw scripts/zo_opd/results/es_token_lr.txt

## [2026-08-24] sync | es_token docs merged from the OPD-estoken worktree into main

Branch `feat/es-token-trainer` (worktree `/home/yequan/Project/compression/OPD-estoken`, HEAD 687ddaa)
carried the whole ZO-ES-token thread in its own copy of the wiki. Synced into main without
clobbering main's diverged pages: `results/zo_opd.md` gained the 632-line **§ZO-ES-token** section
(sessions 2026-06-09 build/gates/headline, 08-21 profiling, 08-22 fused rail kernel, 08-22b direct
Rademacher noise, 08-23 budget-sized scratch-KV, 08-23b LR bound + measurement trap, 08-23c the
150-step negative result) spliced above main's existing §ZO-NP, which was kept verbatim (the two
copies differed only in markdown formatting). New pages `wiki/es_token_trainer.md` and
`plans/es_token_trainer.md` copied over so the results page's links resolve; `index.md` got those
two rows plus the expanded zo_opd summary, while main-only rows (`fura_grpo`, `ES/es_results`)
were preserved; the eight es_token log entries were interleaved by date rather than appended.
Pre-existing stale link noticed, not touched: `index.md` still lists `results/fura_opd.md`, deleted
in 365857a.

## [2026-08-24] correction | BP-OPD's teacher was never slow — every ES/BP ratio used a COLD BP step

User asked why BP-OPD's teacher phase costs 35.61 s when es_token's costs 4.2 s. It does not.
**(1) The reward path cannot cost that.** `RewardModelWorker._forward_micro_batch` always
materialises full-vocab logits (`use_fused_kernels=False`, and `compute_entropy` is hard-coded True
for a logging metric, so `need_logits` is always set) — 2.52 GiB per micro-batch. Replaying one
micro-batch (8 seqs x 1112 tok = the shipped `reward.micro_batch_size_per_gpu=8`) on the real
Qwen3-4B teacher prices the whole thing at **3.03 s/step**: transformer fwd + lm_head 2.60, top-K +
overlap 0.24, `_compute_entropy_safe` 0.17, `logprobs_from_logits` 0.01, `div_` 0.01. The FSDP1
`CPUOffload(offload_params=True)` the reward worker hard-wires (fsdp_workers.py:1883) is not it
either — 333 ms warm vs 325 unwrapped. **(2) It is cold start.** In the 138-step BP run
`timing_s/compute_rm_score` is **29.15 s at step 1 and 3.80 s median over the next 137** (min 3.46,
max 7.31), matching the microbenchmark: step 1 pays the teacher's first CPU→GPU param fetch, kernel
autotune, and first-touch allocation of the 2.52 GiB logits + 2.49 GiB fp32 entropy buffers into an
allocator already holding vLLM's 55% reservation. **(3) The headline was wrong.** Every ES/BP ratio
on the page divided by that 61.86 s cold step. Steady-state medians, both runs from
`launch_zo_opd_q34b_1p7b.sh` (batch 64x1024, T=1.0, same GPU, first 3 steps dropped): **ES 37.17 s
vs BP 25.11 s = 1.48x**, not 0.69x — decode 23.06 vs 9.85 (2.34x), teacher 3.10 vs 3.80 (**0.82x**,
not 8-10x), grad+update 10.89 vs 9.63 (1.13x). Cold penalties differ 13% (ES: teacher is the
already-warm vLLM engine) vs 146% (BP: separate FSDP module firing first inside the timed phase).
WITHDRAWN: "es_token is faster than BP-OPD" and "teacher phase 8.4x/10x faster than BP's".
UNAFFECTED: the 3.46x step / 5.10x decode optimisation gains (cold-vs-cold on one harness), the
gradient-cosine results, and the negative learning result. Corrected in results/zo_opd.md §10 (new)
+ summary/wall-clock tables + §1/§4/§6.4/§7.5/§8.3 relabels, wiki/es_token_trainer.md, index.md.
-> `docs/results/zo_opd.md` §10, raw `scripts/zo_opd/results/es_token_bp_teacher_cold.txt`,
harness `scripts/zo_opd/es_token_checks/bench_rm_stages.py`

## [2026-08-25] ingest | Paper-aligned OPD: two root-cause bugs, six falsified hypotheses, and the student-init finding

Created [results/opd_paper_align.md](results/opd_paper_align.md); cross-linked and superseded
[results/zo_opd.md](results/zo_opd.md) §9/§9.5.

Root causes of the long-standing "neither BP-OPD nor es_token learns" result: (1) 1024-token
truncation from using the *instruct* student with thinking on; (2) bf16 **master** weights
(`fsdp_workers.py:449` + `MODEL_DTYPE=bfloat16`) discarding 98.6% of an Adam step at `lr=1e-6`.
Code: `REWARD_MODEL_DTYPE` / `VAL_BEFORE_TRAIN` / `PPO_MAX_TOKEN_LEN_PER_GPU` env knobs in
`on_policy_distillation.sh`; `es_token.fp32_master`; `es_token.teacher_max_model_len`; greedy
heldout probe in `es_token/ray_trainer.py`; ES sweep teardown now reaps leaked vLLM engines.
Launchers under `scripts/zo_opd/paper_align/`.

Finding: BP-OPD at the paper's LR destroys `Qwen3-1.7B-Base` (six alternative explanations tested
and falsified), but the identical recipe is stable from `lllyx/Qwen3-1.7B-SFT`. The discriminator is
initial policy entropy (1.18 vs 0.31), not the paper's overlap ratio. The same split governs
es_token, so it is a property of the setting rather than the gradient estimator.
