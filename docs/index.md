# Knowledge-base index

Content catalog for the `docs/` knowledge base. See `docs/llm-wiki.md` for the pattern
and the **Knowledge system** section of `CLAUDE.md` for the maintenance workflow.
`log.md` is the chronological companion to this file.

Every page below is LLM-maintained. When you ingest a source or file a query answer,
add/update its row here and append a line to `log.md`.

---

## Wiki — design docs (`docs/wiki/`)

How a subsystem works: architecture, invariants, knobs, file map. Stable; updated when the design changes.

| Page | One-line summary |
|---|---|
| [compressed_opd](wiki/compressed_opd.md) | BTT-compressed Qwen3-4B → ~1.7B student for OPD math: launchers, calib cache, FSDP2 + BlockTT fixes, C4-PPL audit. |
| [ZO](wiki/ZO.md) | The two zeroth-order OPD trainers — Weight Perturbation (ES) vs Node Perturbation (NP); what's perturbed, variance scaling, memory, update shape. |
| [reasoning_aware_compress_calib](wiki/reasoning_aware_compress_calib.md) | Why one-shot structured compression collapses Qwen3-4B reasoning + the two levers that fix it: **M1 rank floor** (full-rank sparse residual, MATH 72→82% at fixed budget; M2 objective is null) and **sequence-reweighted full-seq calibration** (now the repo default). Failure = loss of *termination* before reasoning (looping cliff ~0.65). Synthesizes A/B/D + ratio-sweep + calib study. |
| [reweighted_compress](wiki/reweighted_compress.md) | **NEGATIVE result.** Within-sequence token reweighting of calibration covariances: compress once → per-token forward KL(uncompressed teacher ‖ compressed student) on the trace → exp-tilt upweights damaged tokens → recompress. Clean derivation + verified impl (β=0 reproduces §5 baseline), **but −5pp MATH @ retain 0.7 (67→62%)** — KL chases teacher-uncertain not task-leverage tokens. Uniform sequence-reweight stays the recipe; kept as a documented dead end + reusable weighted-covariance machinery. |
| [reweighted_compress_v2](wiki/reweighted_compress_v2.md) | **Design (corrects v1's negative).** Derives the *right* damage-aware lever from a Taylor expansion of end-to-end teacher–student KL: reweight the per-layer **output** error by the **KL-Fisher curvature `G_ℓ`** (= backward cov of a KL loss = task-leverage, opposite of v1's teacher-uncertainty), via the **doubly-whitened SVD already in the repo**, inside an **initialize→measure-realized-gap→refit→iterate** loop (SRC / lr_sparse refine harnesses). Per-token "balance" = the left metric, not a scalar weight; optional `δ_t^α` minimax tilt for worst tokens. Not yet run; honest prior = gain (if any) lives below the cliff. |
| [zo_np_trainer](wiki/zo_np_trainer.md) | NP trainer internals: 1+n_sample-wide perturbed vLLM decode, teacher reverse-KL scoring, rank-1 δW accumulation. §8 = V1 throughput (forward is 99%). **§9 = V2 buffer-in-graph design + initial results (branch `np-v2-cudagraph-rails`): all GPU gates PASS (σ=0 incl. graph capture, m1/m2 parity u-bit-identical → graphed≡V1); throughput-vs-N (N≤16 ~free, +19% @ N=8; compute-bound past 32); cosine-vs-N (0.205/0.276/0.356/0.407 @ N=8/16/32/64)**. **§10 = V2.1 packed multi-prompt decode (`B_pack` prompts/forward, per-prompt scratch-KV; gates PASS, u bit-identical to serial): throughput grid (`scripts/zo_opd/results/packed_grid_1024.txt`) — ~2.6× decode tok/s batch 1→8 @ rails=8, plateaus by batch≈8; NEGATIVE result — NP one-step (865s) is ~14× slower than BP-OPD first-step / ~38× steady-state (zeroth-order forward-count tax, not closable by tiling)**. **§11 = V3 fully-CUDA-graphed ALL-LAYER packed decode (branch worktree-np-alllayer-graphed): all parity gates PASS (graphed u bit-identical to eager via torch.equal, logits rtol 0, staggered-EOS bit-for-bit); learns at LR=1e-3 (heldout-KL 0.53→0.27; single-layer default 3e-2 diverges); GOAL PROOF NEGATIVE — NP one-step 2472s vs BP 54s = ~46× slower at batch64/1024 (decode 1368s was an artifact: noise-refill = 896 draw_noise/token; see plans/2026-06-07-np-decode-host-glue-optimization.md). Results scripts/zo_opd/results/np_vs_bp_alllayer_graphed.txt**. |

## Results — experiments & mid-conclusions (`docs/results/`)

What we ran and what we learned. Append-mostly; each session adds a dated block.

| Page | One-line summary |
|---|---|
| [compressed_opd](results/compressed_opd.md) | Post-train compression of Qwen3-4B→1.7B one-shot: SparseGPT/SVD_V2/Nystrom × C4/OpenThought3 calib vs C4-PPL + MATH-500. SparseGPT+math-calib = 45%; structured + one-shot SVD collapse to 0%. **[2026-06-03]** `nystrom_combined` (joint fwd+bwd kernel) ≈ forward-only `nystrom` on Llama-3-8B 60%-MLP C4-PPL (20.94 vs 19.38, +8%). |
| [compress_sft](results/compress_sft.md) | **Compress-then-train** (branch `compress_sft`): new LlamaFactory `finetuning_type: svd_nystrom` compresses in-process at model-init (SVD-attn + Nystrom-MLP, retain 0.7, seq-reweighted full-seq calib) keeping factors **trainable**, then SFT on OpenThoughts3; eval MATH-500@4096 + MMLU-Pro → wandb `compress_sft_{model}`. Qwen3-4B fwd+combined smoke PASS (intermediate_size 9728→6810, save/reload clean). OLMoE blocked: fused 3D experts on tfm 5.2 → fast-fail guard + follow-up. |
| [zo_opd](results/zo_opd.md) | ZO-NP OPD runs (Qwen3-1.7B student): NP-vs-BP gradient scaling, LR search, self-amplifying divergence. |
| [fura_opd](results/fura_opd.md) | _(placeholder — FurA/LoRA OPD results, empty)_ |
| [fura_grpo](results/fura_grpo.md) | **Full vs FURA (BlockTT) zero-RL GRPO on Qwen2.5-7B** (MATH lv3–5, 138 steps), wandb `nersc_grpo_qwen2p5_7b`. New `slurm/grpo/` infra. Fixes: grpo.sh RAY_EXTERNAL + overridable ckpt/exp + exit-code propagation; `fsdp_workers.py` real-weight init for blocktt/svd (meta-tensor crash on 7B). **[2026-07-18 COMPLETE]** MATH-500 acc@4 Full 0.526→0.668 / FURA 0.498→0.633. Benchmarks mean@16 (Full/FURA): AMC23 0.362/0.345, AIME24 0.060/0.081, Minerva 0.276/0.247, Olympiad 0.307/0.281 — FURA ≈ full at a fraction of trainable params. |

## ARIS threads — agent-generated research docs (`docs/aris/{project}/`)

Idea-discovery → refine → experiment-plan → results pipelines, one subfolder per research thread.

### [moe_compress](aris/moe_compress/) — which MoE expert-compression method matters AFTER short training (OLMoE-1B-7B)

| Page | One-line summary |
|---|---|
| [MANIFEST](aris/moe_compress/MANIFEST.md) | Thread index + read order + user-confirmed decisions + trace pointers. **Read first.** |
| [PIPELINE_SUMMARY](aris/moe_compress/PIPELINE_SUMMARY.md) | TL;DR: thesis (compression-method choice is a *trajectory* question), first runs, risks, key engineering facts (OLMoE experts per-Linear in verl env). |
| [FINAL_PROPOSAL](aris/moe_compress/FINAL_PROPOSAL.md) | Problem anchor + thesis + frozen-vs-varied table + claims C1/C1-sharp/C2. Dominant = cross-family **inversion**; no new compressor. |
| [EXPERIMENT_PLAN](aris/moe_compress/EXPERIMENT_PLAN.md) | Phased roadmap: Phase 0 smoke → 6-method×2-retain×3-seed (36-run) atlas → variance decomposition + pre-registered inversion test → LOFO diagnostics. Run order, gates, ~210-320 GPU-h (eval-dominated), results-to-claims matrix. |
| [EXPERIMENT_TRACKER](aris/moe_compress/EXPERIMENT_TRACKER.md) | Run table + gates (G0/G2a/G3) + prereqs; all ☐ pending. |
| [NOVELTY_CHECK](aris/moe_compress/NOVELTY_CHECK.md) | **5.5/10 PROCEED-WITH-CAUTION**. "Methods converge" headline taken (SlimQwen + A Free Lunch); edge = cross-family + low-rank/unstructured + variance decomp + short-horizon curve. Claim-1=paper, Claim-2=support; **drop effective rank** (2602.20433). |
| [RESEARCH_REVIEW](aris/moe_compress/RESEARCH_REVIEW.md) | **Design 4.5/10** → 3 identifiability killers (n=1/family; non-common budget; frozen-router confound) + 2 (thin tasks; underpowered Claim-2). MVP design + 5 fixes (all folded into the plan). |
| [IDEA_REPORT](aris/moe_compress/IDEA_REPORT.md) | 10 GPT-5.4 ideas → 3 pillars (trajectory atlas / family>criterion / predictive diagnostic); audit trail. |
| [EXPERIMENT_RESULTS](aris/moe_compress/EXPERIMENT_RESULTS.md) | **Training-free leg DONE (2026-06-08, OLMoE-Instruct), 10 methods × 2 retains.** Step-0: **SparseGPT near-lossless** (@0.75 MMLU 0.549 ≥ uncompressed 0.542; @0.50 0.521) ≈ naive magnitude; **drop/merge collapse at 0.50**; **MoBE DEAD LAST**. Weight-approx ordering @0.75: sparsegpt 0.549 > nystrom_combined 0.499 > nystrom 0.474 > svd_llm_v2 0.450 ≫ mobe 0.246. Findings: **granularity (weight vs expert) ≫ criterion**; within structured low-rank **fwd+bwd > fwd-only Nyström > per-matrix SVD**; MoBE shared-basis collapses small MoE. Recovery-SFT (inversion test) pending. |
| [LITERATURE](aris/moe_compress/LITERATURE.md) | 5-agent survey of expert prune/merge/low-rank/unstructured + the train-free-vs-finetuned gap. **The cell "which MoE expert method matters after short training" is empirically empty**; closest prior = **SlimQwen** (gap closes at 400B tokens, whole-expert only). |

### [reason_aware_compress](aris/reason_aware_compress/) — reasoning-aware structured compression (Qwen3-4B→1.7B one-shot)

| Page | One-line summary |
|---|---|
| [MANIFEST](aris/reason_aware_compress/MANIFEST.md) | Output manifest + v1/v2 pipeline provenance and per-file version status. **Read first.** |
| [DIAGNOSIS](aris/reason_aware_compress/DIAGNOSIS.md) | 5 ranked failure mechanisms (M1–M5) for why structured pruning collapses reasoning. (v1, still valid.) |
| [LITERATURE](aris/reason_aware_compress/LITERATURE.md) | 3-agent survey: prior-art collisions (SAES-SVD, OBD-LLM, PGSVD) + ~40 verified arXiv refs. (v2.) |
| [FINAL_PROPOSAL](aris/reason_aware_compress/FINAL_PROPOSAL.md) | **TRACER** — transition/steering-subspace-preserving structured compression; method math C1/C2/C3. (v2, current.) |
| [EXPERIMENT_PLAN](aris/reason_aware_compress/EXPERIMENT_PLAN.md) | Primary track = **Blocks A/B/D** (M1/M2/M3 fixes: LR+sparse residual, SRC, OPD bi-whitened SVD) **+ Block T** (dense-vs-compressed reasoning-trace diff); diagnostics = Block 0 (SER, done/falsified), Block 1 (C1/C3 knobs), Block 2 (phase-transition, sweep 0.8/0.7/0.6/0.5), Block 4 headline, Block 5 robustness. Operating point: retain **0.8** first (last decoder layer skipped), sweep down. **Long-context block + attn/mlp split removed.** (v3, current.) |
| [EXPERIMENT_RESULTS](aris/reason_aware_compress/EXPERIMENT_RESULTS.md) | Block 0 SER probe — **central thesis falsified** (steering subspace is best-preserved, not eroded); pivot to M1/M2/M3 → A/B/D. |
| [INITIAL_RESULTS_ABD](aris/reason_aware_compress/INITIAL_RESULTS_ABD.md) | **A/B/D + ratio-sweep results (2026-06-03).** M2(objective)=**null** (D2≈D0); **M1(rank floor)=HEADLINE**: full-rank sparse residual recovers MATH **72→82%** (beats dense) at same budget where tail-rescue failed. Forward-only cliff **r\*≈0.65**; trace-diff → failure = **late-trace convergence loss / looping**, not early divergence. B(M3)/D3 skipped per user. |
| [FULLSEQ_CALIB_RESULTS](aris/reason_aware_compress/FULLSEQ_CALIB_RESULTS.md) | **Calibration-data lever (2026-06-04).** Full (un-windowed) sequences + **sequence-level reweighting** beats the 2048-window scheme: r=0.7 **71% vs 66%**; sweep **47/36% vs 37/20%** @0.6/0.5 (Δ +10/+16pp, gap grows with compression). Bonus: bounded gen_len + relaxed==strict → also fixes the looping/termination failure. Pure calib change, orthogonal to M1. |
| [EXPERIMENT_TRACKER](aris/reason_aware_compress/EXPERIMENT_TRACKER.md) | Run table → plan blocks → GPU/status/key result; **A/B/D + sweep DONE**, B1/B2/D3 skipped. |
| [IDEA_CANDIDATES](aris/reason_aware_compress/IDEA_CANDIDATES.md) | Compact candidate table + active thesis. **v3**: TRACER C2 falsified → A/B/D re-promoted to method candidates (session recovery). |
| [IDEA_REPORT](aris/reason_aware_compress/IDEA_REPORT.md) | Full idea-discovery report; v2 header points to TRACER, v1 diagnosis preserved as audit trail. |

## Plans — implementation specs (`docs/plans/`)

Step-by-step build plans for trainer changes (companion to wiki design docs).

| Page | One-line summary |
|---|---|
| [np_trainer](plans/np_trainer.md) | Implementation plan for the Node-Perturbation trainer in the verl fork: config interfaces, modules, acceptance tests. |
| [np-v2-cudagraph-rails (spec)](superpowers/specs/2026-06-03-np-v2-cudagraph-rails.md) | **Current V2 spec** — CUDA-graph the single-prompt 1+N decode step via a host-refilled `u_buf` (perturbation = captured `y += σ·u_buf`, RNG moves outside the graph). Path B (buffer-in-graph); keeps V1 eager as parity oracle. M0 capture spike → M1 noise-relocation → M2 graph → M3 N-scaling bench. |
| [np-v2-design (spec, SUPERSEDED)](superpowers/specs/2026-06-03-np-v2-design.md) | ~~Prompt-packing-first V2 plan (`B_pack` across prompts)~~ — superseded 2026-06-03 by np-v2-cudagraph-rails (cross-prompt packing dropped; goal is graphing the 1+N step, not batching prompts). |

## Papers — reference PDFs (`docs/papers/`)

Read-only source PDFs. Cite by filename; do not edit. Key ones:

| File | Relevance |
|---|---|
| `26_Rethinking On-Policy Distillation...pdf` | The paper this repo implements (OPD phenomenology/mechanism/recipe). |
| `25_ICLR_MoDeGPT...pdf`, `25_SVD-LLM V2...pdf` | Structured-compression baselines used by `compressed_opd` / `reason_aware_compress`. |
| `26_ICLR_RAC...pdf`, `26_ICLR_When Reasoning Meets Compression...pdf` | Reasoning-aware compression / compression-on-reasoning effects. |
| `26_ICML_Evolution Strategies at Scale...pdf`, `25_Dr.GRPO.pdf` | ZO/ES fine-tuning and GRPO background for the `ZO` trainers. |
