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
| [zo_np_trainer](wiki/zo_np_trainer.md) | NP trainer internals: 1+n_sample-wide perturbed vLLM decode, teacher reverse-KL scoring, rank-1 δW accumulation. §8 = V1 throughput (forward is 99%) + V2 plan (CUDA-graphed one-wide-forward-per-token). Branch `feat/np-trainer`. |

## Results — experiments & mid-conclusions (`docs/results/`)

What we ran and what we learned. Append-mostly; each session adds a dated block.

| Page | One-line summary |
|---|---|
| [compressed_opd](results/compressed_opd.md) | Post-train compression of Qwen3-4B→1.7B one-shot: SparseGPT/SVD_V2/Nystrom × C4/OpenThought3 calib vs C4-PPL + MATH-500. SparseGPT+math-calib = 45%; structured + one-shot SVD collapse to 0%. **[2026-06-03]** `nystrom_combined` (joint fwd+bwd kernel) ≈ forward-only `nystrom` on Llama-3-8B 60%-MLP C4-PPL (20.94 vs 19.38, +8%). |
| [zo_opd](results/zo_opd.md) | ZO-NP OPD runs (Qwen3-1.7B student): NP-vs-BP gradient scaling, LR search, self-amplifying divergence. |
| [fura_opd](results/fura_opd.md) | _(placeholder — FurA/LoRA OPD results, empty)_ |

## ARIS threads — agent-generated research docs (`docs/aris/{project}/`)

Idea-discovery → refine → experiment-plan → results pipelines, one subfolder per research thread.

### [reason_aware_compress](aris/reason_aware_compress/) — reasoning-aware structured compression (Qwen3-4B→1.7B one-shot)

| Page | One-line summary |
|---|---|
| [MANIFEST](aris/reason_aware_compress/MANIFEST.md) | Output manifest + v1/v2 pipeline provenance and per-file version status. **Read first.** |
| [DIAGNOSIS](aris/reason_aware_compress/DIAGNOSIS.md) | 5 ranked failure mechanisms (M1–M5) for why structured pruning collapses reasoning. (v1, still valid.) |
| [LITERATURE](aris/reason_aware_compress/LITERATURE.md) | 3-agent survey: prior-art collisions (SAES-SVD, OBD-LLM, PGSVD) + ~40 verified arXiv refs. (v2.) |
| [FINAL_PROPOSAL](aris/reason_aware_compress/FINAL_PROPOSAL.md) | **TRACER** — transition/steering-subspace-preserving structured compression; method math C1/C2/C3. (v2, current.) |
| [EXPERIMENT_PLAN](aris/reason_aware_compress/EXPERIMENT_PLAN.md) | Primary track = **Blocks A/B/D** (M1/M2/M3 fixes: LR+sparse residual, SRC, OPD bi-whitened SVD) **+ Block T** (dense-vs-compressed reasoning-trace diff); diagnostics = Block 0 (SER, done/falsified), Block 1 (C1/C3 knobs), Block 2 (phase-transition, sweep 0.8/0.7/0.6/0.5), Block 4 headline, Block 5 robustness. Operating point: retain **0.8** first (last decoder layer skipped), sweep down. **Long-context block + attn/mlp split removed.** (v3, current.) |
| [EXPERIMENT_RESULTS](aris/reason_aware_compress/EXPERIMENT_RESULTS.md) | Block 0 SER probe — **central thesis falsified** (steering subspace is best-preserved, not eroded); pivot to M1/M2/M3 → A/B/D. |
| [EXPERIMENT_TRACKER](aris/reason_aware_compress/EXPERIMENT_TRACKER.md) | Run table mapping run IDs → plan blocks → GPU/status/key result; A/B/D rows PENDING. |
| [IDEA_CANDIDATES](aris/reason_aware_compress/IDEA_CANDIDATES.md) | Compact candidate table + active thesis. **v3**: TRACER C2 falsified → A/B/D re-promoted to method candidates (session recovery). |
| [IDEA_REPORT](aris/reason_aware_compress/IDEA_REPORT.md) | Full idea-discovery report; v2 header points to TRACER, v1 diagnosis preserved as audit trail. |

## Plans — implementation specs (`docs/plans/`)

Step-by-step build plans for trainer changes (companion to wiki design docs).

| Page | One-line summary |
|---|---|
| [np_trainer](plans/np_trainer.md) | Implementation plan for the Node-Perturbation trainer in the verl fork: config interfaces, modules, acceptance tests. |

## Papers — reference PDFs (`docs/papers/`)

Read-only source PDFs. Cite by filename; do not edit. Key ones:

| File | Relevance |
|---|---|
| `26_Rethinking On-Policy Distillation...pdf` | The paper this repo implements (OPD phenomenology/mechanism/recipe). |
| `25_ICLR_MoDeGPT...pdf`, `25_SVD-LLM V2...pdf` | Structured-compression baselines used by `compressed_opd` / `reason_aware_compress`. |
| `26_ICLR_RAC...pdf`, `26_ICLR_When Reasoning Meets Compression...pdf` | Reasoning-aware compression / compression-on-reasoning effects. |
| `26_ICML_Evolution Strategies at Scale...pdf`, `25_Dr.GRPO.pdf` | ZO/ES fine-tuning and GRPO background for the `ZO` trainers. |
