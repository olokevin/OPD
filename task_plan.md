# Task Plan: Shared Knowledge System for 3 OPD Sub-Projects

<!--
  This is NOT a step-level plan. Detailed per-project step plans live elsewhere.
  These root files are the SHARED BRAIN across three sibling projects that share
  this one codebase: a status board (here) + cross-project findings (findings.md)
  + a session log (progress.md). Goal: never get stuck on the same problem twice.
-->

## Goal
Maintain a single shared knowledge + status system across the three OPD sub-projects
that share this repo, so findings (gotchas, env/FSDP/GPU facts) flow between them and
no project re-hits a problem another already solved.

## The Three Projects (shared codebase)

| ID | Project | One-liner | Primary scripts |
|----|---------|-----------|-----------------|
| **P1 compressed-opd** | Compressed-teacher OPD | Compress 4B→1.7B (BTT), SFT student on 4B-*generated* sequences, THEN OPD with original 4B as teacher | `scripts/compress_opd/math/btt_v2_combined_opd.sh`, `scripts/opd/math/compressed_opd/{btt_v2,btt_v2_combined,_common}.sh`, `eval_c4_ppl.py` |
| **P2 peft-opd** | PEFT-OPD comparison | OPD training under LoRA / QLoRA / FurA(BlockTT) / qFurA vs full FT | `scripts/opd/math/{full,lora,qlora,fura,qfura,lr_search}.sh` |
| **P3 param-space-opd** | Parameter-space OPD | OPD via param-space exploration: ES + node-perturbation (NP) trainers in verl; future combine w/ FurA | `scripts/zo_opd/{es.sh,opd_np.sh,np_checks/*}` |

## Current Phase
Ongoing — this is a living knowledge system, not a finite task. No fixed end state.

## Status Board (high-level per project)
<!-- Update the Status cell + Next as each project moves. Keep to ONE line each. -->

### P1 compressed-opd
- **Status:** in_progress — calibrated BTT 4B→~1.7B teacher compression landed (commit 15f9d45); FSDP2 forced (8240edc); LR lowered to 1e-6 + memory knobs for shared GPU (af170f5). New `eval_c4_ppl.py` (untracked) for C4 perplexity eval of compressed teacher.
- **Next:** validate compressed teacher quality (C4 PPL) → confirm SFT-on-4B-generated step → run OPD with 4B teacher.
- **Open risks:** does the BTT-compressed 1.7B teacher retain enough signal to be a useful OPD reward model? PPL eval is the gate.

### P2 peft-opd
- **Status:** in_progress — FurA (plain BlockTT) made to train end-to-end on FSDP2 (7 fixes, VERIFIED 2026-05-31, see findings). full/lora/fura run concurrently one-per-GPU. qFurA path NOT yet verified.
- **Next:** verify qFurA (BTT_QFURA=True streaming path); run LR sweep (`lr_search.sh`); ensure all arms route to real MATH data (see trainset-name bug).
- **Open risks:** qFurA untested; trainset-name bug may have silently mis-routed earlier "math" runs to DAPO.

### P3 param-space-opd
- **Status:** in_progress — NP trainer fully scaffolded (main_np.py, RayNPTrainer, NCCL broadcast, perturb layers, gradient estimator); ES trainer ported; final-review remediation done (ef6bc73); grad cosine-sim + sigma=0 smoke checks exist.
- **Next:** run real NP/ES OPD training beyond smoke; later combine NP with FurA cores.
- **Open risks:** correctness of gradient estimator at scale; integration cost with FurA.

## Cross-Cutting Concerns (affect ≥2 projects)
<!-- These are the shared-brain payoff. Full detail in findings.md. -->
1. **FSDP2 is the working backend** for BlockTT/mixed-requires_grad modules (P1 + P2). FSDP1 FlatParameter writeback breaks on frozen embeddings.
2. **Concurrent one-GPU-per-run harness** (CUDA pinning + RAY_ISOLATE) — needed whenever P1/P2/P3 runs share the node.
3. **SimpleRL-Zoo MATH settings** are the canonical train/eval config all math OPD runs must match (P1, P2, and P3 when on math).
4. **Trainset-name routing bug** (`math-500k` → silent DAPO fallback) — affects every `*.sh` that sets `TRAIN_DATASET_NAME=math-500k`.
5. **Single-GPU OPD step timing** ~6 min/step, ~13h/run — budget all three projects' runs against this.

## Key Questions
1. Does the BTT-compressed teacher (P1) retain enough quality (C4 PPL) to give useful OPD rewards?
2. Is the trainset-name bug fixed across all wrappers, or are some still mis-routing to DAPO?
3. Does qFurA (P2) train end-to-end like plain FurA, or does its streaming path need separate fixes?
4. Can the NP/ES param-space trainer (P3) match dense-OPD reward trajectories at real scale?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| One shared root planning set + per-project sections | Single source of truth; cross-project findings shared (the whole point) |
| Root files = knowledge & status only, not step plans | Detailed step plans live per-project elsewhere; avoid duplication |
| Seed findings from memory + git history | Immediate value; encodes hard-won gotchas so no project re-hits them |
| FSDP2 as default for BlockTT arms | Verified-working; FSDP1 fundamentally breaks on frozen-embed writeback |

## Errors Encountered
<!-- High-signal cross-project failures only. Full detail + per-attempt log in findings.md / progress.md. -->
| Error | Project | Resolution |
|-------|---------|------------|
| FSDP1 writeback fails on frozen embed (BlockTT) | P1, P2 | Switch to FSDP2 (per-param DTensor sharding) |
| Concurrent runs collide on one GPU | all | CUDA_VISIBLE_DEVICES default + RAY_ISOLATE per-GPU |
| `math-500k` trains on DAPO not MATH | all (math) | Add `MATH)` branch to launcher case |

## Notes
- Update the Status Board line + Next after any session that moves a project.
- Before starting work on ANY project, re-read this board + findings.md cross-cutting section.
- Log ALL errors to findings.md "Issues Encountered" with which project + which attempt.
- A finding solved in one project that could bite another → add to Cross-Cutting Concerns.
