# MANIFEST — moe_compress ARIS thread

> Idea-discovery pipeline for "which MoE expert-compression method matters AFTER short training" (OLMoE-1B-7B). **Read [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) first.** Generated 2026-06-07/08 via idea-discovery (research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline).

## Pages (read order)
| # | Page | Role |
|---|---|---|
| 1 | [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) | TL;DR + first runs + risks. **Start here.** |
| 2 | [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md) | Problem anchor, thesis, frozen-vs-varied, claims. |
| 3 | [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Phased roadmap, 36-run matrix, stats, claims matrix, budget. |
| 4 | [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) | Run table + gates + status. |
| 5 | [NOVELTY_CHECK.md](NOVELTY_CHECK.md) | 5.5/10; Claim-1=paper, Claim-2=support; drop effective rank. |
| 6 | [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) | 4.5/10 design → 5 identifiability fixes (all in the plan). |
| 7 | [IDEA_REPORT.md](IDEA_REPORT.md) | 10 ideas → 3 pillars; audit trail. |
| 8 | [LITERATURE.md](LITERATURE.md) | 5-family survey + SlimQwen framing + the empty cell. |

## Provenance / key decisions (user-confirmed)
- Base = **OLMoE-1B-7B-0924-Instruct** (both MMLU+GSM8K discriminate). Naive baseline = **SlimQwen** expert-compression part.
- **All 3 pillars** in scope; **breadth across families** slate.
- Router **trainable during recovery** (experts-only compression, but router re-adapts; frozen=ablation) — *relaxes original "router frozen"* per reviewer confound.
- **Standardized calibration** primary (native → appendix sensitivity).

## Traces
- `.aris/traces/idea-creator/2026-06-08_run01/` (brainstorm)
- `.aris/traces/novelty-check/2026-06-08_run01/` (novelty verdict)
- `.aris/traces/research-review/2026-06-08_run01/` (design review)

## Status: PLANNING COMPLETE → ready for `/run-experiment` (Phase 0 smoke) or `/experiment-bridge`.
