# Pipeline Summary — MoE Expert-Compression Recovery Atlas

**Problem**: In a controlled experts-only compression study of OLMoE-1B-7B-Instruct, does short recovery training reorganize compression-method quality at the FAMILY level rather than the step-0 score level, and do MoE-specific step-0 diagnostics predict post-recovery value better than reconstruction error?
**Final Method Thesis**: Treat "which expert-compression method is best" as a *trajectory* question — hold model/budget/calibration/recovery fixed, vary only the method across 4 families, and measure whether the training-free ranking **inverts** during short recovery and at what granularity.
**Final Verdict**: READY (novelty 5.5/10 PROCEED-WITH-CAUTION; design defensible after 5 fixes)
**Date**: 2026-06-08

## Final Deliverables (all in `docs/aris/moe_compress/`)
- Literature: [LITERATURE.md](LITERATURE.md)
- Idea report: [IDEA_REPORT.md](IDEA_REPORT.md)
- Novelty: [NOVELTY_CHECK.md](NOVELTY_CHECK.md)
- Design review: [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md)
- Proposal: [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md)
- Experiment plan: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- Tracker: [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md)

## Contribution Snapshot
- **Dominant**: a controlled cross-family recovery atlas establishing/refuting a **family-level inversion** of MoE expert-compression quality after short recovery (the axis SlimQwen + A Free Lunch leave open).
- **Supporting**: a MoE-specific step-0 recoverability diagnostic that out-predicts reconstruction error (leave-one-family-out).
- **Rejected complexity**: no new compressor; no depth/width/quant; effective rank → negative control only.

## Must-Prove Claims
- **C1**: ranking not rank-stable; reorganization is family-level (between-family variance > within-family criterion variance).
- **C1-sharp**: robust cross-family inversion (step-0 winner family ≠ 2k winner; sign-flip 95% bootstrap CI excludes 0; ≥75% task×seed cells; both retains).
- **C2 (support)**: MoE step-0 diagnostic predicts AURC₀₋₂ₖ > recon error / step-0 acc (LOFO).

## First Runs to Launch
1. Download OLMoE-1B-7B-0924-Instruct + Phase 0 smoke (SVD @0.75 → reload → 200-step SFT → eval) — prove the loop (Gate G0).
2. Phase 1: implement the 6 method plugins at matched calibration + dual budget axes.
3. Phase 2 pilot: 6 methods @ retain 0.50, 1 seed — check degeneracy (Gate G2a), then launch the 36-run matrix.

## Main Risks
- **Identifiability** (reviewer's #1): mitigated by 2 methods/family + dual budget axes + trainable router. If a family ports to 1 usable method → downgrade to method-level claim.
- **Budget non-commensurability**: report both storage + active-capacity axes; an inversion holding under both is the strong result.
- **Null result**: rankings stay stable → well-powered negative (still publishable); C2 may survive.

## Key engineering facts (de-risked this session)
- OLMoE experts = per-expert `nn.Linear` in **verl env (tfm 4.56)** → existing `src/compress` tooling applies directly; fused-3D blocker is **sft-env (tfm 5.2) only**. Compress in verl env, recover-SFT in sft env on the dense compressed ckpt.
- Expert params = 93% of the 6.9B model; whole-expert drop 64→48 @0.75; intra-expert shrink intermediate 1024→768 @0.75.
- LlamaFactory `olmoe_compressed_fwd_sft.yaml` is the recovery-harness starting point (template `olmo`, cutoff 4096, ds_z2_offload).
- ⚠️ 10k samples @ eff-batch 8 ≈ 1250 steps < 2000 → recovery length must be ≥16k samples to reach the step-2000 primary horizon.

## Next Action
- `/run-experiment` (Phase 0 smoke first) — or `/experiment-bridge` to implement `src/moe_compress/` from this plan.
