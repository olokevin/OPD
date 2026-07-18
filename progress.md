# Progress Log — Shared Across 3 OPD Sub-Projects

<!--
  Chronological session log for ALL THREE projects. Tag each entry with the project
  (P1 compressed-opd / P2 peft-opd / P3 param-space-opd / X cross-cutting).
  This answers "what have I done?" across the shared codebase.
-->

## Session: 2026-05-31 — Set up shared knowledge system

### Bootstrap (X cross-cutting)
- **Status:** complete
- Actions taken:
  - Invoked planning-with-files; confirmed no prior session context, no existing planning files.
  - Clarified scope with user: 3 sibling projects share this repo; want a SHARED knowledge/status system (not per-project step plans — those live elsewhere).
  - Surveyed auto-memory (5 memories) + git log (30 commits) + script dirs to map projects → scripts.
  - Seeded the three root files: status board (task_plan.md), cross-cutting + per-project findings (findings.md), this log.
- Files created/modified:
  - `task_plan.md` (created — status board + 3 projects + cross-cutting)
  - `findings.md` (created — seeded C1–C5 cross-cutting, F2 FurA fixes, P1/P3 state)
  - `progress.md` (created — this file)

### Known state captured at bootstrap (from git history, NOT this session's work)
| Project | Last known state | Source |
|---------|------------------|--------|
| P1 compressed-opd | Calibrated BTT 4B→1.7B teacher + FSDP2 + LR 1e-6; eval_c4_ppl.py untracked | commits 15f9d45, 8240edc, af170f5 |
| P2 peft-opd | FurA OPD trains e2e on FSDP2 (7 fixes, VERIFIED 2026-05-31, step1 true_reward 0.543); qFurA untested | commit 7a8a61c + memory |
| P3 param-space-opd | NP+ES trainers scaffolded, smoke checks pass, final-review remediation done | commits 9df753f→ef6bc73 |

## Session: 2026-05-31 — Get OpenThoughts3 math dataset for teacher rollout (P1)

### Acquire + filter OpenThoughts3-1.2M-math (P1 compressed-opd, SFT-step input)
- **Status:** complete
- User asked: get the README's `OpenThoughts3-1.2M-math.parquet` and save to `/data/yequan/datasets`. Chose "download full 1.2M, filter to math".
- Actions:
  - Found repo already has `datasets/OpenThoughts3_opd.parquet` (30k complement slice) + `datasets/OpenThought3-Qwen3-4B/` (305k rolled-out SFT outputs) — neither is the full corpus. (See findings P1.D.)
  - Confirmed `vllm_rollout.py` only needs a `prompt` chat-list column.
  - Downloaded upstream `open-thoughts/OpenThoughts3-1.2M` (120 parquet shards, 27GB) → `/data/yequan/datasets/OpenThoughts3-1.2M/` via `hf download --include "data/*.parquet" --local-dir ...` (hf_transfer, 8 workers).
  - Wrote `scripts/infer/build_openthoughts3_math.py` (filter domain==math, conversations→prompt, suffix boxed-instruction, best-effort \boxed{} ground_truth).
  - Built `/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet` = **850,000 rows**, 130MB.
  - Validated: schema matches existing slice; `apply_chat_template` with local Qwen3-4B tokenizer works; 0 malformed in 5k sample.
- Files created: `scripts/infer/build_openthoughts3_math.py`; data under `/data/yequan/datasets/`.
- **Rollout note:** pass the ABSOLUTE path to `--input-parquet` (README's repo-relative `datasets/OpenThoughts3-1.2M-math.parquet` does not exist).

## Test Results (real run results — fill as runs complete)
| Test | Project | Input | Expected | Actual | Status |
|------|---------|-------|----------|--------|--------|
| FurA OPD e2e | P2 | fura.sh on FSDP2 | trains, grads flow | step1 true_reward 0.543, grad_norm 1.016, pg_loss 0.0045, 0 crashes | ✓ (2026-05-31) |
| C4 PPL of compressed teacher | P1 | eval_c4_ppl.py | reasonable PPL | not yet run | pending |
| qFurA OPD e2e | P2 | qfura.sh | trains like FurA | not yet run | pending |
| NP/ES real OPD train | P3 | opd_np.sh / es.sh | matches dense-OPD reward | smoke only | pending |
| SparseGPT V1 (OpenThought3 cached) re-build | P1 | build_sparsegpt_student.py | match prior 45% / PPL 82 | nz 1.697B, C4 PPL 82.05, MATH-500 45.0% | ✓ (2026-06-01) |
| SparseGPT V2 (Qwen3-4B fresh gen) build | P1 | 128 user prompts → Qwen3-4B T=0.6,top_p=1.0,max=3072 | reasonable / +V1 | nz 1.697B, C4 PPL 82.69, MATH-500 49.0% (+4pp vs V1) | ✓ (2026-06-01) |
| OPD-V1 train step 2 lands | P1 | opd_sparsegpt.sh + optim_offload + ppo_max=16384 | no OOM | step1 t-rew 0.500 / step2 0.502; ~7min/step | ✓ (2026-06-01) |
| OPD-V2 train step 3 lands | P1 | same w/ V2 student | no OOM | step1 t-rew 0.539 → step2 0.533 → step3 ongoing; ~5min/step | ✓ (2026-06-01) |
| SparseGPT mask-preservation patch | P1 | sparsity_mask.py + dp_actor + fsdp_workers | weights stay zero after step | smoke test: 252 modules masked, post-step ratio z_now/z_expected=1.0 | ✓ (2026-06-01) |
| OPD-V2 finishes 138 steps (masked) | P1 | opd_sparsegpt.sh w/ SPARSEGPT_PRESERVE_MASK=1 | clean finish, sparsity preserved | step 138 mean@4=53.05%, maj@4=54.58%, best@4=66.06% | ✓ (2026-06-02) |
| V2 ckpt apples-to-apples eval | P1 | merge FSDP→HF, eval_opd_ckpt.py | match build-script recipe | linear_zero_frac=64.00%, C4 PPL 82.67, MATH-500 greedy 51.0% (+2.0pp vs 49.0% pre-OPD) | ✓ (2026-06-02) |
| OPD-V1 finishes 138 steps | P1 | opd_sparsegpt.sh w/ mask | clean finish | CRASHED at step 90 (ActorUnavailableError / raylet term); 18 mid-evals plateau 49-52% mean@4; no ckpt saved | ✗ (2026-06-02) |

## Error Log (ALL errors, with project + attempt — avoid repetition)
| Timestamp | Project | Error | Attempt | Resolution |
|-----------|---------|-------|---------|------------|
| 2026-05-30 | P2 | param_offload loads model to CPU; BTT needs CUDA | 1 | ACTOR_PARAM_OFFLOAD=False (F2.1) |
| 2026-05-30 | P2 | Linear→BTT conversion requires CUDA weights | 2 | `.to(cuda)` before convert (F2.2) |
| 2026-05-30 | P2 | FSDP: must flatten uniform requires_grad | 2 | use_orig_params=True (F2.3) |
| 2026-05-30 | P2 | KeyError ..._fsdp_wrapped_module... on weight-sync | 2 | complete weight export + strip qualname (F2.4) |
| 2026-05-31 | P2 | element 0 does not require grad (gradckpt+frozen embed) | 3 | enable_input_require_grads() (F2.5) |
| 2026-05-31 | P1,P2 | FSDP1 Cannot writeback when param shape changes (frozen embed) | 3 | FSDP2 (C1 / F2.7) — decisive |
| 2026-06-01 | P1 | vLLM Engine init fails: ninja build flashinfer JIT (math.h not found) | 1 | VLLM_USE_FLASHINFER_SAMPLER=0 + VLLM_ATTENTION_BACKEND=FLASH_ATTN (known repo workaround) |
| 2026-06-01 | P1 | OPD step 2 update_actor OOM at 121 GB reserved (4B-dense SparseGPT student + 4B teacher on 93 GB H100) | 1 | optim_offload=True + ppo_max_token_len_per_gpu=16384 + max_num_batched_tokens=16384 + gpu_mem_util=0.40 (mirror btt_v2_combined config) |
| 2026-06-01 | P1 | verl actor Adam silently re-densifies SparseGPT zeros (∂L/∂W on zero weight ≠ 0; Adam moments amplify) — dense-run "OPD" was actually 4B-effective by step 5 | 1 | Patch: sparsity_mask.py + post-step reapply_masks hook in dp_actor._optimizer_step + attach_masks hook in fsdp_workers (gated by SPARSEGPT_PRESERVE_MASK=1). Restart from saved SparseGPT students. |
| 2026-06-02 | P1 | OPD-V1 raylet termination / ActorUnavailableError at step ~90 (transient Ray bug, possibly CPU mem pressure from concurrent V1+V2+eval) | 1 | No ckpt saved; descope V1 to its 18 mid-evals (clear plateau 49-52% mean@4). V2 finished cleanly. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Shared knowledge system live; all 3 projects in_progress (see task_plan.md Status Board) |
| Where am I going? | P1: C4 PPL gate → SFT → OPD. P2: verify qFurA + LR sweep. P3: real NP/ES runs, then +FurA. |
| What's the goal? | One shared brain across 3 OPD sub-projects so none re-hits a solved problem |
| What have I learned? | See findings.md — esp. CROSS-CUTTING C1–C5 + F2 |
| What have I done? | Bootstrapped the 3 root files from memory + git (this session) |

---
*Tag every entry with project (P1/P2/P3/X). Update after each phase or error.*
