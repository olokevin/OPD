# Experiment Tracker — MoE Expert-Compression Recovery Atlas

> Run table → plan blocks → status. Companion to [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). Update status as runs land.

## Gates
| Gate | Condition | Status |
|---|---|---|
| G0 | End-to-end compress→reload→eval loop works on OLMoE-Instruct | ✅ **PASS** (2026-06-08: random_drop/svd_llm_v2/slimqwen_merge all compress→reload→eval; arc_challenge runs) |
| G2a | Retain 0.50 not degenerate for ≥4/6 methods (else → 0.625) | ☐ pending (in the launched atlas) |
| G3 | Inversion verdict (i/ii/iii) computed via pre-registered test | ☐ pending (needs recovery phase) |

## Prereqs
| Item | Status |
|---|---|
| Download `allenai/OLMoE-1B-7B-0924-Instruct` | ✅ local |
| `src/moe_compress/compress_olmoe.py` (verl env, per-Linear experts) | ✅ driver + config-sync (num_experts/intermediate_size) |
| 7 method plugins `src/moe_compress/methods/` | ✅ random_drop/reap_drop/slimqwen_merge/hcsmoe_merge/svd_llm_v2/sparsegpt/magnitude |
| Standardized calib loader (256×2048, OpenThoughts3+C4) | ✅ `calib.py` + coverage check (min tokens/expert) |
| `src/moe_compress/budget.py` (storage + active-capacity axes) | ✅ verified: drop→storage .75/active 1.0; svd→.75/.75 |
| `src/moe_compress/eval_tasks.py` (lm-eval, 4 tasks) | ✅ lm-eval 0.4.12 installed; mmlu/gsm8k/arc_challenge/hellaswag confirmed |
| `scripts/moe_compress/run_trainfree_atlas.sh` (GPU 1,2,3 round-robin) | ✅ |
| OLMoE recovery YAML (full FT, attn frozen, experts+router trainable) | ☐ recovery phase (adapt `olmoe_compressed_fwd_sft.yaml`) |

## Training-free leg (LAUNCHED 2026-06-08, GPU 1/2/3)
`scripts/moe_compress/run_trainfree_atlas.sh "1 2 3" 200` — 7 methods × 2 retains, seed 0, **step-0 only (no recovery)**, eval limit 200/task (fast first pass; full eval is a follow-up). Metrics → `/data/yequan/moe_compress/metrics/<method>_r<ret>_s0.json`.

| Reference | value |
|---|---|
| Uncompressed OLMoE-Instruct MMLU (5-shot, limit 500) | **0.542** ✅ matches paper (~54) — eval pipeline validated |
| Per-method compress timing | drop/merge ~10-40s, svd ~100s, **sparsegpt ~7min** (mem_gb=2.5, per-layer groups) |
| Budget axes verified | drop/merge: storage 0.75 / **active 1.0**; svd/sparsegpt: storage 0.75 / **active 0.75** |

## Phase 2 — 36-run atlas (6 methods × 2 retains × 3 seeds)
| Family | Method | retain | seeds | step-0 done | recovery {0,100,500,2k} | eval 4-task | GPU |
|---|---|---|---|---|---|---|---|
| expert-removal | random-drop | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| expert-removal | REAP-drop | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| merge | SlimQwen-merge+PP | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| merge | HC-SMoE-cluster | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| weight-approx | SVD-LLM-V2 | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| weight-approx | SparseGPT | 0.75 / 0.50 | 0,1,2 | ☐ | ☐ | ☐ | — |
| (aux) | magnitude | 0.75 | 0 | ☐ | ☐ | ☐ | — |

## Phase 3 — stats
| Output | Status |
|---|---|
| recovery-curve figure (acc vs step by family, CI) | ☐ |
| AURC variance-decomposition table (family vs criterion) | ☐ |
| pre-registered inversion verdict | ☐ |

## Phase 4 — diagnostics (Claim 2)
| Predictor | computed @ step0 | LOFO result |
|---|---|---|
| routed-token curvature (MoE) | ☐ | ☐ |
| inter-expert diversity-retention (MoE) | ☐ | ☐ |
| reconstruction error (baseline) | ☐ | ☐ |
| step-0 accuracy (baseline) | ☐ | ☐ |
| effective rank (negative control) | ☐ | ☐ |

## Phase 5 — ablations
| Ablation | Status |
|---|---|
| router-frozen (best+worst family ×0.75) | ☐ |
| native-calibration sensitivity (6 methods ×0.75) | ☐ |
| 5k/10k long horizon (subset) | ☐ |
