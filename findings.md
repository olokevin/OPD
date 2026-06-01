# Findings & Decisions — Shared Across 3 OPD Sub-Projects

<!--
  This is the SHARED BRAIN. Anything one project learns that another could trip on
  goes here. Organized: Cross-Cutting (affects ≥2 projects) FIRST, then per-project.
  Seeded 2026-05-31 from auto-memory + git history.
-->

## Requirements (what the shared system must do)
- Serve all three projects: P1 compressed-opd, P2 peft-opd, P3 param-space-opd.
- Surface cross-cutting gotchas so no project re-hits a solved problem.
- Stay knowledge+status focused; detailed step plans live per-project elsewhere.

---

## CROSS-CUTTING FINDINGS (read these before touching any project)

### C1. FSDP2 is the working backend for BlockTT / mixed-requires_grad  (P1, P2)
- FSDP1 + `use_orig_params=True` (required for BlockTT's mixed trainable/frozen params) FAILS the per-step actor-update writeback on the frozen embedding:
  `RuntimeError: Cannot writeback when the parameter shape changes / Expects torch.Size([311164928]) but got torch.Size([151936, 2048])` (311164928 = 151936*2048, the embed_tokens FlatParameter).
- Untying embeddings did NOT fix it — the FlatParameter writeback constraint is fundamental to FSDP1.
- **FSDP2 (`fully_shard`, per-parameter DTensor sharding) has no such constraint.** Set via `actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2`.
- Commit 8240edc forced FSDP2 for compressed_opd (P1) explicitly to "match fura's verified-working config" — i.e. P1 inherited P2's fix. This is the shared-brain working.

### C2. Concurrent one-GPU-per-run harness  (all projects sharing the node)
- To run multiple wrappers concurrently, one per GPU:
  1. **CUDA pinning:** wrappers use `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N}` so an external pin wins (defaults full→1, lora→2, fura→3). The old hardcoded `=4/5` under `set -a` silently piled every run onto GPU 4/5 and OOM'd.
  2. **`RAY_ISOLATE=1`** (knob in `on_policy_distillation.sh`): private Ray head per GPU (port 6379 + gpu*100 + 21 → GPU1=6500, 2=6600, 3=6700), temp-dir `/tmp/ray_opd_gpu<N>`, SKIPS global `ray stop --force`. Without it, a 2nd run's `ray start/stop` kills the 1st run's cluster (ActorUnavailableError / "Socket closed").
  3. **Stagger launches ~30-40s** so each Ray head comes up first.
  4. Also needs `RAY_memory_monitor_refresh_ms=0` or runs crash ~step 4.
  5. **Cleanup:** global `ray stop --force` then `rm -rf /tmp/ray_opd_gpu*`.
- Verified 2026-05-29: full→GPU1/LR1e-6, lora→GPU2/LR1e-5, fura→GPU3/LR1e-4 coexisting, 0 errors.

### C3. SimpleRL-Zoo MATH settings = canonical config for all math OPD runs  (P1, P2, P3-on-math)
- Align to SimpleRL-Zoo (arXiv:2503.18892, hkust-nlp/simpleRL-reason).
- Train: MATH level 3-5 (~8,890 rows) at `datasets/train_data/math-lv3to5/train.parquet` (filtered from `datasets/test_data/MATH/train.parquet`, has `level` col).
- `max_prompt_length=1024`, `max_response_length=3000` (OPD uses 3072, treated aligned), train rollout `temperature=0.6`, rollout `n=8`, actor `lr=5e-7` (sweep around it), init KL coef 0.01 (OPD launcher defaults USE_KL=False).
- Eval: `temperature=1.0, top_p=0.95, max_tokens=16000`; **MATH-500 pass@1** is the eval target (`datasets/test_data/MATH-500/test.parquet`). AIME = avg@32.

### C4. Trainset-name routing bug: `math-500k` → silent DAPO fallback  (all math wrappers)
- `scripts/opd/math/{full,lora,fura,qfura,qlora}.sh` + `scripts/compress_opd/math/btt_v2_combined_opd.sh` set `TRAIN_DATASET_NAME=math-500k`.
- The `on_policy_distillation.sh` case only matches `gsm8k|math-500`; `math-500k` hits the `*)` default → trains on `datasets/dapo-math-17k.parquet`, evals AIME25/AMC23/AIME24. **So those "math" runs were NOT on MATH.**
- **Fix:** add a `MATH)` branch (train=`datasets/train_data/math-lv3to5/train.parquet` per C3, val=`datasets/test_data/MATH-500/test.parquet`) and set `TRAIN_DATASET_NAME=MATH` in the wrappers. **VERIFY this is applied repo-wide before trusting any "math" result.**

### C5. Single-GPU OPD step timing (budget all runs against this)  (all)
- Measured on free H100 NVL (95GB), Qwen3-1.7B actor + Qwen3-4B teacher, MATH lv3-5, MAX_RESP_LENGTH=3072, N_RESPONSES=8, MINI_BATCH_SIZE=64, gpu_mem_util 0.55:
- `timing_s/step` ≈ **355–390s (~6 min/step)** → 138 steps ≈ **13–14h/run**.
- Breakdown: generate_sequences≈161s, compute_rm_score (teacher)≈80s, reward≈116s, update_actor≈95–100s.
- `ACTOR_OPTIM_OFFLOAD=False` does NOT speed it up — cost is rollout + teacher scoring, inherent to OPD. To go faster you'd cut N_RESPONSES/max_resp/batch (breaks SimpleRL alignment) or add GPUs.

---

## P1 — compressed-opd findings
<!-- Compress 4B→1.7B (BTT), SFT student on 4B-generated seqs, then OPD with 4B teacher. -->
- Calibrated BTT compression of 4B→~1.7B teacher landed (commit 15f9d45). The CALIBRATED BTT path runs on CUDA already (contrast the plain path bug under P2/F2).
- `calib.loss=opd` wired through PEFTConfig + BlockTT/SVD adapters (commit 2b2395e).
- FSDP2 forced (8240edc) + LR→1e-6 + memory-fit knobs for shared GPU (af170f5).
- **New, untracked:** `scripts/opd/math/compressed_opd/eval_c4_ppl.py` — C4 perplexity eval of the compressed teacher (the quality gate). Not yet committed; behavior/results not yet logged here.
- **Pipeline order (per user):** (1) compress 4B→1.7B, (2) SFT student on sequences GENERATED by the 4B teacher, (3) OPD with the ORIGINAL 4B as teacher.
- **Open:** has the SFT-on-4B-generated step been run/verified? Has C4 PPL been measured? — log results here when known.

### P1.D — OpenThoughts3 math prompts for teacher rollout (SFT step input)  (2026-05-31)
- README's teacher-rollout cmd reads `datasets/OpenThoughts3-1.2M-math.parquet`; `vllm_rollout.py` only uses `df["prompt"]` (a chat list) → `tokenizer.apply_chat_template`. So the input parquet just needs a `prompt` column of `[{"role":"user","content":...}]`.
- Repo already had `datasets/OpenThoughts3_opd.parquet` = 30k rows, `data_source=open-thoughts/OpenThoughts3-1.2M-**complement**` (held-out slice WITH ground_truth), prompts already suffixed with "Please reason step by step ... \boxed{}.". That is a drop-in rollout input.
- `datasets/OpenThought3-Qwen3-4B/` = 305k rows of ALREADY-rolled-out teacher responses (`messages` w/ assistant) = the released SFT dataset (OUTPUT of the rollout step), not the input.
- **User asked (2026-05-31) for the FULL upstream corpus, math-filtered, saved to `/data/yequan/datasets`.**
  - Upstream `open-thoughts/OpenThoughts3-1.2M`: 120 parquet shards, 10k rows each = 1.2M total, **28.2 GB**. Schema: `difficulty(float)`, `source(str)`, `domain(str∈{code,math,science})`, `conversations(list[{from,value}])`. **Shards are grouped by domain** — code shards first (0–~22), then math (~34–67+), etc. `domain=="math"` ≈ 850k rows.
  - math `conversations` = `[{from:'human', value:question}, {from:'gpt', value:<think>...\boxed{}}]`. Human turn does NOT include the boxed-instruction suffix (the complement slice added it). `\boxed{}` extractable from gpt turn for ground_truth.
  - Downloaded to `/data/yequan/datasets/OpenThoughts3-1.2M/data/train-*.parquet` via `hf download ... --include "data/*.parquet" --local-dir ...` (hf_transfer on, 8 workers).
  - Filter+format script: `scripts/infer/build_openthoughts3_math.py` → writes `/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet` in the verl prompt format (prompt suffixed w/ boxed instruction, best-effort ground_truth from gpt \boxed{}).
- **/data is 95–96% full (≈1.3T free of 28T)** — fine for 28GB but watch headroom.
- **DONE 2026-05-31:** full 1.2M downloaded (120 shards, 27GB at `/data/yequan/datasets/OpenThoughts3-1.2M/data/`), filtered → **`/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet`** = **850,000 math rows**, 130MB, 5 cols (prompt/data_source/ability/reward_model/extra_info). ground_truth (\boxed{}) on 321k/850k. Validated: `apply_chat_template` with local Qwen3-4B tokenizer works; 0 malformed in 5k sample.
- **To run the README rollout, point `--input-parquet` at the ABSOLUTE path** `/data/yequan/datasets/OpenThoughts3-1.2M-math.parquet` (README uses the repo-relative `datasets/OpenThoughts3-1.2M-math.parquet`, which does NOT exist — the prepared file is under /data per user request). Note `vllm_rollout.py` ignores ground_truth/reward_model; only `prompt` is used.

## P2 — peft-opd findings
<!-- OPD under LoRA / QLoRA / FurA(BlockTT) / qFurA vs full FT. -->

### F2. FurA (plain BlockTT) OPD — 7 fixes to train end-to-end (VERIFIED 2026-05-31)
Running `scripts/opd/math/fura.sh` (PEFT_MODE=blocktt, BTT_CONVERT_MODE=svd, BTT_QFURA=False, CALIB_MODE=none) failed at init 3+ ways; the plain non-calibrated non-qfura BlockTT path had never been exercised. Fixes:
1. `ACTOR_PARAM_OFFLOAD=False` in fura.sh (BTT conversion needs CUDA weights; necessary not sufficient).
2. In `verl/verl/workers/peft/blocktt.py` plain-convert `else` branch: `model = model.to(torch.cuda.current_device())` before `convert_linear_to_btt_compress(...)`. (calibrated/qfura branches already on CUDA.)
3. `actor.fsdp_config.use_orig_params=True` for blocktt (mixed requires_grad → FSDP needs it; LoRA avoids via lora-aware wrap policy).
4. `BlockTTAdapter.export_for_vllm` must export the COMPLETE weight set: each BTTLinear's `materialize_dense_weight()` as `{name}.weight` (+bias), THEN every other named_parameter verbatim, stripping `_fsdp_wrapped_module.` from keys. Old version dropped embed/layernorm/lm_head and left FSDP qualname → KeyError on first rollout weight-sync. vLLM Qwen3 fuses split q/k/v/gate/up, so emitting split *_proj.weight is correct.
5. `model.enable_input_require_grads()` in BlockTTAdapter.apply() (grad checkpointing + frozen embeds → backward "element 0 does not require grad" without it).
6. Untie input/output embeddings for ALL blocktt branches (`_untie_embeddings` at apply() start).
7. **Use FSDP2** (the decisive fix — see C1). Fixes 5,6 alone don't resolve the writeback failure; FSDP2 does.
- **VERIFIED 2026-05-31:** step:1 true_reward 0.543, actor/grad_norm 1.016, actor/pg_loss 0.0045, 0 crashes. Gradients flow through BlockTT cores.
- Reference impl: `/home/yequan/Project/lora/lora-without-regret/run_rl.py` (`export_weights_for_vllm` ~L753, `_build_factored_dense_state_dict` ~L912; also `btt_layer.py`, `compress_integration.py`). That repo uses single-GPU `device_map`, sidestepping all FSDP issues.
- **qFurA (BTT_QFURA=True) uses a DIFFERENT streaming path — NOT yet verified, may need separate fixes.**
- Reward sanity baseline: `critic/true_reward/mean` ~0.47–0.53 at steps 1–2 (Qwen3-1.7B base, MATH lv3-5).

## P3 — param-space-opd findings
<!-- OPD via ES + node-perturbation (NP) trainers in verl; future combine w/ FurA. -->
- NP trainer scaffold (from git, commits 9df753f→ef6bc73): package skeleton, deterministic seeding + gaussian/bernoulli/uniform noise, perturb_rules regex + per-step layer schedule, node-perturbation gradient estimator (sample_scale + rank-1 accumulate), reverse-KL top-k kernel for per-token teacher scoring, PerturbedLinear shim + install_perturb_layers, np_trainer.yaml Hydra config (np.* interface), re-exports ES `get_task_components`, assemble_layer_delta + apply_node_update, per-layer NCCL broadcast + inter-engine group init, n_sample-wide custom decode driver + sigma=0 verification, RayNPTrainer + NPNcclLLM + teacher scorer + per-step fit loop, main_np.py Hydra entry, opd_np.sh launcher.
- Final-review remediation (ef6bc73): sign descent, top_k_strategy, grpo guard.
- Verification tooling: `scripts/zo_opd/np_checks/check_grad_cosine.py` (gradient cosine-sim vs true grad), `check_decode_sigma0.py` (sigma=0 smoke). 1% training slice + e2e smoke fixes (8a8ab80).
- ES trainer ported (merge d59f38d feat/es-trainer-port).
- **Open:** real (non-smoke) NP/ES OPD training results not yet logged. Future: combine NP perturbation with FurA cores (cross-link to F2/C1 when attempted).

---

## Technical Decisions (shared)
| Decision | Rationale |
|----------|-----------|
| FSDP2 default for any BlockTT/frozen-embed arm | C1 — FSDP1 writeback is fundamentally broken for these |
| Teacher LLM occupies verl's `reward_model.*` slot | Repo convention — "reward model" = teacher, not scalar RM |
| ADV_ESTIMATOR=token_reward_direct = the OPD method | Per-token teacher-logprob reward; do not swap for GRPO |

## Issues Encountered (cross-project log; add project + attempt)
| Issue | Project | Resolution |
|-------|---------|------------|
| FSDP1 FlatParameter writeback fails on frozen embed | P1, P2 | FSDP2 (C1 / F2.7) |
| Plain BlockTT convert with weights on CPU | P2 | `.to(cuda)` before convert (F2.2) |
| Incomplete vLLM weight export → KeyError on weight-sync | P2 | Export complete weight set, strip FSDP qualname (F2.4) |
| Concurrent runs collide / kill each other's Ray | all | CUDA default pin + RAY_ISOLATE (C2) |
| "math" runs silently train on DAPO | all (math) | MATH) launcher branch (C4) |

## Resources
- verl OPD algorithm: `verl/verl/trainer/ppo/core_algos.py` (@register_adv_est).
- OPD knobs plumbing: `verl/verl/workers/config/rollout.py`, `fsdp_workers.py`, `actor/dp_actor.py`.
- BlockTT adapter: `verl/verl/workers/peft/blocktt.py`.
- Math reward: `verl/verl/utils/reward_score/ttrl_math/`.
- Reference BlockTT impl: `/home/yequan/Project/lora/lora-without-regret/run_rl.py` (+ btt_layer.py, compress_integration.py).
- Paper: arXiv:2604.13016 (OPD). SimpleRL-Zoo: arXiv:2503.18892.
- Auto-memory dir: `/home/yequan/.claude/projects/-home-yequan-Project-compression-OPD/memory/`.

## Visual/Browser Findings
- (none yet)

---
*Cross-cutting findings come FIRST — read them before touching any project.*
*A fix in one project that could bite another → promote it to CROSS-CUTTING.*
