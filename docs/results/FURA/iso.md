# ISO-AdamW reproduction (Qwen3-4B-Base math) and `isobtt` (fura + ISO)

> Reproduces the ISO-AdamW curve of the ISO paper's Fig. 6 (right) for 50 steps on 2 GPUs, then runs
> `isobtt` (ISO on fura's block-wise SVD) with the same recipe and a tuned LR. Both done (2026-10-05).

## Key result

| run (50 steps, 2× H100) | LR | step 0 | 10 | 20 | 30 | 40 | **50** | train time |
|---|---|---|---|---|---|---|---|---|
| paper ISO-AdamW (Fig. 6 right, read-off) | 7.5e-7 | 16.2 | – | 29.5 | – | 32.2 | ~34 | – |
| **ISO-AdamW** (ours) | 7.5e-7 | 20.0 | 29.9 | 30.6 | 31.4 | 31.9 | **32.2** | 14.3 h |
| **isobtt** (ours) | 2.5e-6 | 20.0 | 30.6 | 34.0 | 35.3 | 37.6 | **37.7** | 35.6 h |

Avg = unweighted mean of AIME24/25, AMC23, Minerva, OlympiadBench, avg@16.

- **The ISO reproduction matches the paper:** our ISO-AdamW tracks the paper's absolute curve within
  ~1 pp at steps 20 and 40 (30.6 vs 29.5, 31.9 vs 32.2), same shape (fast jump, then slow climb).
- **`isobtt` beats ISO by +5.5 pp at step 50** (37.7 vs 32.2) and is ahead at every eval from step 20;
  the gain is concentrated on the hard sets (AIME24 21.0 vs 14.8, AIME25 17.7 vs 10.4).
- Caveat: `isobtt` reaches it with ~3× longer answers (3.6K vs 1.1K tokens, 15–27% hitting the 8K cap
  vs <1%), so it is 2.5× slower per step, and its LR is 3.3× ISO's — it is not an equal-step-size
  comparison. Its 8K truncation is now the bottleneck.

## How to launch (exact commands used for the runs above)

Launcher: `scripts/iso/dapo_math_4b.sh` (verl DAPO recipe, paper App. H settings baked in; knobs via
env vars). Run from the repo root; it starts a private Ray head for the GPU set, uses the `verl`
env, and saves under `/data/yequan/opd/opd/DeepMath-103K/<exp>/` (latest ckpt only, every eval).

```bash
cd /home/yequan/Project/compression/OPD
mkdir -p logs/iso

# 0. one-time (already done): build datasets/DeepMath-103K.parquet (103,022 problems);
#    prompt = question + " Please reason step by step, and put your final answer within \boxed{}."
HF_HOME=/data/yequan/huggingface /home/yequan/miniconda3/envs/verl/bin/python scripts/iso/prep_deepmath.py

# 1. optional end-to-end check (2 tiny steps, ~40 min incl. init)
SMOKE=1 DEVICES=5,7 PEFT_MODE=iso LR=7.5e-7 \
  setsid nohup bash scripts/iso/dapo_math_4b.sh > logs/iso/smoke_iso.log 2>&1 < /dev/null &

# 2. ISO-AdamW (paper LR), 50 steps, evals at steps 0,10,...,50 (step 0 = base model, ~1h45m).
#    (The logged run used VAL_BEFORE=False and took its step-0 eval from the first, OOM'd attempt
#    with identical settings; this single command does both.)
PPO_TOK=9216 DEVICES=5,7 PEFT_MODE=iso LR=7.5e-7 TOTAL_STEPS=50 \
  setsid nohup bash scripts/iso/dapo_math_4b.sh > logs/iso/iso_lr7.5e-7.log 2>&1 < /dev/null &

# 3. isobtt (tuned LR), same recipe. Step 0 equals the base model bit-for-bit, so the
#    step-0 eval is skipped and reused from the ISO run (20.01).
PPO_TOK=9216 VAL_BEFORE=False DEVICES=5,7 PEFT_MODE=isobtt LR=2.5e-6 TOTAL_STEPS=50 \
  setsid nohup bash scripts/iso/dapo_math_4b.sh > logs/iso/isobtt_lr2.5e-6.log 2>&1 < /dev/null &
#    (to run 3 automatically after 2: `bash scripts/iso/chain_isobtt.sh <pid of step-2 launcher>`;
#     note it defaults to LR=7.5e-6 and VAL_BEFORE=True — override with LR=2.5e-6 VAL_BEFORE=False)

# 4. read results
python3 scripts/iso/collect_val.py logs/iso/iso_lr7.5e-7.log logs/iso/isobtt_lr2.5e-6.log
#    the last step's console line can be lost when the run exits -> read it from the eval dump:
/home/yequan/miniconda3/envs/verl/bin/python scripts/iso/val_from_jsonl.py \
  /data/yequan/opd/opd/DeepMath-103K/dapo_q3-4b-base_deepmath_<PEFT_MODE>_lr<LR>/val_generations/50.jsonl
```

Launcher knobs: `PEFT_MODE` (`iso` | `isobtt` | `none` = weight-space AdamW), `LR`, `DEVICES`
(2 GPUs used here), `TOTAL_STEPS` (50), `TEST_FREQ` (10; also the save interval), `PPO_TOK`
(**set 9216** — the default 16384 OOMs the 4B ISO actor backward on 94 GB cards), `VAL_BEFORE`
(True), `THINKING` (True = stock Qwen3 template; keep it — False drops the base score to 9.8),
`GPU_UTIL` (vLLM, 0.55), `SMOKE`. wandb project `iso_rlvr_qwen3_4b`.

Gotchas:
- One run per GPU set; before relaunching on the same GPUs, check `nvidia-smi` is empty and kill
  leftover `ray_iso_gpu<first GPU>` processes **by PID** (a finished run can leave its Ray GCS alive).
- To stop a run: `kill -TERM -<pgid of the launcher bash>`, then kill remaining `ray_iso_gpu5` PIDs.
- Cost on 2× H100 NVL: ISO ~16 min/step + ~45 min/eval (≈17 h total for 50 steps); `isobtt`
  13 → 70 min/step as its answers lengthen (≈36 h train + evals).
- Requires the `iso` / `isobtt` PEFT modes (`verl/verl/workers/peft/iso.py`), the ref-policy skip in
  `fsdp_workers.py` and the `true_reward_score` line in `verl/recipe/dapo/dapo_ray_trainer.py`.

## Setup

Paper recipe (`docs/papers/26_ISO.pdf`, App. H), launcher `scripts/iso/dapo_math_4b.sh`:

| knob | value |
|---|---|
| model / data | `Qwen/Qwen3-4B-Base` / DeepMath-103K (`datasets/DeepMath-103K.parquet`, 103,022 problems) |
| algorithm | DAPO (`recipe.dapo.main_dapo`): GRPO advantages, clip 0.2/0.28, token-mean loss, KL loss β=1e-3 (low_var_kl), dynamic filtering of all-correct/all-wrong groups with batch refill |
| batch | 256 prompts × 12 rollouts = 3072 seqs / update, 1 optimizer step per update |
| lengths | prompt 1024, response 8192 |
| rollout | T=1.0, top-p 1.0 |
| optimizer | AdamW, wd 0, grad clip 1.0, constant LR |
| ISO-AdamW | `peft.mode=iso`, LR **7.5e-7** on (U, V), fp64 polar retraction every step; embeddings/norms trained by plain AdamW |
| eval | AIME24/25, AMC23, Minerva, OlympiadBench; avg@16, T=1.0, top-p 0.8, 8192 tokens; every 10 steps; **avg = unweighted mean of the 5** (paper's "Avg") |
| hardware | 2× H100 NVL (GPUs 5, 7), FSDP1 fp32 master + bf16 compute, `ppo_max_token_len_per_gpu=9216` |

Not stated in the paper, my choices: prompt template (below), KL type `low_var_kl`, reward = our
`ttrl_math` grader (sympy/math_verify-level equivalence), 1 optimizer step per update (from
"400 updates ≈ 1 pass over 103K at batch 256").

## Key read 1: the prompt template sets the base score

The paper's base row is 16.15 avg. The base model's score depends mostly on the chat template:

| template (same grader, same sampling) | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | **avg** | mean len | hits 8K cap |
|---|---|---|---|---|---|---|---|---|
| paper (Table 3, base) | 6.88 | 5.00 | 25.45 | 20.97 | 22.43 | **16.15** | – | – |
| A: Qwen3 template, `enable_thinking=False` (avg@16) | 6.04 | 2.08 | 16.34 | 9.24 | 15.26 | **9.79** | ~3.8K | 42% |
| **B: Qwen3 template default (stock verl)** (avg@16, used) | 8.12 | 6.88 | 31.25 | 24.06 | 29.73 | **20.01** | ~1.5K | 10% |

- A makes the base model ramble (only ~40% of answers are boxed). B boxes 88%.
- B is what stock verl does for Qwen3 tokenizers, so it is the most likely paper setting; it is also
  2.5× cheaper to roll out. The +4 pp gap to the paper is spread evenly over the five benchmarks,
  which points at a grading difference, not a different model.
- The grader is not the cause of A's low score: `math_verify` agrees with ours within 0.3 pp.
  verl's strict `math_dapo` grader is far too strict here (AMC23 → 0.1%, gts stored as `142.0`).
- So we match the paper on **gain over base**, not on absolute score.

## Results

### ISO-AdamW, LR 7.5e-7 (template B)

wandb: [pzcd6ihk](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/iso_rlvr_qwen3_4b/runs/pzcd6ihk)
(step-0 eval: [ku0ofhj8](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/iso_rlvr_qwen3_4b/runs/ku0ofhj8), same config, crashed at step 1 by OOM).
Logs `logs/iso/iso_lr7.5e-7.log`; per-sample eval dumps
`/data/yequan/opd/opd/DeepMath-103K/dapo_q3-4b-base_deepmath_iso_lr7.5e-7/val_generations/<step>.jsonl`.

| step | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | **avg** | gain | paper ISO gain |
|---|---|---|---|---|---|---|---|---|
| 0 | 8.12 | 6.88 | 31.25 | 24.06 | 29.73 | **20.01** | – | – |
| 10 | 10.83 | 11.46 | 46.16 | 39.43 | 41.59 | **29.89** | +9.9 | – |
| 20 | 11.04 | 10.62 | 47.59 | 40.79 | 42.77 | **30.56** | +10.6 | +13.3 (0.162→0.295) |
| 30 | 13.54 | 10.42 | 48.34 | 41.11 | 43.38 | **31.36** | +11.4 | – |
| 40 | 13.75 | 10.00 | 49.92 | 40.97 | 44.77 | **31.88** | +11.9 | +16.0 (→0.322) |
| 50 | 14.79 | 10.42 | 48.42 | 42.07 | 45.31 | **32.20** | +12.2 | ~+18 (0.322→0.370 by 60) |

Train side (`scripts/iso/collect_val.py`): batch reward 0.33 → 0.57, grad norm 0.27 → 0.18, KL to
ref 0 → 1.1e-4, entropy 1.41 → 0.09, response length 1225 → ~1130 tokens, truncation 3% → 0.5%,
2 generation batches per step (≈50% of groups filtered), ~16 min/step on 2× H100.

**Read:** absolute scores track the paper closely (30.6 vs 29.5 at step 20, 31.9 vs 32.2 at step 40);
the gain over base is 2–4 pp smaller because our base starts 4 pp higher. The shape matches:
a big jump in the first 10–20 steps (format/termination), then a slow climb (+0.6 pp / 10 steps here).

### `isobtt` (ISO on fura's block-wise SVD) — LR tuning

Same recipe, `peft.mode=isobtt` (per input block `W[:, blk_j] = U_j S_j V_jᵀ`, U_j/V_j trained +
retracted, S_j frozen; 40–76 blocks of width 64–128). Step 0 is bit-identical to the base, so the
step-0 eval is reused (20.01). Health signals: response length and 8K-truncation rate (the failure
mode is an entropy collapse that turns into ever-longer, truncated answers).

| LR | per-step KL @ step 2 | what happened | verdict |
|---|---|---|---|
| ISO 7.5e-7 (ref) | 3.2e-6 | length flat ~1K, truncation ≤0.5% | – |
| **7.5e-6** (10× ISO, start point) | 3.0e-5 | entropy 1.41→0.16 in **one** step; length 0.9K→2.2K, truncation 0.1%→9.6% by step 6 | unstable, killed @6 ([ylvzdf2e](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/iso_rlvr_qwen3_4b/runs/ylvzdf2e), `logs/iso/isobtt_lr7.5e-6_unstable.log`) |
| **2.5e-6** (3.3× ISO) | 6.1e-6 | steps 2–10 clean (len ~850, trunc ≤0.3%); then length grows ~+90 tok/step (3.0K, 18% trunc @29) **while eval keeps rising** | used |

### `isobtt`, LR 2.5e-6

wandb: [eixguy6f](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/iso_rlvr_qwen3_4b/runs/eixguy6f).
Logs `logs/iso/isobtt_lr2.5e-6.log`; eval dumps
`/data/yequan/opd/opd/DeepMath-103K/dapo_q3-4b-base_deepmath_isobtt_lr2.5e-6/val_generations/<step>.jsonl`.

| step | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | **avg** | vs ISO same step |
|---|---|---|---|---|---|---|---|
| 0 | 8.12 | 6.88 | 31.25 | 24.06 | 29.73 | **20.01** | = |
| 10 | 12.08 | 10.21 | 46.76 | 41.48 | 42.51 | **30.61** | +0.7 |
| 20 | 14.58 | 14.79 | 51.05 | 43.15 | 46.28 | **33.97** | +3.4 |
| 30 | 19.17 | 17.50 | 53.16 | 39.64 | 46.92 | **35.28** | +3.9 |
| 40 | 22.29 | 18.75 | 56.10 | 41.27 | 49.75 | **37.63** | +5.8 |
| 50 | 21.04 | 17.71 | 58.58 | 41.27 | 50.09 | **37.74** | +5.5 |

Train side: batch reward 0.33 → ~0.58; KL to ref ~8e-5 (≈ ISO's at step 50); entropy 0.09;
response length 0.85K (steps 3–10) → 3.9K peak @35 → 3.6K @50; truncation peak 27% @35, then
self-limits to ~15%; 2–3 generation batches/step; 13 min/step early → ~70 min/step late.

**Read:** at 2.5e-6 `isobtt` learns faster and further than ISO, mostly by learning to reason
longer. That is the same direction the paper reports for ISO at 8B (longer answers, cap raised to 16K).
The length growth did not turn into the 7.5e-6 failure: truncation peaked and came back down while
eval kept rising.

## [2026-10-06] Full 400-step reproduction on NERSC (2 nodes × 4 A100-80GB, 4 runs)

> Same recipe as above (paper App. H: 256 prompts × 12 rollouts, 8K, DAPO) on 8 GPUs per run, wandb
> `iso_rlvr_qwen3_4b`. Launched for 400 steps; **cut to 160 steps for all four runs on 2026-10-07** (cost:
> answers lengthened to 2.3K+, 23–30 min/step; 400 steps ≈ 1,400–1,600 node-hours, 160 ≈ ~40% of that).

| run | arm | LR | wandb id | status |
|---|---|---|---|---|
| full | weight-space AdamW (`PEFT_MODE=none`) | 2e-6 (paper's best AdamW) | `q3-4b_full_lr2e-6_c` | running (phase 1) |
| fura | BlockTT small core only, output_one_block, S merged into frozen core, FSDP2 | 2e-5 (10× full) | `q3-4b_fura_lr2e-5_c` | running (phase 1) |
| iso | ISO-AdamW | 7.5e-7 | `q3-4b_iso_lr7p5e-7_c` | regular-QOS batch, submitted 2026-10-08 (jobs 59567885 → 916 → 917) |
| isobtt | ISO on block-wise SVD (U_j, V_j trained) | 2.5e-6 | `q3-4b_isobtt_lr2p5e-6_c` | regular-QOS batch, submitted 2026-10-08 (jobs 59567919 → 921 → 923 → 926) |

wandb runs are the `_c` ids (offline segments appended by `slurm/iso/wandb_append_daemon.sh`;
display name = id without `_c`), in project
[iso_rlvr_qwen3_4b](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/iso_rlvr_qwen3_4b).

**Changes vs the 2-GPU runs:** 8 GPUs (2 nodes) instead of 2; eval every **20** steps (was 10),
ckpt every step (keep 1; saves take 4–12 s; was 10, then 2 — at 25–36 min/step an unsaved odd step cost ~30 min per 4 h cutoff); step-0 eval only for full and fura (iso/isobtt step 0 = base bit-for-bit, reuse full's).
fura's blocks = closest factor pair (`factorize_by_head=False`), the same blocks isobtt uses.

**First reads (full, step 0–1):** step-0 avg **20.17** (AIME24 11.3 / AIME25 6.0 / AMC23 28.8 /
Minerva 25.1 / Olympiad 29.6) = the 2-GPU base 20.01, so the setup transfers. Step 1: 13.6 min
(gen 377 s, actor update 306 s), 61 GB reserved/GPU, score 0.36, length 1.19K, 2.7% truncated,
2 gen batches. ⇒ ~90 h train per run + evals ≈ 25 four-hour segments; phases run back to back.

**Eval (avg@16, 5-benchmark mean; paper read-offs: AdamW 2e-6 28.8 @20 / 30.1 @40, ISO 29.5 @20 / 32.2 @40):**

| step | full AIME24/25, AMC, Minerva, Oly | **full avg** | fura | **fura avg** | iso | isobtt |
|---|---|---|---|---|---|---|
| 0 | 11.3 / 6.0 / 28.8 / 25.1 / 29.6 | **20.17** | 12.1 / 5.2 / 29.4 / 24.1 / 30.2 | **20.18** | = full | = full |
| 20 | 11.5 / 10.0 / 46.9 / 40.6 / 42.0 | **30.18** | 11.7 / 7.9 / 47.5 / 41.3 / 41.8 | **30.03** | | |
| 40 | 14.2 / 11.7 / 48.5 / 41.6 / 44.0 | **31.98** | 15.0 / 11.3 / 47.2 / 41.7 / 43.6 | **31.75** | | |
| 60 | 16.3 / 12.3 / 50.4 / 41.3 / 46.0 | **33.24** | 20.0 / 15.6 / 51.1 / 41.4 / 46.2 | **34.87** | | |
| 80 | 16.5 / 14.2 / 56.0 / 42.8 / 50.1 | **35.88** | 19.6 / 15.8 / 52.0 / 41.5 / 46.6 | **35.10** | | |
| 100 | 23.1 / 17.7 / 57.2 / 42.5 / 51.3 | **38.37** | | (pending) | | |

Read: full AdamW 30.2 @20 / 32.0 @40 (paper AdamW 28.8 / 30.1; our 2-GPU ISO 30.6 / 31.9); fura tracks it within 0.2 pp (30.0 / 31.8) while training only the small BTT cores; @60 fura pulls ahead (34.9 vs 33.2, gain on AIME24/25).
Watch: from ~step 45 fura's answers lengthen (2.5K tokens, 11% truncated at 8K, 25 min/step @60)
while full stays at ~1.6K / 2% — the same length growth isobtt showed at 2.5e-6 above. @80 they are level
again (full 35.9, fura 35.1); full's answers now grow too (3.3K, 14% truncated, 36 min/step @95), fura flat ~2.4K / 10%. Tables are
rebuilt with `python3 scripts/iso/collect_val.py $PSCRATCH/opd/logs/iso/q3-4b_<arm>_attempt*.log`.

**Smoke-verified on 2 nodes before launch** (2 steps → ckpt → resume to 4): full, fura, iso all
train, save, eval and resume (model + optimizer + rng + lr_scheduler loaded; old ckpt pruned).
isobtt not smoked separately (same code path as iso).

How to launch / monitor / stop:

```bash
bash slurm/iso/launch.sh full      # or fura | iso | isobtt; SMOKE=1 [SMOKE_STEPS=4] for a smoke test
setsid nohup bash slurm/iso/chain_phase2.sh > $PSCRATCH/opd/logs/iso/chain_phase2.log 2>&1 &
setsid nohup bash slurm/iso/wandb_append_daemon.sh > $PSCRATCH/opd/logs/iso/wandb_append_daemon.log 2>&1 &
tail -f $PSCRATCH/opd/logs/iso/ctl_q3-4b_<arm>_*.log          # segments / resume points
python3 scripts/iso/collect_val.py $PSCRATCH/opd/logs/iso/q3-4b_<arm>_attempt*.log
# stop: kill the controller PID (pgrep -af slurm/iso/controller.sh), then scancel its job
```

iso / isobtt run as **regular-QOS batch jobs** instead (2026-10-08; phase-2 interactive chain stopped):
`bash slurm/iso/submit_batch.sh <arm> [n_jobs]` submits n × 24 h jobs (`slurm/iso/batch.sbatch`, 2 nodes)
chained `afterany`; each resumes from the ckpt and exits at once if step 160 is done. Logs
`$PSCRATCH/opd/logs/iso/<wandb id>_batch_<jobid>.log`; cancel with `scancel -n <wandb id>`.

Paths: ckpts `$PSCRATCH/opd/checkpoints/iso/dapo_q3-4b-base_deepmath_<arm>_lr<LR>_2node/`
(eval dumps in `val_generations/<step>.jsonl`), logs `$PSCRATCH/opd/logs/iso/`.

Code changes for these runs (details in Reference):
- `peft/blocktt.py` export: gather DTensor cores before contracting. Multi-rank FSDP2 with
  `s_merged_to=frozen` otherwise fails ("Attempted to flatten sharded dimension").
- `peft/iso.py retract_iso`: no-op when the model has no ISO frames. It used to raise
  "FSDP1 only" on any FSDP2 model, e.g. fura.
- DAPO trainer: skip `val_before_train` when resuming (it re-ran the step-0 eval every 4 h segment).
- `scripts/iso/dapo_math_4b.sh`: env-overridable REPO/CKPT/NNODES/RESUME_MODE/strategy,
  `RAY_EXTERNAL=1`, new `PEFT_MODE=fura`, `SMOKE_STEPS`; `slurm/opd/opd_2node_inside.sh` takes
  `RAYENV_SCRIPT` (iso needs `verl/` on the raylets' PYTHONPATH for `recipe.dapo`).

## Next steps

1. **Weight-space AdamW (2e-6) baseline** on the same 2 GPUs — needed to state ISO-vs-AdamW
   ourselves instead of leaning on the paper's curve.
2. **Equal-motion `isobtt`**: rerun at ~1.25e-6 (per-step KL ≈ ISO's) to separate "block-wise
   frames" from "bigger step size".
3. Raise the response cap to 16K for `isobtt` (as the paper did at 8B), since 15–27% truncation caps it.

## Reference

- Fig. 6 (right) read-off, ISO-AdamW 7.5e-7: 0.162 @0, 0.295 @20, 0.322 @40, 0.370 @60, 0.42 @180.
  AdamW 2e-6: 0.288 @20, 0.301 @40.
- Failed attempts: (1) template A run, step-0 eval 9.79, aborted after step 1 (42 min/step, 3.7 h eval);
  (2) first template-B run OOM'd in the actor backward at `ppo_max_token_len_per_gpu=16384`
  (4.6 GiB logits alloc); fixed with 9216.
- Scripts: `scripts/iso/dapo_math_4b.sh` (launcher; `PEFT_MODE`, `LR`, `SMOKE=1`, `PPO_TOK`,
  `VAL_BEFORE`), `scripts/iso/collect_val.py` (eval + train tables from a log),
  `scripts/iso/val_from_jsonl.py` (eval from `val_generations/<step>.jsonl`; checked equal to the
  logged step-40 numbers), `scripts/iso/chain_isobtt.sh`. Final checkpoints: `.../global_step_50` in
  each run dir (FSDP shards + `actor/merged_hf` dense bf16).
- Code changes for this run: DAPO trainer sets `true_reward_score` (this fork's metrics need it);
  the reference policy skips the ISO conversion (`fsdp_workers.py`, it is the base model).
