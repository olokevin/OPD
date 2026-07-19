# FURA vs Full-model GRPO on Qwen2.5-7B (NERSC)

> Zero-RL GRPO on `Qwen2.5-7B` (base) over MATH lv3–5, comparing **full-model**
> fine-tuning against **FURA** (BlockTT parameter-efficient). Two 2-node
> interactive jobs on Perlmutter, logged to wandb project
> [`nersc_grpo_qwen2p5_7b`](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/nersc_grpo_qwen2p5_7b).
> Started 2026-07-17. Builds on the multi-node infra in memory
> `nersc-multinode-opd` / `fura-blocktt-multinode`.

## Setup

| | Full | FURA |
|---|---|---|
| Actor | `Qwen/Qwen2.5-7B` (base, `Qwen2ForCausalLM`, 28 layers) | same |
| Estimator | GRPO (`adv_estimator=grpo`), rule-based math reward, **no teacher** | GRPO |
| PEFT | none (full FSDP) | `blocktt` (FurA: bf16 frozen core, small BTT side trains) on **FSDP2** |
| BTT knobs | — | decomp `output_one_block`, rank full, train `small`, `s_merged_to=keep_trainable`, `convert_mode=svd`, `factorize_by_head=True` |
| Train / eval-during | MATH lv3–5 (`math-lv3to5/train.parquet`, 8890 rows) / MATH-500 | same |
| LR | 1e-6 | 1e-5 |
| N responses / mbs | 8 / 64 | 8 / 64 |
| Max prompt / resp / val | 1024 / 4096 / 4096 | same |
| Nodes | 2 × 4 A100-80GB (8 GPU) | 2 × 4 A100-80GB |
| Total steps | 138 (1 epoch) | 138 |
| wandb run | `grpo_full_qwen2p5_7b_math` | `grpo_fura_qwen2p5_7b_math` |

Memory-fit knobs (7B actor on 8×A100): `MODEL_DTYPE=bfloat16`,
`PPO_MAX_TOKEN_LEN_PER_GPU=16384`; full: `ACTOR_OPTIM_OFFLOAD=True` +
`gpu_mem_util=0.55`; fura: `ACTOR_PARAM/OPTIM_OFFLOAD=False` (BTT conversion needs
weights on CUDA) + `gpu_mem_util=0.5`. `REF_PARAM_OFFLOAD=True` for both.

## Infrastructure (new scripts under `slurm/grpo/`)

Reuses the validated OPD multi-node bootstrap (`slurm/opd/opd_2node_inside.sh` +
`opd_2node_rayenv.sh`) via the `ENV_SCRIPT` hook.

- `slurm/grpo/full/grpo_2node_env.sh`, `slurm/grpo/fura/grpo_2node_env_fura.sh` — GRPO driver envs (call `grpo.sh`).
- `slurm/grpo/full/grpo_full_controller.sh`, `slurm/grpo/fura/grpo_fura_controller.sh` — 2-node auto-resume controllers (4h interactive cap).
- `slurm/grpo/grpo_wandb_sync_daemon.sh` — login-node daemon pushing the offline wandb runs online.
- `slurm/grpo/eval/grpo_eval.sh` — post-training `val_only` eval on the 4 held-out benchmarks.

## Fixes applied to run GRPO / FURA at 7B under the multi-node infra

1. **`grpo.sh` `RAY_EXTERNAL`/`RAY_ISOLATE` support** — it did its own `ray stop --force` + `ray start --head`, which would tear down the pre-built 2-node Ray cluster. Now skips both when `RAY_EXTERNAL=1` (mirrors `on_policy_distillation.sh`).
2. **`grpo.sh` overridable `CKPT_PATH` / `EXPERIMENT_NAME`** — both were unconditionally overwritten with timestamps → breaks auto-resume (fresh dir per 4h segment) + wandb run pinning. Changed to `${VAR:-<default>}`. (Caught on the first launch; killed + fixed + relaunched.)
3. **`grpo.sh` exit-code propagation** — the trailing `if [ -z "$SLURM_JOB_ID" ]` block became the script's last statement, so a driver **crash exited 0** (false success) and the auto-resume controller *stopped* instead of retrying. Now captures `TRAINER_RC` and `exit`s with it.
4. **`fsdp_workers.py` meta-tensor init vs BlockTT/SVD** — large models with `tie_word_embeddings=False` (Qwen2.5-7B) meta-init the weights; BlockTT/SVD then crash in `blocktt.apply` `model.to(cuda)` with *"Cannot copy out of meta tensor"* (SVD needs real weight values). Now forces real-weight init when `peft.mode ∈ {blocktt, svd}`. Full (mode none) and the previously-validated Qwen3-1.7B FURA (tied embeddings) are unaffected.
5. **`grpo.sh` env knobs** — `TRAINER_LOGGER` (→ `[console,wandb]`), `PPO_MAX_TOKEN_LEN_PER_GPU` (OOM control; 32768 floor too large for 7B), `VAL_ONLY`/`VAL_BEFORE_TRAIN` (post-training eval).

## Results

### MATH-500 during training (val-core acc, avg@4, TEST_FREQ=5)

Both runs completed 138 steps (1 epoch) with auto-resume across the 4h interactive
cap (full: 3 segments, fura: 2). MATH-500 acc@4 climbs steadily under GRPO:

| step | 5 | 20 | 40 | 60 | 80 | 100 | 120 | 130 | 138 (final) |
|---|---|---|---|---|---|---|---|---|---|
| **Full** | 0.526 | 0.570 | 0.603 | 0.653 | 0.653 | 0.670 | 0.677 | **0.682** | 0.668 |
| **FURA** | 0.498 | 0.534 | 0.567 | 0.572 | 0.597 | 0.603 | 0.612 | 0.625 | **0.633** |

- **Full**: 0.526 → 0.682 peak (**+15.6 pts**), 0.668 final.
- **FURA**: 0.498 → 0.633 final (**+13.5 pts**). Starts ~3 pts below full (the BlockTT
  SVD decomposition is a lossy approximation of the base weights at init) and ends
  ~3.5 pts below full — a small PEFT gap for training only the compact BTT factors.

### Post-training benchmarks (`val_only` on the step-138 checkpoint, avg@16)

`mean@16` = pass@1 averaged over 16 samples (temp 1.0, top_p 0.95, max 4096 tok);
`maj@16` = majority vote; `best@16` = pass@16.

| Benchmark | Full mean@16 | Full maj@16 | Full best@16 | FURA mean@16 | FURA maj@16 | FURA best@16 |
|---|---|---|---|---|---|---|
| AMC23 | **0.362** | 0.477 | 0.704 | 0.345 | 0.464 | 0.667 |
| AIME24 | 0.060 | 0.111 | 0.239 | **0.081** | 0.157 | 0.315 |
| Minerva | **0.276** | 0.350 | 0.543 | 0.247 | 0.317 | 0.558 |
| Olympiad-Bench | **0.307** | 0.394 | 0.575 | 0.281 | 0.367 | 0.572 |

**Takeaway:** FURA (training only the compact BlockTT factors, ~a few % of params)
is within ~2–3 pts of full-model GRPO on AMC23/Minerva/Olympiad-Bench, and actually
*ahead* on AIME24 (0.081 vs 0.060 mean@16; best@16 0.315 vs 0.239). Combined with the
MATH-500 curve (FURA 0.633 vs full 0.668), FURA recovers most of full-model GRPO's
gains at a fraction of the trainable parameters. (FURA's first eval `salloc` hit a
transient NERSC "Connection timed out"; a retry wrapper cleared it — not a model issue.)

## Notes / observations

- Full step-5 baseline MATH-500 acc@4 = **0.526** (maj@4 0.555, best@4 0.72) — strong start for Qwen2.5-7B base; expect GRPO to lift it.
- Full: `max_memory_reserved ≈ 76.4/80 GB`, throughput ~480 tok/s, ~100 s/step train + ~220 s per MATH-500 eval.
- FURA is the first BlockTT run on a **GQA** model (Qwen2.5-7B: 28 heads / 4 KV heads); the meta-tensor fix (#4) was required to get it past model init.
- FURA memory footprint is *lower* than full (60.8 vs 76.4 GB reserved) — the optimizer state is tiny (only BTT factors train), which more than offsets keeping the frozen bf16 core on-GPU.
- Auto-resume across the 4h interactive cap was exercised for real: full = 3 segments (resumed at steps 90), fura = 2 segments (resumed at step 100), both continued with a continuous loss/curve.

_Status: **COMPLETE** (2026-07-18). Both runs trained 138 steps and were evaluated on all
4 held-out benchmarks. wandb: [`grpo_full_qwen2p5_7b_math`](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/nersc_grpo_qwen2p5_7b/runs/grpo_full_qwen2p5_7b_math),
[`grpo_fura_qwen2p5_7b_math`](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/nersc_grpo_qwen2p5_7b/runs/grpo_fura_qwen2p5_7b_math),
plus `_eval` runs for the benchmark pass._
