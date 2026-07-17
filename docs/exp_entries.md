# Experiment entries

> Per-experiment runbook: the **setting** (calibration, data, key hyperparameters) and
> the **run entry** (standard / single-node, and on SLURM / NERSC multi-node) for each
> active thread. Companion to the design docs in `wiki/` and the result logs in
> `results/`. When a recipe changes, update the relevant section here.

**Shared default calibration** (compress threads — `svd_nystrom` / SVD-V2 + Nystrom):
**128** reasoning traces from OpenThought3-Qwen3-4B, kept as **full sequences, NEVER
truncated**, **sequence-reweighted** covariance, last decoder layer left dense
(`skip_last_layers=1`). A length cap **DROPS** (does not truncate) any trace longer than it,
so kept traces stay intact. Per objective:
- **forward** (`svd_v2`): no cap (`max_seq_len=None`) — every trace at full length, `bs=2`.
- **combined** (`svd_v2_combined`): a full-length CE **backward** over a 4B model OOMs on the
  longest traces, so **drop traces > 8192 tokens** and use `bs=1`. **Verified on an 80 GB A100**
  (`slurm/compress_sft/test_combined_calib_mem.py`): full fwd+bwd + Nystrom-stats calibration
  completes at **~62 GB alloc / 64 GB reserved** (host RSS ~55 GB; the covariance dicts are
  d×d, size-independent of num_seqs/length).

Set in `compress_setup.py`, both `build_svd_nystrom_student.py`, and
`compress_common.build_calib_loader`; the drop-vs-truncate behavior lives in
`build_fullseq_calib_loader` (`src/compress/loaders.py`, submodule).

---

## compress_sft — reasoning-aware compress → SFT

Compress a well-trained Qwen3-4B (SVD-LLM-V2 on `self_attn` + Nystrom on MLP, retain 0.7,
in-process at LlamaFactory model-init so factors stay **trainable**), then SFT on
OpenThought3 to recover reasoning. Design: [wiki/reasoning_aware_compress_calib.md](wiki/reasoning_aware_compress_calib.md);
results: [results/compress_sft.md](results/compress_sft.md).

### Setting

| | |
|---|---|
| **Model** | `Qwen/Qwen3-4B` (non-thinking, `enable_thinking=false`) |
| **Method** | `finetuning_type: svd_nystrom`, `compression_ratio: 0.7` (retain), `skip_last_layers: 1`, `trainable_type: all` |
| **calib_mode** | `svd_v2` (forward) / `svd_v2_combined` (forward+backward) |
| **Calibration** | shared default above — `calib_num_seqs: 128`, full-seq never-truncated, sequence-reweighted, traces from `OpenThought3-Qwen3-4B/data/train.jsonl` |
| **Train data** | `openthought3_qwen3_4b` (305k-row `lllyx/OpenThought3-Qwen3-4B`), `cutoff_len: 10240`, 2 epochs |
| **Optim** | global batch **64**, `lr 1e-5`, cosine, warmup 0.05, bf16, `flash_attn: sdpa`, liger off |
| **Eval** | MATH-500 + AIME24 + MMLU-Pro (post-hoc, verl env); val-loss in-trainer |
| **Configs** | `LlamaFactory/examples/compress_train/qwen3_4b_{compressed,nersc_*}_{fwd,combined}_sft.yaml` |

### Run — standard (local, sft env)

```bash
# compress + train + save (one sft-env process), both objectives:
bash scripts/compress_sft/run_compress_sft.sh train all
# post-hoc benchmark eval on the merged ckpts (verl env):
bash scripts/compress_sft/run_compress_sft.sh eval all
# standalone compressed-student build (no SFT loop), e.g. for compress_opd:
python scripts/compress_sft/build_svd_nystrom_student.py \
  --model Qwen/Qwen3-4B --objective forward --ratio 0.7 --save-dir <out>
```

### Run — SLURM (NERSC, 4 nodes/16 GPU each, retain 0.7, current run)

```bash
# two concurrent jobs, 4 nodes each (interactive QOS allows it: node=4 is per-JOB,
# MaxJobsPU=2, no per-user node cap). Detached on a login node:
RATIO=0.7 nohup bash slurm/compress_sft/compress_sft_fwd_controller.sh \
  > /pscratch/sd/$USER/opd/compress_sft/logs/fwd_controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &
RATIO=0.7 nohup bash slurm/compress_sft/compress_sft_combined_controller.sh \
  > /pscratch/sd/$USER/opd/compress_sft/logs/combined_controller_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# unified wandb run (train curve + eval points) — ONE online logger per objective:
for o in forward combined; do
  PYTHONUNBUFFERED=1 nohup python -u scripts/compress_sft/log_train_to_wandb.py \
    --objective $o --ratio 0.7 > .../trainwandb_${o}.log 2>&1 &
done
# benchmark eval daemon (gpu_shared 1-GPU jobs; step-0 baseline + every ~500 steps):
RATIO=0.7 nohup bash slurm/compress_sft/compress_sft_eval_daemon.sh > .../eval_daemon.log 2>&1 &
```

- Auto-resume across the 4h interactive cap is in `_compress_sft_controller_core.sh`
  (`salloc -N4 ... → compress_sft_inside.sh → compress_sft_env.sh`, `save_steps=100`,
  `save_total_limit=1`). Jobs change id each re-alloc.
- wandb project `nersc_compress_sft_qwen4b`, runs `qwen3_4b_nersc_compress_{obj}_r0.7_sft`.
- **Stop-all** (controllers FIRST so they don't re-salloc):
  `pkill -f 'compress_sft_fwd_controller|compress_sft_combined_controller|log_train_to_wandb|compress_sft_eval_daemon'` then `scancel` the interactive jobs.

### Eval recipes

Each benchmark follows a fixed contract (so numbers are comparable across the project):

**MATH-500** — aligns with the wiki eval contract
([wiki/reasoning_aware_compress_calib.md](wiki/reasoning_aware_compress_calib.md) §Eval contract):
- **greedy** (`do_sample=False`, `max_new_tokens=2048`), Qwen3 **non-thinking** chat
  template (`enable_thinking=False`); graded by `ttrl_math.compute_score` against the
  dataset gold (`reward_model.ground_truth`, never a model output).
- companion metric: C4 sliding-window PPL (seqlen 2048, seed 0).
- standing references: dense Qwen3-4B = **80.5% / PPL 19.9**.

**AIME24 / AIME25 / AMC23** — same benchmark set + sampling distribution as the
**OPD-training validation** (SimpleRL-Zoo; `on_policy_distillation.sh` `val_kwargs`), but the
**sample budget differs per launched run** (the env scripts override the script defaults):
- shared: `do_sample=True`, **temperature 1.0**, **top_p 0.95**, vLLM; report mean acc over N
  samples. Benchmark set AIME24+AIME25+AMC23 = the DAPO / `*)` branch of the TEST_DATASET case.
- **as actually launched:**
  - opd full / LoRA / FURA (`slurm/opd/{full,lora,fura}/`): **avg@8** (`VAL_N=8`), `max_tokens=7168`.
  - compress_opd (`scripts/compress_opd/math/opd_svd_nystrom.sh`): **avg@4** (`VAL_N=4`),
    `max_tokens=3072`, and **adds MATH-500** to the set (AIME24+AIME25+AMC23+MATH-500).
  - `on_policy_distillation.sh` defaults (no override): avg@16, `max_tokens=31744`.
- (the standalone `scripts/val/eval` pipeline uses `n=16` / `top_p 0.95` but `temperature 0.7`.)

**MMLU** — matches LlamaFactory's **common evaluator** setting
(`hparams/evaluation_args.py` + `eval/evaluator.py`):
- MMLU (57 subjects), **5-shot** (`n_shot=5`), `lang=en`, `batch_size=4`, `seed=42`.
- multiple-choice scoring: softmax over the **A/B/C/D** answer-token logits at the answer
  position → argmax letter (NOT free-form generation).

**Implementation status** — the current post-hoc eval (`scripts/compress_sft/eval_opd_ckpt.py`
+ `eval_mmlu_pro.py`, HF greedy generate in the verl env) matches MATH but **diverges** on
the other two; to fully honor the contracts above:
- MATH-500 — ✓ matches (greedy 2048, non-thinking, ttrl grader).
- AIME — ✗ currently greedy + single sample @4096; align by switching to vLLM **avg@16**
  (temp 1.0, top_p 0.95, ~31744 tok) and adding AIME25 + AMC23.
- MMLU — ✗ currently **MMLU-Pro** via generation+`\boxed`; align by using **MMLU** (not Pro),
  5-shot, A–D choice-logprob (the fork disables LlamaFactory's built-in MMLU CLI, so this
  needs a small standalone harness or re-enabling it).

---

## compress_opd — compress → on-policy distillation

Two-stage: (1) build a reasoning-aware-compressed Qwen3-4B student (same `svd_nystrom`
recipe as compress_sft), (2) OPD on top of it with the **uncompressed** Qwen3-4B as teacher.
Scripts under `scripts/compress_opd/math/`; design: [wiki/compressed_opd.md](wiki/compressed_opd.md);
results: [results/compressed_opd.md](results/compressed_opd.md).

### Setting

| | |
|---|---|
| **Stage 1 (compress)** | `build_svd_nystrom_student.py`: SVD-V2 attn + Nystrom MLP, retain `0.7`, last layer dense, **shared default calibration** (128 full-seq never-truncated seq-reweighted OpenThought3). `--objective forward` (wiki D0) / `combined` (D2) |
| **Stage 2 (OPD)** | `opd_svd_nystrom.sh` → `on_policy_distillation.sh`. Student = the compressed HF dir; **teacher = `Qwen/Qwen3-4B`** (uncompressed, non-thinking) |
| **Estimator** | `token_reward_direct` (token-level teacher log-prob reward) |
| **Train data** | `datasets/OpenThoughts3_opd.parquet` (30k math prompts, verl OPD schema) |
| **Optim** | `LR 5e-7` (default, SimpleRL-Zoo); no structural mask to preserve (dense low-rank weights) |

### Run — standard (single GPU, verl env)

```bash
# full pipeline (stage1 compress → stage2 OPD), forward-only (wiki D0 default):
bash scripts/compress_opd/math/run_svd_nystrom_opd.sh
# combined (wiki D2):
OBJECTIVE=combined bash scripts/compress_opd/math/run_svd_nystrom_opd.sh
# stage control + concurrent GPUs:
STAGE=stage1 bash scripts/compress_opd/math/run_svd_nystrom_opd.sh        # build student only
GPU=1 RAY_ISOLATE=1 STAGE=stage2 STUDENT_DIR=<dir> bash scripts/compress_opd/math/run_svd_nystrom_opd.sh
```

### Run — SLURM (NERSC multi-node)

The compress_opd scripts target a single H100 with `/data/yequan` paths. On NERSC, run
**stage 1** a single-GPU build of the compressed student, then **stage 2 OPD** on 4
interactive nodes via `slurm/opd/compress/opd_compress_combined_controller.sh`.

**Shared-model design (2026-06-09): SFT and OPD start from the SAME compressed model.**
Both compress Qwen3-4B with combined (fwd+bwd) calibration on Qwen3-4B-generated OpenThought3
traces, **last layer dense** (`skip_last_layers=1`), and train OpenThought3 prompts. The
catch: that compression has a **heterogeneous MLP** (shrunk layers + a full last layer), which
neither verl's `from_pretrained` nor vLLM can load. Resolution = **Option B (zero-pad + freeze)**:
- `build_svd_nystrom_student.py --skip-last-layers 1 --save-zero-padded` merges SVD-attn to dense
  and **zero-pads the shrunk MLPs up to `intermediate_size`** → a **stock Qwen3** (vLLM/verl load
  it normally, no patch). The padding is exactly zero.
- OPD runs with **`SPARSEGPT_PRESERVE_MASK=1`** (`verl/verl/workers/sparsity_mask.py`), which
  snapshots the zero positions at load and keeps them zero every step (grad-mask + post-step
  re-zero, already wired into `dp_actor._optimizer_step`) → the model stays effectively compressed.

**Why keep vLLM (rollout benchmark, 2026-06-09).** Qwen3-4B, our rollout params, 2048-token cap
(HF at the real 7168 takes hours): **vLLM 1452 tok/s vs HF `generate` 97 tok/s → ~15× faster**
(understated — vLLM's paged-attn/batching edge grows with length + n). So OPD keeps `rollout.name=vllm`
on the zero-padded stock student rather than switching to `HFRollout` for the heterogeneous model.

---

## opd — on-policy distillation (full / LoRA / FURA)

The core OPD/RL trainer (`on_policy_distillation.sh` → `verl.trainer.main_ppo`): a teacher
gives dense per-token reward; a student is trained PPO-style. PEFT variants: **full**
(`PEFT_MODE=none`), **LoRA** (`lora`), **FURA** (`blocktt`, BlockTT cores). Knobs:
[../README.md](../README.md); memories `nersc-opd-setup`, `nersc-multinode-opd`, `fura-blocktt-multinode`.

### Setting

| | |
|---|---|
| **Estimator** | `ADV_ESTIMATOR=token_reward_direct` (OPD); `grpo` for zero-RL (`grpo.sh`) |
| **Models** | student `Qwen3-1.7B`; teacher `Qwen3-4B` (RL-Math, non-thinking) in the `reward_model.*` slot |
| **Train data** | `TRAIN_DATASET=datasets/dapo-math-17k.parquet` (or MATH lv3–5); eval `MATH-500`/`AMC23` |
| **OPD knobs** | `LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `REWARD_WEIGHT_MODE`, `TEACHER_TEMPERATURE` |
| **PEFT** | `none` (full, lr 1e-6) / `lora` (rank 128, alpha 256) / `blocktt` FURA (lr 1e-5; `strategy=fsdp2`, `ACTOR_PARAM_OFFLOAD=False`, `PYTHONPATH+=src/compress`) |

### Run — standard

```bash
# full OPD (single node, 4× A100): defaults token_reward_direct
bash on_policy_distillation.sh
# variant knobs (override env, keep the script diff-free):
PEFT_MODE=lora   LORA_RANK=128 LORA_ALPHA=256 bash on_policy_distillation.sh
PEFT_MODE=blocktt BTT_QFURA=False             bash on_policy_distillation.sh   # FURA
# single Perlmutter node via sbatch:
sbatch slurm/opd_math_1node.sbatch          # NERSC plumbing + MATH knobs, 4× A100
```

### Run — SLURM (NERSC, 2–4 nodes, interactive auto-resume)

```bash
# full run (job1), 4 nodes:
nohup bash slurm/opd/full/opd_job1_full_controller.sh   > .../job1_controller.log 2>&1 &
# FURA run (job2), 4 nodes, concurrent with job1 (2 jobs × 4 nodes):
nohup bash slurm/opd/fura/opd_job2_fura_controller.sh   > .../job2_controller.log 2>&1 &
# LoRA run:
nohup bash slurm/opd/lora/opd_lora_controller.sh        > .../lora_controller.log 2>&1 &
```

- Stack: `*_controller.sh` (login-node, loops `salloc -N4 --qos interactive --time 4:00:00
  → opd/opd_2node_inside.sh`) → bootstraps Ray (head+workers via `srun --block &`) → driver
  with `RAY_EXTERNAL=1`. The env EVERY ray-start step sources is `opd/opd_2node_rayenv.sh`
  (+ the recipe env `opd/{full,lora,fura}/opd_2node_env_*.sh`) — **Ray actors inherit the
  RAYLET env, not the driver**, so HF/WANDB/NCCL/PYTHONPATH must live there.
- Resume: `trainer.resume_mode=auto` via `latest_checkpointed_iteration.txt`; controllers run
  a background pruner (`max_actor_ckpt_to_keep=1` only rotates in-process saves).
- wandb: offline, one pinned `WANDB_RUN_ID` per run; `wandb sync` from a login node.
- ⚠️ **Never `pkill -f opd_2node_inside.sh`** — job1 and job2 share it; that kills both.
  Stop a run by `scancel <jobid>` + `pkill -f <unique_controller_name>`.
