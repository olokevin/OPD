# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the paper *"Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"* ([arXiv:2604.13016](https://arxiv.org/pdf/2604.13016)). The repo studies token-level on-policy distillation (OPD): a teacher model provides dense per-token reward signals (typically log-probs over a Top-K set at student-visited states), and a student is trained against them using a PPO-style trainer.

The codebase is **two vendored frameworks plus thin glue**:

- `verl/` — fork of [verl](https://github.com/verl-project/verl) v0.7.0 used for OPD + RL training. **All OPD-specific algorithm code lives here**, not in a separate module.
- `LlamaFactory/` — fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) v0.9.5 used for SFT (cold-start checkpoints, off-policy distillation baselines).
- `src/compress/` — **git submodule** ([olokevin/compress](https://github.com/olokevin/compress)) holding the model-compression code (BTT/BlockTT, SVD, structured layers, ZO grad estimators) used by the `compressed_opd` teacher-compression experiments.
- Top-level `*.sh`, `scripts/`, `datasets/` — entry points and data plumbing that drive the frameworks above.

The two training frameworks use **different conda envs** (`verl` py3.12 vs `sft` py3.11) and should not share dependencies.

### `src/compress` is a git submodule — commit it separately

`src/compress` is a nested git repo wired in as a submodule (`.gitmodules` → `git@github.com:olokevin/compress.git`). The rest of `src/` is gitignored; only the `src/compress` gitlink is tracked. This changes the commit workflow:

- **Edits inside `src/compress` are NOT committed by the main repo.** `git add`/`git commit` from the OPD root only records the submodule's *commit pointer*, never its file changes. You must commit and push from inside the submodule first:
  ```bash
  cd src/compress
  git add -A && git commit -m "..."   # commit in the submodule's own repo
  git push origin master              # push BEFORE pinning, or clones fail to fetch the pinned SHA
  cd ../..
  git add src/compress                # stage the new pointer in OPD
  git commit -m "compress: bump submodule"
  ```
- **Always push the submodule's commits before committing the bumped pointer in OPD** — the main repo pins a remote SHA, so an unpushed pointer breaks `git clone --recurse-submodules`.
- **Clone with** `git clone --recurse-submodules ...`, or run `git submodule update --init --recursive` in an existing checkout.
- A dirty submodule shows up in the OPD root as `modified: src/compress (modified content/new commits)` — that is the signal you have un-pinned submodule work to handle as above.

## Environment setup

OPD / RL environment (used by `on_policy_distillation.sh`, `grpo.sh`, `scripts/infer/*`, `scripts/val/eval/*`):

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

SFT environment (used by `llamafactory-cli`):

```bash
conda create -n sft python==3.11
conda activate sft
cd LlamaFactory/
pip install -e .
pip install -r requirements/metrics.txt
```

All paper experiments assume **8× A800 80GB GPUs** on a single node. Both top-level scripts (`on_policy_distillation.sh`, `grpo.sh`) start with SBATCH headers but auto-fall-back to local execution (with tee'd log files under `logs/`) when `$SLURM_JOB_ID` is unset.

## Common commands

### Training

```bash
# On-policy distillation (token-level teacher reward). Defaults: ADV_ESTIMATOR=token_reward_direct.
bash on_policy_distillation.sh

# Zero-RL via GRPO. Defaults: ADV_ESTIMATOR=grpo, LOG_PROB_TOP_K=0.
bash grpo.sh

# SFT (teacher-rolled-out responses → student fine-tuning)
llamafactory-cli train LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml
```

Both top-level scripts source most knobs from env vars (e.g. `LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `REWARD_WEIGHT_MODE`, `ACTOR_MODEL_PATH`, `REWARD_MODEL_PATH`, `TRAIN_DATASET`, `MINI_BATCH_SIZE`, `TEMPERATURE`, `MAX_RESP_LENGTH`) — prefer overriding via `KEY=value bash on_policy_distillation.sh` to keep the script diff-free across experiments. They internally invoke `python3 -m verl.trainer.main_ppo ...` with Hydra-style overrides.

Checkpoint paths and SwanLab experiment names are auto-derived from those env vars; expect long, hyperparameter-tagged directory names under `checkpoint/`.

**Non-thinking models** (e.g. `Qwen3-1.7B (Non-thinking)`): add `+data.apply_chat_template_kwargs.enable_thinking=False` to the launch command.

### Teacher rollout for SFT data

```bash
python scripts/infer/vllm_rollout.py \
  --input-parquet datasets/OpenThoughts3-1.2M-math.parquet \
  --model-path model/Qwen3-4B \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3
```

`scripts/infer/dedup_deepmath.py` deduplicates DeepMath against DAPO-Math-17K (Section 5.2 of the paper).

### Evaluation

Reuses the [JustRL](https://github.com/thunlp/JustRL) pipeline:

```bash
cd scripts/val/eval
python gen_vllm.py                # generation — set MODEL_NAMES + available_workers inside the script first
python grade.py                   # rule-based grading; --enable_model_verifier adds LLM verifier
```

Generations land as JSONL alongside `grading_results.json`. Math benchmarks under `scripts/val/data/`: AIME24/25, AMC23, BRUMO25, CMIMC25, HMMT25, MATH-500, Minerva, Olympiad-Bench.

### Tests

`verl/tests/` is the upstream verl test suite (pytest). No project-level test runner is wired up — there is no top-level `Makefile`/`pyproject` for the repo root. When debugging trainer changes, the practical loop is to run the relevant SBATCH script on a small dataset (the commented-out `dapo-math-17k-1percent.parquet` lines in the `.sh` files exist for this).

## OPD algorithm: where the custom code lives

The OPD-specific logic is a **set of patches to verl**, not a separate package. When investigating or modifying OPD behavior, start here:

- `verl/verl/trainer/ppo/core_algos.py` — registers the custom advantage estimators via `@register_adv_est(...)`:
  - `token_reward_direct` — pure token-level teacher-log-prob reward (the "OPD" estimator).
  - `token_reward_direct_plus_grpo` — combines token-level dense reward with GRPO outcome reward; weighted by `GRPO_OUTCOME_WEIGHT`.
  - `token_grpo` — token-level variant of GRPO.
  - `grpo` — stock GRPO, used by `grpo.sh`.
  `ADV_ESTIMATOR` env var → `algorithm.adv_estimator` Hydra key picks one of these.

- `verl/verl/workers/config/rollout.py`, `verl/verl/workers/fsdp_workers.py`, `verl/verl/workers/actor/dp_actor.py` — plumbing for the OPD-specific rollout/actor knobs `log_prob_top_k`, `top_k_strategy`, `reward_weight_mode`, `teacher_temperature`. These are passed as `+actor_rollout_ref.rollout.<key>=...` overrides in `on_policy_distillation.sh`.

- `verl/verl/utils/reward_score/ttrl_math/` — custom math reward function (`reward_func` from `__init__.py`), pulled in via `custom_reward_function.path` / `.name`. Uses `math_normalize.py` + `grader.py` for answer extraction and equivalence checking.

Key OPD knobs (full table in `README.md`):

| Env var | Meaning |
|---|---|
| `LOG_PROB_TOP_K` | Top-K size for teacher-reward token set; `0` falls back to sampled-token OPD |
| `TOP_K_STRATEGY` | `only_stu` / `only_tch` / `intersection` / `union` / `union-intersection` — how to pick the K tokens that get the teacher's log-prob |
| `REWARD_WEIGHT_MODE` | `student_p` / `teacher_p` / `none` — how token rewards are reweighted |
| `TEACHER_TEMPERATURE` | Temperature applied to teacher logits before computing log-probs |

The teacher model is wired through verl's `reward_model.*` config — i.e. the "reward model" slot is repurposed to hold the **teacher LLM**, not a scalar RM. `reward_model.model.path=$REWARD_MODEL_PATH` in the launch scripts.

## Data layout

- `datasets/*.parquet` — training datasets in verl's expected parquet format (DAPO-Math-17k, DeepMath-103K-deduped, OpenThoughts3 OPD slice).
- `datasets/test_data/<bench>/test.parquet` — held-out math benchmarks consumed via `TEST_DATASET` env var.
- `scripts/val/data/<bench>/` — separate eval-pipeline data dir used by `gen_vllm.py` / `grade.py`.

When adding a new train set, append its path to `TRAIN_DATASET` and a short identifier to `TRAIN_DATASET_NAME` (the latter only affects the auto-generated experiment/checkpoint name).

## Released checkpoints (from README)

- `lllyx/Qwen3-1.7B-SFT` — SFT student from `Qwen3-1.7B-Base`.
- `lllyx/Qwen3-4B-Base-GRPO` — zero-RL student from `Qwen3-4B-Base` (produced by `grpo.sh`).

The HF collection lives at `huggingface.co/collections/lllyx/rethinking-opd`.
