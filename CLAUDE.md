# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not invoke superpowers related skills unless explicitly instructed to do so.

## Project

Research code for the paper *"Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"* ([arXiv:2604.13016](https://arxiv.org/pdf/2604.13016)). The repo studies token-level on-policy distillation (OPD): a teacher model provides dense per-token reward signals (typically log-probs over a Top-K set at student-visited states), and a student is trained against them using a PPO-style trainer.

The codebase is **two vendored frameworks plus thin glue**:

- `verl/` — fork of [verl](https://github.com/verl-project/verl) v0.7.0 used for OPD + RL training. **All OPD-specific algorithm code lives here**, not in a separate module.
- `LlamaFactory/` — fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) v0.9.5 used for SFT (cold-start checkpoints, off-policy distillation baselines).
- `src/compress/` — **git submodule** ([olokevin/compress](https://github.com/olokevin/compress)) holding the model-compression code (BTT/BlockTT, SVD, structured layers, ZO grad estimators) used by the `compressed_opd` teacher-compression experiments.
- Top-level `*.sh`, `scripts/`, `datasets/` — entry points and data plumbing that drive the frameworks above.

The two training frameworks use **different conda envs** (`verl` py3.12 vs `sft` py3.11) and should not share dependencies.

## Knowledge system (`docs/`)

`docs/` is an **LLM-maintained wiki** following the pattern in `docs/llm-wiki.md`: a persistent, interlinked markdown knowledge base that you (Claude) own and keep current, sitting between the raw repo/papers and any question asked about this project. Read `docs/llm-wiki.md` once for the philosophy; this section is the operational schema.

**Layout:**

| Dir                           | What lives there                                                                                                                                                                                            | Who owns it               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `docs/index.md`             | **Content catalog** — every wiki/results/aris/plans page with a one-line summary. **Read this first** when answering a question or deciding where new knowledge goes.                          | LLM (update every ingest) |
| `docs/log.md`               | **Chronological log** — append-only `## [YYYY-MM-DD] <op> \| <title>` entries for every ingest/query/lint. `grep '^## \[' docs/log.md \| tail -5` shows recent activity.                           | LLM (append every op)     |
| `docs/wiki/`                | **Design docs** — how a subsystem works (architecture, invariants, knobs, file map). One page per subsystem (`compressed_opd`, `ZO`, `zo_np_trainer`). Stable; revise when the design changes. | LLM                       |
| `docs/results/`             | **Experiments & mid-conclusions** — what we ran and what we learned, per thread (`compressed_opd`, `zo_opd`, `fura_opd`). Append-mostly; each session adds a dated block.                      | LLM                       |
| `docs/aris/{project_name}/` | **ARIS research threads** — agent-pipeline outputs (see subsection below).                                                                                                                           | ARIS skills + LLM         |
| `docs/plans/`               | **Implementation specs** — step-by-step build plans for trainer changes (companion to wiki design docs).                                                                                             | LLM                       |
| `docs/papers/`              | **Reference PDFs** — read-only source of truth. Cite by filename; **never edit**.                                                                                                              | human (curates)           |

**Workflow (the three operations from `docs/llm-wiki.md`):**

- **Ingest** — when new findings, a finished experiment, or a design change land, file them into the right `wiki/` or `results/` page (create the page if missing), add/update its row in `docs/index.md`, link related pages with relative markdown links, and append one `docs/log.md` line. A single ingest may touch several pages — keep cross-references consistent.
- **Query** — to answer a question about this project, read `docs/index.md` first to locate relevant pages, drill into them, then synthesize with citations to `docs/...` paths. **File substantial answers back** as a new/updated page (a comparison, an analysis, a discovered connection) so explorations compound instead of vanishing into chat history.
- **Lint** — when asked to health-check, scan for contradictions between pages, stale claims newer results superseded, orphan pages with no inbound links, concepts that deserve their own page, and missing cross-references; report findings and propose fixes.

**Conventions:** start each page with an H1 and a one-line/blockquote summary; prefer relative links between docs (`results/zo_opd.md`); keep `docs/index.md` and `docs/log.md` in sync with every change; convert relative dates to absolute (today is in the session context). The wiki is just a git repo of markdown — commit doc changes alongside the work they describe.

This knowledge base and the auto-memory at `~/.claude/.../memory/` are complementary: **memory** holds short cross-session facts/preferences/gotchas; the **wiki** holds the durable, interlinked project knowledge. When a memory and a wiki page overlap, the wiki page is the fuller source — point the memory at it.

### ARIS / agent-generated docs go under `docs/aris/{project_name}/`

All docs produced by ARIS pipelines (idea-discovery, research-refine, experiment-plan, reviews, etc.) — `IDEA_REPORT.md`, `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, `DIAGNOSIS.md`, `MANIFEST.md`, and friends — live in **`docs/aris/{project_name}/`**, one subfolder per research thread (e.g. `docs/aris/reason_aware_compress/`). Do **not** scatter them at the repo root or in skill-default dirs like `idea-stage/` / `refine-logs/`; point those skills' output dirs at `docs/aris/{project_name}/` (or move their output there when done). `{project_name}` is a short kebab/snake-case slug for the thread. When an ARIS thread produces a durable result, also catalog its key pages in `docs/index.md` and log the ingest.

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

| Env var                 | Meaning                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `LOG_PROB_TOP_K`      | Top-K size for teacher-reward token set;`0` falls back to sampled-token OPD                                                                   |
| `TOP_K_STRATEGY`      | `only_stu` / `only_tch` / `intersection` / `union` / `union-intersection` — how to pick the K tokens that get the teacher's log-prob |
| `REWARD_WEIGHT_MODE`  | `student_p` / `teacher_p` / `none` — how token rewards are reweighted                                                                    |
| `TEACHER_TEMPERATURE` | Temperature applied to teacher logits before computing log-probs                                                                                |

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

# Claude Code Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes
