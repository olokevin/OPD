# Repository Guidelines

## Project Structure & Module Organization

This repository contains research code for on-policy distillation (OPD) and related LLM compression experiments. The main components are:

- `verl/`: vendored verl v0.7.0 framework; OPD and GRPO trainer changes live here.
- `LlamaFactory/`: vendored LLaMA-Factory v0.9.5 used for supervised fine-tuning.
- `src/compress/`: a separate Git submodule containing BlockTT, SVD, pruning, and compression code. Commit changes there before updating its pointer in this repository.
- `scripts/`: rollout, evaluation, experiment, and Slurm launch helpers.
- `datasets/`: Parquet training data and held-out benchmark data.
- `docs/`: maintained project wiki, design notes, and experiment results.

Keep framework-specific changes within their framework and put reusable compression code in `src/compress/`.

## Build, Test, and Development Commands

Use separate environments: Python 3.12 for OPD/RL (`verl`) and Python 3.11 for SFT (`LlamaFactory`).

```bash
bash on_policy_distillation.sh                  # launch OPD training
bash grpo.sh                                    # launch GRPO training
llamafactory-cli train LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml
cd scripts/val/eval && python gen_vllm.py       # generate evaluation outputs
cd scripts/val/eval && python grade.py          # grade generated JSONL files
pytest verl/tests/<relevant_test>.py            # test a verl change
```

Prefer environment-variable overrides, such as `LOG_PROB_TOP_K=16 bash on_policy_distillation.sh`, instead of editing shared launch defaults for one experiment.

## Coding Style & Naming Conventions

Follow the style already used in the target component. Python uses four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and concise module names. Shell scripts should use descriptive uppercase environment variables and `.sh` filenames. Keep changes narrow; avoid unrelated reformatting of vendored code.

## Testing Guidelines

There is no root-level test suite. Add or update focused pytest coverage under `verl/tests/` for trainer behavior and under `src/compress/tests/` for compression code. Name tests `test_<behavior>.py` and test functions `test_<expected_result>`. For GPU/distributed changes, validate with the smallest suitable dataset and record material outcomes in `docs/results/`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, scope-oriented subjects, for example `es-decode: fix OOB loads` or `docs: record cleanup`. Keep each commit focused. PRs should state the experiment or behavior changed, affected scripts/configuration, validation performed, and links to relevant documentation or issues. Include logs, metrics, or screenshots only when they substantiate a training or evaluation result.
