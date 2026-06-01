# Progress

## Done
- P1 SparseGPT impl: `src/compress/unstructured/{__init__,sparsegpt,pruning}.py`
  - `SparseGPT` (Hessian accum + OBS fasterprune w/ adaptive damping), `sparsegpt_prune`
    (memory-budgeted grouping, scope=all/mlp/attn, skip_layers, thirds), helpers
    `compute_sparsity`, `compute_linear_sparsity`, `sparsity_for_param_ratio`.
  - Tests `src/compress/tests/test_sparsegpt.py` — 3 pass (incl. OBS < magnitude MSE).
- P2 Nystrom verify: WORKS. Added `src/compress/tests/test_nystrom.py` — 3 pass.
- Env: OPD env = conda `verl` (py3.12). Run tests with `PYTHONPATH=src <verl python> -m pytest`.

## Validation results (pre-full-run)
- Uncompressed Qwen3-4B: C4 PPL = 19.86 (4.022B).
- SparseGPT (c4, 64% per-linear): attn 64.00% + mlp 64.00% sparsity, 1.697B nonzero,
  C4 PPL = 34.50. lm_head/embed correctly skipped. Per-layer OBS passes clean (no Cholesky retries).
- Mixed SVD_V2(attn)+Nystrom(mlp): compresses to 1.697B (target hit).
- Driver bug fixed: compute_linear_sparsity now respects skip_layers (was diluting with lm_head).
- SparseGPT OOM fix: --sparsegpt-mem-gb (default 8) chunks Hessian groups → low peak mem on shared GPU.
- 6/6 tests pass.

## Next
- P4 RUN FULL MATRIX (baseline + 2 calib × 3 strategies, C4 PPL + MATH-500 acc).
- P3 driver `scripts/opd/math/compressed_opd/compare_compression.py`:
  - load Qwen/Qwen3-4B (non-thinking) base; for each (calib in {c4, openthought3}) ×
    (strategy in {sparsegpt~64%, svd_llm_v2 all, mixed svd_llm_v2 attn + nystrom mlp}):
    compress a fresh copy, eval C4 PPL (ppl_eval) + MATH-500 acc (HF generate + ttrl grader).
  - Also eval uncompressed baseline once.
- P4 run matrix; P5 report + memory.

## Notes / gotchas
- SVD_V2 mixed path: pass method JSON `{"attn":"svd_llm_v2","mlp":"nystrom"}` to
  compress_model_with_loader. SVD_V2 all = method "svd_llm_v2".
- compress_model_with_loader does in-place; must reload model per strategy.
- SparseGPT keeps 4B param count (dense zeros) — report nonzero-equivalent params too.
