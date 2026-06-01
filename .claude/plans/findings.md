# Findings

## Reference SparseGPT (recomp)
- Core: `recomp/src/open_r1/open_r1_trl/trl/sparsegpt/sparsegpt.py`
  - `class SparseGPT(layer)`: builds running Hessian H = (2/n) Σ xxᵀ over forward-hook inputs.
    - `add_batch(inp, out, weights=None)`: accumulates H in fp32.
    - `fasterprune(sparsity, prunen, prunem, blocksize=128, percdamp=0.01, ...)`:
      adaptive damping retry loop; Cholesky-based OBS error compensation; supports n:m.
    - `free()`.
  - All fp32, tf32 disabled.
- Driver: `recomp/.../trl/pruner/pruning.py`
  - `sparsegpt_prune(model, calib_loader, sparsity, scope='all'|mlp, memory_limit_gb, thirds_to_prune)`
    - groups Linear/Conv layers under a Hessian-memory budget, hooks them, one calib forward pass,
      then `fasterprune` each, `compute_sparsity` reports realised.
  - Also has WANDA + magnitude pruning + `_subset_by_thirds`, `find_layers`, `Catcher`.
  - `compute_sparsity(model)`: fraction of exactly-zero params.
  - depends on `..sparsegpt.quant import *` (quantizer optional; gated by hasattr(self,'quantizer')).

## OPD src/compress architecture
- Covariance-based structured methods route through:
  `compress_model.py: compress_model_with_loader(model, calib_loader, method, compression_ratio, ...)`
  - methods: svd, svd_llm, svd_llm_v2, svd_llm_v2_bp/_combined, svd_als, svd_twosteps,
    btt, btt_llm_v2, ...; plus MethodSpec JSON `{"attn": "...", "mlp": "nystrom"}` for mixed.
  - `nystrom` is ONLY valid for MLP via MethodSpec dict. `{"attn":"svd_llm_v2","mlp":"nystrom"}`
    is exactly strategy #3.
- Nystrom: `structured/nystrom.py` — `nystrom_compress_model(model, statistics, sparsity, ...)`
  uses C_sigma (input cov of down_proj, dint×dint) to select neurons. sparsity = 1 - retain.
- Calibration covariance: `calibration.py` (C4 streaming + loader path), `loaders.py`
  (`build_c4_calib_loader`, `build_traces_jsonl_calib_loader`, `build_text_calib_loader`).
- PPL eval: `ppl_eval.py: evaluate_model_ppl(model, tokenizer, datasets=('c4',))`.
- Existing eval scripts: `scripts/opd/math/compressed_opd/eval_c4_ppl.py` (BTT cache loader),
  `eval_c4_calib_ppl*.py`.

## SparseGPT vs structured target-1.7B (CRITICAL)
- SparseGPT does UNSTRUCTURED weight zeroing → param COUNT on disk unchanged (dense storage of zeros).
  Structured (SVD_V2/Nystrom) genuinely shrinks param count to ~1.7B.
- Qwen3-4B sizing (from _common.sh): embed/tied lm_head ≈ 0.389B, compressible linears ≈ 3.633B,
  total ≈ 4.02B. Structured retain ratio on linears = 0.36 → 1.70B total. VERIFIED by arithmetic.
- For SparseGPT to be "equivalent to a 1.7B model": set per-layer unstructured sparsity so
  #nonzero linear params == 0.36 × 3.633B → sparsity ≈ 0.639 (≈ 64%). This is the comparability knob.

## Datasets/models on disk
- OpenThought3-Qwen3-4B: `datasets/OpenThought3-Qwen3-4B/data/train.jsonl` (~2GB jsonl, conversational/opd).
- MATH-500 eval: `datasets/test_data/MATH-500/test.parquet`.
- Teacher/student default: `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` (HF_HOME=/data/yequan/huggingface).
