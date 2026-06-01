# Task: SparseGPT impl + nystrom verify + 4B→1.7B compression comparison

## Goal
1. Implement SparseGPT unstructured pruning under `src/compress/unstructured/`, following the
   reference at `recomp/src/open_r1/open_r1_trl/trl/{sparsegpt,pruner}/`.
2. Verify `src/compress/structured/nystrom.py` works.
3. For Qwen3-4B Non-Thinking model, with two calibration datasets:
   - (1) C4 calibration
   - (2) OpenThought3-Qwen3-4B (sampled) calibration
   Run a comparison of compression strategies targeting ~1.7B:
   - SparseGPT (unstructured)
   - SVD_V2 for all layers
   - SVD_V2 for attention layers + Nystrom for MLP layers
   Metrics: (1) C4 val PPL, (2) MATH val accuracy.
   Compare: uncompressed + 3 compression strategies.

## Phases
- [ ] P0: Brainstorm/confirm design decisions (SparseGPT comparability, model target, eval harness)
- [ ] P1: Implement `src/compress/unstructured/{__init__,sparsegpt,pruning}.py` (TDD)
- [ ] P2: Verify nystrom.py (existing tests + a fresh smoke test on real small layer)
- [ ] P3: Build a single comparison driver that: loads Qwen3-4B-NT, applies each strategy
        (per calib dataset), evaluates C4 PPL + MATH accuracy, writes a results table.
- [ ] P4: Run the matrix (2 calib × 3 strategies + 1 uncompressed baseline), collect results.
- [ ] P5: Report table; save memory notes.

## Key decisions (RESOLVED with user)
- SparseGPT comparability: **~64% unstructured sparsity** (iso-nonzero with 1.7B structured target;
  retain 0.36 of the 3.633B compressible linears). Run per-output-row (per-layer) sparsity.
- Base model: **Qwen/Qwen3-4B base, non-thinking** (NOT the RL-Math step500 teacher).
- MATH eval: **vLLM-free HF model.generate on MATH-500 + ttrl_math grader** (works on in-memory
  compressed models, incl. dense-zeroed SparseGPT; no checkpoint export needed).
