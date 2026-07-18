# Qwen3-4B → ~1.7B Compression Comparison

**Model:** Qwen/Qwen3-4B (non-thinking, base). **Target:** ~1.7B effective params (structured: retain 0.36 of compressible linears; SparseGPT: 64% unstructured sparsity, iso-nonzero).

**Metrics:** C4 validation perplexity (seqlen 2048) · MATH-500 accuracy (200 problems, greedy, ttrl_math grader).

| Strategy | Calib | nz params | C4 PPL | MATH-500 |
|---|---|---:|---:|---:|
| Uncompressed (4B) | — | 4.022B | 19.9 | 80.5% |
| SparseGPT (64% unstruct.) | c4 | 1.697B | 34.5 | 0.0% |
| SparseGPT (64% unstruct.) | openthought3 | 1.697B | 82.0 | 45.0% |
| SVD_V2 (all layers) | c4 | 1.696B | 4,443.3 | 0.0% |
| SVD_V2 (all layers) | openthought3 | 1.696B | 33,464.4 | 0.0% |
| SVD_V2 attn + Nystrom MLP | c4 | 1.697B | 914.9 | 0.0% |
| SVD_V2 attn + Nystrom MLP | openthought3 | 1.697B | 4,979.9 | 0.0% |

## Key findings

- **Calibration domain is decisive for SparseGPT.** At 64% unstructured sparsity, C4 (generic web) calibration collapses MATH to **0%** — the model falls into repetition loops and never emits a boxed answer — while OpenThought3 (math-trace) calibration preserves **45%**. Same compression, only the calibration set differs.
- **C4 PPL hides the reasoning collapse.** The C4-calibrated SparseGPT model has the *best* C4 PPL (34.5) of all compressed models yet scores 0% on MATH: perplexity on generic text rewards exactly the fluency C4 calibration preserves, but does not measure reasoning.
- **One-shot SVD low-rank at 36% retain is not viable without fine-tuning.** SVD_V2 (all) and SVD_V2+Nystrom give catastrophic PPL (hundreds–tens of thousands) and 0% MATH under both calibration sets. At this aggressive ratio, low-rank error compounds across 36 layers; SVD-LLM-style methods need a post-decomposition LoRA/SFT recovery step to be usable here.
- **SparseGPT is far more robust than one-shot SVD at the same ~1.7B budget**, because OBS weight-compensation keeps layers full-rank; the best compressed model overall is **SparseGPT + OpenThought3 calibration (45% MATH @ ~1.7B nz, vs 80.5% uncompressed)**.