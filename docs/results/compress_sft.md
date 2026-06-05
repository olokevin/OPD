# compress_sft: reasoning-aware compress-then-train (SVD-attn + Nystrom-MLP → SFT)

> Compress a well-trained model with reasoning-aware structured compression
> (SVD-LLM-V2 on `self_attn` + Nystrom/MoDeGPT on MLP), retain ratio 0.7, then SFT
> on OpenThoughts3 to recover reasoning. Compression runs **in-process at
> LlamaFactory model-init** so the compressed factors stay **trainable**. Eval =
> MATH-500 (max 4096 new tokens) + MMLU-Pro; wandb project `compress_sft_{model}`.
> Built on branch `compress_sft` (worktree `OPD-compress-sft`), based on
> `np-v2-cudagraph-rails`. Companion design: [reasoning_aware_compress_calib](../wiki/reasoning_aware_compress_calib.md).

## What was built (2026-06-04/05)

A new LlamaFactory `finetuning_type: svd_nystrom` that, at model-init, runs the
SVD-attn / Nystrom-MLP split using the productionized **sequence-reweighted
full-sequence** OpenThought3 calibration (wiki §5 default), then leaves the SVD
factors (`SVDCompressedLinear.U_r/V_r`) and the Nystrom-shrunk MLP `nn.Linear`s
**trainable** for SFT. On save, `CompressSaveCallback` materializes SVD→dense and
passes the smaller dense MLPs through, writing a plain (smaller) HF checkpoint.

**Files** (worktree `OPD-compress-sft`):
- `LlamaFactory/src/llamafactory/model/compress_setup.py` — `_init_compress_svd_nystrom`
  (the split + trainability + `config.intermediate_size` update + fast-fail
  `_assert_no_fused_experts`).
- `LlamaFactory/src/llamafactory/hparams/finetuning_args.py` — `svd_nystrom` in the
  `finetuning_type` Literal + `skip_last_layers` knob + calib validation.
- `LlamaFactory/src/llamafactory/model/adapter.py` — dispatch `svd_nystrom` →
  `init_compress_model`.
- `LlamaFactory/src/llamafactory/hparams/parser.py` — ZeRO-3 ban extended to
  `svd_nystrom` (the save callback reads a full per-rank state_dict; ZeRO-3 shards it).
- `LlamaFactory/examples/compress_train/{qwen3_4b,olmoe}_compressed_{fwd,combined}_sft.yaml`
  + `_smoke_qwen3_4b_fwd.yaml`.
- `scripts/opd/math/compressed_opd/run_compress_sft.sh` (launcher, sft-env training +
  verl-env eval), `eval_mmlu_pro.py` (MMLU-Pro via ttrl grader), `sweep_sft_ckpts.sh`
  (post-hoc MATH growth curve), `_smoke_svd_nystrom.py` (standalone compress-init smoke).

### Key design decisions
- **In-process, factors trainable** (user choice): compression at model-init, not
  offline-dense-then-full-FT. The whole compress+train+save is ONE `sft`-env process.
- **Training always in the `sft` env**; the MATH/MMLU graders (`verl/.../ttrl_math`,
  needs ray) run as a *separate* post-hoc eval step in the `verl` env.
- **Last-layer skip is ATTN-ONLY.** SVD-attn materializes back to full-size dense
  (shape-preserving), so skipping the last layer's attn is a pure quality knob. The
  MLP must shrink **uniformly** across all layers so the saved checkpoint keeps one
  global `intermediate_size` (we update `config.intermediate_size` to the Nystrom `k`).
- **MMLU = MMLU-Pro via the generation+ttrl grader** — LlamaFactory's built-in MMLU
  evaluator CLI is disabled in this fork (`launcher.py:147` `NotImplementedError`).
  MMLU-Pro prompts already ask for `\boxed{letter}`, so the MATH-500 path applies as-is.

## Validation (compress-init smoke, Qwen3-4B-Base, retain 0.7)

`_smoke_svd_nystrom.py` exercises `_init_compress_svd_nystrom` + the save roundtrip
directly (bypassing the SFT loop). **Both objectives PASS:**

| objective | triplets | SVD attn (trainable) | MLP width | config update | reload | save |
|---|---|---|---|---|---|---|
| forward  | 36 | 140 (✓) | 9728→6810 (=⌈0.7·9728⌉) | ✓ | 0 missing / 0 unexpected | ✓ model.safetensors |
| combined | 36 | 140 (✓) | 9728→6810 | ✓ | 0 missing / 0 unexpected | ✓ |

(0 triplets missing cov; Nystrom params count 1.88B/2.69B = 0.70 retain on MLP.)
Pre-SFT MATH-500 sanity baseline (wiki §5): forward @0.7 ≈ 66% / 100. Full runs +
final MATH-500@4096 / MMLU-Pro are pending.

## OLMoE-1B-7B — BLOCKED follow-up (fused experts)

OLMoE on **transformers 5.2.0** stores its 64 experts as **fused 3D tensors**
(`OlmoeExperts.gate_up_proj` `(64, 2·inter, hidden)` + `down_proj`
`(64, hidden, inter)`, applied via a functional per-expert loop), **not** per-expert
`nn.Linear` `gate_proj/up_proj/down_proj`. The `src/compress` Nystrom path
(`find_mlp_triplets` + `nystrom_compress_mlp`) only handles `nn.Linear` triplets, so it
finds **0 triplets** on OLMoE. `_init_compress_svd_nystrom` now **fails fast** with an
actionable `NotImplementedError` (`_assert_no_fused_experts`) rather than a deep trace.

**Follow-up options** (not yet implemented; user chose to ship Qwen3 first):
1. **Unfuse at load** — replace `OlmoeExperts` with a `ModuleList` of per-expert
   gate/up/down `nn.Linear` (+ matching forward) at model-init, so `find_mlp_triplets`
   / Nystrom / SFT all work unchanged. Adds a 64-expert python loop in the forward.
2. **Fused-tensor Nystrom** — collect per-expert covariance through the functional
   forward and shrink the 3D tensors in place. Keeps OLMoE's fast forward; most code.

Per-expert under-sampling (top-8/64 routing → rare experts undersampled) remains a
real concern for either path; the configs use `calib_num_seqs: 512` for OLMoE and the
compress step guards on `n_missing == 0` (every triplet must be routed ≥ once).

## How to run

```bash
# Qwen3-4B both objectives, GPUs 1 & 2, sft env (compress+train+save in one process):
bash scripts/opd/math/compressed_opd/run_compress_sft.sh train all
# Final eval (MATH-500@4096 + MMLU-Pro) on the final-merged ckpts, verl env:
bash scripts/opd/math/compressed_opd/run_compress_sft.sh eval all
# Mid-training MATH growth curve (post-hoc, over checkpoint-*-merged), verl env:
bash scripts/opd/math/compressed_opd/sweep_sft_ckpts.sh \
  /data/yequan/compress_sft/sft/qwen3_4b_base/forward_r0.7 1 100
```
Mid-training **val-loss** growth is live in wandb via `val_size`/`eval_steps` (tier-1).
