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
- `scripts/compress_sft/run_compress_sft.sh` (launcher, sft-env training +
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
bash scripts/compress_sft/run_compress_sft.sh train all
# Final eval (MATH-500@4096 + MMLU-Pro) on the final-merged ckpts, verl env:
bash scripts/compress_sft/run_compress_sft.sh eval all
# Mid-training MATH growth curve (post-hoc, over checkpoint-*-merged), verl env:
bash scripts/compress_sft/sweep_sft_ckpts.sh \
  /data/yequan/compress_sft/sft/qwen3_4b_base/forward_r0.7 1 100
```
Mid-training **val-loss** growth is live in wandb via `val_size`/`eval_steps` (tier-1).

## NERSC port + full run (2026-06-08)

Ported the recipe to Perlmutter and launched the two **retain-0.6** jobs (the paper
operating point this thread now uses), forward (`svd_v2`) + combined (`svd_v2_combined`),
**2 nodes/8 GPUs each, run concurrently** (~~interactive QOS caps a user at 4 nodes total,
so the requested "4 nodes each" is impossible concurrently — went 2+2~~ **CORRECTED
2026-06-08: that cap claim is WRONG — `node=4` is per-JOB (`MaxTRESPerJob`), `MaxJobsPU=2`,
no per-user node cap, so 2 jobs × 4 nodes = 8 nodes IS allowed; verified live. See the
retain-0.7 subsection below**). Global batch 64 =
per_device 4 × accum 2 × world 8; lr 1e-5; cutoff 10240; `gc=false` (probed: bs4 peaks
~66 GB/80 GB). Train+calib data = the full **305k-row `lllyx/OpenThought3-Qwen3-4B`** HF
dataset (the in-repo `OpenThought3-Qwen3-4B-Calibration/train.jsonl` is only a 512-row
calib stub). S is sqrt-split onto both cores and both train — inherent to `svd_nystrom`
(`svd_llm_v2.py`); `s_merged_to` is *not consulted* on that path. The last decoder layer
is left fully dense (`skip_last_layers=1`) → the merged MLP is heterogeneous
(layers 0..34 at 5837, layer 35 at 9728), `config.intermediate_size` stays 9728.

**Files:** configs `examples/compress_train/qwen3_4b_nersc_{fwd,combined}_r0.6_sft.yaml`;
launcher `slurm/compress_sft/` (`compress_sft_{fwd,combined}_controller.sh` →
`_compress_sft_controller_core.sh` → `compress_sft_inside.sh` (srun → torchrun, 1 task/node,
`NODE_RANK=$SLURM_NODEID`, `MASTER_ADDR`=head hsn IP) → `compress_sft_env.sh`); validation
`smoke_test.sh` + `probe_mem.sh`; eval `compress_sft_eval.sh` + `scripts/compress_sft/hetero_load.py`.

**Three fixes the new stack required** (torch 2.12+cu130, transformers 5.2.0, sft env on
`/pscratch/.../envs/sft`):
1. **flash-attn has no cu130 wheel** → YAML uses `flash_attn: sdpa`.
2. **Merged-save hang** (`callbacks.py::_materialize_and_save`): the dense `-merged` save
   did `V_r@U_r` in **bf16 on CPU**, which hangs for ~140 SVD linears on torch≥2.12
   (py-spy showed `materialize_dense_weight`). Patched to materialize in **fp32 on CPU**
   then cast back (~7 s). The factored `checkpoint-N` (resume path) was never affected.
3. **Heterogeneous-MLP reload** (`hetero_load.py`): the merged ckpt is not
   `from_pretrained`-loadable (scalar `intermediate_size` ≠ per-layer widths);
   `load_compressed_merged` rebuilds from config, resizes per-layer MLP `nn.Linear`s, loads.
   Eval runs in the **verl env** (grader ⇒ verl ⇒ ray); pass `--tokenizer Qwen/Qwen3-4B`
   (the 5.2.0-saved tokenizer config breaks verl's older transformers).

**Validated end-to-end** before launch: compress init (2.609 B trainable) + train (loss
drops) + factored save + `resume_from_checkpoint` + **8-way multi-node DDP** rendezvous +
MATH-500/MMLU-Pro eval pipeline. Mid-training resume across the 4 h interactive cap is
handled by the controllers (save_steps=100, save_total_limit=1, auto re-salloc). See the
auto-memory `nersc-compress-sft` for the operational runbook.

## NERSC retain-0.7 run + unified train/eval wandb (2026-06-08)

Relaunched both objectives at **retain 0.7**, **4 nodes/16 GPU each, concurrent** (8 nodes
total — proven allowed: `sacctmgr show qos interactive` → `MaxTRESPerJob=node=4`,
`MaxJobsPU=2`, **no** per-user node cap; two `salloc -N4 --qos interactive` both ran at once).
The serialize-compress fix scales to 16-way (16 serial SVD compresses, ~20 min). **Global
batch kept at 64** = per_device 2 × grad_accum **2** × world 16 (accum lowered 8→2 vs the
1-node run because world went 4→16). New configs `qwen3_4b_nersc_{fwd,combined}_r0.7_sft.yaml`;
output `compress_sft/sft/qwen3_4b/{forward,combined}_r0.7`; the r0.6 dirs/runs are untouched.

**Eval = MATH-500 + AIME24 + MMLU-Pro** (AIME24 added — same parquet schema as MATH-500, so
`eval_opd_ckpt.py` now drives a generic `eval_bench`), at two cadences:
1. **Right after compression** — `CompressSaveCallback.on_train_begin` writes
   `checkpoint-0-merged` on a fresh start (`global_step==0`); `_rotate_merged` preserves step-0
   so the post-compression baseline survives `save_total_limit=1`.
2. **Each saved checkpoint** — `compress_sft_eval_daemon.sh` evals the latest
   `checkpoint-N-merged`, throttled to every `EVAL_MIN_GAP=500` steps, via gpu_shared 1-GPU
   jobs (off the 2-job interactive cap).

**Train and eval log to the SAME wandb run** (`qwen3_4b_nersc_compress_{obj}_r0.7_sft` in
`nersc_compress_sft_qwen4b`): the single login-node ONLINE `log_train_to_wandb.py --ratio 0.7`
logs `train/*` from `trainer_log.jsonl` **and** `eval/{math500,aime24,mmlu_pro}_acc` from the
eval JSONs (`compress_sft/metrics/{obj}_r0.7/step<N>/`, gated by an `EVAL_DONE` marker). The
eval job no longer touches wandb; the separate `_eval` run and the wandb-sync daemon are
retired. See the auto-memory `nersc-compress-sft` for the updated runbook.
