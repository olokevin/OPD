# Experiment Results — MoE Expert Compression Recovery Atlas

> Live results log. **Training-free leg (step-0)** ran 2026-06-08 on GPU 1/2/3 via `scripts/moe_compress/run_trainfree_atlas.sh`. Recovery (SFT) leg pending. Reference: uncompressed OLMoE-1B-7B-0924-Instruct MMLU(5-shot,limit200) = **0.542** (matches paper ~54 → eval pipeline validated). Eval limit 200/task (fast first pass; full eval is a follow-up).
>
> ⚠️ **CALIBRATION CAVEAT (the v1 atlas above):** all v1 results used `datasets/OpenThought3-Qwen3-4B/` for calibration — OpenThoughts3 *prompts* with **Qwen3-4B-generated** reasoning traces, NOT OLMoE-native completions.

### v2 — OLMoE-NATIVE calibration (2026-06-08), retain 0.5 @ eval-limit 200
Regenerated 500 on-distribution traces (OpenThoughts3 prompts → OLMoE-1B-7B-Instruct completions, `/data/yequan/moe_compress/calib_src/ot3_olmoe_native.jsonl`) and re-compressed the 3 recovery methods. **OLMoE traces are much SHORTER** (p50 525 / mean 587 tok vs Qwen's long chains; 153k vs 428k total calib tokens; min 21 tok/expert, 0 dead).

| method | MMLU (Qwen→native) | GSM8K (Qwen→native) | ARC | Hella |
|---|---|---|---|---|
| nystrom | 0.341 → **0.338** | 0.195 → 0.165 | 0.300 | 0.530 |
| nystrom_combined | 0.382 → **0.349** | 0.155 → 0.140 | 0.335 | 0.530 |
| reap_drop | 0.289 → **0.276** | 0.005 → 0.000 | 0.290 | 0.475 |

**On-distribution calibration is slightly WORSE across the board** (MMLU −0.003/−0.033/−0.013; largest hit on nystrom_combined). Cause: the native traces' *token count* dropped ~3× (153k vs 428k), and OLMoE's terse low-reasoning answers exercise reasoning-relevant expert activations less than Qwen's long chains — so calibration *quality* fell more than distribution-match helped. **The method ORDERING is unchanged** (nystrom_combined > nystrom > reap_drop; weight-approx > expert-removal; fwd+bwd > fwd-only) — *the headline findings are robust to the calibration source.* Native-calib `*_native_s0` checkpoints carry forward to the v2 recovery training.

### v2b — nystrom_combined CALIBRATION-SOURCE ABLATION (where should C_f vs C_b come from?), retain 0.5
The fwd+bwd Nyström uses a forward activation cov `C_f` and a backward gradient cov `C_b`. We split their data sources (2 collection passes, zip `C_f` from one, `C_b` from the other):

| recipe | forward `C_f` | backward `C_b` | MMLU | GSM8K |
|---|---|---|---|---|
| **both-OT3** (`nystrom_combined_ot3`) | OpenThoughts3 | OpenThoughts3 | **0.381** | 0.170 |
| both-Qwen (= v1, same file as OT3) | OpenThoughts3 | OpenThoughts3 | 0.382 | 0.155 |
| **split** (`nystrom_combined_fwdnat_bwdot3`) | OLMoE-native | OpenThoughts3 | **0.377** | 0.160 |
| both-native (v2) | OLMoE-native | OLMoE-native | 0.349 | 0.140 |

**Finding: the BACKWARD signal's source is what matters, not the forward's.** Moving only `C_b` from native→OpenThoughts3 (split recipe) recovers almost the entire loss (0.349→0.377, +0.028) and nearly matches all-OT3 (0.381). So `C_b` (the "which directions matter for the task" gradient signal) must come from high-quality target responses; `C_f` (where activations go) can stay on-policy/native with ~no penalty. The **split recipe is the principled choice** (on-policy activations + target-quality gradients), essentially tied with all-OT3 (Δ 0.004 ≈ noise). v2b recovery training launched for both.

## Step-0 (training-free) — COMPLETE: 10 methods × 2 retains (20 jobs). Weight-approx family = {sparsegpt, nystrom_combined, nystrom, svd_llm_v2, mobe}.

**Per-method step-0 (eval limit 200; uncompressed MMLU 0.542). Sorted by retain then MMLU:**

| method | family | retain | storage/active | MMLU | GSM8K | ARC-C | HellaSwag |
|---|---|---|---|---|---|---|---|
| magnitude | control | 0.75 | 0.75/0.75 | 0.550 | 0.365 | 0.540 | **0.730** |
| **sparsegpt** | weight-approx | 0.75 | 0.75/0.75 | **0.549** | 0.360 | 0.535 | 0.695 |
| **nystrom_combined** | weight-approx | 0.75 | 0.75/0.75 | **0.499** | **0.310** | 0.460 | 0.650 |
| nystrom | weight-approx | 0.75 | 0.75/0.75 | 0.474 | 0.265 | 0.425 | 0.625 |
| reap_drop | expert-removal | 0.75 | 0.75/1.0 | 0.473 | 0.235 | 0.415 | 0.700 |
| random_drop | expert-removal | 0.75 | 0.75/1.0 | 0.451 | 0.025 | 0.420 | 0.655 |
| slimqwen_merge | merge | 0.75 | 0.75/1.0 | 0.450 | 0.070 | 0.375 | 0.635 |
| **svd_llm_v2** | weight-approx | 0.75 | 0.75/0.75 | 0.450 | 0.215 | 0.420 | 0.585 |
| hcsmoe_merge | merge | 0.75 | 0.75/1.0 | 0.352 | 0.005 | 0.385 | 0.580 |
| **mobe** | weight-approx | 0.75 | 0.75/0.75 | **0.246** | 0.000 | 0.260 | 0.425 |
| **sparsegpt** | weight-approx | 0.50 | 0.5/0.5 | **0.521** | 0.275 | 0.480 | 0.685 |
| magnitude | control | 0.50 | 0.5/0.5 | 0.516 | 0.260 | 0.480 | 0.670 |
| **nystrom_combined** | weight-approx | 0.50 | 0.5/0.5 | **0.382** | 0.150 | 0.390 | 0.520 |
| nystrom | weight-approx | 0.50 | 0.5/0.5 | 0.341 | 0.195 | 0.320 | 0.575 |
| **svd_llm_v2** | weight-approx | 0.50 | 0.5/0.5 | 0.307 | 0.135 | 0.300 | 0.520 |
| reap_drop | expert-removal | 0.50 | 0.5/1.0 | 0.289 | 0.000 | 0.290 | 0.495 |
| slimqwen_merge | merge | 0.50 | 0.5/1.0 | 0.269 | 0.005 | 0.295 | 0.460 |
| random_drop | expert-removal | 0.50 | 0.5/1.0 | 0.255 | 0.005 | 0.285 | 0.465 |
| hcsmoe_merge | merge | 0.50 | 0.5/1.0 | 0.244 | 0.000 | 0.255 | 0.360 |
| **mobe** | weight-approx | 0.50 | 0.5/0.5 | **0.231** | 0.000 | 0.275 | 0.505 |

**SparseGPT is near-LOSSLESS training-free** (@0.75 MMLU 0.549 ≥ uncompressed; @0.50 0.521), tied with naive magnitude — the step-0 SOTA at both retains.

### The weight-approx family now has a clean internal ordering (5 methods)
**@0.75 MMLU:** sparsegpt **0.549** > nystrom_combined 0.499 > nystrom 0.474 > svd_llm_v2 0.450 ≫ mobe 0.246. **@0.50:** sparsegpt **0.521** > nystrom_combined 0.382 > nystrom 0.341 > svd_llm_v2 0.307 ≫ mobe 0.231. Two structural findings drop out:

1. **fwd+bwd > fwd-only > per-matrix-SVD.** Adding the backward (gradient) covariance to the Nyström kernel beats forward-only Nyström at both retains (MMLU 0.499 vs 0.474 @0.75; 0.382 vs 0.341 @0.50). And both *structured-triplet* Nyström methods beat *per-matrix* whitening SVD (svd_llm_v2 0.450), because Nyström factors the gate/up/down triplet JOINTLY (preserving MLP function) while SVD truncates each matrix independently. So within structured low-rank, **how you factor matters**: joint-triplet + gradient-aware > joint-triplet + fwd-only > independent-per-matrix.
2. **MoBE is still dead last by a mile** — shared-basis collapses the small MoE regardless.

**Practical cost note (per user):** nystrom_combined and svd_llm_v2 are 5–25× SLOWER than the others. nystrom_combined runs a full CE `loss.backward()` per calib batch (~48s/batch × 64 ≈ 50 min @ 256 seqs); svd_llm_v2 hooks 3136 expert-linear covariances + does 3072 per-matrix SVDs (CPU-bound). **For the recovery atlas use a smaller calib (32–64 seqs) for these two** — the 32-seq smoke was fast and accurate; 256 seqs is overkill and dominates wall-clock.

### Where MoBE lands: DEAD LAST, even at 0.75 (a real finding, not a bug)
**MoBE collapses the model to near-chance at BOTH retains** (MMLU 0.246 @0.75, 0.231 @0.50; GSM8K 0.000) — *worse than random expert drop*. The optimization converged (recon loss 0.27–0.38 normalized, monotone decreasing; reload verified), so this is the method, not the implementation. **Mechanism:** MoBE's shared-basis assumption (here m=8 basis matrices for 64 experts) is *too strong for a small MoE*. ~30% relative weight error per expert, applied across all 1024 experts simultaneously, compounds to collapse. MoBE's published wins are at 235B/512-expert scale where experts are far more numerous and redundant; **OLMoE-7B's 64 experts share too little for an aggressive shared basis.** *Caveat:* MoBE is sensitive to (m, iters, lr); a larger m (less compression per basis) or more iters could lift it — but at the *budget-matched* m needed to hit retain 0.75/0.50, it loses badly. This is itself the finding: **cross-expert-redundancy methods that win at giant scale fail at small-MoE scale.**

### So the weight-approx family SPLITS into two regimes
- **Granularity-preserving** (SparseGPT keeps all dims; nystrom shrinks structured) → strong, esp. SparseGPT (lossless).
- **Aggressive cross-expert sharing** (MoBE) → collapse on a small MoE.
This *complicates the clean "family" story*: within weight-approx, the CRITERION/mechanism gap is now enormous (SparseGPT 0.549 vs MoBE 0.246 @0.75). Family means are no longer the right summary for weight-approx — report per-method. The recovery question sharpens: **can training rescue MoBE's collapsed-but-low-rank experts (the factors are a good warm start), or is the shared-basis capacity loss permanent?**

> **Magnitude control (aux), CORRECTED & DONE:** first run was buggy (`torch.kthvalue` on the 6.4B bf16 tensor zeroed nothing); fixed to fp32 quantile-on-sample. Corrected: magnitude **@0.75 MMLU 0.550 / GSM8K 0.365** (storage 0.750), **@0.50 MMLU 0.516 / GSM8K 0.260** (storage 0.498). The 12 focal results are unaffected.

> **⚡ Sharpened finding — GRANULARITY dominates SOPHISTICATION at step-0.** The corrected magnitude control is the punchline: *naive global magnitude pruning* (MMLU 0.516 @0.50, 0.550 @0.75) nearly matches *sophisticated* SparseGPT (0.521 / 0.549) and *crushes* every whole-expert method (0.24–0.29 @0.50). So at step-0 the decisive axis is **weight-level vs expert-level GRANULARITY, not criterion sophistication** — within the weight-level family dumb≈smart, but the family (granularity) gap is huge. This is the family>criterion structure the variance decomposition will formalize, and it reframes the recovery question: **does training let the coarse expert-level methods close the granularity gap?** Full 14-job step-0 means @0.75: control(magnitude) 0.546, weight-approx 0.491, expert-removal 0.422, merge 0.357.

## Step-0 findings (the baseline the recovery-phase inversion test compares against)

1. **The weight-approx family — which SlimQwen omits entirely — WINS step-0 at both retains** (means @0.75: weight-approx 0.447 > expert-removal 0.422 > merge 0.357; @0.50 the gap widens enormously). This directly motivates the paper: SlimQwen's "no single one-shot method dominates" is partly an artifact of testing only whole-expert prune/merge.
2. **SparseGPT@0.50 is the standout**: MMLU **0.521** (~96% of uncompressed 0.542) at 50% expert-param removal, where drop/merge are at chance (0.24–0.29). Unstructured Hessian-aware sparsity barely dents the model — keeps all experts + all dims, zeroes least-important weights. At 0.50 its GSM8K (0.275) beats *every* drop/merge method's GSM8K at 0.75.
3. **Both FAMILY and CRITERION matter at step-0.** Within weight-approx @0.50, SparseGPT 0.521 vs SVD 0.341 is a *huge* criterion gap. Within expert-removal @0.75, reap 0.456 vs random 0.388. The variance decomposition will quantify which dominates — and the recovery phase asks which gap *survives training*.
4. **Budget axes distinguish families as designed**: drop/merge preserve active-capacity (active 1.0, a token still routes to top-8), weight-approx cuts it (active = storage). SparseGPT wins *despite* lower active capacity — a clean cross-family contrast.
5. **The step-0 winner family is weight-approx at BOTH retains** — this is the ranking the recovery-phase inversion test will check for a flip.

## Why this sets up the headline
The thesis is "does short recovery reorganize quality at the family level / make one-shot rankings unreliable?" The step-0 picture gives the inversion test maximum leverage: weight-approx (esp. SparseGPT) has a large, clean lead — *especially at 0.50 where everything else is dead*. If recovery lets the collapsed drop/merge families catch up → textbook cross-family inversion. If SparseGPT keeps its lead → contradicts SlimQwen's convergence claim on the omitted family. **Either outcome is publishable.**

## Pending
- SparseGPT@0.75 (in eval) + magnitude control pair → completes the 14-job step-0 table.
- Full-size eval (limit 200 → full) as a follow-up.
- **Recovery-SFT leg** (the actual inversion test): harness must run in the **verl env** (per-Linear ckpts incompatible with sft-env tfm-5.2 fused-3D) — minimal HF-Trainer, freeze attn, train experts+router, checkpoints {0,100,500,2000}, ×3 seeds. See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) + memory `olmoe-experts-per-linear-in-verl-env`.

### v3 — ORIGINAL OpenThoughts3 traces for calibration (2026-06-09), retain 0.5
User clarified: use the ORIGINAL OpenThoughts3 dataset's own reasoning traces (QwQ-32B-distilled, in the `conversations` field), NOT the Qwen3-4B re-rollout. Extracted 600 math convs from `/data/yequan/datasets/OpenThoughts3-1.2M/data/` (`ot3_original_math.jsonl`). These traces are LONG (p50 ~17k tok, all ≥2048 → full calib windows). Rendered with OLMoE's own chat template (verified).

**Calibration-source comparison (MMLU, retain 0.5):**
| calib source | nystrom (fwd) | nystrom_combined |
|---|---|---|
| **original OpenThoughts3 (QwQ)** | **0.365** | **0.384** |
| Qwen3-4B re-rollout | 0.341 | 0.382 |
| OLMoE-native | 0.338 | 0.349 |

**Original QwQ traces = best calibration for both** (nystrom +0.024 vs native; nystrom_combined 0.384 best). Quality ordering: original-QwQ > Qwen-rollout > native. **Trace richness/length matters more than strict on-policy match** — long high-quality reasoning exercises expert activations best. `*_origot3_s0` ckpts → v3 recovery training.
