# MoE Expert Compression — Literature Landscape

> 5-agent survey (2026-06-07) mapping SOTA expert pruning / merging / low-rank / unstructured methods for MoE LLMs, their calibration recipes, and whether their advantage survives (or requires) short recovery training. **Central finding: the cell "which MoE expert-compression method matters AFTER short training" is empirically empty.**

Thread: `docs/aris/moe_compress/`. Target model: **OLMoE-1B-7B-0924** (6.9B total / 1.3B active, 16 layers, 64 experts/layer, top-8 routing, hidden 2048). Compress experts only; attn + router untouched; retain 0.75.

---

## The research gap (the crux)

The hypothesis — *a "smart" training-free MoE compression method may lose its lead over naive compression after a short amount of training* — sits in a genuinely **open, unstudied cell**:

- **No paper** systematically studies "which MoE expert-compression method matters AFTER short training." Every MoE method reports recovery numbers (if any) **only for itself**, never against a common recovery protocol applied to all baselines.
- The closest **dense-model** evidence — **"A Free Lunch in LLM Compression"** ([2510.14444](https://arxiv.org/abs/2510.14444)) — states outright that *"reconstruction reduces the relative importance of the pruning criterion; performance gaps between sophisticated criteria and simple baselines shrink with model scale"* and that **Wanda matches/surpasses SparseGPT once you reconstruct**. It predicts the gap *should* close, and close *more* at scale — but it is dense-only.
- The literature is **split** along two axes: (1) granularity — gap **closes** for unstructured *weight* pruning, **persists** for coarse *layer/expert* dropping (*Reassessing Layer Pruning* [2411.15558] keeps a ~5% metric lead through FT); (2) sparsity — gap closes at moderate (~50%) sparsity, persists at extreme sparsity (lottery-ticket regime).
- **MoE adds a novel confound**: the router. Closure could be *stronger* (under-used experts are pure redundancy) or *weaker* (the router locks in which experts matter). Nobody has measured it.

**Closest direct competitor / concurrent work:** **"Is Retraining-Free Enough?"** ([2603.02217](https://arxiv.org/abs/2603.02217), Mar 2026) — argues one-shot expert dropping leaves the router mis-calibrated and proposes lightweight **router recalibration** as cheap recovery. It compares NAEE/REAP/HC-SMoE/MoE-SVD etc. **This is the paper to differentiate against** (it studies *router* recovery, not the *full-method ranking under SFT recovery on the original training data*, which is our angle). Code availability unconfirmed.

### ⭐ The single most important prior: SlimQwen ([2605.08738](https://arxiv.org/abs/2605.08738), May 2026)

**SlimQwen already asserts our hypothesis — at a different scale and scope — which makes our study a targeted, controlled test rather than a blind shot.** It prunes Qwen3-Next-80A3B → SlimQwen-23A2B via *depth (drop last ¼ layers) + width + expert* compression, then **continual-pretrains for 120B–400B tokens** with a KD+LM loss.

- **Expert-compression menu (this is the user's chosen "plain/SlimQwen" baseline — expert part only, no depth/width):**
  - *Pruning importance metrics:* **frequency** `E_x[𝟙[i∈A(x)]]`, **soft-logits** `E_x[𝟙[i∈A(x)]·z_i/Σz_j]`, **REAP** `(1/|X_i|)Σ z_i(x)‖E_i(x)‖₂`.
  - *Merging:* cosine-similarity pairing, importance-weighted convex merge `Ê = (I_i·E_i + I_{m(i)}·E_{m(i)})/(I_i+I_{m(i)})`.
  - *Partial-preservation (their innovation):* keep top-⌊Ñ/2⌋ experts **intact**, build the rest by merging each discarded expert into its most-similar retained base — "prevents representation homogenization."
  - *Calibration:* **1024 samples** from the pretraining set.
- **Their headline finding (= our hypothesis):** *"no single one-shot pruning or merging method establishes consistent superiority across all downstream tasks"* and after 400B-token continual pretraining "performance differences are **marginal** … one-shot choices matter less than recovery training" (§4.2, Table 2).

**Why our study is still novel and worth running despite SlimQwen:**
1. **Scale of recovery:** SlimQwen needs **120B–400B tokens** to wash out the differences. Our question is whether the gap closes after a **SHORT** recovery (~10k samples / a few hundred M tokens) — i.e. *how much* training is needed before method choice stops mattering. SlimQwen does not chart the *trajectory*; it only reports the 400B endpoint.
2. **Family coverage:** SlimQwen tests only *whole-expert* prune/merge (frequency/soft-logits/REAP/partial-preservation). It does **not** include **intra-expert low-rank / shared-base (D²-MoE, MoBE)** or **unstructured (SparseGPT/Wanda) expert compression** — entire families whose post-training behavior is unknown.
3. **Confound isolation:** SlimQwen co-varies depth+width+expert+distillation; ours isolates **experts only** (attn + router frozen), so any ranking change is attributable to the expert-compression method alone.
4. **Model regime:** SlimQwen is 80B→23B with a shared expert and 512 experts. OLMoE is 7B, 64 experts, **no shared expert** — a regime where cross-expert redundancy and the "plain catches up" effect may differ.

**Takeaway for idea generation:** the contribution is no longer "does the gap close?" (SlimQwen says yes at 400B) but **"how FAST does it close, across ALL compression families, with experts isolated, on a small MoE — and is there a *trajectory* / *crossover* structure?"** The interesting science is the *recovery curve*, not just the endpoint.

---

## Family 1 — Expert pruning (drop whole experts)

| Method | arXiv / venue | Selection | Calib | Recovery? | Code | OLMoE? |
|---|---|---|---|---|---|---|
| **NAEE** ("Not All Experts Are Equal") | 2402.14800 / ACL'24 | Enumerate expert subsets per layer, keep combo min ‖F′−F‖_F; + dynamic skip | C4 (or MATH) 128×2048 | training-free; opt. MetaMathQA FT restores GSM8K | [Lucky-Lance/Expert_Sparsity](https://github.com/Lucky-Lance/Expert_Sparsity) | port |
| **REAP** | 2510.13999 / ICLR'26 | Saliency Sⱼ = mean gⱼ(x)·‖fⱼ(x)‖₂ (router gate × act norm); **argues prune > merge** | C4(+code) 1024×2048 | one-shot, training-free | [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) | port |
| **MoNE** | 2507.00390 | Replace expert w/ constant mean output; pick by freq + output variance. **Tested on OLMoE!** | C4/Zyda2 100–1000 | both; 2B-token continued-pretrain recovery exp | [zxgx/mode-pd](https://github.com/zxgx/mode-pd) | **native OLMoE** |
| **Demystifying MoE Compression** | 2406.02500 / TMLR | Expert/Layer/Block Drop unified | C4 | light tuning noted | [CASE-Lab-UMD/Unified-MoE-Compression](https://github.com/CASE-Lab-UMD/Unified-MoE-Compression) | port |
| MoE-I² | 2411.01016 / EMNLP'24 | genetic inter-expert prune + intra-expert low-rank | C4+Alpaca 2048 | **LoRA recovery central** (2 ep, 50K) | [xiaochengsky/MoEI-2](https://github.com/xiaochengsky/MoEI-2) | port |
| Diversifying Expert Knowledge | 2407.09590 / NeurIPS'24 | CKA-group similar experts, merge group+router | C4 128 | opt. coeff SGD | promised | port |
| DERN | 2509.10377 | drop experts, re-graft neuron triplets | C4 128×2048 | retraining-free | HF only | port |

**Canonical recipe:** 128 C4 sequences × 2048 tokens, cache router gates + expert activation norms. **MoNE is the one method already validated on OLMoE-7B** → lowest-friction whole-expert baseline.

## Family 2 — Expert merging (fuse/cluster experts)

| Method | arXiv / venue | Grouping | Calib | Recovery? | Code |
|---|---|---|---|---|---|
| **MC-SMoE** ("Merge, then Compress") | 2310.01334 / ICLR'24 Spotlight | freq-dominant pick + router-logit sim + permutation align; merged→low-rank UV+sparse | subset of train data | **headline gains REQUIRE post-merge FT+KD** | [UNITES-Lab/MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) |
| **HC-SMoE** | 2410.08589 / ICML'25 | agglomerative clustering on mean expert outputs (router-independent), freq-weighted merge | C4 32×2048 | fully retraining-free | [wazenmai/HC-SMoE](https://github.com/wazenmai/HC-SMoE) |
| **Sub-MoE** | 2506.23266 / AAAI'26 | K-means on output cos-sim; joint/Union SVD shared-U + freq-weighted V | WikiText-2 128×2048 | training-free; **+FT → +4–6%** | [lliai/MoERazor](https://github.com/lliai/MoERazor) |
| EEP | 2407.00945 | learned expert-merge + router-map matrices via evolutionary search | task subset | gradient-free | [imagination-research/EEP](https://github.com/imagination-research/EEP) |
| MergeMoE / PuzzleMoE | 2510.14436 / 2511.04805 | output-merge least-squares / dual-mask sparse | act stats | training-free | check |

**Note:** MC-SMoE's famous gains *require* fine-tuning — a direct prior data point for our hypothesis. Modern methods (HC-SMoE, Sub-MoE) moved the advantage out of FT into smarter grouping. **REAP (2510.13999) counter-argues pruning > merging one-shot.**

## Family 3 — Intra-expert low-rank / shared-base decomposition

| Method | arXiv / venue | Per-expert op | Cross-expert redundancy | Calib | Recovery? | Code |
|---|---|---|---|---|---|---|
| **D²-MoE** | 2502.17298 / ICML'25 | shared base (Fisher merge) + low-rank delta SVD + struct prune | **yes (the canonical shared-base+delta)** | WikiText-2/C4 512 | training-free | [lliai/D2MoE](https://github.com/lliai/D2MoE) |
| **MoBE** | 2508.05257 | Wⁱ = Aⁱ·f(Σ αⁱʲ Bʲ), m shared basis matrices | **yes, explicit shared basis (SOTA at scale)** | none (weight reconstruction, Adam) | training-free | [inclusionAI/MoBE](https://github.com/inclusionAI/MoBE) |
| MoLAE | 2503.23100 | shared latent proj B + per-expert A | yes | closed-form SVD | training-free | — |
| MoE-SVD | ICML'25 (acJ3vdFljk) | selective SVD; **shared V across experts**, per-expert top-k U | yes | act stats | training-free | suppl. |
| SVD-LLM / V2 | 2403.07378 / 2503.12340 | whitening-aware SVD truncation (per-matrix, general LLM) | no | opt. LoRA | [AIoT-MLSys-Lab/SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) |
| MoDeGPT | 2408.09632 / ICLR'25 | Nyström+CR+SVD module reconstruction (general LLM FFN) | no | training-free | released |

**Cross-expert redundancy is the decisive lever for the best training-free results** (experts are CKA 0.3–0.5 similar; deltas are low-rank). **D²-MoE = shared-base+delta canonical; MoBE = SOTA at scale.** This family maps directly onto the repo's existing `src/compress/svd` (SVD-LLM-V2, Nyström) tooling.

## Family 4 — Unstructured / semi-structured expert pruning

| Method | arXiv / venue | What | Calib | Recovery? | Code |
|---|---|---|---|---|---|
| **MoE-Pruner** | 2410.12013 | Wanda × router gate: S=|W|·‖X·gate‖ (unstructured + N:M) | C4 128 | KD or 2B-token CPT → 99% dense | none released |
| **STUN** | 2409.06211 / ACL'25 | structured expert-drop THEN unstructured Wanda/SparseGPT | small | training-free | check |
| SparseGPT / Wanda | 2301.00774 / 2306.11695 | Hessian sparse-regress / |W|·‖X‖ (substrate, dense) | C4 128 | both one-shot | [locuslab/wanda](https://github.com/locuslab/wanda) |

**Repo already has SparseGPT** (`src/compress/unstructured/sparsegpt.py`) and the verl-actor mask-preservation hook → near-zero-friction unstructured baseline.

## Family 5 — The train-free-vs-finetuned gap (dense-model evidence)

| Paper | arXiv | Verdict on our hypothesis |
|---|---|---|
| **A Free Lunch in LLM Compression** | 2510.14444 | **GAP CLOSES** — reconstruction shrinks criterion importance; Wanda≈SparseGPT after; *stronger at scale*. Strongest support. |
| Rethinking the Value of Network Pruning | 1810.05270 / ICLR'19 | gap closes — pruned *structure* matters, not inherited weights, after retrain |
| Unreasonable Ineffectiveness of Deeper Layers | 2403.17887 / ICLR'25 | cheap QLoRA heal rescues crude layer-drop |
| Reassessing Layer Pruning | 2411.15558 | **GAP PERSISTS** for layer pruning — metric lead survives FT |
| Lottery Ticket / rewinding | 1803.03635 / 1903.01611 | gap persists at extreme sparsity |

---

## Implications for our study

1. **Method shortlist (one strong representative per family), ranked by OLMoE-portability:**
   - **Unstructured:** SparseGPT on experts (repo-native) + router-weighted Wanda (MoE-Pruner-style).
   - **Whole-expert prune:** MoNE (native OLMoE) and/or REAP saliency (prune>merge claim).
   - **Merge:** HC-SMoE (router-independent, most portable) — and its FT-dependence is the historical prior.
   - **Low-rank / shared-base:** D²-MoE-style shared-base+delta, built on repo `src/compress/svd` + Nyström.
   - **Naive control:** uniform random/magnitude expert drop + plain per-matrix SVD (no calibration) — the "plain compress" the hypothesis says might catch up.
2. **Calibration:** each method uses its own paper recipe (mostly 128×2048 C4/WikiText). Hold the recovery protocol fixed (OpenThoughts3, 10k samples) across all.
3. **The contribution is the 2-D table** {method × {training-free, +short-SFT}} on OLMoE — empirically empty in the literature. Whichever way it lands (gap closes → "method choice is wasted effort, just train"; gap persists → "method choice compounds"), it's a clean result.
4. **Differentiate from 2603.02217** ("Is Retraining-Free Enough?"): they recover the *router*; we recover the *whole model on the original SFT data* and rank *all five families* under one protocol.

## Sources
NAEE 2402.14800 · REAP 2510.13999 · MoNE 2507.00390 · Demystifying 2406.02500 · MoE-I² 2411.01016 · Diversifying 2407.09590 · DERN 2509.10377 · MC-SMoE 2310.01334 · HC-SMoE 2410.08589 · Sub-MoE 2506.23266 · EEP 2407.00945 · MergeMoE 2510.14436 · PuzzleMoE 2511.04805 · D²-MoE 2502.17298 · MoBE 2508.05257 · MoLAE 2503.23100 · SVD-LLM 2403.07378 · SVD-LLM-V2 2503.12340 · MoDeGPT 2408.09632 · MoE-Pruner 2410.12013 · STUN 2409.06211 · SparseGPT 2301.00774 · Wanda 2306.11695 · A Free Lunch 2510.14444 · Reassessing Layer Pruning 2411.15558 · Deeper Layers 2403.17887 · Rethinking Pruning 1810.05270 · Is Retraining-Free Enough 2603.02217 · MoE Survey 2407.06204 · OpenThoughts3 2506.04178 · OLMoE 2409.02060
