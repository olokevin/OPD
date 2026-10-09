# Parameter-space exploration for BP-OPD (pn-explore)

> Does exploring in **parameter space** help on-policy distillation? Each step we perturb the student's
> weights, greedy-decode new sequences with the perturbed model, let the teacher score every token, and
> update the **clean** weights by backward, with importance sampling for the off-policy (perturbed)
> sequences. Setting = the es-prefill / BP-OPD ds15b pair; compared against the 2026-08-31 BP-OPD
> baseline at the same rollouts per update and learning rate. Started 2026-10-08.

## Status

| run | GPU | wandb | state |
|---|---|---|---|
| **full** — all params, Adam 1e-6, perturb decoder blocks, σ=1e-3 | 2 | [kt8hbgk9](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/es_opd_JustRL_1p5b_0903/runs/kt8hbgk9) | running (relaunched 2026-10-08 19:20 PDT after an OOM at step 1) |
| **isobtt** — only the small cores V_j trained + perturbed, Adam 1e-5, σ=2e-3 | 3 | [aupz6xbl](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/es_opd_JustRL_1p5b_0903/runs/aupz6xbl) | running (launched 2026-10-08 18:19 PDT) |
| BP baseline (reference, 2026-08-31) — T=1 sampling, all params, Adam 1e-6 | — | [s7x9sidi](https://wandb.ai/yequan_zhao-university-of-california-santa-barbara/es_opd_JustRL_1p5b_0903/runs/s7x9sidi) | finished |

## Results

Greedy pass@1 (`val-core/<bench>/acc/mean@1`, n=1, T=0, 7168 tokens), every 20 steps. The BP baseline's
in-run evals are sampled (n=2 @ T=0.6); its only greedy number is the step-220 re-score.

| step | full MATH-500 | full AIME24 | isobtt MATH-500 | isobtt AIME24 | BP MATH-500 (sampled) | BP AIME24 (sampled) |
|---|---|---|---|---|---|---|
| 0 | 0.670 | 0.233 | 0.670 | 0.133 | 0.751 | 0.150 |
| 20 | **0.818** | 0.200 | **0.766** | 0.300 | 0.823 | 0.317 |
| 40 | 0.766 | 0.333 | 0.776 | 0.233 | 0.843 | 0.267 |
| 60 | | | | | 0.846 | 0.267 |
| 80 | | | | | 0.848 | 0.367 |
| 100 | | | | | 0.844 | 0.317 |
| 120 | | | | | 0.859 | 0.250 |
| 140 | | | | | 0.850 | 0.367 |
| 160 | | | | | 0.851 | 0.333 |
| 180 | | | | | 0.861 | 0.400 |
| 200 | | | | | 0.858 | 0.350 |
| 220 | | | | | 0.845 (greedy 0.678) | 0.283 (greedy 0.267) |

Greedy references on the same evaluator (from [zo_opd.md § aligned pair](zo_opd.md)): base 0.670 / 0.133;
es-prefill peak 0.808 @140, final 0.784 @260; BP @220 greedy re-score 0.678 / 0.267.

**Step-20 read (2026-10-09).** Both arms learn fast under greedy evaluation. full +14.8 pp (0.818),
already above es-prefill's best (0.808 @140); isobtt +9.6 pp (0.766), versus es-prefill's 0.676 @20.
Same-step greedy curves for comparison: es-prefill 0.676 @20, es-decode 0.696 @20.

| step | arm | entropy | \|log π_W − log π_pert\| / tok | first divergence (tok) | IS ESS | tokens with IS > 2 | mean len | cap-hit |
|---|---|---|---|---|---|---|---|---|
| 1 | full | 0.30 | 0.012 | 67 | 0.999 | 0.0 % | 6.6 k | 0.84 |
| 20 | full | 0.86 | 0.098 | 16 | 0.976 | 1.2 % | 6.6 k | 0.72 |
| 1 | isobtt | 0.30 | 0.009 | 99 | 1.000 | 0.0 % | 6.5 k | 0.83 |
| 20 | isobtt | 0.84 | 0.022 | 54 | 0.999 | 0.0 % | 6.3 k | 0.72 |

Student entropy rises 0.30 → 0.85 in both arms (teacher entropy ≈ 0.68). A flatter policy has more
near-ties, so a fixed σ flips more greedy tokens: the full arm's effective exploration grew 8× by step 20
with no change to σ. IS stays healthy so far.

**Step-40 read.** full falls back to 0.766 (−5.2 pp from @20, about 2.7 SE on 500 problems) while AIME24
rises to 0.333. isobtt holds at 0.776. Training-rollout looping keeps falling (cap-hit 0.72 → 0.49 full,
0.72 → 0.61 isobtt). Policy gap: full 0.085 / divergence at token 12, entropy 0.74; isobtt 0.019 /
token 65, entropy 0.88. IS ESS 0.98 / 1.00.

Step 0 reads 0.670 on MATH-500 for both arms, the same as the base on the old A800 box. AIME24 differs by
3 of 30 problems between the two runs on identical weights (greedy bf16 nondeterminism).

## Method

Per step, 64 prompts × 4 rollouts, all **greedy**:

1. **Slot 0 = clean decode** of the current student W.
2. **Slots 1–3 = perturbed decodes.** For each slot k, draw a fresh seeded Gaussian ε_k, set
   W + σ·ε_k, greedy-decode the 64 prompts, compute the behaviour log-probs log π_{W+σε_k}(y_t) by a
   teacher-forced forward under the same perturbation, then remove the perturbation. One perturbation per
   slot per step, shared by the slot's 64 prompts. This reuses the ES machinery (`es_perturb_weights`), and
   the hybrid engine re-syncs FSDP weights into vLLM on every generate.
3. **Teacher scores every token** of all 256 sequences. Advantage = `log q(y_t) − log π_W(y_t)` (k1, clean
   student log-probs), the same as BP-OPD.
4. **BP update on the clean weights.** Decoupled PPO: the PPO ratio and clip stay around the clean policy
   (π_θ / π_W), and each token carries the truncated IS weight `min(π_W(y_t) / π_{W+σε_k}(y_t), 2)`
   (verl `rollout_correction`, token level). Clean rows get weight exactly 1.

This follows Plappert et al. (2018): perturb the parameters to explore, learn with the unperturbed policy,
and correct the off-policy data by importance sampling. Greedy decoding makes each slot deterministic,
so all the exploration comes from parameter space. Without the perturbation, the 4 rollouts per prompt
would be identical copies.

**Noise size.** σ is sized by its effect on the policy (Plappert's adaptive-noise idea, set once at
step 1, not adapted). Smoke test, 64 prompts × 1024 tokens:

| arm | σ | perturbed decode = clean | first divergence (tokens) | mean \|log π_W − log π_pert\| per token | IS weight mean |
|---|---|---|---|---|---|
| full (decoder blocks) | 1e-3 | 0 % | 67 | 0.021 | 1.0007 |
| isobtt (V_j only) | 1e-3 | 0 % | 151 | 0.011 | 1.0001 |

At the same σ, perturbing only the V_j cores moves the policy about half as much. So isobtt uses
**σ=2e-3**, which roughly matches the full arm's policy gap. At full length (step 1, 7168 tokens):
full |log-ratio| 0.012, first divergence at token 67; isobtt 0.009, token 99. Neither arm ever
produced a perturbed decode identical to the clean one.

**Cost** (step 1, RTX 6000 Ada): ~990 s/step, split as generation 470 s (4 greedy passes + 58 s
behaviour log-probs), teacher 104 s, clean log-probs 78 s, update 334 s. The base student loops on
DAPO prompts under greedy decoding (mean length 6.5 k, 83 % hit the 7168 cap), so 279 steps take
about 3.3 days at the current lengths.

## Setup

| knob | BP baseline (s7x9sidi) | pn-explore full | pn-explore isobtt |
|---|---|---|---|
| student / teacher / data | R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B, DAPO-Math-17k (17,917 prompts) | same | same |
| prompts × rollouts / step | 64 × 4 = 256 | same | same |
| updates / step | 4 Adam steps × 64 seqs (ppo_mini 16 prompts) | same | same |
| rollout | vLLM, T=1.0, top-p 1.0 | **greedy (T=0)**: 1 clean + 3 perturbed | same |
| perturbation | — | σ=1e-3, decoder blocks (`layers`, 1.31 B elements, = es-prefill aligned) | σ=2e-3 on the V_j deltas (40.5 M elements) |
| off-policy correction | none | token TIS, threshold 2 | same |
| trained params | all 1.78 B | all 1.78 B | isobtt small cores V_j only (40.5 M); U_j, S_j, embeddings, norms, biases, lm_head frozen |
| optimizer | Adam lr 1e-6, wd 0.01 | Adam **1e-6**, wd 0.01 | Adam **1e-5** (10×), wd 0 (ISO needs 0), polar retraction of V_j every step |
| loss | k1 token reward, PPO clip 0.2, token-mean | same | same |
| lengths | prompt 1024, response 7168 | same | same |
| epochs / steps | 1 / 279 | 1 / 279 | 1 / 279 |
| eval | MATH-500 + AIME24, n=2 @ T=0.6, every 20 | **greedy n=1**, every 20 (standing ES setting) | same |
| checkpoints | every 20, keep 1 | every 20, keep 1 (`/data`) | same |
| hardware | 1 GPU (A800/H100 box) | 1× RTX 6000 Ada 48 GB | same |

isobtt = ISO on fura's block-wise SVD: per input block `W[:, blk_j] = U_j S_j V_jᵀ` (blocks from the
closest factor pair, b = 48 / 112). U_j (out × b) is the large core, V_j (b × b) the small core. Here
only V_j moves, so each block's singular values stay fixed.

## Next steps

1. Fill the results table as evals land. At the end, re-score the final checkpoints with the sampled
   protocol (n=2 @ T=0.6) to compare against BP's sampled curve directly.
2. Missing control: **BP with greedy rollouts and no perturbation** (64 × 1 clean). This separates "the
   exploration helps" from "greedy training rollouts help" (es-prefill showed that greedy training alone
   cuts looping).

## Reference

- **Code** (commit `6dd292a`): `verl/verl/trainer/ppo/pn_explore.py` (slot generation, behaviour
  log-probs, diagnostics `pn/*`); hooks in `ray_trainer.py` (gen step; `pn_fill_clean` after the clean
  `old_log_probs`); config `algorithm.pn_*`; `peft.iso.train_factors=v` in `verl/workers/peft/iso.py`.
- **Launcher**: `scripts/zo_opd/ds15b/pn_explore.sh`.
  ```bash
  ARM=full TRAIN_GPU=2 bash scripts/zo_opd/ds15b/pn_explore.sh
  ARM=isobtt PN_SIGMA=2e-3 TRAIN_GPU=3 bash scripts/zo_opd/ds15b/pn_explore.sh
  SMOKE=1 ARM=full TRAIN_GPU=2 bash scripts/zo_opd/ds15b/pn_explore.sh   # 2 steps, 1024 tokens
  ```
- **Logs**: `logs/ds15b/pn_explore/{full,isobtt}.out` (+ `run_*.log`); smoke `smoke_{full,isobtt}.out`.
  Checkpoints: `/data/yequan/opd/opd/DAPO-Math-17k/ds15b_pn-explore_{full_T0_b64n4_sig1e-3_lr1e-6,isobtt_T0_b64n4_sig2e-3_lr1e-5}/`.
  Eval dumps: `validation_log/<experiment>/`.
- **Bugs fixed on the way**:
  1. With greedy rollouts, the BP update divided the logits by T=0, giving a NaN loss and every step
     skipped. The batch inherits `meta_info["temperature"]` from the rollout; `update_actor` now uses
     T=1 like the log-prob path.
  2. `algorithm.rollout_correction` defaults never load (the `algorithm@@…` defaults entry), so the keys
     are created with `++`.
  3. OOM in the full arm's update on the 48 GB card. The offloaded Adam states (14 GB) were loaded
     before forward/backward. `OPTIM_LOAD_AT_STEP=1` now loads them only around each
     `optimizer.step()`. Log of the OOM'd attempt: `logs/ds15b/pn_explore/full_oom_step1.out`
     (wandb `1sz7lpbq`).
  4. `scripts/opd/run_systemd.py` `Path.open(opener=)` bug. Bypassed with `OPD_SYSTEMD=0`; the Ray temp
     dir moved to `/data/yequan/temp` (root disk is 96 % full).
- **Aborted first launch** (wrong wandb project, stopped after the step-0 eval): `kquhpetn` / `eiqkcxa4`
  in `es_opd_JustRL_1p5b`.
