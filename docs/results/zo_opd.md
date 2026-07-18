# ZO-NP (zeroth-order node-perturbation) OPD — results

Student `Qwen/Qwen3-1.7B`, teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`.
Loss = per-token reverse-KL to the teacher over the student top-K=16 set (`reward_weight_mode=student_p`).
Trainer: `verl/verl/trainer/np/` (custom n_sample-wide perturbed vLLM decode); driver
`scripts/zo_opd/zo_np_train.sh`. Offline gradient harness: `verl/verl/trainer/zo_np/grad_check.py`
(`scripts/zo_opd/zo_np.sh`). Full working notes: `scripts/zo_opd/results/{ANALYSIS,SCALING_FIX_AND_LR}.md`.

---

## Session 2026-06-02 — gradient scaling, LR search, and a self-amplifying divergence

### 1. NP estimate vs the true BP gradient (offline, `grad_check.py`)
For one perturb layer (`model.layers.0.mlp.down_proj`, d_out=2048) on a frozen (prompt, greedy-response),
the harness computes the NP δW (reusing the **shipping** estimator math) and the true `dL/dW` via
`loss.backward()` of the same OPD loss.

- **cos(NP δW, BP dL/dW) ≈ 0.01–0.02** at the trainer's 64 perturbations/token — the δW *matrix* direction
  is variance-starved. NOT a bug: a per-token `dL/dy` probe shows cos rising 0.03 → 0.18 as N: 16 → 4096.
- **‖NP‖/‖BP‖ tracks √(d_out/N)** exactly (≫1 at small N, → 1 as N → d_out), on both d_out=2048 and 1024.
- Binding constraint is the **rank-1-assembled weight matrix** (12.6 M elements over ~24 noisy g_t), far
  more sample-hungry than a single node-gradient vector.

### 2. Scaling fixes (`grad_estimator.py`, `ray_trainer.py`)
- ANP `1/‖u‖²` normalization made a config (`np.normalize_anp`, default **false**); it was hardcoded
  `True` and shrank the update by `1/d_out ≈ 1/2048`.
- `grad_estimate_sample=grpo` scale, two iterations:
  - `(L_q−mean)/σ` — restores the `1/σ` finite-difference scale (drops `/std`).
  - `((L_q−mean)/std)/σ` — **current code, per request** — keeps BOTH the z-score (`1/std`) and `1/σ`.
- Offline: the fixed estimators put ‖δW‖ on the true-gradient scale (vs the old `2e-4` ratio).
- **Key invariant:** the *assembled* δW norm is ≈28–57 for the `/std`, `/σ`, AND `/std/σ` forms alike,
  because `token_agg=mean` cancels the per-token scale. So the per-token 100× difference (`1/σ`) does **not**
  reach the weight update — the bf16-effective LR is similar across all three forms.

### 3. Training infrastructure built this session
- `fit()` restructured: **batch_size prompts/update** (1 rollout/prompt, n_sample=64), greedy clean decode.
- **Student + teacher co-located on one GPU** (one LR per GPU): needs `distributed_executor_backend="uni"`
  + keep `CUDA_VISIBLE_DEVICES` + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + mem-util 0.30 each.
- **Update-propagation verification** every step: `train/weight_changed_frac` (fraction of weight elements
  that flip in bf16 — the true "did it land" signal) + `train/weight_sync_ok` (all engines hold the same
  weight after broadcast → the next rollout reads the update). `apply_node_update` returns the changed frac.
- **Fixed held-out teacher-KL probe** (`eval/heldout_kl`) — per-step `train/L_clean_mean` is on shifting
  prompts so it can't show learning.
- **Bug fixes that made training valid:** (a) the MATH/GSM8K prompt processor only recognized `list/tuple`
  prompts but the parquet `prompt` is a `numpy.ndarray` → it silently fed an empty `"Problem: "` prompt;
  **both eval and training ran on blank prompts** (teacher-KL ~0.81 blank vs ~0.33 real) — fixed in
  `task_utils.py`, affects all np/es opd_math runs. (b) removed a global `ray stop --force` that killed
  concurrent runs' Ray sessions.

### 4. The bf16 reality
vLLM student weights are bf16. An update lands only if `lr·δW_elem` clears the mantissa step. Two
consequences that shaped the whole LR search:
- **`weight_delta` (the ‖W‖-norm difference) badly UNDER-reports the update** — element changes partly
  cancel in the norm, so a 20 %-of-elements update can show ~0 norm-delta and *look* like a no-op. Use
  `weight_changed_frac`, not the norm difference.
- For the production δW (norm ≈ 28–57), the LR → fraction-of-weights-changed map is roughly:
  lr 2e-5 → 0.1–0.3 %/step, 2e-4 → 3–12 %, 6e-4 → 7–31 %, 2e-3 → 22–57 %.

### 5. LR search for grpo = `((L_q−mean)/std)/σ` (wandb project `zo_opd_qwen4b_1p7b`)
The proper LR is `≈ ÷100` vs the `/std`-only form (the `1/σ=100×` per-token factor): the analog of a good
`/std @ 2e-3` is `/std/σ @ 2e-5`, etc. Swept the meaningful-update band (2e-4 / 6e-4) at batch=8.

| phase | observation |
|---|---|
| steps 0–10 | both 2e-4 & 6e-4 **dip the KL** (e.g. 6e-4: 0.336→0.322→0.318) — looks like training |
| steps 10–25 | KL **oscillates in a 0.31–0.35 band** = the probe's own ~±0.03 noise (greedy NP-decode is not bit-deterministic: two runs gave step-0 KL 0.306 vs 0.336 with identical weights) |
| **steps ~28–35** | **both runs DIVERGE**: 2e-4 KL → 0.47→0.48; 6e-4 KL → **0.93→1.14**. dW had grown 57→~2000 across the round-robin and chg% had climbed to 40–65 % before the layer-cycle reset |

**Honest verdict:** `/std/σ` *lands valid updates* in the 2e-4–6e-4 band (update signal is clean: chg%
rises monotonically, no no-ops), but the **held-out KL never sustainably decreases** — early steps are
buried in probe noise and by ~step 30 (one full 28-layer round-robin) the run **diverges**. This is the
`1/std` self-amplification (low-signal tokens, `std→0 ⇒ 1/std→∞`, +1e-8 floor insufficient) playing out
over a longer horizon than the wildly-too-high LRs did. No LR in the tested band gives stable training.

### 6. Cross-check: grpo = `(L_q−mean)/σ` (drop `/std`)
The `/σ`-only form trained **cleanly and monotonically** at **lr=3e-2** over the first ~16 steps
(held-out KL 0.335 → 0.322 → 0.319, bounded dW). It is the cleanest demonstrated training curve.
(A long run to check whether it too eventually diverges was not done this session.)

### 7. Important measurement caveats discovered (so future runs don't repeat them)
- **`weight_delta` norm-diff ≠ no-op** — use `weight_changed_frac`.
- **dW grows step-over-step from the `en_layerwise` round-robin**, not (only) from divergence — each step
  perturbs a *different* layer with its own δW norm. PROOF: the dW sequence `28,38,37,46,51…` is identical
  at lr=2e-5 and lr=2e-3. Compare dW only at the **same layer** across cycles before calling divergence.
- **The held-out KL probe is noisy (~±0.03)** because it re-runs the nondeterministic NP-decode. To rank
  LRs cleanly, either run ≥100–200 steps (cumulative signal > noise) or replace it with a **deterministic
  teacher-forced NLL/KL on a larger fixed set**.

### 8. Recommendation / open items
- **For a clean, demonstrably-training config:** grpo `(L_q−mean)/σ` at **lr=3e-2** (drop `/std`).
- **If keeping `/std/σ`** (current code): no tested LR trains stably past ~30 steps; needs either a hard
  std floor (`std.clamp_min(~0.05)`) or **global** (batch-level, not per-token) standardization to stop the
  `1/std` blow-up — untested.
- **Before any further LR pick:** add the deterministic teacher-forced loss probe; the current KL probe's
  noise was the single biggest obstacle to ranking LRs this session.

**Code touched:** `verl/verl/trainer/np/{grad_estimator.py,ray_trainer.py}`,
`verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`,
`verl/verl/trainer/config/np_trainer.yaml`, `verl/verl/trainer/es/task_utils.py`,
`verl/verl/trainer/zo_np/grad_check.py` (new), `scripts/zo_opd/{zo_np.sh,zo_np_train.sh}` (new).

**wandb** (`zo_opd_qwen4b_1p7b`): `/std/σ` — a1rmd3vt(2e-5), l0gsgnc6(6e-5), ul4tt5n3(2e-4), pz36he7i(6e-4),
6bjqk1a7(2e-3), mt9un3ge(2e-4 b8), bkbs4fms(6e-4 b8). `(L_q−mean)/σ` — h4hk3tex(3e-2) and siblings.
