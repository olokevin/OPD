# reasoning_aware_compress_calib: what fixes one-shot structured compression of reasoning LLMs

> Goal: understand *why* one-shot structured low-rank compression (SVD-LLM-V2 attn
>
> + Nystrom MLP) collapses Qwen3-4B's math reasoning, and which levers recover it.
>   Two independent levers landed: **(1) the rank floor** — a small full-rank sparse
>   residual restores the "escape edges" pure low-rank truncation kills (MATH 72→82%
>   at fixed budget); **(2) the calibration format** — sequence-reweighted full-length
>   covariances beat the legacy 2048-token-window/token-pooled scheme and push the
>   compression cliff lower (now the repo default).

This page synthesizes the A/B/D mechanism search, the forward-only ratio sweep +
reasoning-trace diff, and the full-sequence-calibration study. Source results:
[INITIAL_RESULTS_ABD](../aris/reason_aware_compress/INITIAL_RESULTS_ABD.md) and
[FULLSEQ_CALIB_RESULTS](../aris/reason_aware_compress/FULLSEQ_CALIB_RESULTS.md);
plan [EXPERIMENT_PLAN](../aris/reason_aware_compress/EXPERIMENT_PLAN.md); the
preceding falsification is in
[EXPERIMENT_RESULTS](../aris/reason_aware_compress/EXPERIMENT_RESULTS.md).
Related: [compressed_opd](compressed_opd.md) (the standing compression pipeline).

---

## 1. Problem & operating point

One-shot, no-SGD structured compression of **Qwen3-4B (non-thinking)**: SVD-LLM-V2
input-whitening on attention linears + Nystrom/MoDeGPT neuron-subsampling on MLP.
At the aggressive 0.36 retain ratio this collapses MATH-500 to **0% / PPL ~5000**
even with in-domain math calibration, while iso-param SparseGPT (full-rank
unstructured) keeps 45% — single-module compression is ~free, so the failure is
**distributed across depth**.

**Operating point for the whole study**: retain **0.8 first, then sweep down**;
the **last decoder layer's linears are never compressed** (layer 35 of 36 left
dense — it feeds the LM head directly); bf16, 1×H100.

**Standing references** (same eval contract): dense 4B **80.5% / 19.9** · native
1.7B 50.0% / 15.4 · SparseGPT+math 45.0% / 82.0 · SVD+Nystrom collapse @0.36
**0.0% / 4,980**.

### Eval contract

MATH-500, greedy (`do_sample=False`, `max_new_tokens=2048`), Qwen3 non-thinking
chat template (`enable_thinking=False`), graded by `ttrl_math.compute_score`
against the **dataset gold** (`reward_model.ground_truth`, never a model output).
Companion metric: C4 sliding-window PPL (seqlen 2048, seed 0). Driver
`eval_math500` (`scripts/.../layer_sensitivity.py`); `eval_math_capture`
(`compress_common.py`) additionally saves every response to compute the relaxed
metrics below.

### Calibration data

`datasets/OpenThought3-Qwen3-4B/data/train.jsonl` — each row = a **user math
problem + the full reasoning response rolled out by the uncompressed Qwen3-4B**,
rendered with the *same* chat template (`add_generation_prompt=False` so the
assistant trace is included). The covariances are collected over the model's
forward (and, for the bilateral objective, backward) pass on these complete
**prompt+reasoning** sequences. Held fixed (seed 3) across cells. §4 covers the
format (windowing/reweighting) that turned out to matter.

## 2. Three candidate mechanisms (M1/M2/M3)

The hypothesis space after the **steering-subspace thesis was falsified** (Block 0:
difference-of-means steering directions are the *best-preserved* part of the
collapsed model, not the most eroded — see EXPERIMENT_RESULTS):

- **M1 — rank floor**: low-rank truncation drops the full-rank "escape edges"
  SparseGPT keeps; the residual stream needs that off-subspace/tail mass.
- **M2 — objective**: the reconstruction is variance-weighted (input activations),
  not loss-weighted; a grad/loss-weighted objective should help.
- **M3 — accumulation**: covariances are collected in one *dense* pass, so no
  layer sees its compressed upstream; errors compound across depth.

The A/B/D blocks are the three direct fixes (A→M1, D→M2, B→M3), all one-shot.

## 3. Results — which mechanism is the lever

All at retain 0.8, last layer dense, MATH-500/100 + C4 PPL, reasoning-trace calib
(2048-window/token-pooled — the format current at the time).

### Block D — objective (M2): **NULL**

| Cell | Attn objective                              | C4 PPL | MATH/100      |
| ---- | ------------------------------------------- | ------ | ------------- |
| D0   | forward-only (input whitening)              | 52.1   | **73%** |
| D1   | backward-only (CE grad)                     | 294    | **0%**  |
| D2   | bilateral (input + CE grad) = OBD-LLM-style | 52.1   | **70%** |

**D0 ≈ D2 ≫ D1.** Adding the CE-gradient (bilateral, the OBD-LLM prior-art
baseline) gives **no gain** over plain input-whitened SVD (70 vs 73, within noise).
Backward-only whitening *destroys* attention (0% / PPL 294) — a grad-weighted
objective without input whitening picks the wrong subspace. **The objective is not
a separable lever.** (The OPD/teacher-gradient cell D3 was deferred — it requires a
*distinct* teacher; with teacher==student the OPD KL≡0 and the cell is degenerate,
a fail-fast guard now blocks it. The CE-bilateral null makes a large OPD-gradient
effect unlikely.)

### Block A — rank floor (M1): **THE HEADLINE**

`Ŵ = UV + S`: low-rank SVD at a reduced budget + a SparseGPT/OBS-pruned **full-rank**
residual of `R = W − UV` at the leftover budget; total stays at the retain ratio.
(`src/compress/hybrid/lr_sparse.py`, `LRPlusSparse`; budget split via `sparse_frac`,
default 0.075 → LR 0.74 + S 0.06 at ρ=0.8.)

| Cell | System                                                                        | C4 PPL | MATH/100      |
| ---- | ----------------------------------------------------------------------------- | ------ | ------------- |
| A0   | pure SVD-V2 (= D0 baseline)                                                   | 52.1   | 72%           |
| A1   | + sparse residual, fit vs**dense** acts                                 | 42.4   | 80%           |
| A2   | + sparse residual, fit vs**compressed-upstream** acts (refine_passes=1) | 42.4   | **82%** |

**A2 ≥ A1 > A0** (the plan's success criterion). A small (~6% density) **full-rank**
sparse residual jumps MATH **72→82%** (*beats dense 4B 80.5%*) and drops PPL 52→42,
**at the same budget**. Crucially this is *exactly* where the earlier
attention-tail-rescue failed (0→4%, which re-added only the discarded *low-rank*
tail): the missing ingredient is **full-rank escape edges**, not more low-rank. M1
confirmed causally.

### Block B — accumulation (M3): baseline only

B0 (dense-pass = D0) reproduced the 71% baseline; B1/B2 (sequential
re-linearization, `src/compress/sequential/relinearized.py`) were **skipped per
user direction** — M3 not pursued this pass.

## 4. The cliff and the failure mode (forward-only ratio sweep + trace diff)

Plain forward-only SVD-V2+Nystrom across descending retain ratios
(`ratio_sweep_trace.py`), with the 5 frozen dense-correct MATH probes regenerated
on each compressed model.

| ratio | C4 PPL | MATH/100      | trace len (med×dense) |
| ----- | ------ | ------------- | ---------------------- |
| 0.8   | 52.1   | **72%** | 1.1×                  |
| 0.7   | 96.6   | **66%** | 1.1×                  |
| 0.6   | 223.8  | **37%** | 5.1×                  |
| 0.5   | 1,157  | 20%           | 6.3×                  |
| 0.4   | 6,553  | 4%            | 7.3×                  |

**Sharp cliff at r\* ≈ 0.65.** Above it, traces stay ~dense length and accuracy
holds; below it, traces **balloon 5–7×** while accuracy collapses.

### Failure = loss of termination, not loss of reasoning

The break is **not an early wrong step**. Splitting "did the model ever reach the
correct answer" (relaxed/`reached✓`, any correct `\boxed{}` anywhere —
`analyze_len_to_correct.py`) from "graded on the full output" (strict MATH) reveals
**two cliffs**:

|                                                 | r=0.7 | r=0.6           | r=0.5 |
| ----------------------------------------------- | ----- | --------------- | ----- |
| **reached** the correct answer (5 probes) | 4/5   | **4/5**   | 1/5   |
| **strict MATH** (full output)             | 66%   | **37%**   | 20%   |
| trace blow-up                                   | 1.1× | **5.1×** | 6.3× |

- **Termination cliff ≈ 0.65**: at r=0.6 the model *still reaches the correct
  answer as often as at r=0.7* (reasoning intact) but strict accuracy halves —
  because the trace balloons and **loops past the answer until the token cap**, so
  the graded output is degenerate. What breaks first is the ability to **stop**.
- **Reasoning cliff ≈ 0.55**: only below r=0.6 does `reached✓` itself collapse
  (4/5→1/5) — the model genuinely stops getting the answer.

Worked example (probe pid=2, `f(-2)+f(-1)+f(0)`, gold `14/3`): at r=0.6 the
compressed model computes `14/3` correctly at ~char 700, then repeats
`### Final Answer \boxed{14/3}` until the 2048-token cap (trace 5× dense) and is
scored wrong. This is the **RAC length↑/acc↓ looping signature** — compression
first raises the per-token error floor past the model's **termination/EOS capacity**,
and only later past its **answer-reaching capacity**. (The char-level
first-divergence metric is uninformative — greedy decoding rephrases the opening
cosmetically; the discriminating signal is trace-length blow-up + non-termination.)

## 5. The calibration-format lever (now the default)

A *pure calibration change* — same forward-only SVD+Nystrom method, same budget —
turned out to be a second independent lever. Two axes
(`fullseq_calib_sweep.py`, `build_fullseq_calib_loader`, `collect_covariances_reweighted`):

- **reweight**: `token` (every token equal → long traces dominate the pool) vs
  `sequence` (every *conversation* equal: `C = mean_seq[(Σ_t v_t v_tᵀ)/N_seq]`).
- **length**: `full` (whole conversation) vs `lt2048` (only convs <2048 tokens) vs
  the legacy `window2048` (2048-token windows). All mask-aware (pad excluded); the
  legacy window scheme used all-ones masks, dropped short convs, and discarded
  long-trace tails.

**Stage 1 @ retain 0.7 (strict MATH, pick-best):**

| setting                      | strict        | C4 PPL |
| ---------------------------- | ------------- | ------ |
| **sequence · lt2048** | **71%** | 92.4   |
| sequence · full             | 69%           | 98.7   |
| token · full                | 66%           | 103.0  |
| token · lt2048              | 65%           | 355.2  |

→ **Sequence-reweighting is the dominant axis** (both sequence settings beat both
token settings by +4–6pp); +5pp over the 2048-window baseline (66→71%). Full-length
mainly helps PPL (token:full 103 vs token:lt2048 355).

**Stage 2 — winner `sequence:lt2048` across the cliff vs the 2048-window sweep:**

| ratio | full-seq seq·lt2048 | 2048-window | Δ  |
| ----- | -------------------- | ----------- | --- |
| 0.7   | **71%**        | 66%         | +5  |
| 0.6   | **47%**        | 37%         | +10 |
| 0.5   | **36%**        | 20%         | +16 |
| 0.4   | **13%**        | 4%          | +9  |

**The gain grows as compression gets more aggressive** (peak +16pp at r=0.5,
nearly doubling). And it **fixes the looping pathology**: at every ratio
`relaxed == strict` and gen_len stays bounded (~800–1000 tokens, no 5–7× blow-up)
— better calibration makes the model both reach the answer *and* stop. Because it's
a calibration change, it is **orthogonal to the M1 sparse-residual headline** and
should stack (an apples-to-apples M1 × full-seq run is a follow-up).

## 6. How to run — entry points

All drivers live in `scripts/reasoning_aware_compress/`, run in the **`verl` conda
env**, and share the env prefix (the eval grader needs `verl` on the path; ray needs
the verl python):

```bash
PY=/home/yequan/miniconda3/envs/verl/bin/python
ENV="CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl"
# run from the OPD repo root
```

Each driver takes `--cells`/`--ratios`, `--ratio`, `--math-limit`, `--out`, and
emits a JSON of `{cell, math500_acc, c4_ppl, params_nonzero_B, ...}`. Operating
point defaults (retain 0.8, last layer dense, OpenThought3 calib) are baked in;
override via flags.

| Experiment (§)                | Entry point                    | Cells / args                                                    | One-shot launcher              |
| ------------------------------ | ------------------------------ | --------------------------------------------------------------- | ------------------------------ |
| Block D — objective (§3)     | `bi_whitened_svd.py`         | `--cells D0 D1 D2 [D3]`                                       | `GPU=N bash run_abd.sh D`    |
| Block A — rank floor (§3)    | `lr_sparse_residual.py`      | `--cells A0 A1 A2`, `--sparse-frac 0.075`, `--a3-sweep`   | `GPU=N bash run_abd.sh A`    |
| Block B — accumulation (§3)  | `sequential_src.py`          | `--cells B0 B1 B2` (B2 needs `--teacher`)                   | `GPU=N bash run_abd.sh B`    |
| Ratio sweep + trace diff (§4) | `ratio_sweep_trace.py`       | `--ratios 0.8 0.7 0.6 0.5 0.4`, `--probe-set …`            | —                             |
| Trace probe set (§4)          | `trace_diff.py --mode build` | `--n-probes 5 --scan-limit 60` (run ONCE first)               | —                             |
| Trace len-to-correct (§4)     | `analyze_len_to_correct.py`  | `--sweep-dir … --probe-set …` (CPU, post-hoc)               | —                             |
| Full-seq calib study (§5)     | `fullseq_calib_sweep.py`     | `--stage tune --settings …` / `--stage sweep --setting …` | `bash run_fullseq.sh stage1` |

### Canonical invocations

**Block D / A / B** (the mechanism search, retain 0.8):

```bash
$ENV $PY scripts/reasoning_aware_compress/bi_whitened_svd.py \
    --cells D0 D1 D2 --ratio 0.8 --math-limit 100 \
    --out scripts/reasoning_aware_compress/results/blockD/bi_whitened_r0.8.json
$ENV $PY scripts/reasoning_aware_compress/lr_sparse_residual.py \
    --cells A0 A1 A2 --ratio 0.8 --sparse-frac 0.075 --math-limit 100 \
    --out scripts/reasoning_aware_compress/results/blockA/lr_sparse_r0.8.json
# or the chained launcher (D→A→B sequential on one GPU):
GPU=5 bash scripts/reasoning_aware_compress/run_abd.sh all
```

D3/B2 (OPD/teacher) require a **distinct** teacher (else fail-fast):
`--teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`.

**Forward-only ratio sweep + trace diff** (§4) — build the probe set ONCE, then sweep:

```bash
$ENV $PY scripts/reasoning_aware_compress/trace_diff.py --mode build --n-probes 5 \
    --scan-limit 60 --probe-set scripts/reasoning_aware_compress/results/blockT/trace_probe_set.json
$ENV $PY scripts/reasoning_aware_compress/ratio_sweep_trace.py \
    --ratios 0.8 0.7 0.6 0.5 0.4 --math-limit 100 \
    --probe-set scripts/reasoning_aware_compress/results/blockT/trace_probe_set.json \
    --out-dir scripts/reasoning_aware_compress/results/sweep
# post-hoc termination-vs-reasoning metrics (CPU):
PYTHONPATH=src:verl $PY scripts/reasoning_aware_compress/analyze_len_to_correct.py \
    --sweep-dir scripts/reasoning_aware_compress/results/sweep \
    --probe-set scripts/reasoning_aware_compress/results/blockT/trace_probe_set.json
```

**Full-seq calibration study** (§5) — two stages (tune @0.7 → sweep the winner):

```bash
# stage 1: 4 settings (token/sequence × full/lt2048) @0.7, split GPU 2 & 3
bash scripts/reasoning_aware_compress/run_fullseq.sh stage1
# stage 2: best setting at 0.6/0.5/0.4
GPU=2 SET=sequence:lt2048 bash scripts/reasoning_aware_compress/run_fullseq.sh stage2
```

### Shared infra (reused by every driver)

- `compress_common.py` — `build_calib_loader` (calib format), `eval_cell` /
  `eval_math_capture` (MATH + C4 PPL + relaxed/length metrics), `drop_protected_stats`
  (last-layer skip), `load_model`/`count_params`.
- `eval_math500` (`layer_sensitivity.py`) — the standing MATH-500 greedy eval.
- Core compression: `compress_model_with_loader` (dispatcher) →
  `collect_*` (calibration.py) → `svd_llm_v2_compress_model` / `nystrom_compress_model`.

## 7. Status & key knobs

**Headline takeaways**

1. **M1 (rank floor) is the mechanism**; M2 (objective) is null; M3 not tested.
2. The failure is **loss of termination before loss of reasoning** (looping cliff
   ~0.65, reasoning cliff ~0.55).
3. **Sequence-reweighted full-seq calibration** is a second, orthogonal lever that
   pushes the cliff lower and bounds generation length.

**Productionized default (2026-06-04)**: sequence-reweighting + full-length is the
**default calibration format** for all SVD/Nystrom compression — forward, backward,
combined. `reweight="sequence"` on every collector in `src/compress/calibration.py`
(backward hooks now mask `grad_output`; shared mask-aware `_accumulate_cov`, CPU
accumulators + GPU matmul); threaded through `compress_model_with_loader`; loaders
default `length="full"`. **Escape hatch** to reproduce pre-2026-06-04 baselines:
`reweight="token"` + `length="window2048"`. Memory caveat: `full` **truncates to
`max_seq_len=4096`** at batch_size=1 for the backward/combined path (a ~10k-token
backward over the 4B OOMs 96GB). See
[memory: calib-default-sequence-fullseq].

**File map**

- `scripts/reasoning_aware_compress/` drivers: `bi_whitened_svd.py` (D),
  `lr_sparse_residual.py` (A), `sequential_src.py` (B), `ratio_sweep_trace.py`
  (cliff + trace diff), `fullseq_calib_sweep.py` (calib study),
  `trace_diff.py` / `analyze_len_to_correct.py` (trace metrics),
  `compress_common.py` (shared eval/loader/last-layer-skip helpers).
- `src/compress/` core: `hybrid/lr_sparse.py` (M1), `sequential/relinearized.py`
  (M3), `calibration.py` (reweighting + masking), `svd/svd_llm_v2.py`,
  `structured/nystrom.py`.

**Open follow-ups** (not run): A2-method ratio sweep (does the sparse residual push
r\* below 0.65?), A3 budget-split, D3 OPD/teacher cell with a distinct teacher,
M1 × full-seq-calib combination, headline table vs prior-art on
AIME24/AMC23/OlympiadBench.

**Reviewer-risk note**: keep `S` small and ablate it (A0 vs A2) so the M1 claim
stays "we fixed *structured* compression", not "a sparse tail did the work".
