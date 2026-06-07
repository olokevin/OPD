# NP packed_graphed Decode — Host-Glue Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-token host-glue that makes the fully-CUDA-graphed all-layer ZO-NP decode ~46× slower than BP-OPD — by (1) killing the 896 per-token `draw_noise` GPU-kernel launches, (2) removing the per-token `cuda.synchronize()` and batching the per-token D2H u/x captures, and (3) folding `compute_logits`+top-k off the per-token critical path — so decode falls from ~1368s toward the memory-bound floor (~100–140s) **with bit-identical gradients and clean-token parity preserved.**

**Architecture:** The captured CUDA graph wraps only `model()→hidden`; everything expensive (noise refill, sync, D2H captures, eager full-vocab `compute_logits`+`topk`) is eager per token and runs *serially* because of a per-token `torch.cuda.synchronize()`. Measured isolation (`bench_noise_refill_isolation.py`, commit `03bf7bd`): the 896-`draw_noise`/token refill is **74% of decode** (48.0→12.4 ms/token, 3.9× when skipped). This plan replaces the three per-token host-glue bottlenecks with batched/hoisted equivalents, each gated behind the existing F1 bit-parity gate so correctness can't regress. The **single hard invariant** across every task: the noise written to `u_buf[layer][p*n_sample+q]` must stay **bit-identical** to `draw_noise(noise_seed(global_seed, step, layer, slot_rollout_ids[p], q), (d_out,), …)` — the parity-by-construction contract that the whole estimator's correctness rests on.

**Tech Stack:** vLLM 0.11.0 (V1, FLASH_ATTN, enforce_eager), PyTorch CUDA graphs, the existing NP worker extension. conda env `verl` (`/home/yequan/miniconda3/envs/verl/bin/python`). Tests are CPU pytest under `verl/tests/np/`; GPU gates under `scripts/zo_opd/np_checks/` on GPUs 1/2/3 only.

---

## Context the implementer needs (read before starting)

**Worktree:** `/home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed`. Run all commands from there.

**CRITICAL env fact:** `verl` is pip editable-installed pointing at the **MAIN checkout** (`/home/yequan/Project/compression/OPD/verl`), NOT the worktree. CPU pytest from the worktree root works (pytest inserts rootdir → worktree `verl/` shadows the install). **GPU runs via Ray/subprocess import the STALE main-repo verl** unless you prepend `PYTHONPATH=/home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed/verl`. Always print `np_worker_extension.__file__` and assert `"np-alllayer-graphed" in` it before trusting any GPU number.

**GPU policy:** GPUs **1, 2, 3 ONLY** (check `nvidia-smi`; 4–7 are off-limits). One job per GPU.

**The model snapshot the GPU gates use:** `/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (HF_HOME=/data/yequan/huggingface; teacher `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` also cached).

**The files (all paths from worktree root):**
- `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` — the hot loop. Key sites:
  - `_np_fill_u_buf_all_layers_packed` (1322–1351): the C-6 per-token refill — the triple loop (`layers × bucket × n_sample`) of `draw_noise` → **Fix 1 target**.
  - `_np_replay_step_packed` (1353–1430+): per-token body. Line ~1432: the refill call (env-gated by the existing `NP_BENCH_SKIP_NOISE` harness). Line ~1428 area inside: `gs["graph"].replay()` then **`torch.cuda.synchronize()`** then `model.compute_logits(...)` → **Fix 2 + Fix 3 target**.
  - `run_np_decode_packed_graphed` (622–746): the orchestrator token loop. Lines 721–731: per-active-prompt, per-layer `captured_u[p][ln][t] = gs["u_buf"][ln][…].detach().to("cpu").clone()` (and x) — the **28 D2H clones/token** → **Fix 2 target**. Line 724–725: `self._topk_store(block, topk_store_k)` per token → **Fix 3 target**.
  - `_np_capture_step_packed` (1153+): captures the graph (`model()→hidden_buf`); `compute_logits` is OUTSIDE the graph by design.
  - `_topk_store` (1729): `log_softmax(full vocab).topk` + D2H, per token.
  - `_alloc_layer_buffers` (166): allocates the per-layer u/x dicts.
- `verl/verl/trainer/np/seeding.py` — `noise_seed` (15) + `draw_noise` (26): `draw_noise` creates a fresh `torch.Generator(device=device)` + `manual_seed` + a `randn`/`randint` kernel + casts, PER CALL. On `device=cuda` that's ~3–5 tiny kernel launches each; 896/token is the bottleneck.
- `scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py` — the **F1 parity gate** (σ=0 routing, u bit-identity via `torch.equal`, logits rtol≤1e-2, staggered-EOS bit-for-bit). **This is the regression gate for every fix** — it must stay green after each.
- `scripts/zo_opd/np_checks/bench_noise_refill_isolation.py` — the isolation harness (refill ON vs OFF). Reuse its worker-boot + timing pattern for the per-fix microbenchmark.
- `scripts/zo_opd/bench_np_vs_bp.sh` — the full NP-vs-BP one-step bench (re-run at the end for the headline post-fix number).
- `verl/tests/np/test_replay_packed_meta.py` — pins the C-6 refill parity (`test_fill_u_buf_all_layers_packed_parity_with_eager_loop`) and the C-4 pad-meta. **The parity test is the contract Fix 1 must not break.**

**The bit-parity invariant (DO NOT violate in any task):** the F1 gate (b) feeds matching `rollout_ids` to the graphed and eager paths and asserts the captured `u` is `torch.equal` across all (prompt, layer, token). Any change to how/when noise is generated must keep `u_buf[ln][p*n_sample+q] == draw_noise(noise_seed(global_seed, step, ln, slot_rollout_ids[p], q), (d_out,), buf.device, buf.dtype, sample_method)` byte-for-byte. The proven-safe transformation is to **move WHERE the draw happens (CPU vs GPU) and WHEN (hoisted/batched), never the seed key or the math.**

**Locked design decisions (do not re-litigate):**
1. **Fix 1 = CPU-stage + one H2D per layer.** Draw each layer's `[bucket·n_sample, d_out]` block on the **host** (CPU `draw_noise` — cheap, no kernel launch), assemble into one contiguous CPU staging tensor, then **one** `buf.copy_(staging, non_blocking=True)` H2D per layer. 28 H2D copies/token instead of 896 GPU-generator kernels. Bit-identical because the per-(slot,q) `draw_noise` math is unchanged — only `device` moves from cuda→cpu then a copy. (NOT the "one big `randn` per layer" fusion — that would change the seeds and break parity. NOT full-wave pre-generation — ~3.7 GB/wave is too big to materialize.)
2. **Fix 2 = remove the per-token full sync + batch the D2H captures.** Capture u/x into preallocated per-layer device buffers `[bucket·n_sample, max_tokens, d_out]` (u) and `[bucket, max_tokens, d_out]` (x) written in place each token (a cheap device→device slice copy, no sync), then do **one** D2H copy per layer at wave end. Replace the per-token `torch.cuda.synchronize()` with the single sync needed before the host reads logits for sampling (sampling needs row-0 logits on host → keep a minimal per-token logits read, but drop the *full-device* sync in favor of only what sampling requires). The sampling read is the one unavoidable per-token host dependency; everything else (noise refill via Fix 1's H2D, u/x captures) can be issued without blocking.
3. **Fix 3 = keep `compute_logits` per token (sampling needs it) but only over what's needed, and move `_topk_store` off the critical path.** `compute_logits` over full vocab is one GPU GEMM (~ms) and is needed for greedy sampling of row 0; leave it but ensure it's the only per-token host-blocking read. `_topk_store` (log_softmax over full vocab + topk + D2H) can be deferred/batched: store the raw row-block logits into a device buffer per token and run the top-k log-prob extraction in one batched pass at wave end. This is the smallest-impact fix; do it last and only if Fix 1+2 don't already hit the target.
4. **Each fix is independently gated.** After each fix: F1 parity gate MUST stay green (bit-identical u, bit-for-bit tokens), and the isolation/full bench measures the speedup. A fix that breaks F1 parity is reverted, not shipped.

**Estimator-validity reminder:** perturbation is added to layer **output y**, the clean row (index 0 per slot) is never perturbed, and x is captured from the clean row. None of these fixes touch that — they only change *when/where* noise is drawn and *when* signals are copied to host. The captured graph (`model()→hidden`) and the `PerturbedLinear.perturb_all_layers` packed-scatter branch are NOT modified by any task here.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `np_worker_extension.py` | Modify | Fix 1: `_np_fill_u_buf_all_layers_packed` → CPU-stage + single H2D/layer. Fix 2: `_np_replay_step_packed` drop full sync; `run_np_decode_packed_graphed` batch u/x captures into `[…, T, …]` device buffers + one D2H at wave end. Fix 3: defer `_topk_store` to a batched wave-end pass. |
| `verl/tests/np/test_noise_refill_batched.py` | Create | CPU parity tests: the batched/CPU-staged refill is byte-identical to the per-element eager loop (the parity contract), across gaussian/bernoulli/uniform. |
| `verl/tests/np/test_replay_packed_meta.py` | Modify | (Only if a helper signature changes) keep the existing C-6 parity test green; add a regression case for the staged path if a new helper is introduced. |
| `scripts/zo_opd/np_checks/bench_noise_refill_isolation.py` | Modify | Add a third timing mode "refill BATCHED" so the harness reports ON / BATCHED / OFF side-by-side, proving the batched refill closes most of the ON→OFF gap. |
| `scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py` | Reuse (run) | The F1 regression gate — run after every fix; must stay green. |
| `scripts/zo_opd/bench_np_vs_bp.sh` | Reuse (run) | Re-run for the final post-fix NP-vs-BP headline number. |
| `scripts/zo_opd/results/np_vs_bp_alllayer_graphed_postfix.txt` | Create | The post-fix measured result + before/after table. |
| `docs/wiki/zo_np_trainer.md`, `docs/index.md`, `docs/log.md` | Modify | Correct §11.4 (decode was a fixable host-glue artifact, not a fundamental floor) + record the post-fix numbers. |

---

## Dependency order (critical path)

```
Fix 1 (noise refill: CPU-stage + 1 H2D/layer)  ── highest leverage (measured 74% of decode)
  T1.1 CPU parity test  → T1.2 implement  → T1.3 F1 gate green  → T1.4 isolation bench (ON/BATCHED/OFF)
        │
        ▼
Fix 2 (drop per-token full sync + batch u/x D2H captures)
  T2.1 batch-capture buffers  → T2.2 drop full sync  → T2.3 F1 gate green  → T2.4 isolation bench
        │
        ▼
Fix 3 (defer topk_store to wave-end batched pass)   ── do ONLY if Fix1+2 miss the target
  T3.1 defer topk  → T3.2 F1 gate green  → T3.3 isolation bench
        │
        ▼
F (proof + docs)
  F.1 full NP-vs-BP re-bench (post-fix headline)  →  F.2 docs correction (§11.4) + index + log
```

**Hard rule:** the F1 parity gate (`check_alllayer_graphed_parity.py`) must pass after EVERY fix task before moving on. A fix that turns any `torch.equal` u-comparison false, or any clean-token bit-for-bit mismatch, is **reverted** — these are correctness, not perf, regressions.

---

# FIX 1 — Noise refill: CPU-stage + one H2D per layer (the 74% win)

### Task T1.1: CPU parity test — batched refill == per-element eager loop

**Files:**
- Create: `verl/tests/np/test_noise_refill_batched.py`

- [ ] **Step 1: Write the failing parity test.** It asserts a NEW method `_np_fill_u_buf_all_layers_packed_staged` writes bytes bit-identical to the existing per-element `draw_noise` loop, for all three sample methods.

```python
"""Fix 1 parity: the CPU-staged batched noise refill must write byte-identical
noise to the per-element eager loop -- the parity-by-construction contract the
whole ZO-NP estimator rests on (F1 gate (b) asserts torch.equal on captured u)."""
import pytest
import torch

from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
from verl.trainer.np.seeding import noise_seed, draw_noise


@pytest.mark.parametrize("method", ["gaussian", "bernoulli", "uniform"])
def test_staged_refill_bit_identical_to_eager_loop(method):
    we = WorkerExtension.__new__(WorkerExtension)
    bucket_b_pack, n_sample, d_out = 3, 4, 16
    layers = ["L0", "L1"]
    cfg = dict(global_seed=42, sample_method=method)
    slot_rollout_ids = [10, 23, 7]
    step = 5

    staged = {ln: torch.zeros(bucket_b_pack * n_sample, d_out) for ln in layers}
    we._np_fill_u_buf_all_layers_packed_staged(
        staged, cfg, layers, step, slot_rollout_ids, n_sample)

    for ln in layers:
        for p, rid in enumerate(slot_rollout_ids):
            for q in range(n_sample):
                exp = draw_noise(noise_seed(42, step, ln, rid, q), (d_out,),
                                 torch.device("cpu"), torch.float32, method)
                got = staged[ln][p * n_sample + q]
                assert torch.equal(got, exp), f"{method} {ln} slot{p} q{q} mismatch"


def test_staged_refill_matches_existing_unstaged():
    # The staged method must produce the SAME buffer as the existing
    # _np_fill_u_buf_all_layers_packed (the production refill it replaces).
    we = WorkerExtension.__new__(WorkerExtension)
    bucket_b_pack, n_sample, d_out = 2, 8, 32
    layers = ["L0", "L1", "L2"]
    cfg = dict(global_seed=7, sample_method="bernoulli")
    slot_rollout_ids = [0, 1]
    a = {ln: torch.zeros(bucket_b_pack * n_sample, d_out) for ln in layers}
    b = {ln: torch.zeros(bucket_b_pack * n_sample, d_out) for ln in layers}
    we._np_fill_u_buf_all_layers_packed(a, cfg, layers, 3, slot_rollout_ids, n_sample)
    we._np_fill_u_buf_all_layers_packed_staged(b, cfg, layers, 3, slot_rollout_ids, n_sample)
    for ln in layers:
        assert torch.equal(a[ln], b[ln]), f"{ln}: staged != unstaged"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed && CUDA_VISIBLE_DEVICES="" /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_noise_refill_batched.py -q`
Expected: FAIL — `AttributeError: 'WorkerExtension' object has no attribute '_np_fill_u_buf_all_layers_packed_staged'`.

- [ ] **Step 3: Commit the failing test**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/tests/np/test_noise_refill_batched.py
git commit -m "np: failing parity test for CPU-staged batched noise refill (Fix 1)"
```

### Task T1.2: Implement the CPU-staged refill + wire it into the replay

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` (add method after `_np_fill_u_buf_all_layers_packed` ~1351; change the call site in `_np_replay_step_packed` ~1432)

- [ ] **Step 1: Add the staged method.** Insert immediately after `_np_fill_u_buf_all_layers_packed` (after line 1351). It draws each (slot, q) row on the **host** into one contiguous CPU `[bucket·n_sample, d_out]` staging tensor per layer, then does ONE `buf.copy_(staging, non_blocking=True)` H2D per layer. Same `noise_seed`/`draw_noise` math, same seed key → bit-identical; only the device of the draw (cpu) and a single batched copy change.

```python
    def _np_fill_u_buf_all_layers_packed_staged(self, u_buf_dict, np_cfg,
                                                layer_names, step,
                                                slot_rollout_ids, n_sample):
        """Fix 1: CPU-staged batched analog of _np_fill_u_buf_all_layers_packed.

        Identical noise (same noise_seed(global_seed, step, layer, rollout, q)
        key, same draw_noise math, parity-by-construction) but drawn on the HOST
        into one contiguous [bucket_b_pack*n_sample, d_out] CPU staging tensor per
        layer, then copied to the layer's u_buf in ONE H2D transfer. Replaces the
        896 per-(layer,slot,q) cuda-Generator kernel launches/token (the measured
        74%-of-decode bottleneck) with len(layer_names) H2D copies/token. The CPU
        draws are cheap (no kernel launch); only the location of the RNG moved.
        """
        bucket_b_pack = len(slot_rollout_ids)
        cpu = torch.device("cpu")
        gseed = int(np_cfg["global_seed"])
        method = np_cfg["sample_method"]
        for layer_name in layer_names:
            buf = u_buf_dict[layer_name]          # [bucket_b_pack*n_sample, d_out] (device)
            d_out = buf.shape[1]
            # Draw all rows for this layer on the host (cheap), bit-identical to
            # the eager per-element loop, into one contiguous CPU tensor.
            staging = torch.empty(bucket_b_pack * n_sample, d_out,
                                  dtype=buf.dtype, device=cpu)
            for p in range(bucket_b_pack):
                rid = int(slot_rollout_ids[p])
                for q in range(n_sample):
                    seed = noise_seed(gseed, int(step), layer_name, rid, q)
                    staging[p * n_sample + q] = draw_noise(
                        seed, (d_out,), cpu, buf.dtype, method)
            buf.copy_(staging, non_blocking=True)  # ONE H2D per layer
```

- [ ] **Step 2: Run the T1.1 parity tests — expect PASS**

Run: `cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed && CUDA_VISIBLE_DEVICES="" /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_noise_refill_batched.py -q`
Expected: PASS (4 passed: 3 parametrized + 1 staged-vs-unstaged).

- [ ] **Step 3: Wire the staged refill into the replay hot loop.** In `_np_replay_step_packed`, change the C-6 refill call (currently `self._np_fill_u_buf_all_layers_packed(...)` inside the `if not os.environ.get("NP_BENCH_SKIP_NOISE"):` block, ~line 1432) to call the staged version:

Find this block (~1432–1439):
```python
        if not os.environ.get("NP_BENCH_SKIP_NOISE"):
            self._np_fill_u_buf_all_layers_packed(
                gs["u_buf"], np_cfg, layer_names, step, slot_rollout_ids, n_sample)
```
Replace the inner call with:
```python
        if not os.environ.get("NP_BENCH_SKIP_NOISE"):
            self._np_fill_u_buf_all_layers_packed_staged(
                gs["u_buf"], np_cfg, layer_names, step, slot_rollout_ids, n_sample)
```
(Keep the env-gate exactly as-is — the isolation harness still works. Leave the original `_np_fill_u_buf_all_layers_packed` in place: it's still the CPU-parity oracle the staged method is tested against, and the orchestrator's one-time pre-fill under `NP_BENCH_SKIP_NOISE` still calls it.)

- [ ] **Step 4: Run the full CPU suite — expect no regression**

Run: `cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed && CUDA_VISIBLE_DEVICES="" /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/ -q`
Expected: all pass (the new file adds tests; existing parity tests for `_np_fill_u_buf_all_layers_packed` stay green since that method is unchanged).

- [ ] **Step 5: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py verl/tests/np/test_noise_refill_batched.py
git commit -m "np: Fix 1 -- CPU-staged batched noise refill (1 H2D/layer vs 896 GPU kernels/token); bit-identical"
```

### Task T1.3: GPU parity gate — Fix 1 must stay bit-identical

**Files:** Run `scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py` (no edits)

- [ ] **Step 1: Pick a free GPU among 1/2/3** — `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed -n '2,4p'`.

- [ ] **Step 2: Run the F1 parity gate**

Run:
```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=<free 1/2/3> NP_KEEP_CUDA_VISIBLE=1 \
  /home/yequan/miniconda3/envs/verl/bin/python \
  scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py
```
Expected: all three gates PASS — specifically gate (b) `u BIT-IDENTICAL (256/256 torch.equal)` and logits rtol 0.000e+00, and gate (c) staggered-EOS bit-for-bit. Verify the printed `np_worker_extension.__file__` is in the worktree.

- [ ] **Step 3: If gate (b) u-bit-identity FAILS:** the staged refill diverged from the eager seeding — STOP, this is a correctness regression. Diff the staging order vs the eager loop (row index `p*n_sample+q` must match; dtype must be `buf.dtype` on both sides; `draw_noise` device=cpu must produce the same bytes as device=cpu in the test). Do NOT proceed. If it PASSES, the bit-identity holds end-to-end on GPU — proceed.

- [ ] **Step 4: Commit a note (no code)** — record the gate pass in the task log; nothing to commit if no code changed. (Gate scripts already committed.)

### Task T1.4: Microbenchmark — add a BATCHED mode to the isolation harness

**Files:** Modify `scripts/zo_opd/np_checks/bench_noise_refill_isolation.py`

- [ ] **Step 1: Add a "refill BATCHED" timing mode.** The harness currently times refill ON vs OFF via `NP_BENCH_SKIP_NOISE`. The ON path NOW uses the staged refill (Fix 1 wired it in), so re-running already measures BATCHED-vs-OFF. Update the harness labels/output to make this explicit and report the three-way comparison by ALSO timing the OLD unstaged path behind a second env gate.

Add an env gate `NP_BENCH_UNSTAGED_NOISE` read in `_np_replay_step_packed`'s refill block so the harness can force the old per-element path for the "ON (unstaged)" baseline. In `np_worker_extension.py`, change the refill block to:
```python
        if not os.environ.get("NP_BENCH_SKIP_NOISE"):
            if os.environ.get("NP_BENCH_UNSTAGED_NOISE"):
                self._np_fill_u_buf_all_layers_packed(
                    gs["u_buf"], np_cfg, layer_names, step, slot_rollout_ids, n_sample)
            else:
                self._np_fill_u_buf_all_layers_packed_staged(
                    gs["u_buf"], np_cfg, layer_names, step, slot_rollout_ids, n_sample)
```
Then in the harness, add a third call that sets `NP_BENCH_UNSTAGED_NOISE=1` (the old 896-kernel path) and report: `ON-unstaged (896 kernels)` vs `ON-staged (28 H2D)` vs `OFF (no refill)`.

- [ ] **Step 2: Run the three-way isolation on a free GPU**

Run:
```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=<free 1/2/3> NP_KEEP_CUDA_VISIBLE=1 \
  /home/yequan/miniconda3/envs/verl/bin/python \
  scripts/zo_opd/np_checks/bench_noise_refill_isolation.py \
  --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --n-layers 28 --n-prompts 4 --n-sample 8 --max-tokens 128
```
Expected: `ON-staged` ms/token is far below `ON-unstaged` (~48 ms) and approaches `OFF` (~12.4 ms). Paste the three numbers. (Baseline from commit `03bf7bd`: unstaged 48.0, OFF 12.4 → staged should land near ~13–18 ms/token, i.e. most of the 35.6 ms refill tax removed.)

- [ ] **Step 3: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py scripts/zo_opd/np_checks/bench_noise_refill_isolation.py
git commit -m "np: Fix 1 bench -- three-way refill isolation (unstaged/staged/off); staged closes most of the 35.6ms/token tax"
```

---

# FIX 2 — Drop the per-token full sync + batch the D2H u/x captures

### Task T2.1: Pre-allocate per-layer capture buffers; write captures in place per token

**Files:** Modify `np_worker_extension.py` (`run_np_decode_packed_graphed` 622–746)

- [ ] **Step 1: Allocate device-side capture buffers once, before the token loop.** Currently (lines 721–731) each token clones `gs["u_buf"][ln][…].to("cpu").clone()` and `gs["x_buf"][ln][…].to("cpu").clone()` per active prompt per layer = 28 D2H clones/token. Instead, pre-allocate per-layer device buffers `[bucket·n_sample, max_tokens, d_out]` (u) and `[bucket, max_tokens, d_out]` (x), and per token write the just-refilled `gs["u_buf"][ln]` / forward-written `gs["x_buf"][ln]` slices into column `t` in place (device→device, no host sync).

Replace the capture block. Before the `try:` token loop (after line 705), add:
```python
        # Fix 2: capture u/x into per-layer DEVICE buffers indexed by token, then
        # do ONE D2H per layer at wave end -- instead of 28 per-token .to("cpu")
        # clones (which each force a host sync). Sized to max_tokens; sliced to the
        # real per-prompt token count at the end.
        dev = gs["u_buf"][layer_names[0]].device
        cap_u_dev = {ln: torch.empty(
            gs["u_buf"][ln].shape[0], max_tokens, gs["u_buf"][ln].shape[1],
            dtype=gs["u_buf"][ln].dtype, device=dev) for ln in layer_names}
        cap_x_dev = {ln: torch.empty(
            gs["x_buf"][ln].shape[0], max_tokens, gs["x_buf"][ln].shape[1],
            dtype=gs["x_buf"][ln].dtype, device=dev) for ln in layer_names}
```

Then inside the per-active-prompt loop, REPLACE the per-layer clone lines (721–731) so the capture is a device-side slice write (no `.to("cpu")`):
```python
                for p in active_idx:
                    base = p * width
                    block = logits[base:base + width]  # [1+N, vocab]
                    candidate_logits[p].append(
                        self._topk_store(block, topk_store_k))
                    for ln in layer_names:
                        # device->device copy into token-column t (no host sync).
                        cap_u_dev[ln][p * n_sample:(p + 1) * n_sample, t, :].copy_(
                            gs["u_buf"][ln][p * n_sample:(p + 1) * n_sample])
                        cap_x_dev[ln][p, t, :].copy_(gs["x_buf"][ln][p])
                    next_tok = self._np_sample_clean(block[0], sampling_params)
                    clean_tokens[p].append(int(next_tok))
                    if self._np_is_eos(next_tok, sampling_params):
                        states[p]["active"] = False
                    else:
                        self._np_commit_clean(states[p], next_tok)
```

- [ ] **Step 2: After the token loop, do ONE D2H per layer and reshape into the per-prompt `{layer:{t:tensor}}` return shape** the rest of the pipeline expects. Replace the `finally:`/return block (738–746) so it first materializes `captured_u`/`captured_x` from the device buffers with one D2H per layer:
```python
        finally:
            st["mode"] = "off"

        # Fix 2: ONE D2H per layer (whole [rows, max_tokens, d_out] block), then
        # slice into the per-prompt {layer:{t:tensor}} shape on the host. The
        # tensors are identical to the old per-token clones (same values), just
        # copied once instead of 28x/token.
        for ln in layer_names:
            u_host = cap_u_dev[ln].to("cpu")        # [bucket*n_sample, max_tokens, d_out]
            x_host = cap_x_dev[ln].to("cpu")        # [bucket, max_tokens, d_out]
            for p in range(B):
                T_p = len(clean_tokens[p])
                for t in range(T_p):
                    captured_u[p][ln][t] = u_host[
                        p * n_sample:(p + 1) * n_sample, t, :].clone()
                    captured_x[p][ln][t] = x_host[p, t, :].clone()

        return {
            "clean_tokens": clean_tokens,
            "candidate_logits": candidate_logits,
            "captured_x": captured_x,
            "captured_u": captured_u,
        }
```
(Note: `captured_u`/`captured_x` are still initialized as `[{ln: {} for ln in layer_names} for _ in range(B)]` at lines 702–703 — keep those. The per-token in-loop writes to them are removed; they're filled in this wave-end pass instead.)

- [ ] **Step 3: Run the full CPU suite — expect no regression** (this change is GPU-path only; CPU tests don't exercise the orchestrator, but confirm nothing imports-broke).

Run: `cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed && CUDA_VISIBLE_DEVICES="" /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: Fix 2a -- batch u/x captures into per-layer device buffers, one D2H/layer at wave end"
```

### Task T2.2: Remove the per-token full `cuda.synchronize()`

**Files:** Modify `np_worker_extension.py` (`_np_replay_step_packed` — the sync after `gs["graph"].replay()`)

- [ ] **Step 1: Drop the full-device sync; keep only what sampling needs.** In `_np_replay_step_packed`, after `gs["graph"].replay()` there is `torch.cuda.synchronize()` then `logits = model.compute_logits(gs["hidden_buf"])`. The full-device sync forces *everything* serial. The only true per-token host dependency is reading row-0 logits for greedy sampling (`_np_sample_clean` does `int(torch.argmax(...).item())` in the orchestrator). `compute_logits` after `replay()` already orders correctly on the stream; the `.item()` in sampling is itself a sync point that blocks only until the needed result is ready. So remove the blanket `torch.cuda.synchronize()`:

Find (in `_np_replay_step_packed`, the replay block ~after 1440):
```python
        gs["graph"].replay()
        torch.cuda.synchronize()
        logits = model.compute_logits(gs["hidden_buf"])   # [R, vocab]
        return logits
```
Replace with:
```python
        gs["graph"].replay()
        # Fix 2: NO full-device synchronize() here. compute_logits is enqueued on
        # the same stream after replay; the only host read that needs the result
        # is greedy sampling's argmax(...).item() in the orchestrator, which blocks
        # exactly as long as needed -- without forcing the noise refill (H2D),
        # the u/x device-copies, and the next token's metadata writes to serialize.
        logits = model.compute_logits(gs["hidden_buf"])   # [R, vocab]
        return logits
```

- [ ] **Step 2: GPU parity gate — Fix 2 must stay bit-for-bit.** Removing a sync must not change values (only timing), but a sync removal can expose a real ordering bug if any buffer the graph reads is written without proper stream ordering. Run F1:

Run:
```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=<free 1/2/3> NP_KEEP_CUDA_VISIBLE=1 \
  /home/yequan/miniconda3/envs/verl/bin/python \
  scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py
```
Expected: all three gates PASS, u still bit-identical (256/256), staggered-EOS still bit-for-bit. **If tokens diverge or u differs, the sync was load-bearing for an ordering hazard — STOP and report; do not ship.** (Most likely culprit if it fails: the H2D `non_blocking=True` noise copy or the metadata `fill_` racing the replay. If so, the fix is a scoped per-buffer stream-wait, not the blanket sync — but only add that if the gate actually fails.)

- [ ] **Step 3: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: Fix 2b -- drop per-token full cuda.synchronize() (sampling argmax.item() is the only needed host read)"
```

### Task T2.3: Isolation bench — Fix 1+2 combined

**Files:** Run `bench_noise_refill_isolation.py` (no edits)

- [ ] **Step 1: Re-run the isolation harness** (now measuring staged-refill + no-full-sync + batched-captures). Same command as T1.4 Step 2.

- [ ] **Step 2: Record the per-token decode ms** vs the original 48 ms and the 12.4 ms OFF floor. Expect the ON path to be at or below the previous OFF number (the captures and sync that OFF still paid are now also cheaper). Paste the number.

- [ ] **Step 3: Commit a results note** (append to a scratch results file if desired; no code change required).

---

# FIX 3 — Defer top-k store to a batched wave-end pass (smallest impact; do only if needed)

**Decision gate:** Only do Fix 3 if after Fix 1+2 the decode per-token is still meaningfully above the memory-bound floor (i.e. `_topk_store`'s per-token full-vocab `log_softmax`+`topk`+D2H is a measurable fraction). If Fix 1+2 already land decode near the floor, SKIP Fix 3 and note it as unnecessary.

### Task T3.1: Capture raw row-block logits per token; run top-k extraction once at wave end

**Files:** Modify `np_worker_extension.py` (`run_np_decode_packed_graphed` — the `_topk_store` call site)

- [ ] **Step 1: Defer `_topk_store`.** Currently per token: `candidate_logits[p].append(self._topk_store(block, topk_store_k))` — a full-vocab `log_softmax` + `topk` + D2H per active prompt per token. Instead, stash each active prompt's row-block logits into a per-prompt device list per token, and after the token loop run `_topk_store` over the batched stack once.

Replace the per-token append (in the active-prompt loop):
```python
                    candidate_logits[p].append(
                        self._topk_store(block, topk_store_k))
```
with stashing the block (keep a small device buffer; row-block is `[1+N, vocab]`):
```python
                    logits_blocks[p].append(block.clone())  # defer topk to wave end
```
(Allocate `logits_blocks = [[] for _ in range(B)]` before the loop.)

Then after the token loop (in the wave-end pass added in T2.1), build `candidate_logits` once per prompt:
```python
        for p in range(B):
            for blk in logits_blocks[p]:
                candidate_logits[p].append(self._topk_store(blk, topk_store_k))
```

**Parity note:** `_topk_store` is per-(token, row-block) and order-independent across tokens, so deferring it produces byte-identical `(topk_logp, ids)` tuples in the same per-prompt order. **Caution:** stashing `[1+N, vocab].clone()` per token holds `T × (1+N) × vocab` logits on device — for vocab=151936, N=8, that's ~9·151936·2 bytes ≈ 2.7 MB/token, ~2.7 GB at 1024 tokens per prompt. If that's too much memory, instead D2H the raw block per token (one copy) and run `_topk_store` on CPU at the end, OR keep `_topk_store` per-token (Fix 3 skipped). Measure first; prefer skipping Fix 3 if Fix 1+2 suffice.

- [ ] **Step 2: GPU parity gate** — run F1; `candidate_logits` tuples must be byte-identical (gate (b) logits rtol stays 0). Same command as T1.3 Step 2.
Expected: PASS. If memory OOMs at batch64/1024, revert to per-token `_topk_store` (Fix 3 not viable as written) and note it.

- [ ] **Step 3: Commit (only if it passed and helped)**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: Fix 3 -- defer top-k store to a batched wave-end pass (byte-identical tuples)"
```

---

# F — Post-fix proof + docs correction

### Task F.1: Full NP-vs-BP re-bench (the post-fix headline)

**Files:** Run `scripts/zo_opd/bench_np_vs_bp.sh`; Create `scripts/zo_opd/results/np_vs_bp_alllayer_graphed_postfix.txt`

- [ ] **Step 1: Re-run the full one-step bench** at the real scale (batch=64, max_tokens=1024, all 28 layers, packed_graphed, PACK_WIDTH=4) on GPUs 1 (NP) + 2 (BP):
```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
NP_GPU=1 BP_GPU=2 PACK_WIDTH=4 bash scripts/zo_opd/bench_np_vs_bp.sh
```
(Confirm the bench uses `DECODE_MODE=packed_graphed EN_LAYERWISE=false` and the worktree PYTHONPATH for the NP side — it was set in the F3 commit `71c71af`. Verify the worktree module is loaded.)

- [ ] **Step 2: Record the post-fix decode time and total NP one-step**, vs the pre-fix 1368s decode / 2472s total. Compute the new NP/BP ratio. Write `scripts/zo_opd/results/np_vs_bp_alllayer_graphed_postfix.txt` with a BEFORE/AFTER table:
```
                         BEFORE (commit 71c71af)   AFTER (Fix 1+2[+3])
NP decode                1368 s                    <measured>
NP assemble              835 s                     835 s (unchanged; separate fix)
NP one-step total        2472 s                    <measured>
NP/BP ratio (vs 54s)     45.97x                    <measured>
```
Note explicitly: the 835s assemble is a SEPARATE CPU-bound residual (not touched by this plan); the decode improvement is what this plan delivers.

- [ ] **Step 3: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add scripts/zo_opd/bench_np_vs_bp.sh scripts/zo_opd/results/np_vs_bp_alllayer_graphed_postfix.txt
git commit -m "np: post-fix NP-vs-BP re-bench -- decode after host-glue removal (Fix 1+2[+3])"
```

### Task F.2: Correct the docs — decode was a fixable artifact, not a fundamental floor

**Files:** Modify `docs/wiki/zo_np_trainer.md` (§11.4/§11.5), `docs/index.md`, `docs/log.md`

- [ ] **Step 1: Correct §11.4/§11.5.** The pre-fix §11.4 framed decode (1368s) as "the fundamental (1+N)=9× forward floor." Replace that framing with the measured truth: the (1+N) rows are nearly free (memory-bound batch); the 1368s decode was **host glue** — 74% was the 896-`draw_noise`/token refill (isolation: 48→12.4 ms/token, commit `03bf7bd`), plus the per-token full sync + 28 D2H captures/token + eager full-vocab topk. Record the post-fix decode number from F.1 and the Fix 1/2/3 mechanism. Keep the assemble (835s) correctly noted as a separate, still-open CPU-bound residual. Do NOT overclaim — use the verbatim measured post-fix numbers.

- [ ] **Step 2: Update `docs/index.md`** zo_np_trainer row — append a clause: `§11 post-fix: decode host-glue removed (CPU-staged noise refill + no per-token sync + batched captures), decode 1368s→<N>s, NP/BP <ratio>×; the earlier "forward-count floor" framing was an artifact, corrected.` Keep the existing §8–11 text.

- [ ] **Step 3: Append one `docs/log.md` line** — `## [<today>] ingest | NP decode host-glue optimization -- noise refill was 74% of decode; CPU-stage+no-sync+batched-captures cut decode <N>x, NP/BP <ratio>x (was 46x)`.

- [ ] **Step 4: Commit**

```bash
cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
git add docs/wiki/zo_np_trainer.md docs/index.md docs/log.md
git commit -m "docs: correct NP decode framing -- host-glue artifact not forward floor; record post-fix numbers (wiki §11)"
```

---

## Test Plan (verification deliverables)

- **CPU parity (must stay green after every fix):** `pytest verl/tests/np/ -q`. New: `test_noise_refill_batched.py` (staged refill byte-identical to eager loop across gaussian/bernoulli/uniform AND identical to the existing unstaged method).
- **GPU correctness gate (the regression gate — run after EVERY fix task):** `check_alllayer_graphed_parity.py` — σ=0 routing, **u bit-identical (torch.equal, 256/256)**, logits rtol≤1e-2, staggered-EOS bit-for-bit. A fix that turns any of these false is reverted.
- **Per-fix microbenchmark:** `bench_noise_refill_isolation.py` (three-way: unstaged/staged/off) — proves each fix's per-token decode-ms reduction.
- **The headline proof:** F.1 — full NP-vs-BP one-step at batch=64/max_tokens=1024, post-fix decode time + new NP/BP ratio, with the BEFORE/AFTER table.

---

## Self-Review notes

- **Spec coverage:** all three discovered fixes are tasks — Fix 1 (noise refill, the measured 74%/3.9× win: T1.1–T1.4), Fix 2 (per-token sync removal + batched D2H captures: T2.1–T2.3), Fix 3 (deferred topk, gated/optional: T3.1), plus the post-fix proof (F.1) and the docs correction (F.2). The separate 835s CPU-bound assemble is explicitly OUT of scope and flagged as a distinct residual in F.1/F.2.
- **The one invariant, enforced everywhere:** bit-identical noise (`u_buf[ln][p*n_sample+q] == draw_noise(noise_seed(global_seed, step, ln, slot_rollout_ids[p], q), …)`) and bit-for-bit clean tokens. Fix 1 has a dedicated CPU parity test (T1.1) AND the GPU `torch.equal` gate (T1.3); Fix 2/3 are value-preserving (timing/ordering only) and re-gated (T2.2, T3.1). The F1 gate is the hard stop on every fix.
- **No placeholders:** every code step shows the exact insert/replace. The capture-buffer reshape, the staged-refill loop, and the sync removal are all concrete.
- **Memory caveats called out:** Fix 1 full-wave pre-gen rejected (~3.7 GB) in favor of CPU-stage + 1 H2D/layer; Fix 2 capture buffers are `[rows, max_tokens, d_out]` (bounded, sliced to real token count); Fix 3's logits-stash memory risk (~2.7 GB) is flagged with a fallback (skip Fix 3 / D2H-per-token) — measure before committing.
- **Type/signature consistency:** the new method `_np_fill_u_buf_all_layers_packed_staged` has the SAME signature as `_np_fill_u_buf_all_layers_packed`; the capture buffers `cap_u_dev`/`cap_x_dev` and `logits_blocks` are local to the orchestrator; `captured_u`/`captured_x` keep their `{layer:{t:tensor}}` shape so the fit() consumer and `assemble_all_layers_and_apply` are unchanged.
- **Reused machinery (not rebuilt):** `noise_seed`/`draw_noise` (only the device of the draw moves), the F1 parity gate, the isolation harness, the captured graph + `PerturbedLinear` packed-scatter branch (untouched), the `assemble_all_layers_and_apply` consumer contract.
