# NP Packed Multi-Prompt Decode + Throughput Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NP (node-perturbation) trainer decode `B_pack` distinct prompts **simultaneously** in one wide forward (each with its own `1+N` clean+perturbed rails), turning the serial 64-prompt loop into `ceil(64/B_pack)` waves — then benchmark throughput / memory / compute-util across batch sizes {1,2,4,8,16} × rails {8,16,64} and compare one NP step against one standard BP-based OPD step.

**Architecture:** Add a **packed** decode driver beside the existing per-prompt `graphed` driver (which stays callable as the parity oracle). The packed driver builds a flat `R = B_pack × (1+N)`-row batch per token, where each prompt owns a contiguous row block, keeps its **own** disjoint scratch-KV slice and per-row `attn_metadata` (seq_lens / block_table / slot_mapping), and its `N` perturbed rails write no KV (slot `-1`). The token loop stays autoregressive (unavoidable); the rails are already parallel within a prompt; packing adds the cross-prompt parallelism. NP math (`assemble_layer_delta`, `apply_node_update`, teacher scoring, seeding) is **untouched** — packing is a student-side decode-tiling change only, verified by serial-vs-packed numerical parity.

**Tech Stack:** vLLM 0.11.0 (V1 engine, FLASH_ATTN backend, `enforce_eager`), PyTorch CUDA, Ray single-controller, Hydra config. Student Qwen3-1.7B, teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500. conda env `verl` (`/home/yequan/miniconda3/envs/verl/bin/python`).

---

## Context the implementer needs (read before starting)

**Where the code lives (all paths from repo root `/home/yequan/Project/compression/OPD`):**

- `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` — the vLLM `WorkerExtension` (runs on the GPU worker). Holds `PerturbedLinear`, the per-prompt decode drivers, the attn-metadata builders, and the NP math entry points. **This file is where ~all new decode code goes.**
- `verl/verl/trainer/np/ray_trainer.py` — the Ray single-controller trainer. `RayNPTrainer.fit()` has the serial `for b in range(batch_size)` loop (line ~515) that this plan replaces with a wave loop.
- `verl/verl/trainer/np/grad_estimator.py` — `sample_scale`, `accumulate_delta_w`. **Do not touch** (NP math).
- `verl/verl/trainer/np/teacher_scorer.py` — `TeacherScorer.score_rollout`. **Do not touch** (per-prompt, called once per clean rollout).
- `verl/verl/trainer/np/seeding.py` — `noise_seed(global_seed, step, layer, rollout, q)` + `draw_noise(seed, shape, device, dtype, method)`. **Do not touch** (parity-by-construction depends on identical calls).
- `verl/verl/trainer/config/np_trainer.yaml` — Hydra config; add `pack_width` here.
- `scripts/zo_opd/opd_math_np.sh` — NP launcher (env-var driven; already has `DECODE_MODE`, `USE_CUDA_GRAPH`, `BATCH_SIZE`).
- `scripts/zo_opd/opd_math_ref.sh` — the **BP-OPD baseline** launcher (standard verl PPO, `token_reward_direct`), greedy, 64 prompts/step. The throughput comparison's "standard BP-based OPD" point.
- `scripts/zo_opd/np_checks/` — GPU check/bench scripts (`check_decode_sigma0.py`, `check_graphed_parity.py`, `bench_n_scaling.py`). New benchmark scripts go here.
- `verl/tests/np/` — CPU pytest suite (no GPU). Pure-Python helpers get unit tests here.

**Key existing methods the packed path generalizes (single-prompt → B_pack-prompt):**

| Single-prompt (exists) | Packed (new) | What changes |
|---|---|---|
| `run_np_decode_graphed(pid, sp, layer, np_cfg, rollout, use_cuda_graph)` | `run_np_decode_packed(pids, sp, layer, np_cfg, rollout_ids)` | takes a **list** of prompts + rollout_ids; returns **lists** of per-prompt outputs |
| `_np_prefill(model, device, prompt_token_ids)` → `state` | `_np_prefill_packed(model, device, list_of_pids)` → `list_of_states` | B_pack disjoint scratch-KV slices; prefill each prompt |
| `_np_step_forward_graph(model, device, state, n_sample)` → `[1+N, vocab]` | `_np_step_forward_packed(model, device, states, n_sample, active)` → `[R, vocab]` | R = sum over active prompts of `(1+N)` rows |
| `_np_build_attn_metadata(state, query_lens, seq_lens, slot_mapping, positions)` | `_np_build_attn_metadata_packed(states, ...)` | per-row (not shared) seq_lens / block_table / positions |

**The `(1+N)` row layout per prompt (from `_np_step_forward`, lines 745-758):** row 0 = clean (writes KV at `clean_slot`), rows 1..N = perturbed rails (slot `-1` = PAD, write no KV), all rows feed the same `q_token` at the same `q_pos`, share the prompt's prefix KV. Packed simply concatenates these blocks across prompts.

**Scope decision (from the superseded packing spec `docs/superpowers/specs/2026-06-03-np-v2-design.md`, §3 M1):** implement **eager packed** only (no CUDA-graph capture of the packed forward). Rationale: the per-prompt graph already removed eager-dispatch overhead for the rails; the dominant cost at scale is the *serial repetition across prompts*, which packing fixes without graphs. CUDA-graphing the packed forward is a possible follow-up, explicitly **out of scope** here. The existing per-prompt `graphed` driver is the parity oracle.

**Memory budget (verified for this plan):** KV per token (Qwen3-1.7B, bf16, all 28 layers) = 112 KiB. Packed scratch KV at 2048 tok/prompt: B_pack=8 → 1.9 GB, B_pack=16 → 3.8 GB — fits the student's 0.30-fraction (~28 GB) with room. The batched-assemble GEMM workspace (already on GPU as of the prior fix) adds ~2.2 GB transient at nsig≈65k.

**Parity tolerances (from `check_graphed_parity.py`):** `u` must be **bit-identical** (same `noise_seed`+`draw_noise` key → same bytes). `logits` / `x` within `rtol=1e-2, atol=1e-2` (bf16 reduction-order between serial and batched matmul is the only legitimate difference).

**Env to run anything on GPU:** from `verl/`, set `CUDA_VISIBLE_DEVICES=N NP_KEEP_CUDA_VISIBLE=1` (the uni/in-process executor needs the pin kept), `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0`. Free GPUs at plan time: 0,1,4,5,6,7 (2,3 in use by another user). Model path: `Qwen/Qwen3-1.7B` (HF cache at `/data/yequan/huggingface`).

**Run tests with:** `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/ -q` (CPU unit tests don't need the GPU but the import path resolves under the verl env).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` | Modify | New: `run_np_decode_packed`, `_np_prefill_packed`, `_np_step_forward_packed`, `_np_build_attn_metadata_packed`, and pure-Python helpers `_packed_row_blocks`, `_assign_rollout_ids`. Untouched: all V1/V2 single-prompt methods, `PerturbedLinear`, `assemble_layer_delta`, `apply_node_update`, broadcast. |
| `verl/verl/trainer/np/ray_trainer.py` | Modify | `fit()` gains a `decode_mode == "packed"` branch: chunk `batch_size` prompts into waves of `pack_width`, one `run_np_decode_packed` RPC per wave, accumulate all waves' signals into the same four lists → one `assemble_and_apply` per step (math identical to serial). |
| `verl/verl/trainer/config/np_trainer.yaml` | Modify | Add `pack_width: 8`. Extend `decode_mode` comment to include `packed`. |
| `scripts/zo_opd/opd_math_np.sh` | Modify | Add `PACK_WIDTH` env → `np.pack_width`; allow `DECODE_MODE=packed`. |
| `verl/tests/np/test_packed_helpers.py` | Create | CPU unit tests for `_packed_row_blocks`, `_assign_rollout_ids` (row indexing, wave chunking, seed identity vs serial). |
| `scripts/zo_opd/np_checks/check_packed_parity.py` | Create | GPU parity gate: serial (per-prompt `graphed`) vs `packed` on the same prompts/seeds — per-prompt per-token `clean_tokens` identical, `u` bit-identical, `logits`/`x` within tol. |
| `scripts/zo_opd/np_checks/check_packed_sigma0.py` | Create | GPU σ=0 gate: packed clean tokens match stock greedy `LLM.generate` for **every** prompt in the wave (proves per-prompt prefix routing). |
| `scripts/zo_opd/np_checks/bench_packed_grid.py` | Create | Throughput/memory/util grid: batch∈{1,2,4,8,16} × rails∈{8,16,64}; reports s/step, tok/s, peak GPU mem, SM util. |
| `scripts/zo_opd/bench_np_vs_bp.sh` | Create | One-step wall-clock + peak-mem comparison: NP packed (best config) vs BP-OPD (`opd_math_ref.sh`), both batch=64 / max_tokens=1024 / greedy. |
| `docs/wiki/zo_np_trainer.md` | Modify | Add §10 documenting the packed driver + the benchmark results once produced. |

---

## Task 1: Pure-Python packed-layout helpers (CPU, TDD)

These two functions encode the row-block indexing and the seed-identity rule. Isolating them as pure functions lets us unit-test the fiddly index math with no GPU.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` (add two module-level functions near `assemble_layer_delta`)
- Test: `verl/tests/np/test_packed_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/np/test_packed_helpers.py`:

```python
import pytest
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _packed_row_blocks,
    _assign_rollout_ids,
)


def test_packed_row_blocks_layout():
    # B_pack=3 prompts, N=2 perturbed rails -> each prompt owns (1+N)=3 rows.
    blocks = _packed_row_blocks(b_pack=3, n_sample=2)
    # prompt p: clean row = p*(1+N), perturbed rows = next N.
    assert blocks == [
        {"clean": 0, "perturbed": [1, 2]},
        {"clean": 3, "perturbed": [4, 5]},
        {"clean": 6, "perturbed": [7, 8]},
    ]
    # total rows R = B_pack*(1+N)
    assert blocks[-1]["perturbed"][-1] + 1 == 3 * (1 + 2)


def test_assign_rollout_ids_matches_serial_global_index():
    # Serial loop seeds prompt b with rollout_idx = step*batch_size + b (spec §4.6).
    # Wave-chunked packing must reproduce the SAME per-prompt rollout_idx so the
    # noise draw is identical -> parity-by-construction.
    ids = _assign_rollout_ids(step=2, batch_size=8, n_rollout=1)
    assert ids == [16, 17, 18, 19, 20, 21, 22, 23]  # 2*8 + b


def test_assign_rollout_ids_n_rollout_gt_1():
    # n_rollout>1: batch_size*n_rollout slots, each (prompt,rollout) a distinct id.
    ids = _assign_rollout_ids(step=0, batch_size=2, n_rollout=2)
    # 2 prompts x 2 rollouts = 4 slots; ids must be distinct and stable.
    assert len(ids) == 4
    assert len(set(ids)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_packed_helpers.py -q`
Expected: FAIL with `ImportError: cannot import name '_packed_row_blocks'`.

- [ ] **Step 3: Write minimal implementation**

In `np_worker_extension.py`, add near the bottom (module level, after `assemble_layer_delta`):

```python
def _packed_row_blocks(b_pack, n_sample):
    """Row layout for a packed wave (spec §4.1). Each prompt p owns a contiguous
    block of (1+n_sample) rows: row p*(1+n_sample) is its clean row, the next
    n_sample are its perturbed rails. Returns a list (len b_pack) of
    {"clean": int, "perturbed": [int, ...]}."""
    width = 1 + int(n_sample)
    blocks = []
    for p in range(int(b_pack)):
        base = p * width
        blocks.append({"clean": base,
                       "perturbed": list(range(base + 1, base + width))})
    return blocks


def _assign_rollout_ids(step, batch_size, n_rollout):
    """Stable per-(prompt,rollout) seed identity (spec §4.6). The serial loop
    seeds prompt b with rollout_idx = step*batch_size + b; packing must reproduce
    the SAME id per prompt so draw_noise is identical (parity-by-construction).
    For n_rollout>1, each (prompt,rollout) slot gets a distinct id."""
    base = int(step) * int(batch_size)
    if int(n_rollout) <= 1:
        return [base + b for b in range(int(batch_size))]
    ids = []
    for b in range(int(batch_size)):
        for r in range(int(n_rollout)):
            ids.append((base + b) * int(n_rollout) + r)
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -m pytest verl/tests/np/test_packed_helpers.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add verl/tests/np/test_packed_helpers.py verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: packed-layout helpers (_packed_row_blocks, _assign_rollout_ids) + CPU tests"
```

---

## Task 2: `_np_prefill_packed` — B_pack disjoint scratch-KV slices

Generalize `_np_prefill` to lay out `b_pack` **disjoint** high-indexed scratch-KV regions (one per prompt) and prefill each prompt's prefix into its own region. This is the correctness keystone: prompts must not share KV blocks.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`

- [ ] **Step 1: Write the implementation**

In `np_worker_extension.py`, add this method to `WorkerExtension` (right after `_np_prefill`):

```python
def _np_prefill_packed(self, model, device, list_of_prompt_ids):
    """Prefill B_pack prompts, each into its OWN disjoint high-indexed scratch-KV
    slice. Returns a list of per-prompt state dicts (same shape as _np_prefill's
    state, plus 'active': True). Fails fast if the B_pack slices don't fit the
    GPU block pool.

    Each prompt p gets blocks_per_prompt = ceil(max_model_len/block_size) blocks,
    carved from the TOP of the pool downward and disjoint across prompts:
      prompt 0: [num_gpu_blocks - 1*bpp, num_gpu_blocks)
      prompt 1: [num_gpu_blocks - 2*bpp, num_gpu_blocks - 1*bpp)
      ...
    so no two prompts' clean rows ever write the same KV slot."""
    mr = self.model_runner
    block_size = int(mr.cache_config.block_size)
    num_gpu_blocks = int(mr.cache_config.num_gpu_blocks)
    max_blocks = int(
        mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)

    b_pack = len(list_of_prompt_ids)
    blocks_per_prompt = min(
        (int(mr.max_model_len) + block_size - 1) // block_size, max_blocks)
    # Fail fast rather than corrupt KV (spec §4.2).
    assert b_pack * blocks_per_prompt <= num_gpu_blocks, (
        f"packed scratch KV does not fit: b_pack={b_pack} x "
        f"blocks_per_prompt={blocks_per_prompt} = {b_pack*blocks_per_prompt} "
        f"> num_gpu_blocks={num_gpu_blocks}. Lower pack_width or max_tokens.")

    states = []
    for p, prompt_token_ids in enumerate(list_of_prompt_ids):
        hi = num_gpu_blocks - p * blocks_per_prompt
        lo = hi - blocks_per_prompt
        block_ids = list(range(lo, hi))
        prompt_len = len(prompt_token_ids)
        state = {
            "prompt_token_ids": list(prompt_token_ids),
            "committed_tokens": [],
            "prompt_len": prompt_len,
            "kv_cursor": max(0, prompt_len - 1),
            "block_ids": block_ids,
            "block_size": block_size,
            "active": True,
        }
        if prompt_len > 1:
            pre_len = prompt_len - 1
            slot_mapping = [
                self._np_slot_for_position(block_ids, block_size, pos)
                for pos in range(pre_len)
            ]
            positions = list(range(pre_len))
            attn_meta, total = self._np_build_attn_metadata(
                state, [pre_len], [pre_len], slot_mapping, positions)
            prev_mode = self.np_state.get("mode", "off")
            self.np_state["mode"] = "off"
            try:
                with torch.no_grad():
                    self._np_run_forward(
                        model, device, prompt_token_ids[:pre_len], positions,
                        attn_meta, total)
            finally:
                self.np_state["mode"] = prev_mode
        states.append(state)
    return states
```

- [ ] **Step 2: Sanity-check import (no behavior yet to test in isolation)**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -c "from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension; print(hasattr(WorkerExtension,'_np_prefill_packed'))"`
Expected: prints `True` (and no import error). This method is exercised by the GPU gate in Task 6; it has no standalone CPU test because it touches `model_runner`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: _np_prefill_packed -- B_pack disjoint scratch-KV slices with fail-fast guard"
```

---

## Task 3: `_np_build_attn_metadata_packed` — per-row seq_lens / block_table / slots

Generalize the attn-metadata builder from `num_reqs = 1+N` (one prompt, shared seq/block) to `num_reqs = R = sum_active (1+N)`, where seq_lens, block_table rows, and positions are **per-prompt** (rows in the same prompt block share, different prompts differ).

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`

- [ ] **Step 1: Write the implementation**

Add this method to `WorkerExtension` (after `_np_build_attn_metadata`). It mirrors `_np_build_attn_metadata` but takes already-per-row arrays and per-prompt block_ids:

```python
def _np_build_attn_metadata_packed(self, per_row_block_ids, query_lens,
                                   seq_lens, slot_mapping, positions_cpu):
    """Packed attn_metadata: num_reqs = R rows, each row carrying its OWN
    seq_len / block_table / slot. Generalizes _np_build_attn_metadata (which
    assumed one shared block_ids for all 1+N rows) to B_pack prompts.

    per_row_block_ids: list (len R) of that row's prompt's block_ids list.
    query_lens/seq_lens/slot_mapping/positions_cpu: per-row arrays (len R).
    Rows in the same prompt block carry identical block_ids + seq_len; rows of
    different prompts differ. Returns (attn_metadata, total_tokens)."""
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    mr = self.model_runner
    device = mr.device

    num_reqs = len(query_lens)
    total_tokens = int(sum(query_lens))
    max_query_len = int(max(query_lens))
    max_seq_len = int(max(seq_lens))

    qsl_np = np.zeros(num_reqs + 1, dtype=np.int32)
    qsl_np[1:] = np.cumsum(np.asarray(query_lens, dtype=np.int32))
    qsl_cpu = torch.from_numpy(qsl_np)
    qsl_gpu = qsl_cpu.to(device)

    sl_cpu = torch.from_numpy(np.asarray(seq_lens, dtype=np.int32))
    sl_gpu = sl_cpu.to(device)

    max_blocks = int(
        mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)
    bt = torch.zeros((num_reqs, max_blocks), dtype=torch.int32, device=device)
    for row, bids in enumerate(per_row_block_ids):
        bt[row, : len(bids)] = torch.tensor(
            bids, dtype=torch.int32, device=device)

    slot_mapping_gpu = torch.tensor(
        slot_mapping, dtype=torch.int64, device=device)
    num_computed_tokens_cpu = torch.tensor(
        [s - q for s, q in zip(seq_lens, query_lens)], dtype=torch.int32)

    common = CommonAttentionMetadata(
        query_start_loc=qsl_gpu, query_start_loc_cpu=qsl_cpu,
        seq_lens=sl_gpu, seq_lens_cpu=sl_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs, num_actual_tokens=total_tokens,
        max_query_len=max_query_len, max_seq_len=max_seq_len,
        block_table_tensor=bt, slot_mapping=slot_mapping_gpu, causal=True,
    )

    attn_metadata = {}
    for group_id, _ in enumerate(mr.kv_cache_config.kv_cache_groups):
        for attn_group in mr.attn_groups[group_id]:
            meta = attn_group.get_metadata_builder().build(
                common_prefix_len=0, common_attn_metadata=common,
                fast_build=True)
            for layer in attn_group.layer_names:
                attn_metadata[layer] = meta
    return attn_metadata, total_tokens
```

- [ ] **Step 2: Sanity-check import**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -c "from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension; print(hasattr(WorkerExtension,'_np_build_attn_metadata_packed'))"`
Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: _np_build_attn_metadata_packed -- per-row seq_lens/block_table/slot for B_pack prompts"
```

---

## Task 4: `_np_step_forward_packed` — one wide forward of R rows

Build the `R = sum_active (1+N)`-row input, run one forward in `perturb_graph` mode (reusing the existing `u_buf` perturbation op), and slice out each active prompt's `1+N` logits block. The perturbation buffer `u_buf` must hold `N` rows **per active prompt** with that prompt's noise.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`

- [ ] **Step 1: Extend `PerturbedLinear.perturb_graph` to handle multiple prompt blocks**

The current `perturb_graph` mode (lines ~76-96) adds `sigma*u_buf` to a single `[n_clean : n_clean+N]` slice. For packed, perturbed rows are **scattered** per prompt block. Replace the single-slice add with an index-based add driven by `st["perturbed_row_idx"]` (a LongTensor of all perturbed row indices) and `st["clean_row_idx"]`. Modify the `perturb_graph` branch in `PerturbedLinear.forward`:

Find (lines ~76-96):

```python
        if mode == "perturb_graph" and self.name == st["layer"]:
            st["x_buf"].copy_(x[0])
            sigma = st["sigma"]              # captured scalar (python float)
            u_buf = st["u_buf"]              # [N, d_out], host-refilled, persistent
            y[n_clean:n_clean + u_buf.shape[0]] = (
                y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            return _repack(y, bias, was_tuple)
```

Replace with:

```python
        if mode == "perturb_graph" and self.name == st["layer"]:
            sigma = st["sigma"]              # python float
            u_buf = st["u_buf"]              # [n_pert_rows, d_out] host-refilled
            pri = st.get("perturbed_row_idx")  # LongTensor of perturbed rows, or None
            if pri is None:
                # single-prompt path (unchanged): perturbed rows are [n_clean:n_clean+N]
                st["x_buf"].copy_(x[0])
                y[n_clean:n_clean + u_buf.shape[0]] = (
                    y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            else:
                # packed path: x_buf holds one clean-input row per prompt; capture
                # each prompt's clean-row input, and scatter-add u_buf to the
                # (scattered) perturbed rows. u_buf row i corresponds to pri[i].
                cri = st["clean_row_idx"]    # LongTensor [b_pack] clean rows
                st["x_buf"].copy_(x[cri])    # [b_pack, d_in]
                y[pri] = y[pri] + sigma * u_buf
            return _repack(y, bias, was_tuple)
```

- [ ] **Step 2: Write `_np_step_forward_packed`**

Add to `WorkerExtension` (after `_np_step_forward_graph`):

```python
def _np_step_forward_packed(self, model, device, states, n_sample, u_buf,
                            x_buf, clean_row_idx, perturbed_row_idx):
    """One wide forward of R = (#active prompts)*(1+n_sample) rows.

    states: list of per-prompt state dicts (only ACTIVE prompts passed in).
    u_buf:  [#active*n_sample, d_out] host-refilled perturbation buffer; row order
            matches perturbed_row_idx (prompt-major: prompt0's N rows, then
            prompt1's N, ...). x_buf: [#active, d_in] receives each prompt's
            clean-row input. clean_row_idx/perturbed_row_idx: LongTensors of the
            row positions in the packed batch (from _packed_row_blocks).

    Returns [R, vocab] logits. Caller slices each prompt's clean row (row
    p*(1+n_sample)) for sampling and its N perturbed rows for L_q."""
    width = 1 + n_sample
    input_ids, positions, slot_mapping, seq_lens, query_lens = [], [], [], [], []
    per_row_block_ids = []
    for st_p in states:
        block_ids = st_p["block_ids"]
        block_size = st_p["block_size"]
        prompt_len = st_p["prompt_len"]
        q_pos = st_p["kv_cursor"]
        if q_pos < prompt_len:
            q_token = st_p["prompt_token_ids"][q_pos]
        else:
            q_token = st_p["committed_tokens"][q_pos - prompt_len]
        clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
        # clean row writes KV; perturbed rows PAD(-1) -> reshape_and_cache skips.
        input_ids += [q_token] * width
        positions += [q_pos] * width
        slot_mapping += [clean_slot] + [-1] * n_sample
        seq_lens += [q_pos + 1] * width
        query_lens += [1] * width
        per_row_block_ids += [block_ids] * width

    attn_meta, total = self._np_build_attn_metadata_packed(
        per_row_block_ids, query_lens, seq_lens, slot_mapping, positions)

    # np_state carries the scatter indices so PerturbedLinear adds noise to the
    # right rows. u_buf/x_buf already installed by run_np_decode_packed.
    st = self.np_state
    st["clean_row_idx"] = clean_row_idx
    st["perturbed_row_idx"] = perturbed_row_idx
    with torch.no_grad():
        hidden = self._np_run_forward(
            model, device, input_ids, positions, attn_meta, total)
        logits = model.compute_logits(hidden)
    return logits
```

- [ ] **Step 3: Sanity-check import**

Run: `CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python -c "from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension, PerturbedLinear; print(hasattr(WorkerExtension,'_np_step_forward_packed'))"`
Expected: `True`.

- [ ] **Step 4: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: _np_step_forward_packed + PerturbedLinear scatter-add for packed perturbed rows"
```

---

## Task 5: `run_np_decode_packed` — the packed decode driver (RPC entry)

The top-level RPC: prefill all prompts, then per token refill `u_buf` (per-prompt noise) and `x_buf`, run one packed forward, slice per-prompt clean+perturbed logits, sample each clean row, capture `(candidate_logits, u, x)` per prompt, advance, drop prompts that hit EOS (active mask). Returns **lists** of per-prompt outputs.

**Files:**
- Modify: `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`

- [ ] **Step 1: Write `run_np_decode_packed`**

Add to `WorkerExtension` (after `run_np_decode_graphed`):

```python
def run_np_decode_packed(self, list_of_prompt_ids, sampling_params, layer_name,
                         np_cfg, rollout_ids):
    """Packed decode for B_pack prompts simultaneously (spec §4). Same per-prompt
    output contract as run_np_decode_graphed, but returns LISTS indexed by prompt:
      clean_tokens[p], candidate_logits[p], captured_x[p], captured_u[p].

    All prompts share one wide forward per token; each keeps its own prefix KV
    (disjoint scratch slices) and its own noise (seeded by rollout_ids[p], so
    identical to what the serial loop drew for that prompt -> parity). A prompt
    that hits EOS is marked inactive and dropped from the next forward; its
    captured signals stop at its EOS token."""
    st = self._ensure_np_state()
    mr = self.model_runner
    model = mr.model
    device = mr.device
    n_sample = int(np_cfg["n_sample"])
    max_tokens = int(np_cfg["max_tokens"])
    sigma = float(np_cfg["sigma"])

    states = self._np_prefill_packed(model, device, list_of_prompt_ids)
    b_pack = len(states)

    # Resolve the perturbed layer's output / input widths to size buffers.
    wrapped = self.np_modules[layer_name]
    weight = wrapped.wrapped.weight          # [d_out, d_in]
    assert weight.is_floating_point(), (
        f"perturb_graph needs a floating weight (got {weight.dtype}) for "
        f"u_buf dtype parity. Layer {layer_name!r}.")
    d_out = int(weight.shape[0])
    d_in = int(weight.shape[1])
    buf_dtype = weight.dtype

    # Per-prompt outputs.
    clean_tokens = [[] for _ in range(b_pack)]
    candidate_logits = [[] for _ in range(b_pack)]
    captured_u = [{} for _ in range(b_pack)]
    captured_x = [{} for _ in range(b_pack)]

    st.update({
        "mode": "perturb_graph",
        "layer": layer_name,
        "sigma": sigma,
        "n_clean_rows": 1,
    })
    try:
        for t in range(max_tokens):
            active_idx = [p for p in range(b_pack) if states[p]["active"]]
            if not active_idx:
                break
            n_active = len(active_idx)
            blocks = _packed_row_blocks(n_active, n_sample)  # row layout for ACTIVE prompts

            # Buffers sized to the active set this token.
            u_buf = torch.zeros(n_active * n_sample, d_out, device=device,
                                dtype=buf_dtype)
            x_buf = torch.zeros(n_active, d_in, device=device, dtype=buf_dtype)
            clean_row_idx = torch.tensor([blk["clean"] for blk in blocks],
                                         dtype=torch.long, device=device)
            perturbed_row_idx = torch.tensor(
                [r for blk in blocks for r in blk["perturbed"]],
                dtype=torch.long, device=device)
            st["u_buf"] = u_buf
            st["x_buf"] = x_buf

            # Refill u_buf: prompt-major, each active prompt's N rows seeded by
            # ITS rollout_id (parity with serial). u_buf row (i*n_sample + q).
            for i, p in enumerate(active_idx):
                for q in range(n_sample):
                    seed = noise_seed(int(np_cfg["global_seed"]), int(t),
                                      layer_name, int(rollout_ids[p]), q)
                    u = draw_noise(seed, (d_out,), device, buf_dtype,
                                   np_cfg["sample_method"])
                    u_buf[i * n_sample + q].copy_(u)

            active_states = [states[p] for p in active_idx]
            logits = self._np_step_forward_packed(
                model, device, active_states, n_sample, u_buf, x_buf,
                clean_row_idx, perturbed_row_idx)  # [R, vocab]

            # Per active prompt: slice its (1+N) block, sample, capture, advance.
            for i, p in enumerate(active_idx):
                base = blocks[i]["clean"]
                block = logits[base:base + 1 + n_sample]  # [1+N, vocab]
                candidate_logits[p].append(block.detach().to("cpu"))
                # u for this prompt = its N rows of u_buf.
                captured_u[p][t] = u_buf[i * n_sample:(i + 1) * n_sample
                                         ].detach().to("cpu").clone()
                captured_x[p][t] = x_buf[i].detach().to("cpu").clone()
                next_tok = self._np_sample_clean(block[0], sampling_params)
                clean_tokens[p].append(int(next_tok))
                if self._np_is_eos(next_tok, sampling_params):
                    states[p]["active"] = False
                else:
                    self._np_commit_clean(states[p], next_tok)
    finally:
        st["mode"] = "off"
        for k in ("u_buf", "x_buf", "perturbed_row_idx", "clean_row_idx"):
            st[k] = None

    return {
        "clean_tokens": clean_tokens,
        "candidate_logits": candidate_logits,
        "captured_x": captured_x,
        "captured_u": captured_u,
    }
```

- [ ] **Step 2: Register the new RPC method name**

The worker logs an "Injected ... for extended collective_rpc calls [...]" list; new public methods on `WorkerExtension` are auto-available via `collective_rpc`. No registration edit needed — verify by grepping the method is public (no leading underscore on `run_np_decode_packed`). Confirm:

Run: `grep -n "def run_np_decode_packed" verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py
git commit -m "np: run_np_decode_packed -- B_pack-prompt simultaneous decode with active mask"
```

---

## Task 6: GPU σ=0 packed gate — per-prompt prefix routing is correct

With σ=0 every perturbed row equals the clean row, and each prompt's packed clean tokens must match stock greedy `LLM.generate` — proving the per-prompt block_ids/seq_lens/slot routing sends each prompt to **its own** prefix (no cross-prompt KV bleed). This is the single most important correctness check for the packing.

**Files:**
- Create: `scripts/zo_opd/np_checks/check_packed_sigma0.py`

- [ ] **Step 1: Write the gate script**

Create `scripts/zo_opd/np_checks/check_packed_sigma0.py`:

```python
"""GPU gate: with sigma=0 the packed decode must reproduce stock greedy
generate() for EVERY prompt in the wave -- proving per-prompt prefix routing
(disjoint scratch KV, per-row seq_lens/block_table) is correct.

Usage (1 GPU + small model):
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_packed_sigma0.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --n-sample 4 --b-pack 4
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--b-pack", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    prompts = [
        "What is 2+2? Answer:",
        "Compute 7*8. Answer:",
        "What is the capital of France? Answer:",
        "Differentiate x^3. Answer:",
        "What is 10/2? Answer:",
        "Name a prime number. Answer:",
        "What is 3-1? Answer:",
        "Square root of 81? Answer:",
    ][: args.b_pack]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]

    # Stock greedy references, one per prompt.
    refs = []
    for pid in pids:
        r = llm.generate({"prompt_token_ids": pid},
                         SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                         use_tqdm=False)
        refs.append(list(r[0].outputs[0].token_ids))

    llm.collective_rpc("install_perturb_layers", args=([args.layer],))
    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=42, sigma=0.0, sample_method="gaussian")
    rollout_ids = list(range(len(pids)))
    out = llm.collective_rpc(
        "run_np_decode_packed",
        args=(pids, SamplingParams(temperature=0.0), args.layer, np_cfg,
              rollout_ids))[0]

    for p in range(len(pids)):
        np_tok = out["clean_tokens"][p]
        ref = refs[p]
        assert np_tok[: len(ref)] == ref, (
            f"prompt {p}: packed sigma=0 diverged from greedy:\n"
            f" ref={ref}\n np ={np_tok}")
        for i, cl in enumerate(out["candidate_logits"][p]):
            assert cl.shape[0] == 1 + args.n_sample, (
                f"prompt {p} step {i} width {cl.shape[0]} != {1+args.n_sample}")
    print(f"PASS [packed sigma=0]: all {len(pids)} prompts match greedy; "
          f"width=1+{args.n_sample}; b_pack={args.b_pack}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `cd verl && CUDA_VISIBLE_DEVICES=6 NP_KEEP_CUDA_VISIBLE=1 /home/yequan/miniconda3/envs/verl/bin/python ../scripts/zo_opd/np_checks/check_packed_sigma0.py --model Qwen/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 4 --b-pack 4`
Expected: `PASS [packed sigma=0]: all 4 prompts match greedy; ...`

If it FAILS with token divergence on prompt p>0, the per-prompt routing is wrong (most likely block_table or seq_lens not per-row) — fix Task 3/4 before proceeding. Do not weaken the gate.

- [ ] **Step 3: Commit**

```bash
git add scripts/zo_opd/np_checks/check_packed_sigma0.py
git commit -m "np: GPU sigma=0 packed gate (per-prompt prefix routing correct)"
```

---

## Task 7: GPU parity gate — serial vs packed produce the same signals

The acceptance gate: run the SAME prompts/seeds through the per-prompt `graphed` driver (the oracle) and the new `packed` driver; assert per-prompt per-token `clean_tokens` identical, `u` bit-identical, `logits`/`x` within bf16 tolerance. This proves packing changed *only how rows are tiled*, not the NP signals → δW direction is identical.

**Files:**
- Create: `scripts/zo_opd/np_checks/check_packed_parity.py`

- [ ] **Step 1: Write the parity gate**

Create `scripts/zo_opd/np_checks/check_packed_parity.py`:

```python
"""GPU parity gate: per-prompt graphed (oracle) vs packed driver on the SAME
prompts/seeds. clean_tokens identical, u bit-identical (same noise_seed key),
logits/x within bf16 reduction-order tol. Proves packing is a tiling change only.

Usage:
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_packed_parity.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --n-sample 8 --b-pack 4 --sigma 0.01
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--b-pack", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--logit-rtol", type=float, default=1e-2)
    args = ap.parse_args()

    prompts = [
        "Compute 7*8. Answer:", "Differentiate x^3. Answer:",
        "What is 10/2? Answer:", "Square root of 81? Answer:",
        "What is 3-1? Answer:", "Name a prime. Answer:",
        "Integral of 2x? Answer:", "What is 5! ? Answer:",
    ][: args.b_pack]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=42, sigma=args.sigma,
                  sample_method=args.sample_method)
    rollout_ids = list(range(len(pids)))

    # Oracle: per-prompt graphed driver, seeded with the SAME rollout_id per prompt.
    serial = []
    for p, pid in enumerate(pids):
        o = llm.collective_rpc(
            "run_np_decode_graphed",
            args=(pid, SamplingParams(temperature=0.0), args.layer, np_cfg,
                  rollout_ids[p], False))[0]
        serial.append(o)

    # Packed.
    packed = llm.collective_rpc(
        "run_np_decode_packed",
        args=(pids, SamplingParams(temperature=0.0), args.layer, np_cfg,
              rollout_ids))[0]

    for p in range(len(pids)):
        sa, ta = serial[p]["clean_tokens"], packed["clean_tokens"][p]
        n = min(len(sa), len(ta))
        assert sa[:n] == ta[:n] and len(sa) == len(ta), (
            f"prompt {p}: clean tokens diverged\n serial={sa}\n packed={ta}")
        for t in range(n):
            us = serial[p]["captured_u"][t].detach().cpu().float()
            up = packed["captured_u"][p][t].detach().cpu().float()
            assert torch.equal(us, up), (
                f"prompt {p} step {t}: u NOT bit-identical "
                f"(max {(us-up).abs().max():.3e}) -- seed/key bug")
            ls = serial[p]["candidate_logits"][t].float()
            lp = packed["candidate_logits"][p][t].float()
            assert torch.allclose(ls, lp, rtol=args.logit_rtol, atol=1e-2), (
                f"prompt {p} step {t}: logits beyond tol "
                f"(max {(ls-lp).abs().max():.3e})")
            xs = serial[p]["captured_x"][t].float()
            xp = packed["captured_x"][p][t].float()
            assert torch.allclose(xs, xp, rtol=args.logit_rtol, atol=1e-2), (
                f"prompt {p} step {t}: x beyond tol "
                f"(max {(xs-xp).abs().max():.3e})")
    print(f"PASS [serial vs packed]: {len(pids)} prompts, u bit-identical, "
          f"logits/x within rtol={args.logit_rtol}, b_pack={args.b_pack}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the parity gate**

Run: `cd verl && CUDA_VISIBLE_DEVICES=6 NP_KEEP_CUDA_VISIBLE=1 /home/yequan/miniconda3/envs/verl/bin/python ../scripts/zo_opd/np_checks/check_packed_parity.py --model Qwen/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 8 --b-pack 4 --sigma 0.01`
Expected: `PASS [serial vs packed]: 4 prompts, u bit-identical, ...`

If `u` is not bit-identical: the per-prompt `rollout_id` → `noise_seed` mapping in `run_np_decode_packed` (the u_buf refill loop) doesn't match what the serial `graphed` call drew. Re-check `_assign_rollout_ids` use and the `(i*n_sample+q)` indexing. Do not relax the bit-identity assert — it's the parity-by-construction guarantee.

- [ ] **Step 3: Commit**

```bash
git add scripts/zo_opd/np_checks/check_packed_parity.py
git commit -m "np: GPU serial-vs-packed parity gate (u bit-identical, logits/x within tol)"
```

---

## Task 8: Wire the wave loop into `fit()`

Add a `decode_mode == "packed"` branch to `RayNPTrainer.fit()` that chunks `batch_size` prompts into waves of `pack_width`, issues one `run_np_decode_packed` RPC per wave, and accumulates all per-prompt signals into the same four lists fed once to `assemble_and_apply`. Math identical to serial — `pack_width` is purely a tiling knob.

**Files:**
- Modify: `verl/verl/trainer/np/ray_trainer.py` (the `for b in range(batch_size)` block, ~lines 504-545)

- [ ] **Step 1: Add `pack_width` + `packed` to the decode-mode validation**

Find (ray_trainer.py ~lines 489-493):

```python
        decode_mode = cfg.get("decode_mode", "eager")
        use_cuda_graph = bool(cfg.get("use_cuda_graph", False))
        if decode_mode not in ("eager", "graphed"):
            raise ValueError(
                f"np.decode_mode={decode_mode!r} must be 'eager' or 'graphed'.")
```

Replace with:

```python
        decode_mode = cfg.get("decode_mode", "eager")
        use_cuda_graph = bool(cfg.get("use_cuda_graph", False))
        pack_width = int(cfg.get("pack_width", 8))
        if decode_mode not in ("eager", "graphed", "packed"):
            raise ValueError(
                f"np.decode_mode={decode_mode!r} must be 'eager', 'graphed', "
                f"or 'packed'.")
```

- [ ] **Step 2: Add the packed wave loop branch**

Find the inner accumulation loop (ray_trainer.py ~lines 515-542, the `for b in range(batch_size):` block, INCLUDING the `NP_DEBUG_DECODE` instrumentation added earlier). Wrap it so packed mode uses waves. Replace the whole `for b in range(batch_size):` block with:

```python
                if decode_mode == "packed":
                    from verl.workers.rollout.vllm_rollout.np_worker_extension import (
                        _assign_rollout_ids,
                    )
                    all_pids = [
                        (prompts[(step * batch_size + b) % len(prompts)])
                        for b in range(batch_size)
                    ]
                    all_pids = [
                        (p["prompt_token_ids"] if isinstance(p, dict) else p)
                        for p in all_pids
                    ]
                    rollout_ids_full = _assign_rollout_ids(
                        step, batch_size, int(cfg.n_rollout))
                    # n_rollout>1: expand prompts to (prompt,rollout) slots.
                    if int(cfg.n_rollout) > 1:
                        slot_pids, slot_rids = [], []
                        for b in range(batch_size):
                            for r in range(int(cfg.n_rollout)):
                                slot_pids.append(all_pids[b])
                                slot_rids.append(
                                    rollout_ids_full[b * int(cfg.n_rollout) + r])
                    else:
                        slot_pids, slot_rids = all_pids, rollout_ids_full

                    for w0 in range(0, len(slot_pids), pack_width):
                        wave_pids = slot_pids[w0:w0 + pack_width]
                        wave_rids = slot_rids[w0:w0 + pack_width]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step} L={layer_name} "
                                  f"wave {w0}-{w0+len(wave_pids)}/{len(slot_pids)}] "
                                  f"packed decode start", flush=True)
                            _tw = time.time()
                        out = ray.get(self.engines[0].collective_rpc.remote(
                            "run_np_decode_packed",
                            args=(wave_pids, sp, layer_name, np_cfg, wave_rids),
                        ))[0]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step}] wave decode done "
                                  f"dt={time.time()-_tw:.2f}s", flush=True)
                        for pidx in range(len(wave_pids)):
                            if not out["clean_tokens"][pidx]:
                                continue
                            full = (list(wave_pids[pidx])
                                    + list(out["clean_tokens"][pidx]))
                            L_q, L_clean = self.scorer.score_rollout(
                                full, out["candidate_logits"][pidx])
                            L_q_steps += L_q
                            L_clean_steps += L_clean
                            nT = len(out["candidate_logits"][pidx])
                            u_steps += [out["captured_u"][pidx][t]
                                        for t in range(nT)]
                            x_steps += [out["captured_x"][pidx][t]
                                        for t in range(nT)]
                else:
                    for b in range(batch_size):
                        prompt = prompts[(step * batch_size + b) % len(prompts)]
                        pid = (prompt["prompt_token_ids"]
                               if isinstance(prompt, dict) else prompt)
                        for r in range(int(cfg.n_rollout)):
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} L={layer_name} "
                                      f"b{b}/{batch_size} r{r}] decode start "
                                      f"plen={len(pid)}", flush=True)
                                _td = time.time()
                            if decode_mode == "graphed":
                                out = ray.get(self.engines[0].collective_rpc.remote(
                                    "run_np_decode_graphed",
                                    args=(pid, sp, layer_name, np_cfg, r,
                                          use_cuda_graph),
                                ))[0]
                            else:
                                out = ray.get(self.engines[0].collective_rpc.remote(
                                    "run_np_decode",
                                    args=(pid, sp, layer_name, np_cfg, r),
                                ))[0]
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] decode done "
                                      f"ntok={len(out['clean_tokens'])} "
                                      f"dt={time.time()-_td:.2f}s", flush=True)
                                _ts = time.time()
                            if not out["clean_tokens"]:
                                continue
                            full = list(pid) + list(out["clean_tokens"])
                            L_q, L_clean = self.scorer.score_rollout(
                                full, out["candidate_logits"])
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] score done "
                                      f"dt={time.time()-_ts:.2f}s", flush=True)
                            L_q_steps += L_q
                            L_clean_steps += L_clean
                            u_steps += [out["captured_u"][t]
                                        for t in range(len(out["candidate_logits"]))]
                            x_steps += [out["captured_x"][t]
                                        for t in range(len(out["candidate_logits"]))]
```

(The `else` branch is the existing serial code verbatim, kept as the fallback for `eager`/`graphed`.)

- [ ] **Step 3: Smoke the wiring end-to-end (2 steps, tiny)**

Run:
```bash
cd /home/yequan/Project/compression/OPD && NP_DEBUG_DECODE=1 CUDA_VISIBLE_DEVICES=6 \
  EXP=packsmoke DECODE_MODE=packed PACK_WIDTH=4 BATCH_SIZE=8 MAX_RESP_LENGTH=64 \
  N_SAMPLE=8 N_ROLLOUT=1 NUM_ITERATIONS=2 EVAL_INTERVAL=999 \
  GPU_MEMORY_UTILIZATION=0.30 TEACHER_GPU_MEMORY_UTILIZATION=0.45 \
  LR=3e-2 LOG_DIR=logs/np_pack_dbg NP_LOGGER='["console"]' \
  bash scripts/zo_opd/opd_math_np.sh 2>&1 | grep -E "npdbg|step:|wave|Error|Traceback|assert" | head -40
```
Expected: per-wave `[npdbg ... wave ...]` lines (2 waves of 4 for batch=8), two `step:N` lines with `weight_sync_ok:1.0`, no Traceback. (Requires Task 9's `PACK_WIDTH` env wiring; do Task 9 first or inline `np.pack_width=4` in the launch.)

- [ ] **Step 4: Commit**

```bash
git add verl/verl/trainer/np/ray_trainer.py
git commit -m "np: packed wave loop in fit() -- chunk batch into pack_width waves, one assemble/step"
```

---

## Task 9: Config + launcher wiring (`pack_width`, `DECODE_MODE=packed`)

**Files:**
- Modify: `verl/verl/trainer/config/np_trainer.yaml`
- Modify: `scripts/zo_opd/opd_math_np.sh`

- [ ] **Step 1: Add `pack_width` to the config**

In `np_trainer.yaml`, find:

```yaml
  decode_mode: eager               # eager (V1, parity oracle) | graphed (V2 buffer-in-graph)
  use_cuda_graph: false            # graphed only: false = M1 eager-with-u_buf, true = M2 captured graph
```

Replace with:

```yaml
  decode_mode: eager               # eager (V1) | graphed (V2 per-prompt graph) | packed (B_pack prompts/forward)
  use_cuda_graph: false            # graphed only: false = M1 eager-with-u_buf, true = M2 captured graph
  pack_width: 8                    # packed only: prompts per wide forward (B_pack)
```

- [ ] **Step 2: Add `PACK_WIDTH` env + pass it through in the launcher**

In `scripts/zo_opd/opd_math_np.sh`, find:

```bash
export DECODE_MODE=${DECODE_MODE:-graphed}
export USE_CUDA_GRAPH=${USE_CUDA_GRAPH:-true}
```

Replace with:

```bash
export DECODE_MODE=${DECODE_MODE:-graphed}
export USE_CUDA_GRAPH=${USE_CUDA_GRAPH:-true}
export PACK_WIDTH=${PACK_WIDTH:-8}
```

Then find the `python3 -m verl.trainer.main_np` invocation line:

```bash
    np.decode_mode=${DECODE_MODE} np.use_cuda_graph=${USE_CUDA_GRAPH} \
```

Replace with:

```bash
    np.decode_mode=${DECODE_MODE} np.use_cuda_graph=${USE_CUDA_GRAPH} \
    np.pack_width=${PACK_WIDTH} \
```

- [ ] **Step 3: Verify Hydra accepts the key**

Run: `cd /home/yequan/Project/compression/OPD && /home/yequan/miniconda3/envs/verl/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('verl/verl/trainer/config/np_trainer.yaml'); print(c.np.pack_width)"`
Expected: prints `8`.

- [ ] **Step 4: Commit**

```bash
git add verl/verl/trainer/config/np_trainer.yaml scripts/zo_opd/opd_math_np.sh
git commit -m "np: wire pack_width config + PACK_WIDTH env for packed decode"
```

---

## Task 10: Throughput / memory / util benchmark grid

Benchmark the packed driver across batch∈{1,2,4,8,16} × rails∈{8,16,64} at the real `max_tokens=1024`, reporting s/step, tok/s, peak GPU memory, and SM utilization. This is the deliverable that quantifies the packing win.

**Files:**
- Create: `scripts/zo_opd/np_checks/bench_packed_grid.py`

- [ ] **Step 1: Write the grid benchmark**

Create `scripts/zo_opd/np_checks/bench_packed_grid.py`:

```python
"""Throughput/memory/util grid for the packed NP decode driver.

For each (batch_size, n_sample) cell, decode `batch_size` prompts via the packed
driver in waves of `pack_width`, at max_tokens, and report:
  s/step (whole batch, one layer), tok/s (clean tokens decoded / s),
  peak GPU mem (torch.cuda.max_memory_allocated), and mean SM util sampled via
  nvidia-smi during the run.

This measures the STUDENT decode only (no teacher scoring / no assemble) so the
packing effect is isolated. End-to-end step timing incl. teacher+assemble is
measured separately by bench_np_vs_bp.sh (Task 11).

Usage:
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/bench_packed_grid.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --max-tokens 1024 \
      --batches 1,2,4,8,16 --rails 8,16,64 --pack-width 8
"""
import argparse
import os
import subprocess
import threading
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


class _UtilSampler(threading.Thread):
    """Polls nvidia-smi for this process's visible GPU SM-util while running."""
    def __init__(self, gpu_index, interval=0.1):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                    text=True, timeout=2).strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def _make_prompts(tok, n):
    base = [
        "Solve for x: 2x+3=7. Answer:", "Compute the integral of x^2. Answer:",
        "What is 12*13? Answer:", "Differentiate sin(x). Answer:",
        "Factor x^2-9. Answer:", "What is 100/4? Answer:",
        "Sum 1..10. Answer:", "Square root of 144? Answer:",
    ]
    out = []
    for i in range(n):
        out.append(tok(base[i % len(base)] + f" ({i})")["input_ids"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--rails", default="8,16,64")
    ap.add_argument("--pack-width", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    args = ap.parse_args()

    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    batches = [int(x) for x in args.batches.split(",")]
    rails = [int(x) for x in args.rails.split(",")]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem_util)
    tok = llm.get_tokenizer()
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    print(f"# packed grid: model={args.model} layer={args.layer} "
          f"max_tokens={args.max_tokens} pack_width={args.pack_width}")
    print(f"{'batch':>6} {'rails':>6} {'s/step':>9} {'tok/s':>9} "
          f"{'peakGB':>8} {'SM%':>6}")
    for n_sample in rails:
        for batch in batches:
            pids = _make_prompts(tok, batch)
            rollout_ids = list(range(batch))
            np_cfg = dict(n_sample=n_sample, max_tokens=args.max_tokens,
                          global_seed=42, sigma=args.sigma,
                          sample_method="bernoulli")
            # warmup one small wave
            _ = llm.collective_rpc(
                "run_np_decode_packed",
                args=(pids[: min(args.pack_width, batch)],
                      SamplingParams(temperature=0.0), args.layer,
                      dict(np_cfg, max_tokens=8), rollout_ids[: args.pack_width]))[0]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sampler = _UtilSampler(gpu_index)
            sampler.start()
            t0 = time.time()
            total_tok = 0
            for w0 in range(0, batch, args.pack_width):
                out = llm.collective_rpc(
                    "run_np_decode_packed",
                    args=(pids[w0:w0 + args.pack_width],
                          SamplingParams(temperature=0.0), args.layer, np_cfg,
                          rollout_ids[w0:w0 + args.pack_width]))[0]
                total_tok += sum(len(ct) for ct in out["clean_tokens"])
            torch.cuda.synchronize()
            dt = time.time() - t0
            sampler.stop()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            tok_s = total_tok / dt if dt > 0 else 0.0
            print(f"{batch:>6} {n_sample:>6} {dt:>9.3f} {tok_s:>9.1f} "
                  f"{peak_gb:>8.2f} {sampler.mean():>6.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a quick grid (short max_tokens first to validate)**

Run: `cd verl && CUDA_VISIBLE_DEVICES=6 NP_KEEP_CUDA_VISIBLE=1 /home/yequan/miniconda3/envs/verl/bin/python ../scripts/zo_opd/np_checks/bench_packed_grid.py --model Qwen/Qwen3-1.7B --max-tokens 128 --batches 1,2,4,8 --rails 8,16 --pack-width 8`
Expected: a table with rising `tok/s` as batch grows (the packing win) until SM% saturates; `peakGB` rises with batch×rails. No OOM at these sizes.

- [ ] **Step 3: Run the full grid at max_tokens=1024**

Run: `cd verl && CUDA_VISIBLE_DEVICES=6 NP_KEEP_CUDA_VISIBLE=1 /home/yequan/miniconda3/envs/verl/bin/python ../scripts/zo_opd/np_checks/bench_packed_grid.py --model Qwen/Qwen3-1.7B --max-tokens 1024 --batches 1,2,4,8,16 --rails 8,16,64 --pack-width 8 2>&1 | tee ../scripts/zo_opd/results/packed_grid_1024.txt`
Expected: full 5×3 table saved. Note the batch where tok/s plateaus (SM-bound) and any cell that OOMs (record the ceiling). `pack_width=8` caps the wave width; to test wider waves, re-run with `--pack-width 16`.

- [ ] **Step 4: Commit**

```bash
git add scripts/zo_opd/np_checks/bench_packed_grid.py scripts/zo_opd/results/packed_grid_1024.txt
git commit -m "np: packed throughput/memory/util grid bench (batch x rails) + results"
```

---

## Task 11: NP-vs-BP one-step comparison

Compare one full NP step (packed, batch=64, max_tokens=1024, greedy — incl. teacher scoring + assemble) against one standard BP-based OPD step (`opd_math_ref.sh`, same regime). Both single-GPU. The headline number the user asked for.

**Files:**
- Create: `scripts/zo_opd/bench_np_vs_bp.sh`

- [ ] **Step 1: Write the comparison driver**

Create `scripts/zo_opd/bench_np_vs_bp.sh`:

```bash
#!/bin/bash
# bench_np_vs_bp.sh — one-step wall-clock + peak-mem: NP packed vs BP-OPD.
# Both: Qwen3-1.7B student, Keven16 4B teacher, batch=64, max_tokens=1024, greedy.
#
# NP: packed decode (one step = 64 prompts decoded in waves, scored, one delta_W).
# BP: standard verl PPO token_reward_direct (opd_math_ref.sh), one step.
#
#   NP_GPU=6 BP_GPU=7 PACK_WIDTH=8 bash scripts/zo_opd/bench_np_vs_bp.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
mkdir -p logs/np_vs_bp

NP_GPU=${NP_GPU:-6}
BP_GPU=${BP_GPU:-7}
PACK_WIDTH=${PACK_WIDTH:-8}
TS=$(date +%Y%m%d_%H%M%S)

# --- NP: one packed step (NUM_ITERATIONS=1, eval off), debug timing on ---
NP_DEBUG_DECODE=1 CUDA_VISIBLE_DEVICES=$NP_GPU \
  EXP=npvsbp_np DECODE_MODE=packed PACK_WIDTH=$PACK_WIDTH \
  BATCH_SIZE=64 MAX_RESP_LENGTH=1024 N_SAMPLE=8 N_ROLLOUT=1 \
  NUM_ITERATIONS=1 EVAL_INTERVAL=999 \
  GPU_MEMORY_UTILIZATION=0.30 TEACHER_GPU_MEMORY_UTILIZATION=0.45 \
  LR=3e-2 LOG_DIR=logs/np_vs_bp NP_LOGGER='["console"]' \
  bash scripts/zo_opd/opd_math_np.sh > logs/np_vs_bp/np_${TS}.log 2>&1
echo "NP done. step_time:"
grep -E "step:0 .*step_time" logs/np_vs_bp/np_${TS}.log | head -1

# --- BP: one OPD step (run opd_math_ref.sh, it will print per-step timing) ---
# opd_math_ref.sh runs the full trainer; we capture the first step's timing from
# verl's own step log, then stop. TEST_FREQ high + a 1-step cap via env.
CUDA_VISIBLE_DEVICES=$BP_GPU TEST_FREQ=9999 \
  bash scripts/zo_opd/opd_math_ref.sh > logs/np_vs_bp/bp_${TS}.log 2>&1 &
BP_PID=$!
# Wait for verl to log its first step timing, then kill (one-step measurement).
( while ! grep -qE "step:1|'timing_s/step'|perf/time_per_step" logs/np_vs_bp/bp_${TS}.log 2>/dev/null; do
    sleep 5; if ! kill -0 $BP_PID 2>/dev/null; then break; fi; done
  kill $BP_PID 2>/dev/null ) 
wait $BP_PID 2>/dev/null
echo "BP done. first-step timing:"
grep -E "step:1|time_per_step|timing_s/step" logs/np_vs_bp/bp_${TS}.log | head -3

echo "=== logs: logs/np_vs_bp/np_${TS}.log  logs/np_vs_bp/bp_${TS}.log ==="
```

- [ ] **Step 2: Make it executable and run**

Run:
```bash
chmod +x scripts/zo_opd/bench_np_vs_bp.sh
NP_GPU=6 BP_GPU=7 PACK_WIDTH=8 bash scripts/zo_opd/bench_np_vs_bp.sh
```
Expected: prints NP `step_time` (from `step:0` line) and BP first-step timing. NP one-step should be far below the pre-fix ~25 min; BP is verl's normal ~60s/step (per the memory `opd-math-singlegpu-timing`). Record both. (If BP timing key differs, inspect `logs/np_vs_bp/bp_*.log` for verl's actual per-step metric name and adjust the grep.)

- [ ] **Step 3: Commit**

```bash
git add scripts/zo_opd/bench_np_vs_bp.sh
git commit -m "np: one-step NP-packed vs BP-OPD wall-clock/peak-mem comparison driver"
```

---

## Task 12: Document results in the wiki

**Files:**
- Modify: `docs/wiki/zo_np_trainer.md` (add §10)

- [ ] **Step 1: Add the §10 section**

Append to `docs/wiki/zo_np_trainer.md` a new section `## 10. Packed multi-prompt decode (V2.1)` covering: the packed row layout + per-prompt KV (cite `run_np_decode_packed`), the gates that pass (`check_packed_sigma0.py`, `check_packed_parity.py`), and a results table from `scripts/zo_opd/results/packed_grid_1024.txt` plus the NP-vs-BP one-step numbers. Use the exact measured numbers (do not invent). Include the batch×rails throughput table and the SM-util / peak-mem columns. Note the batch where throughput plateaus and the OOM ceiling.

- [ ] **Step 2: Update the index + log (knowledge-system convention)**

Add a row to `docs/index.md` for the packed-decode wiki section, and append one `docs/log.md` line: `## [2026-06-XX] ingest | NP packed multi-prompt decode + throughput grid`.

- [ ] **Step 3: Commit**

```bash
git add docs/wiki/zo_np_trainer.md docs/index.md docs/log.md
git commit -m "docs: NP packed decode design + throughput grid results (wiki §10)"
```

---

## Test Plan (the measurement deliverable)

This is the explicit answer to "test plan that includes throughput, memory, compute utilization, timing across batch and rails, and compare with BP-OPD."

### A. Correctness gates (must all PASS before any benchmark is trusted)

| Gate | Script | Asserts |
|---|---|---|
| CPU helpers | `pytest verl/tests/np/test_packed_helpers.py` | row layout + seed-id identity (Task 1) |
| CPU regression | `pytest verl/tests/np/` | all 54 existing NP tests still pass |
| σ=0 packed routing | `check_packed_sigma0.py --b-pack 4` | every prompt's packed clean tokens == stock greedy (no cross-prompt KV bleed) |
| serial-vs-packed parity | `check_packed_parity.py --b-pack 4 --n-sample 8` | `u` bit-identical, `logits`/`x` within `rtol=1e-2` → δW direction identical to oracle |
| e2e smoke | Task 8 Step 3 | 2 packed steps, `weight_sync_ok=1.0`, no crash |

### B. Throughput / memory / utilization grid (`bench_packed_grid.py`)

Sweep **batch ∈ {1,2,4,8,16}** × **rails ∈ {8,16,64}** at `max_tokens=1024`, `pack_width=8` (and a second pass at `pack_width=16` to test wider waves). Per cell, report:

- **s/step** — wall-clock to decode the whole batch (student decode only, isolates packing from teacher/assemble)
- **tok/s** — clean tokens decoded per second (the throughput headline; should rise with batch until SM-bound)
- **peak GB** — `torch.cuda.max_memory_allocated` (the memory cost of packing; rises with batch×rails; record OOM ceiling)
- **SM %** — mean GPU utilization sampled via nvidia-smi during the run (compute utilization; low at batch=1 = the under-utilization packing fixes, should climb toward saturation)

Expected shape: at batch=1 rails=8, SM% is low (memory-bound single-prompt forward, the status quo); as batch grows, tok/s climbs and SM% rises until the forward becomes compute-bound (plateau). rails=64 reaches the compute-bound regime at smaller batch. The cell that maximizes tok/s within the memory budget is the recommended operating point.

### C. NP-vs-BP one-step comparison (`bench_np_vs_bp.sh`)

Both at **batch=64, max_tokens=1024, greedy, single GPU, same student+teacher**:

- **NP packed:** one step = 64 prompts decoded in `ceil(64/pack_width)` waves + teacher scoring + GPU assemble + apply. Report `step_time` from the `step:0` log line and peak mem.
- **BP-OPD:** one step of `opd_math_ref.sh` (verl PPO `token_reward_direct`). Report verl's per-step timing and peak mem.

Report the ratio. Context: pre-fix NP was ~25 min/step (serial decode + CPU assemble); BP-OPD is ~60s/step. The deliverable shows where packed NP lands relative to BP — i.e. whether the zeroth-order method is throughput-competitive with backprop for this config.

### D. What is explicitly NOT tested here (scope boundaries)

- CUDA-graphing the **packed** forward (eager packed only this round).
- Teacher-side batching (teacher scoring stays per-prompt; measured but not optimized).
- Multi-GPU packed decode (single engine 0, as today).
- Gradient quality / convergence (unchanged by construction — parity gate proves δW direction identical to the per-prompt oracle; the existing `check_grad_cosine.py` cos≈0.41 result transfers).

---

## Self-Review notes

- **Spec coverage:** packing (Tasks 1-5,8,9), per-prompt KV correctness (Tasks 2,3,6), parity (Task 7), throughput/mem/util grid across batch×rails (Task 10 + Test Plan B), BP comparison (Task 11 + Test Plan C), docs (Task 12). All requested items covered.
- **Type/name consistency:** `run_np_decode_packed(list_of_prompt_ids, sampling_params, layer_name, np_cfg, rollout_ids)` returns dict of **lists** indexed by prompt; the wave loop (Task 8) and both gates (Tasks 6,7) consume that exact shape. `_packed_row_blocks`/`_assign_rollout_ids` signatures match between Task 1 (defined) and Tasks 5,8 (used). `u_buf` row order is prompt-major `(i*n_sample+q)` consistently in Task 5 (refill) and Task 4 (`perturbed_row_idx` build).
- **Parity-by-construction:** per-prompt `rollout_id` → identical `noise_seed` key → bit-identical `u` (Task 7 asserts `torch.equal`). This is the load-bearing invariant; if it fails, the seed mapping is the bug, not the tolerance.
- **No placeholders:** every code step has complete code; every run step has the exact command + expected output.
