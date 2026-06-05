"""Node-Perturbation worker extension (vLLM WorkerExtension).

Registered via np.worker_extension_cls. Runs on each per-GPU vLLM Worker (self
is the Worker; self.model_runner.model is the loaded model). Responsibilities,
added across plan Tasks 8-11:
  - PerturbedLinear shim + install_perturb_layers  (Task 8)
  - n_sample-wide custom decode driver             (Task 9)
  - apply_node_update (weights) + NCCL broadcast   (Task 10-11)

Perturbations are regenerated from seeds, never stored. enforce_eager=True is
mandatory (set by NPNcclLLM) so these eager-Python hooks actually run.
See docs/superpowers/specs/2026-05-28-np-trainer-design.md.
"""
import numpy as np
import torch

from verl.trainer.np.seeding import noise_seed, draw_noise

try:  # vLLM 0.11.0 — required for the custom decode forwards.
    from vllm.forward_context import set_forward_context
except ImportError:  # pragma: no cover - keep importable on stub installs
    set_forward_context = None


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Initialize NCCL communicator for inter-engine weight broadcast."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


def _unpack(output):
    """vLLM linears may return a bare tensor or (tensor, bias). Normalize."""
    if isinstance(output, tuple):
        return output[0], output[1], True
    return output, None, False


def _repack(tensor, bias, was_tuple):
    return (tensor, bias) if was_tuple else tensor


class PerturbedLinear(torch.nn.Module):
    """Wraps a matched vLLM linear. Behavior keyed by worker-local np_state.

    Row layout per decode step (see spec §2): the active sequence contributes
    n_clean_rows clean row(s) followed by n_sample perturbed rows, contiguous.
    np_state_ref() returns the live dict so the trainer can switch modes per RPC
    without reinstalling.
    """

    def __init__(self, wrapped: torch.nn.Module, name: str, np_state_ref):
        super().__init__()
        self.wrapped = wrapped
        self.name = name
        self._np_state_ref = np_state_ref

    def forward(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        st = self._np_state_ref()
        mode = st.get("mode", "off")
        if mode == "off":
            return out
        x = args[0]
        y, bias, was_tuple = _unpack(out)
        n_clean = st["n_clean_rows"]

        if mode == "capture":
            # record the clean row's input x_t (detached) for the rank-1 update
            st["captured_x"][self.name] = x[0].detach().clone()
            return out

        if mode == "perturb_graph" and self.name == st["layer"]:
            # V2 graph-capturable perturbation (spec §5.2). The noise is NOT drawn
            # here -- the host has already filled the persistent buffer u_buf
            # (st["u_buf"]) via _np_fill_u_buf before this forward. The op is a
            # fixed-shape, RNG-free, allocation-free elementwise add on the
            # perturbed-row block -- the only thing a captured CUDA graph records.
            #   y[n_clean : n_clean+N] += sigma * u_buf      (u_buf is [N, d_out])
            # x_t is written into the persistent x_buf (st["x_buf"]) so it survives
            # past graph.replay() (a view into the transient activation would not).
            # x_buf is always installed by run_np_decode_graphed (the only caller
            # of perturb_graph mode); copy x[0] in place so it survives replay.
            st["x_buf"].copy_(x[0])
            sigma = st["sigma"]              # captured scalar (python float)
            u_buf = st["u_buf"]              # [N, d_out], host-refilled, persistent
            # In-place add on the perturbed-row slice. sigma==0 still adds 0*u_buf,
            # which is exactly the no-op the sigma=0 gate expects -- and keeping the
            # op unconditional means the captured graph is identical for sigma>0/=0
            # (no data-dependent branch inside the captured region).
            y[n_clean:n_clean + u_buf.shape[0]] = (
                y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            return _repack(y, bias, was_tuple)

        if mode == "perturb" and self.name == st["layer"]:
            # Capture the clean row's input x_t HERE, in the same forward that does
            # the perturbation -- this is the rank-1 update's x_t and is identical to
            # what a separate capture pass would record, so the redundant re-decode in
            # run_capture_pass is unnecessary. (x[0] is the clean row; perturbations
            # are added to y, not x, so the clean input is unperturbed.)
            st["captured_x"][self.name] = x[0].detach().clone()
            sigma = float(st["sigma"])
            if sigma == 0.0:
                return out
            n_sample = st["n_sample"]
            d_out = y.shape[-1]
            # perturbed rows occupy [n_clean : n_clean + n_sample]
            u_rows = []
            for q in range(n_sample):
                seed = noise_seed(st["global_seed"], st["step"], self.name, st["rollout"], q)
                u = draw_noise(seed, (d_out,), y.device, y.dtype, st["sample_method"])
                u_rows.append(u)
                y[n_clean + q] = y[n_clean + q] + sigma * u
            # stash regenerated u (stacked) so the update step can reuse identical noise
            st["captured_u"][self.name] = torch.stack(u_rows, dim=0)
            return _repack(y, bias, was_tuple)

        return out


class WorkerExtension:
    def _ensure_np_state(self):
        if not hasattr(self, "np_state"):
            self.np_state = {"mode": "off"}
        return self.np_state

    def install_perturb_layers(self, perturb_rules):
        """Wrap every perturb_rules-matched module with PerturbedLinear. Idempotent."""
        from verl.trainer.np.layer_resolve import resolve_modules

        self._ensure_np_state()
        model = self.model_runner.model
        names = [n for n, _ in model.named_modules()]
        matched = resolve_modules(list(perturb_rules), names, error_if_empty=True)
        self.np_modules = {}
        for layer_name in matched:
            parent = model
            *path, leaf = layer_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            child = getattr(parent, leaf)
            if isinstance(child, PerturbedLinear):
                wrapped = child  # already installed
            else:
                wrapped = PerturbedLinear(child, layer_name, lambda: self.np_state)
                setattr(parent, leaf, wrapped)
            self.np_modules[layer_name] = wrapped
        return list(matched)

    # Task 9: n_sample-wide custom decode driver.
    # Mechanics (spec §2): prefill prompt KV once. Per step t, build a
    # (1 + n_sample)-row batch sharing the committed KV; perturbed rows
    # (1..n_sample) get slot_mapping=-1 (PAD_SLOT_ID) so reshape_and_cache
    # skips writing them; row 0 writes its own next-token KV. Sample row 0,
    # commit, advance. Scratch KV blocks are taken from the top of the GPU
    # pool — no scheduler runs concurrently with this RPC.

    def run_np_decode(self, prompt_token_ids, sampling_params, layer_name,
                      np_cfg, rollout_idx):
        """Custom decode for ONE prompt. See module docstring + spec §2."""
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])

        # Prefill prompt (clean, normal KV write).
        state = self._np_prefill(model, device, list(prompt_token_ids))

        clean_tokens, candidate_logits, captured_u, captured_x = [], [], {}, {}
        for t in range(max_tokens):
            st.update({
                "mode": "perturb",
                "layer": layer_name,
                "global_seed": int(np_cfg["global_seed"]),
                "step": t,
                "rollout": int(rollout_idx),
                "sigma": float(np_cfg["sigma"]),
                "n_sample": n_sample,
                "sample_method": np_cfg["sample_method"],
                "n_clean_rows": 1,
                "captured_x": {},
                "captured_u": {},
            })
            logits = self._np_step_forward(model, device, state, n_sample)
            candidate_logits.append(logits.detach().to("cpu"))
            captured_u[t] = st["captured_u"].get(layer_name)
            # x_t for the rank-1 update, captured in the SAME forward (no 2nd pass).
            cx = st["captured_x"].get(layer_name)
            captured_x[t] = cx.detach().to("cpu") if cx is not None else None
            next_tok = self._np_sample_clean(logits[0], sampling_params)
            clean_tokens.append(int(next_tok))
            if self._np_is_eos(next_tok, sampling_params):
                break
            self._np_commit_clean(state, next_tok)

        st["mode"] = "off"
        return {
            "clean_tokens": clean_tokens,
            "candidate_logits": candidate_logits,
            "captured_x": captured_x,
            "captured_u": captured_u,
        }

    # ================================================================== V2 ===
    # CUDA-graphed 1+N decode rails (spec docs/.../2026-06-03-np-v2-cudagraph-rails.md).
    # The perturbation moves from in-forward RNG (perturb mode, above) to a
    # host-refilled persistent buffer + a captured `y += sigma*u_buf` op
    # (perturb_graph mode). Two sub-modes, selected by use_cuda_graph:
    #   use_cuda_graph=False (M1): same eager step forward, but reads u_buf.
    #                              Isolates the noise-relocation from graphing.
    #   use_cuda_graph=True  (M2): the step forward is captured once and replayed
    #                              per token; only buffers are refilled in place.
    # V1's run_np_decode / _np_step_forward (perturb mode) are left UNTOUCHED as
    # the parity oracle.

    def _np_fill_u_buf(self, u_buf, np_cfg, layer_name, step, rollout, n_sample):
        """Refill the persistent perturbation buffer for one token, on the host.

        This is the ONLY place RNG runs in the V2 path, and it calls the SAME
        noise_seed(...)+draw_noise(...) V1 calls in-forward (np_worker_extension
        perturb mode, line ~91) with the SAME (global_seed, step, layer, rollout,
        q) key -> the bytes written into u_buf are bit-identical to V1's u. Only
        the *location* of the draw moved (host, before replay) -- the value did
        not. This is the M1/M2 parity-by-construction guarantee (spec §3, §5.2).

        u_buf: persistent [n_sample, d_out] device tensor (dtype = layer out dtype).
        copy_ writes into it in place so the captured graph (which holds u_buf by
        pointer) sees the new noise on the next replay.
        """
        d_out = u_buf.shape[1]
        for q in range(n_sample):
            seed = noise_seed(int(np_cfg["global_seed"]), int(step), layer_name,
                              int(rollout), q)
            u = draw_noise(seed, (d_out,), u_buf.device, u_buf.dtype,
                           np_cfg["sample_method"])
            u_buf[q].copy_(u)

    def run_np_decode_graphed(self, prompt_token_ids, sampling_params, layer_name,
                              np_cfg, rollout_idx, use_cuda_graph=False):
        """V2 decode for ONE prompt. Same contract as run_np_decode (returns
        clean_tokens / candidate_logits / captured_x / captured_u), but the
        perturbation is applied via the host-refilled u_buf + perturb_graph op
        instead of in-forward RNG. With use_cuda_graph=True the step forward is a
        captured CUDA graph replayed per token."""
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])
        sigma = float(np_cfg["sigma"])

        state = self._np_prefill(model, device, list(prompt_token_ids))
        # Largest seqused_k the decode reaches (prompt_len + max_tokens). Used to
        # freeze FlashAttention's max_seqlen_k at capture (see _np_capture_step).
        state["max_seq_len_cap"] = int(state["prompt_len"]) + max_tokens

        # Resolve the perturbed layer's output width to size u_buf / x_buf.
        wrapped = self.np_modules[layer_name]
        weight = wrapped.wrapped.weight          # [d_out, d_in]
        # draw_noise generates f32 then casts to buf_dtype, so buf_dtype is
        # load-bearing for bit-identity with V1's in-forward draw (which used the
        # layer's OUTPUT dtype y.dtype). For an unquantized linear, F.linear out
        # dtype == weight.dtype, so weight.dtype is the right size. Guard against a
        # quantized/packed weight (int dtype) where weight.dtype != y.dtype.
        assert weight.is_floating_point(), (
            f"perturb_graph needs the perturbed layer's weight to be floating "
            f"(got {weight.dtype}); a quantized weight would size u_buf at the "
            f"wrong dtype and break parity. Layer {layer_name!r}.")
        d_out = int(weight.shape[0])
        d_in = int(weight.shape[1])
        buf_dtype = weight.dtype
        u_buf = torch.zeros(n_sample, d_out, device=device, dtype=buf_dtype)
        x_buf = torch.zeros(d_in, device=device, dtype=buf_dtype)

        # Persistent per-prompt graph state (only used when use_cuda_graph).
        graph_state = None

        # np_state stays in perturb_graph mode for the whole decode; per-token we
        # only change `step` and refill u_buf (no remode, so the captured graph's
        # control flow is stable).
        st.update({
            "mode": "perturb_graph",
            "layer": layer_name,
            "sigma": sigma,
            "n_clean_rows": 1,
            "u_buf": u_buf,
            "x_buf": x_buf,
        })

        clean_tokens, candidate_logits, captured_u, captured_x = [], [], {}, {}
        try:
            for t in range(max_tokens):
                # Host noise refill for this token (the only RNG). sigma=0 still
                # fills u_buf but the perturb_graph op adds 0*u_buf -> no-op.
                self._np_fill_u_buf(u_buf, np_cfg, layer_name, t, rollout_idx,
                                    n_sample)
                st["step"] = t

                if use_cuda_graph:
                    logits, graph_state = self._np_replay_step(
                        model, device, state, n_sample, graph_state)
                else:
                    logits = self._np_step_forward_graph(
                        model, device, state, n_sample)

                candidate_logits.append(logits.detach().to("cpu"))
                # u_t for the rank-1 update: same noise just written to u_buf.
                captured_u[t] = u_buf.detach().to("cpu").clone()
                captured_x[t] = x_buf.detach().to("cpu").clone()
                next_tok = self._np_sample_clean(logits[0], sampling_params)
                clean_tokens.append(int(next_tok))
                if self._np_is_eos(next_tok, sampling_params):
                    break
                self._np_commit_clean(state, next_tok)
        finally:
            st["mode"] = "off"
            st["u_buf"] = None
            st["x_buf"] = None

        return {
            "clean_tokens": clean_tokens,
            "candidate_logits": candidate_logits,
            "captured_x": captured_x,
            "captured_u": captured_u,
        }

    def _np_step_forward_graph(self, model, device, state, n_sample):
        """Eager 1+n_sample step forward for the perturb_graph path (M1).

        Identical row/KV/slot machinery to _np_step_forward (the V1 perturb path)
        -- the ONLY difference is np_state is in perturb_graph mode, so the
        PerturbedLinear reads u_buf instead of drawing noise. Kept separate from
        _np_step_forward so V1's parity oracle is byte-for-byte untouched."""
        block_ids = state["block_ids"]
        block_size = state["block_size"]
        prompt_len = state["prompt_len"]
        q_pos = state["kv_cursor"]

        if q_pos < prompt_len:
            q_token = state["prompt_token_ids"][q_pos]
        else:
            q_token = state["committed_tokens"][q_pos - prompt_len]

        query_lens = [1] * (1 + n_sample)
        seq_lens = [q_pos + 1] * (1 + n_sample)
        clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
        slot_mapping = [clean_slot] + [-1] * n_sample
        positions = [q_pos] * (1 + n_sample)
        input_ids = [q_token] * (1 + n_sample)

        attn_meta, total = self._np_build_attn_metadata(
            state, query_lens, seq_lens, slot_mapping, positions)
        with torch.no_grad():
            hidden = self._np_run_forward(
                model, device, input_ids, positions, attn_meta, total)
            logits = model.compute_logits(hidden)
        return logits

    # ------------------------------------------------- M2: capture / replay ---
    # vLLM-0.11.0 ground truth (verified against source, see spec §5.6 + pointers):
    #  - FlashAttentionMetadataBuilder.build() stores REFERENCES to the tensors in
    #    CommonAttentionMetadata (flash_attn.py:235-239,344-351) -- no copy. So a
    #    captured graph holds those tensors by pointer; mutating them in place (via
    #    copy_) before replay() feeds the kernel new values.
    #  - set_forward_context stashes attn_metadata in a python global read at
    #    forward time (forward_context.py); the captured graph re-reads the SAME
    #    tensor storage on replay (attention/layer.py:318-329).
    #  - compute_logits stays OUTSIDE the graph (gpu_model_runner.py:2286-2331),
    #    matching vLLM's own decode-graph split.
    # Therefore: allocate persistent input buffers ONCE, build attn_metadata once
    # from them, capture model(input_ids,positions) into hidden_buf, then per token
    # copy_ new ids/positions/slot_mapping/seq_lens/u_buf into the SAME buffers and
    # graph.replay(). One graph per prompt (its block_ids are prompt-specific); we
    # could cache across same-length prompts later, but per-prompt capture keeps M2
    # correct and simple (capture cost is amortized over max_tokens replays).

    def _np_capture_step(self, model, device, state, n_sample, max_seq_len_cap):
        """Allocate persistent step buffers, build attn_metadata from them once,
        warm up, then capture one 1+n_sample step forward into a CUDA graph.
        Returns a graph_state dict with the graph + the persistent buffers +
        the attn_metadata (held by the graph by reference).

        max_seq_len_cap: the LARGEST seqused_k the decode will reach
        (prompt_len + max_tokens). FlashAttention's `max_seqlen_k` is a frozen
        Python int that sizes the kernel's KV-iteration grid -- it CANNOT be a
        graph input -- so we bake it at the cap and feed the true per-token
        seqused_k through the seq_lens_gpu tensor (FA tolerates seqused_k <=
        max_seqlen_k). This mirrors how vLLM captures its own decode graphs at the
        padded maximum (build_for_cudagraph_capture). Capturing at the current
        (small) q_pos instead would freeze max_seqlen_k too low and silently
        truncate attention to the prompt -- the blocker this fixes."""
        from vllm.config.compilation import CUDAGraphMode

        block_ids = state["block_ids"]
        block_size = state["block_size"]
        q_pos = state["kv_cursor"]
        R = 1 + n_sample

        # Persistent input/output buffers (fixed shapes for the graph's life).
        input_ids_buf = torch.zeros(R, dtype=torch.long, device=device)
        positions_buf = torch.zeros(R, dtype=torch.long, device=device)

        # Build attn_metadata ONCE. The kernel-bound max_seqlen_k is frozen at the
        # CAP via max_seq_len_override, while seq_lens (the live per-token seqused_k
        # tensor) and num_computed_tokens hold the TRUE token-0 length q_pos+1 -- so
        # the warmup/capture run is numerically the real token-0 step. Per replay we
        # mutate seq_lens_gpu / slot_mapping[0] / input_ids / positions in place;
        # block_table is fixed (shared prefix). The kernel bounds attention by the
        # live seqused_k, not the frozen max_seqlen_k (verified vs vLLM source +
        # gpu_model_runner.py:3057 which captures decode graphs the same way).
        clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
        slot_mapping = [clean_slot] + [-1] * n_sample
        positions = [q_pos] * R
        # TRUE token-0 seq_lens (q_pos+1) for seqused_k / num_computed_tokens; the
        # frozen kernel-bound max_seqlen_k is overridden to the CAP so it stays
        # large enough for every later token (seq_lens_gpu is mutated per replay).
        attn_meta, total, meta_bufs = self._np_build_attn_metadata_persistent(
            state, [1] * R, [q_pos + 1] * R, slot_mapping, positions,
            max_seq_len_override=max_seq_len_cap)

        # Seed the input buffers with the current step's values.
        if q_pos < state["prompt_len"]:
            q_token = state["prompt_token_ids"][q_pos]
        else:
            q_token = state["committed_tokens"][q_pos - state["prompt_len"]]
        input_ids_buf.fill_(int(q_token))
        positions_buf.fill_(int(q_pos))

        # Warm up a few eager steps so cuBLAS workspaces / autotune settle before
        # capture (vLLM does the same before its own capture).
        for _ in range(3):
            with torch.no_grad(), set_forward_context(
                attn_meta, self.model_runner.vllm_config, num_tokens=total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE):
                _ = model(input_ids=input_ids_buf, positions=positions_buf)
        torch.cuda.synchronize()

        # Capture. The forward reads attn_meta from the forward-context global; the
        # captured graph records the kernel ops + tensor pointers (input buffers,
        # u_buf inside PerturbedLinear, the persistent metadata tensors). The
        # attention layer allocates its output tensor inside the captured region --
        # legal via the graph's private mempool. We do NOT share one pool across
        # graphs: a prior graph is still alive (use_count>0) when the next prompt
        # captures, and capturing into a pool a live graph holds trips
        # CUDACachingAllocator's "use_count > 0" assert. Instead we explicitly
        # release the PREVIOUS graph (freeing its pool) before capturing a new one,
        # so each graph owns its own default pool and pools don't accumulate across
        # prompts. M0 spike (spec §7.4): confirmed no in-forward host alloc trips
        # "operation not permitted when stream is capturing".
        prev = getattr(self, "_np_active_graph", None)
        if prev is not None:
            del prev
            self._np_active_graph = None
            import gc as _gc
            _gc.collect()
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), set_forward_context(
            attn_meta, self.model_runner.vllm_config, num_tokens=total,
            cudagraph_runtime_mode=CUDAGraphMode.NONE):
            with torch.cuda.graph(graph):
                hidden_buf = model(input_ids=input_ids_buf, positions=positions_buf)
        self._np_active_graph = graph

        # Pre-fix the perturbed-row slots to PAD (-1) once: only the clean slot
        # (element 0) changes per token, so the replay loop avoids a per-token host
        # alloc + full H2D copy and just writes element 0 in place.
        meta_bufs["slot_mapping"][1:].fill_(-1)

        return {
            "graph": graph,
            "input_ids_buf": input_ids_buf,
            "positions_buf": positions_buf,
            "hidden_buf": hidden_buf,
            "meta_bufs": meta_bufs,   # {slot_mapping, seq_lens_gpu, seq_lens_cpu, qsl_*}
            "attn_meta": attn_meta,
            "total": total,
        }

    def _np_replay_step(self, model, device, state, n_sample, graph_state):
        """Run one 1+n_sample step via CUDA-graph replay. Captures the graph on the
        first call (graph_state is None). Per token: refill the persistent input +
        metadata buffers in place, replay, compute_logits (eager) on hidden_buf."""
        if graph_state is None:
            graph_state = self._np_capture_step(
                model, device, state, n_sample, state["max_seq_len_cap"])
            # The capture run already produced hidden for the FIRST token's q_pos,
            # but u_buf may have been refilled after capture; replay once so the
            # returned logits reflect the current u_buf. Fall through to replay.

        gs = graph_state
        block_ids = state["block_ids"]
        block_size = state["block_size"]
        q_pos = state["kv_cursor"]

        if q_pos < state["prompt_len"]:
            q_token = state["prompt_token_ids"][q_pos]
        else:
            q_token = state["committed_tokens"][q_pos - state["prompt_len"]]

        # Refill persistent inputs in place (no realloc -> pointers stable).
        gs["input_ids_buf"].fill_(int(q_token))
        gs["positions_buf"].fill_(int(q_pos))
        clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
        mb = gs["meta_bufs"]
        # Only the clean-row slot (element 0) changes per token; rows 1..N were
        # fixed to PAD(-1) at capture (perturbed rows never write KV). Update
        # element 0 in place -> no per-token host alloc + H2D copy in the hot loop.
        mb["slot_mapping"][0].fill_(int(clean_slot))
        # Live seqused_k for this token. max_seqlen_k stays frozen at the cap
        # (sized for the whole decode at capture). seq_lens_cpu is NOT consumed by
        # FLASH_ATTN+fast_build (only the use_cascade / aot-scheduler branches read
        # it, both off here), so we mutate ONLY the load-bearing GPU tensor.
        mb["seq_lens_gpu"].fill_(q_pos + 1)

        gs["graph"].replay()
        # Sync before the host reads hidden_buf via compute_logits / before
        # captured_x reads x_buf. (Per-token full sync; if it dominates, batch the
        # host reads -- spec §5.7 buffer-fill / overlap note.)
        torch.cuda.synchronize()
        logits = model.compute_logits(gs["hidden_buf"])
        return logits, gs

    def _np_build_attn_metadata_persistent(self, state, query_lens, seq_lens,
                                           slot_mapping, positions_cpu,
                                           max_seq_len_override=None):
        """Like _np_build_attn_metadata but returns the persistent GPU tensors it
        built (slot_mapping, seq_lens) so the caller can mutate them in place
        across graph replays. The FlashAttention metadata builder stores these by
        reference (verified, flash_attn.py:344-351), so mutating them feeds the
        kernel new values without recapture.

        max_seq_len_override: freeze FlashAttention's max_seqlen_k at this value
        (the decode's CAP) while seq_lens / num_computed_tokens hold the TRUE
        token-0 length. max_seqlen_k must be a frozen int (it sizes the kernel
        grid, not a graph input); the live per-token seqused_k is the seq_lens GPU
        tensor (verified: kernel bounds attention by seqused_k, not max_seqlen_k --
        mirrors vLLM gpu_model_runner.py:3057 capturing at max_model_len)."""
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        mr = self.model_runner
        device = mr.device
        block_ids = state["block_ids"]

        num_reqs = len(query_lens)
        total_tokens = int(sum(query_lens))
        max_query_len = int(max(query_lens))
        max_seq_len = (int(max_seq_len_override) if max_seq_len_override is not None
                       else int(max(seq_lens)))

        qsl_np = np.zeros(num_reqs + 1, dtype=np.int32)
        qsl_np[1:] = np.cumsum(np.asarray(query_lens, dtype=np.int32))
        qsl_cpu = torch.from_numpy(qsl_np)
        qsl_gpu = qsl_cpu.to(device)

        sl_cpu = torch.from_numpy(np.asarray(seq_lens, dtype=np.int32))
        sl_gpu = sl_cpu.to(device)

        max_blocks = int(
            mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)
        bt = torch.zeros((num_reqs, max_blocks), dtype=torch.int32, device=device)
        bt[:, : len(block_ids)] = torch.tensor(
            block_ids, dtype=torch.int32, device=device)

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
        meta_bufs = {
            "slot_mapping": slot_mapping_gpu,
            "seq_lens_gpu": sl_gpu,
            "seq_lens_cpu": sl_cpu,
            "qsl_gpu": qsl_gpu,
            "qsl_cpu": qsl_cpu,
            "block_table": bt,
        }
        return attn_metadata, total_tokens, meta_bufs

    # -- helpers (vLLM-0.11.0-specific, against gpu_model_runner internals) --

    def _np_slot_for_position(self, block_ids, block_size, position):
        """Map an absolute position in the sequence to its KV cache slot."""
        return int(block_ids[position // block_size]) * int(block_size) + int(
            position % block_size)

    def _np_build_attn_metadata(self, state, query_lens, seq_lens, slot_mapping,
                                positions_cpu):
        """Build per-layer attn_metadata via the model_runner's MetadataBuilder.
        All rows share the same block_ids (shared-prefix KV). vLLM internal."""
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        mr = self.model_runner
        device = mr.device
        block_ids = state["block_ids"]

        num_reqs = len(query_lens)
        total_tokens = int(sum(query_lens))
        max_query_len = int(max(query_lens))
        max_seq_len = int(max(seq_lens))

        qsl_np = np.zeros(num_reqs + 1, dtype=np.int32)
        qsl_np[1:] = np.cumsum(np.asarray(query_lens, dtype=np.int32))
        qsl_cpu = torch.from_numpy(qsl_np)
        qsl_gpu = qsl_cpu.to(device, non_blocking=True)

        sl_cpu = torch.from_numpy(np.asarray(seq_lens, dtype=np.int32))
        sl_gpu = sl_cpu.to(device, non_blocking=True)

        max_blocks = int(
            mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)
        bt = torch.zeros((num_reqs, max_blocks), dtype=torch.int32, device=device)
        bt[:, : len(block_ids)] = torch.tensor(
            block_ids, dtype=torch.int32, device=device)

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

    def _np_run_forward(self, model, device, input_ids, positions,
                        attn_metadata, num_input_tokens):
        """Run model forward under set_forward_context. Returns hidden_states."""
        from vllm.config.compilation import CUDAGraphMode
        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        with set_forward_context(
            attn_metadata,
            self.model_runner.vllm_config,
            num_tokens=num_input_tokens,
            # enforce_eager + handcrafted slot_mapping/attn_metadata → CUDA graphs would skip our slot rewrites
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            out = model(input_ids=ids, positions=pos)
        # The model returns hidden_states for text-gen models.
        return out

    def _np_prefill(self, model, device, prompt_token_ids):
        """Prefill prompt KV at positions [0 .. prompt_len-2]; the last prompt
        token is held back to become the query of the first decode step (which
        writes KV at prompt_len-1 in row 0 only)."""
        mr = self.model_runner
        block_size = int(mr.cache_config.block_size)
        num_gpu_blocks = int(mr.cache_config.num_gpu_blocks)
        max_blocks = int(
            mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)

        prompt_len = len(prompt_token_ids)
        # High-indexed scratch slice — disjoint from anything vLLM allocates
        # bottom-up. This RPC owns the worker exclusively.
        n_scratch_blocks = min(
            (int(mr.max_model_len) + block_size - 1) // block_size, max_blocks)
        block_ids = list(range(num_gpu_blocks - n_scratch_blocks,
                                num_gpu_blocks))

        state = {
            "prompt_token_ids": list(prompt_token_ids),
            "committed_tokens": [],
            "prompt_len": prompt_len,
            "kv_cursor": max(0, prompt_len - 1),
            "block_ids": block_ids,
            "block_size": block_size,
        }

        if prompt_len <= 1:  # nothing to prefill; step 0 writes KV at pos 0.
            return state

        pre_len = prompt_len - 1
        slot_mapping = [
            self._np_slot_for_position(block_ids, block_size, p)
            for p in range(pre_len)
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
        return state

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

    def _np_step_forward(self, model, device, state, n_sample):
        """Build a 1+n_sample-row step (prefix-sharing), one query per row at
        position `kv_cursor`. Row 0 writes KV; rows 1..n_sample get -1
        (PAD_SLOT_ID) so reshape_and_cache skips them. Returns
        [1+n_sample, vocab] logits (row 0 = clean)."""
        block_ids = state["block_ids"]
        block_size = state["block_size"]
        prompt_len = state["prompt_len"]
        q_pos = state["kv_cursor"]

        # Input id at q_pos in the running [prompt + committed] sequence.
        if q_pos < prompt_len:
            q_token = state["prompt_token_ids"][q_pos]
        else:
            q_token = state["committed_tokens"][q_pos - prompt_len]

        query_lens = [1] * (1 + n_sample)
        seq_lens = [q_pos + 1] * (1 + n_sample)
        clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
        slot_mapping = [clean_slot] + [-1] * n_sample
        positions = [q_pos] * (1 + n_sample)
        input_ids = [q_token] * (1 + n_sample)

        attn_meta, total = self._np_build_attn_metadata(
            state, query_lens, seq_lens, slot_mapping, positions)
        with torch.no_grad():
            hidden = self._np_run_forward(
                model, device, input_ids, positions, attn_meta, total)
            logits = model.compute_logits(hidden)
        return logits

    def _np_sample_clean(self, logits_row0, sampling_params):
        """Sample / argmax the clean next token. Greedy when temperature==0."""
        temp = getattr(sampling_params, "temperature", 0.0) or 0.0
        if temp == 0.0:
            return int(torch.argmax(logits_row0).item())
        probs = torch.softmax(logits_row0.float() / temp, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _np_is_eos(self, token, sampling_params):
        """Stop-token check. Uses SamplingParams._all_stop_token_ids (filled by
        vLLM with model EOS + user stop ids on __post_init__)."""
        stop = getattr(sampling_params, "_all_stop_token_ids", None) or set()
        if not stop:
            # Fall back to the loaded model's config EOS.
            try:
                cfg = self.model_runner.model_config.hf_config
                eos = cfg.eos_token_id
                if isinstance(eos, int):
                    return int(token) == eos
                if isinstance(eos, (list, tuple, set)):
                    return int(token) in set(eos)
            except Exception:
                pass
            return False
        return int(token) in stop

    def _np_commit_clean(self, state, token):
        """Append clean token to running sequence; shared-prefix KV grows by 1.

        The step-forward we just ran already wrote row-0's KV at kv_cursor.
        We now advance kv_cursor so the next step writes the next position."""
        state["committed_tokens"].append(int(token))
        state["kv_cursor"] += 1

    def apply_node_update(self, layer_name, delta_w_cpu, lr, update_clip=None):
        """W <- W - lr * delta_W (gradient DESCENT). delta_w_cpu: [d_out,d_in].

        Sign: delta_w_cpu ~= +dL/dW (cosine-sim check empirically +0.4 against
        autograd). Combined with the standard convention that `lr > 0` is a
        positive learning rate, we subtract to minimize the loss.
        """
        import torch

        wrapped = self.np_modules[layer_name]
        weight = wrapped.wrapped.weight  # vLLM linear weight [d_out, d_in]
        dw = delta_w_cpu.to(weight.device, weight.dtype)
        if update_clip is not None:
            dw = dw.clamp_(-float(update_clip), float(update_clip))
        with torch.no_grad():
            before = weight.detach().clone()
            weight.add_(dw, alpha=-float(lr))
            # Fraction of weight ELEMENTS that actually changed in bf16. This is the
            # true "did the update land" signal -- ||W||-norm-difference badly
            # under-reports it (elementwise changes partially cancel in the norm),
            # so a 20%-of-elements update can show ~0 norm delta and look like a
            # no-op when it is a perfectly healthy bf16 update.
            changed_frac = (weight != before).float().mean().item()
        self._last_changed_frac = float(changed_frac)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(dw.norm().item())   # return ||delta_W|| for the >0 assertion

    def last_changed_frac(self):
        """Fraction of weight elements changed by the most recent apply_node_update."""
        return float(getattr(self, "_last_changed_frac", 0.0))

    def get_worker_ip(self):
        """Return the worker's IP address for NCCL initialization."""
        from vllm.utils import get_ip
        return get_ip()

    def init_inter_engine_group(self, master_address, master_port, rank, world_size):
        """Initialize NCCL communicator for inter-engine weight synchronization."""
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_layer_weights(self, layer_name, src_rank):
        """Broadcast only the updated layer's weight (+bias if present)."""
        import torch
        wrapped = self.np_modules[layer_name]
        params = [wrapped.wrapped.weight]
        if getattr(wrapped.wrapped, "bias", None) is not None:
            params.append(wrapped.wrapped.bias)
        for p in params:
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def layer_weight_norm(self, layer_name):
        """Return the live ||W||_2 of the (possibly wrapped) layer's weight.

        Used by the trainer to verify the NP update actually mutated the weight
        in-place and that the SAME tensor is what the next vLLM decode reads
        (we read wrapped.wrapped.weight -- the real nn.Linear param the forward uses).
        """
        import torch
        w = self.np_modules[layer_name].wrapped.weight
        return float(w.detach().float().norm().item())

    def assemble_and_apply(self, layer_name, L_q_steps, L_clean_steps, u_steps, x_steps,
                            sigma, sample_mode, normalize, token_agg, lr, update_clip):
        """One-RPC bundle of (assemble δW) + (apply δW locally) for layer_name.

        Returns ‖δW‖ (post-clip) as a float so the trainer can log + assert >0.

        u_steps/x_steps arrive on CPU (moved off-GPU inside run_np_decode*).
        L_q_steps/L_clean_steps come from the teacher scorer on CPU. The assemble
        runs the batched GEMM on this worker's GPU (device=self.device) -- it
        moves the signals back to the device once and reduces with one matmul
        instead of T sequential CPU rank-1 outer products (the multi-minute
        GPU-idle window this fixes). assemble_layer_delta returns a CPU delta_W,
        so apply_node_update's CPU->layer-device contract is unchanged.
        """
        L_q_dev = [lq.detach() if hasattr(lq, "detach") else torch.as_tensor(lq)
                   for lq in L_q_steps]
        dw = assemble_layer_delta(L_q_dev, L_clean_steps, u_steps, x_steps,
                                  sigma=sigma, sample_mode=sample_mode,
                                  normalize=normalize, token_agg=token_agg,
                                  device=self.device)
        return self.apply_node_update(layer_name, dw, lr, update_clip)


def assemble_layer_delta(L_q_per_step, L_clean_per_step, u_per_step, x_per_step,
                         sigma, sample_mode, normalize, token_agg, eps=1e-6,
                         device=None):
    """Build delta_W [d_out, d_in] from per-step signals. Pure math (CPU/GPU).

    Batched GEMM form of the per-token outer-product accumulation: stack the
    per-token gradient g_t into G [T, d_out] and the captured input x_t into
    X [T, d_in], then delta_W = G^T @ X (optionally / T). This is mathematically
    identical to the old `for t: dw += outer(g_t, x_t)` loop but replaces T
    sequential rank-1 updates with one matmul -- on GPU it turns the multi-minute
    CPU-bound assemble (~4.9 ms/token-signal; ~5 min at batch64*1024) into a
    sub-10s GEMM (62x measured, parity to 2.6e-6). Parity gate:
    test_apply_update_math.test_batched_assemble_matches_cpu_loop.

    device: where to run the GEMM ('cuda' to use the GPU; default = the signal
    tensors' device). The returned delta_W is always moved back to CPU so
    apply_node_update's downstream contract (CPU delta -> layer device) is
    unchanged.
    """
    from verl.trainer.np.grad_estimator import sample_scale

    assert len(L_q_per_step) == len(u_per_step) == len(x_per_step)
    T = max(len(L_q_per_step), 1)
    dev = torch.device(device) if device is not None else u_per_step[0].device

    g_rows = []   # [d_out] per token
    x_rows = []   # [d_in]  per token
    for L_q, L_clean, u, x_t in zip(L_q_per_step, L_clean_per_step,
                                    u_per_step, x_per_step):
        L_q = L_q.float().to(dev, non_blocking=True)
        u = u.float().to(dev, non_blocking=True)        # [n_sample, d_out]
        x_t = x_t.float().to(dev, non_blocking=True)    # [d_in]
        scales = sample_scale(L_q, L_clean, sigma, sample_mode)  # [n_sample]
        u_eff = u
        if normalize:
            sq = (u * u).sum(dim=-1, keepdim=True).clamp_min(eps)  # [n_sample,1]
            u_eff = u / sq
        g_t = (scales[:, None] * u_eff).mean(dim=0)     # [d_out]
        g_rows.append(g_t)
        x_rows.append(x_t)

    G = torch.stack(g_rows, dim=0)   # [T, d_out]
    X = torch.stack(x_rows, dim=0)   # [T, d_in]
    dw = G.t() @ X                   # [d_out, d_in]
    if token_agg == "mean":
        dw = dw / T
    return dw.to("cpu", dtype=torch.float32)


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
    For n_rollout==1 the id is simply step*batch_size + b.
    For n_rollout>1, each (prompt b, rollout r) slot gets
    (step*batch_size + b) * n_rollout + r — note this intentionally changes
    absolute seed values vs n_rollout==1 to keep all slots globally distinct."""
    base = int(step) * int(batch_size)
    if int(n_rollout) <= 1:
        return [base + b for b in range(int(batch_size))]
    ids = []
    for b in range(int(batch_size)):
        for r in range(int(n_rollout)):
            ids.append((base + b) * int(n_rollout) + r)
    return ids
