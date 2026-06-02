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

        if mode == "perturb" and self.name == st["layer"]:
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

        clean_tokens, candidate_logits, captured_u = [], [], {}
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
                "captured_x": st.get("captured_x", {}),
                "captured_u": {},
            })
            logits = self._np_step_forward(model, device, state, n_sample)
            candidate_logits.append(logits.detach().to("cpu"))
            captured_u[t] = st["captured_u"].get(layer_name)
            next_tok = self._np_sample_clean(logits[0], sampling_params)
            clean_tokens.append(int(next_tok))
            if self._np_is_eos(next_tok, sampling_params):
                break
            self._np_commit_clean(state, next_tok)

        st["mode"] = "off"
        return {
            "clean_tokens": clean_tokens,
            "candidate_logits": candidate_logits,
            "captured_x": st.get("captured_x", {}),
            "captured_u": captured_u,
        }

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
            weight.add_(dw, alpha=-float(lr))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(dw.norm().item())   # return ||delta_W|| for the >0 assertion

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

    def run_capture_pass(self, prompt_token_ids, clean_tokens, layer_name, np_cfg):
        """Teacher-forced re-run of the committed sequence to capture x_t per response
        step. Returns list[Tensor[d_in]] (CPU), one per token in clean_tokens.

        Mechanics: we re-decode the fixed `prompt_token_ids + clean_tokens` sequence
        one response step at a time. For each step we flip np_state to mode=capture
        (the PerturbedLinear hook records the row-0 input x_t and otherwise leaves
        the forward untouched), call `_np_step_forward(..., n_sample=0)` -- a
        single-row forward (rows = 1 + n_sample, with n_sample=0 -> one clean row,
        no perturbed companions). slot_mapping is [clean_slot] only; no PAD_SLOT_ID
        rows. Then we commit the already-known clean token and advance kv_cursor.
        Returns the list of captured x_t (CPU) -- one per step in clean_tokens.

        Note: n_sample=0 is supported by `_np_step_forward` as written -- the
        perturbed-row branches (`[-1] * n_sample`, `[q_token] * n_sample`) collapse
        to empty lists, so the resulting batch is a single clean row.
        """
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        x_per_step = []

        # Prefill clean prompt KV. Use only the prompt -- the loop below replays
        # clean_tokens one step at a time, writing each clean position's KV in row 0
        # of `_np_step_forward` before advancing kv_cursor via `_np_commit_clean`.
        state = self._np_prefill(model, device, list(prompt_token_ids))
        for t, tok in enumerate(clean_tokens):
            st.update({
                "mode": "capture",
                "layer": layer_name,
                "n_clean_rows": 1,
                "captured_x": {},
            })
            self._np_step_forward(model, device, state, n_sample=0)  # 1-row forward
            x_per_step.append(st["captured_x"][layer_name].detach().to("cpu"))
            self._np_commit_clean(state, tok)
        st["mode"] = "off"
        return x_per_step

    def assemble_and_apply(self, layer_name, L_q_steps, L_clean_steps, u_steps, x_steps,
                            sigma, sample_mode, normalize, token_agg, lr, update_clip):
        """One-RPC bundle of (assemble δW) + (apply δW locally) for layer_name.

        Returns ‖δW‖ (post-clip) as a float so the trainer can log + assert >0.

        u_steps and x_steps may arrive on GPU (captured inside PerturbedLinear).
        L_q_steps/L_clean_steps come from the teacher scorer on CPU. Coerce all
        signal tensors to CPU here so assemble_layer_delta's accumulator (also
        CPU) stays device-consistent.
        """
        u_cpu = [u.detach().cpu() for u in u_steps]
        x_cpu = [x.detach().cpu() for x in x_steps]
        L_q_cpu = [lq.detach().cpu() if hasattr(lq, "detach") else lq for lq in L_q_steps]
        dw = assemble_layer_delta(L_q_cpu, L_clean_steps, u_cpu, x_cpu,
                                  sigma=sigma, sample_mode=sample_mode,
                                  normalize=normalize, token_agg=token_agg)
        return self.apply_node_update(layer_name, dw, lr, update_clip)


def assemble_layer_delta(L_q_per_step, L_clean_per_step, u_per_step, x_per_step,
                         sigma, sample_mode, normalize, token_agg, eps=1e-6):
    """Build delta_W [d_out, d_in] from per-step signals. Pure math (CPU/GPU)."""
    from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w

    assert len(L_q_per_step) == len(u_per_step) == len(x_per_step)
    d_out = u_per_step[0].shape[1]
    d_in = x_per_step[0].shape[0]
    dw = torch.zeros(d_out, d_in, dtype=torch.float32)
    T = max(len(L_q_per_step), 1)
    for L_q, L_clean, u, x_t in zip(L_q_per_step, L_clean_per_step, u_per_step, x_per_step):
        scales = sample_scale(L_q.float(), L_clean, sigma, sample_mode)
        accumulate_delta_w(dw, scales=scales, u=u.float(), x_t=x_t.float(),
                           normalize=normalize, eps=eps)
    if token_agg == "mean":
        dw.div_(T)
    return dw
