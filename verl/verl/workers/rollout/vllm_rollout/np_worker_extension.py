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
import os

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

        if mode == "perturb_all_layers":
            # Every matched layer perturbs with ITS OWN buffer slice and captures
            # ITS OWN clean-row input, all in this one forward. u_buf/x_buf are
            # dicts keyed by layer name (pinned by the caller / graph capture).
            # Perturbation is added to OUTPUT y, never input x, so the clean row
            # stays the genuine unperturbed input at this layer. Two row layouts,
            # selected by st["perturbed_row_idx"] (mirrors the perturb_graph
            # single-vs-packed split, but every layer reads its OWN buffer dict):
            #   contiguous single-prompt (pri is None): perturbed rows are the
            #     slice [n_clean : n_clean+N]; the clean row is row 0 -> x_buf.
            #   packed (pri set): perturbed rows are SCATTERED across prompt blocks
            #     (st["perturbed_row_idx"]); each prompt's clean-row input is
            #     captured into x_buf[b_pack] via st["clean_row_idx"]. u_buf row i
            #     aligns with pri[i]. Used by the all-layer packed graph (Stage E).
            u_buf = st["u_buf"][self.name]          # [n_pert_rows, d_out]
            x_buf = st["x_buf"][self.name]          # contiguous:[d_in] packed:[b_pack,d_in]
            sigma = st["sigma"]
            pri = st.get("perturbed_row_idx")
            if pri is None:
                # contiguous single-prompt eager path (unchanged).
                n_clean = st["n_clean_rows"]
                x_buf.copy_(x[0])
                y[n_clean:n_clean + u_buf.shape[0]] = (
                    y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            else:
                # packed scatter path (mirrors perturb_graph packed branch, but
                # per-layer dicts). x_buf holds one clean-input row per prompt.
                cri = st["clean_row_idx"]            # LongTensor [b_pack] clean rows
                x_buf.copy_(x[cri])                  # [b_pack, d_in]
                y[pri] = y[pri] + sigma * u_buf      # u_buf rows align with pri
            return _repack(y, bias, was_tuple)

        if mode == "perturb_graph" and self.name == st["layer"]:
            # V2 graph-capturable perturbation (spec §5.2). The host has already
            # filled the persistent buffer u_buf (st["u_buf"]) before this forward.
            # The op is a fixed-shape, RNG-free, allocation-free elementwise add on
            # the perturbed rows. Two row layouts:
            #   single-prompt (pri is None): perturbed rows are the contiguous slice
            #     [n_clean : n_clean+N]; one clean row x[0] -> x_buf. (V1/V2 graphed
            #     driver, unchanged -- parity oracle.)
            #   packed (pri set): perturbed rows are SCATTERED across prompt blocks
            #     (st["perturbed_row_idx"]); each prompt's clean-row input is captured
            #     into x_buf row-by-row via st["clean_row_idx"].
            sigma = st["sigma"]              # python float
            u_buf = st["u_buf"]              # [n_pert_rows, d_out] host-refilled
            pri = st.get("perturbed_row_idx")
            if pri is None:
                # single-prompt path (unchanged). sigma==0 still adds 0*u_buf -> the
                # no-op the sigma=0 gate expects; unconditional op keeps a captured
                # graph identical for sigma>0/=0.
                st["x_buf"].copy_(x[0])
                y[n_clean:n_clean + u_buf.shape[0]] = (
                    y[n_clean:n_clean + u_buf.shape[0]] + sigma * u_buf)
            else:
                # packed path: x_buf holds one clean-input row per prompt; capture
                # each prompt's clean-row input, and scatter-add u_buf to the
                # perturbed rows. u_buf row i corresponds to pri[i].
                cri = st["clean_row_idx"]    # LongTensor [b_pack] clean rows
                st["x_buf"].copy_(x[cri])    # [b_pack, d_in]
                y[pri] = y[pri] + sigma * u_buf
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


def _alloc_layer_buffers(np_modules, n_sample, device):
    """Allocate per-layer u_buf [n_sample, d_out] and x_buf [d_in] dicts for the
    all-layer perturbation mode. Shapes are read from each wrapped linear's
    weight [d_out, d_in]. Buffers are zero-initialized on `device` (the noise
    refill / forward overwrites them)."""
    u_buf, x_buf = {}, {}
    for name, wrapped in np_modules.items():
        w = wrapped.wrapped.weight
        d_out, d_in = w.shape[0], w.shape[1]
        u_buf[name] = torch.zeros(n_sample, d_out, device=device, dtype=w.dtype)
        x_buf[name] = torch.zeros(d_in, device=device, dtype=w.dtype)
    return u_buf, x_buf


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
        """Custom decode for ONE prompt. See module docstring + spec §2.

        `layer_name` is EITHER a `str` (single-layer V1 path, the parity oracle)
        OR a `list[str]`/`tuple[str]` (all-layer eager path). In the all-layer
        case every matched layer is perturbed in ONE forward with its own
        per-(layer, q) noise, and `captured_u`/`captured_x` are returned as
        per-layer dicts `{layer: {t: tensor}}` (vs the single-layer `{t: tensor}`)."""
        if isinstance(layer_name, (list, tuple)):
            return self._run_np_decode_all_layers(
                prompt_token_ids, sampling_params, list(layer_name), np_cfg,
                rollout_idx)
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])
        topk_store_k = int(np_cfg.get("topk_store_k", 512))

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
            candidate_logits.append(self._topk_store(logits, topk_store_k))
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

    def _run_np_decode_all_layers(self, prompt_token_ids, sampling_params,
                                  layer_names, np_cfg, rollout_idx):
        """All-layer eager decode for ONE prompt: perturb every layer in
        `layer_names` in ONE forward per token (perturb_all_layers mode), each
        layer with its own per-(layer, q) noise drawn into a per-layer u_buf and
        its own clean-row input captured into a per-layer x_buf. Same per-prompt
        output contract as run_np_decode, but captured_u/captured_x are
        per-layer dicts: {layer: {t: tensor}}."""
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])
        sigma = float(np_cfg["sigma"])
        topk_store_k = int(np_cfg.get("topk_store_k", 512))

        state = self._np_prefill(model, device, list(prompt_token_ids))

        # Per-layer u_buf/x_buf dicts, allocated ONCE and reused across steps
        # (refilled per step). PerturbedLinear (perturb_all_layers branch) reads
        # st["u_buf"][self.name] / writes st["x_buf"][self.name] by reference, so
        # these exact dicts must stay installed for the whole decode.
        subset = {ln: self.np_modules[ln] for ln in layer_names}
        u_buf, x_buf = _alloc_layer_buffers(subset, n_sample, device)
        st.update({
            "mode": "perturb_all_layers",
            "sigma": sigma,
            "n_clean_rows": 1,
            "u_buf": u_buf,
            "x_buf": x_buf,
            # Defensive: ensure the contiguous (non-scatter) perturb_all_layers
            # path; a prior packed decode could have left these behind.
            "perturbed_row_idx": None,
            "clean_row_idx": None,
        })

        clean_tokens, candidate_logits = [], []
        captured_u = {ln: {} for ln in layer_names}
        captured_x = {ln: {} for ln in layer_names}
        try:
            for t in range(max_tokens):
                st["sigma"] = sigma
                st["n_clean_rows"] = 1
                # Refill ALL layers' noise BEFORE the forward (the only RNG site);
                # the forward reads u_buf and writes x_buf for every layer.
                self._np_fill_u_buf_all_layers(
                    st["u_buf"], np_cfg, layer_names, t, rollout_idx, n_sample)
                logits = self._np_step_forward(model, device, state, n_sample)
                candidate_logits.append(self._topk_store(logits, topk_store_k))
                # Capture per-layer u (just refilled) and x (just written by the
                # forward), one CPU clone each, per layer per step.
                for ln in layer_names:
                    captured_u[ln][t] = st["u_buf"][ln].detach().to("cpu").clone()
                    captured_x[ln][t] = st["x_buf"][ln].detach().to("cpu").clone()
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

    def _np_fill_u_buf_all_layers(self, u_buf_dict, np_cfg, matched_layers,
                                  step, rollout, n_sample):
        """Refill every matched layer's u_buf with independent noise per (layer,q),
        seeded identically to V1's single-layer draw -> parity by construction."""
        for layer_name in matched_layers:
            buf = u_buf_dict[layer_name]
            d_out = buf.shape[1]
            for q in range(n_sample):
                seed = noise_seed(int(np_cfg["global_seed"]), int(step),
                                  layer_name, int(rollout), q)
                u = draw_noise(seed, (d_out,), buf.device, buf.dtype,
                               np_cfg["sample_method"])
                buf[q].copy_(u)

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
        topk_store_k = int(np_cfg.get("topk_store_k", 512))

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
            # Defensive: ensure the single-prompt path is taken even if a prior
            # packed decode on this worker left these scatter indices behind
            # (PerturbedLinear reads pri = st.get("perturbed_row_idx")).
            "perturbed_row_idx": None,
            "clean_row_idx": None,
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

                candidate_logits.append(self._topk_store(logits, topk_store_k))
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

    def run_np_decode_packed(self, list_of_prompt_ids, sampling_params, layer_name,
                             np_cfg, rollout_ids):
        """Packed decode for B_pack prompts simultaneously (spec §4). Same per-prompt
        output contract as run_np_decode_graphed, but returns LISTS indexed by prompt:
          clean_tokens[p], candidate_logits[p], captured_x[p], captured_u[p].

        All prompts share one wide forward per token; each keeps its own prefix KV
        (disjoint scratch slices) and its own noise (seeded by rollout_ids[p], so
        identical to what the serial loop drew for that prompt -> parity). A prompt
        that hits EOS is marked inactive and dropped from the next forward; its
        captured signals stop at its EOS token.

        Single-layer only. The all-layer packed path needs a scattered-row,
        per-layer-dict form of PerturbedLinear's perturb_graph branch
        (st["u_buf"][name][pri] / st["x_buf"][name][cri]); that scatter extension
        is added in Stage E (run_np_decode_packed_graphed). Until then a list
        `layer_name` is rejected here rather than silently single-layered."""
        # all-layer packed path: see Stage E (run_np_decode_packed_graphed).
        if isinstance(layer_name, (list, tuple)):
            raise NotImplementedError(
                "run_np_decode_packed is single-layer only; the all-layer packed "
                "path (scattered per-layer u_buf/x_buf) is added in Stage E "
                "(run_np_decode_packed_graphed). Use run_np_decode (eager) for "
                "the all-layer path, or pass a single layer name here.")
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])
        sigma = float(np_cfg["sigma"])
        topk_store_k = int(np_cfg.get("topk_store_k", 512))

        states = self._np_prefill_packed(model, device, list_of_prompt_ids)
        b_pack = len(states)

        # Resolve the perturbed layer's output / input widths to size buffers.
        wrapped = self.np_modules[layer_name]
        weight = wrapped.wrapped.weight          # [d_out, d_in]
        assert weight.is_floating_point(), (
            f"run_np_decode_packed: perturbed layer weight must be floating "
            f"(got {weight.dtype}) for u_buf dtype parity. Layer {layer_name!r}.")
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
            "u_buf": None,
            "x_buf": None,
            "perturbed_row_idx": None,
            "clean_row_idx": None,
        })
        try:
            for t in range(max_tokens):
                active_idx = [p for p in range(b_pack) if states[p]["active"]]
                if not active_idx:
                    break
                n_active = len(active_idx)
                blocks = _packed_row_blocks(n_active, n_sample)  # row layout for ACTIVE prompts

                # Buffers sized to the active set this token.
                # Buffers are re-allocated per token sized to the CURRENT active set
                # (it shrinks as prompts hit EOS); re-alloc is simpler than remapping
                # a fixed buffer across the active-set shrink boundary.
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
                st["clean_row_idx"] = clean_row_idx
                st["perturbed_row_idx"] = perturbed_row_idx

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
                    candidate_logits[p].append(self._topk_store(block, topk_store_k))
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

    # ============================================== E3: graphed orchestrator ==
    def run_np_decode_packed_graphed(self, list_of_prompt_ids, sampling_params,
                                     layer_names, np_cfg, rollout_ids):
        """Fully CUDA-graphed all-layer PACKED decode for B prompts (Stage E3).

        Ties E1 (capture) + E2 (replay) into a full decode driver: prefill all B
        prompts, pick ONE bucket width >= B, capture ONE all-layer packed graph for
        that bucket (cached so a repeat call at the same width never re-captures),
        then loop decode tokens via _np_replay_step_packed, padding finished/unused
        slots as PAD rows (C-4) inside the fixed bucket. Returns the SAME per-prompt
        shape as run_np_decode_packed (lists indexed by prompt) so fit() consumes it
        uniformly:
          clean_tokens[p]      : list[int]
          candidate_logits[p]  : list[(topk_logp, ids)]
          captured_u[p]        : {layer: {t: tensor[n_sample, d_out]}}   (CPU clones)
          captured_x[p]        : {layer: {t: tensor[d_in]}}              (CPU clones)

        Bucket / pad design (the SIMPLEST CORRECT choice, spec): pick the single
        bucket = smallest b in b_pack_buckets with b >= B (clamp/raise if B exceeds
        max). Capture that ONE bucket, pad all B prompts into it, and run the WHOLE
        decode in it -- finished prompts become PAD rows via E2's C-4 logic. We never
        switch buckets mid-decode (no mid-decode recapture), so the captured graph's
        block_table / row layout stay fixed for the whole decode.

        `layer_names` is the LIST of all matched layers (every layer perturbs in one
        forward). `rollout_ids` is the per-prompt rollout id list (from
        _assign_rollout_ids) -- slot p is bound to prompt p, so slot p's noise is
        seeded by rollout_ids[p] (parity with the eager/serial paths)."""
        if not isinstance(layer_names, (list, tuple)):
            raise TypeError(
                "run_np_decode_packed_graphed is the ALL-LAYER packed path; "
                "layer_names must be a list/tuple of matched layer names.")
        layer_names = list(layer_names)
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(np_cfg["n_sample"])
        max_tokens = int(np_cfg["max_tokens"])
        sigma = float(np_cfg["sigma"])
        topk_store_k = int(np_cfg.get("topk_store_k", 512))

        B = len(list_of_prompt_ids)
        assert len(rollout_ids) == B, (
            f"run_np_decode_packed_graphed: {len(rollout_ids)} rollout_ids for "
            f"{B} prompts")
        b_pack_buckets = list(np_cfg.get("b_pack_buckets", [2, 4, 8, 16]))
        bucket = _select_bucket(B, b_pack_buckets)

        # Prefill EXACTLY `bucket` prompt slots. Slots [0..B) are the real prompts;
        # slots [B..bucket) are PAD slots (their KV is prefilled with a real prompt's
        # ids so the graph's block_table is well-defined, but they are NEVER active,
        # so their outputs are discarded -- C-4 keeps their padded rows attention-
        # well-defined). PAD slots reuse prompt 0's ids (any valid prompt works).
        padded_prompt_ids = list(list_of_prompt_ids) + [
            list(list_of_prompt_ids[0]) for _ in range(bucket - B)]
        states = self._np_prefill_packed(model, device, padded_prompt_ids)
        for p in range(B, bucket):
            states[p]["active"] = False  # PAD slot: never decodes.

        # Slot rollout ids: real prompts get their rollout_ids[p]; pad slots get a
        # valid int (their noise is harmless -- those rows are ignored).
        slot_rollout_ids = [int(rollout_ids[p]) for p in range(B)] + [
            int(rollout_ids[0]) for _ in range(bucket - B)]

        max_seq_len_cap = max(s["prompt_len"] for s in states) + max_tokens

        # Capture ONCE per bucket, cached on the worker so repeated calls at the
        # same width skip recapture. C-2: the returned graph_state's u_buf/x_buf
        # dicts are the EXACT pinned objects the replay mutates in place.
        st["sigma"] = sigma
        if not hasattr(self, "_np_graph_by_bucket"):
            self._np_graph_by_bucket = {}
        if bucket not in self._np_graph_by_bucket:
            gs = self._np_capture_step_packed(
                model, device, bucket, n_sample, layer_names, states,
                max_seq_len_cap)
            self._np_graph_by_bucket[bucket] = gs
        gs = self._np_graph_by_bucket[bucket]

        # Per-prompt outputs (only the B real prompts; pad slots contribute none).
        clean_tokens = [[] for _ in range(B)]
        candidate_logits = [[] for _ in range(B)]
        captured_u = [{ln: {} for ln in layer_names} for _ in range(B)]
        captured_x = [{ln: {} for ln in layer_names} for _ in range(B)]

        width = 1 + n_sample
        # TIMING-ISOLATION HARNESS (NP_BENCH_SKIP_NOISE=1): pre-fill every layer's
        # u_buf ONCE so the per-token refill can be skipped while the forward still
        # sees valid noise. Isolates the per-token noise-refill cost (896 draw_noise
        # calls/token) from the rest of decode. Breaks gradient correctness; bench
        # only.
        if os.environ.get("NP_BENCH_SKIP_NOISE"):
            self._np_fill_u_buf_all_layers_packed(
                gs["u_buf"], np_cfg, layer_names, 0, slot_rollout_ids, n_sample)
        try:
            for t in range(max_tokens):
                active_idx = [p for p in range(B) if states[p]["active"]]
                if not active_idx:
                    break
                # ONE noise-refill site lives INSIDE the replay (C-6); the
                # orchestrator never refills noise itself. Replay pads finished/pad
                # slots (C-4) and returns [R, vocab] raw logits.
                logits = self._np_replay_step_packed(
                    model, states, active_idx, n_sample, layer_names, np_cfg, t,
                    slot_rollout_ids, gs)
                # Per ACTIVE real prompt: slice its (1+N) block, sample, capture,
                # advance. u/x are read from the SAME pinned buffers the replay just
                # refilled (u) / the forward just wrote (x), prompt-major sliced and
                # cloned to CPU independently (no aliasing across tokens).
                for p in active_idx:
                    base = p * width
                    block = logits[base:base + width]  # [1+N, vocab]
                    candidate_logits[p].append(
                        self._topk_store(block, topk_store_k))
                    for ln in layer_names:
                        captured_u[p][ln][t] = gs["u_buf"][ln][
                            p * n_sample:(p + 1) * n_sample].detach().to(
                            "cpu").clone()
                        captured_x[p][ln][t] = gs["x_buf"][ln][p].detach().to(
                            "cpu").clone()
                    next_tok = self._np_sample_clean(block[0], sampling_params)
                    clean_tokens[p].append(int(next_tok))
                    if self._np_is_eos(next_tok, sampling_params):
                        states[p]["active"] = False
                    else:
                        self._np_commit_clean(states[p], next_tok)
        finally:
            st["mode"] = "off"

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

    def _np_step_forward_packed(self, model, device, states, n_sample, u_buf,
                                x_buf, clean_row_idx, perturbed_row_idx):
        """One wide forward of R = (#active prompts)*(1+n_sample) rows.

        states: list of per-prompt state dicts (only ACTIVE prompts passed in).
        u_buf:  [#active*n_sample, d_out] host-refilled perturbation buffer;
                row order matches perturbed_row_idx (prompt-major: prompt0's N
                rows, then prompt1's N, ...).
        x_buf:  [#active, d_in] receives each prompt's clean-row input.
        clean_row_idx / perturbed_row_idx: LongTensors of the row positions in
                the packed batch (from _packed_row_blocks).

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

        # Scatter indices (clean_row_idx / perturbed_row_idx) and u_buf/x_buf are
        # installed on np_state by the caller (run_np_decode_packed) before this
        # forward; PerturbedLinear reads them to place the perturbation.
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

    def _np_build_attn_metadata_packed_persistent(self, per_row_block_ids,
                                                  query_lens, seq_lens,
                                                  slot_mapping, positions_cpu,
                                                  max_seq_len_override=None):
        """Persistent packed attn_metadata for the all-layer packed graph (E1).

        Combines _np_build_attn_metadata_packed (num_reqs = R rows, each row
        carrying its OWN seq_len / block_table / slot -- B_pack prompts) with
        _np_build_attn_metadata_persistent (returns the GPU tensors the graph
        pins by reference + freezes FlashAttention's max_seqlen_k at the cap).

        per_row_block_ids: list (len R) of that row's prompt's block_ids list.
        query_lens/seq_lens/slot_mapping/positions_cpu: per-row arrays (len R).
        max_seq_len_override: freeze max_seqlen_k at the decode's CAP (the kernel
        grid is sized by a frozen int; live per-row seqused_k is the seq_lens GPU
        tensor, mutated per replay -- same contract as the single-prompt builder).

        Returns (attn_metadata, total_tokens, meta_bufs). meta_bufs carries the
        per-row slot_mapping / seq_lens GPU tensors the replay (E2) mutates in
        place. Built ONCE; the FlashAttention builder stores these by reference."""
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        mr = self.model_runner
        device = mr.device

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
        meta_bufs = {
            "slot_mapping": slot_mapping_gpu,
            "seq_lens_gpu": sl_gpu,
            "seq_lens_cpu": sl_cpu,
            "qsl_gpu": qsl_gpu,
            "qsl_cpu": qsl_cpu,
            "block_table": bt,
        }
        return attn_metadata, total_tokens, meta_bufs

    def _np_capture_step_packed(self, model, device, bucket_b_pack, n_sample,
                                layer_names, prefill_states, max_seq_len_cap):
        """Capture ONE all-layer packed step forward into a CUDA graph at a FIXED
        bucket width R = bucket_b_pack * (1 + n_sample). The packed analog of
        _np_capture_step: instead of a single prompt's 1+N rows perturbing one
        layer, B_pack prompts' (1+N) blocks all perturb EVERY matched layer in
        one wide forward, each layer with its OWN per-layer u_buf/x_buf dict.

        bucket_b_pack: fixed #prompts this graph serves (the active set is padded
            up to this width at replay -- E2/E3).
        layer_names: the matched layers to perturb (all in this one forward).
        prefill_states: list (len bucket_b_pack) of per-prompt state dicts from
            _np_prefill_packed; supplies block_ids / kv_cursor for the token-0 row.
        max_seq_len_cap: frozen max_seqlen_k (largest seqused_k the decode reaches,
            prompt_len + max_tokens) -- ESSENTIAL, same reason as _np_capture_step.

        ===================== C-2 INVARIANT (READ BEFORE EDITING) =============
        The per-layer u_buf/x_buf DICTS installed on st BELOW are the EXACT tensor
        objects the captured PerturbedLinear.forward pins by pointer. AFTER
        capture you MUST NOT rebind st["u_buf"] / st["x_buf"] to new dict/tensor
        objects -- the graph recorded those storages. The replay (E2) refills
        them IN PLACE via copy_ (u_buf[ln].copy_(...), x_buf[ln] is written by the
        forward). Rebinding -> the graph reads stale buffers -> a graph that
        silently perturbs nothing. This method returns the dicts in graph_state so
        the replay mutates the SAME objects. Likewise clean_row_idx /
        perturbed_row_idx are pinned (scatter indices baked into the captured op).
        =======================================================================

        Returns a graph_state dict (graph + persistent input/meta buffers + the
        per-layer dicts + scatter indices + bucket_b_pack) for E2/E3."""
        from vllm.config.compilation import CUDAGraphMode

        assert len(prefill_states) == bucket_b_pack, (
            f"_np_capture_step_packed: got {len(prefill_states)} prefill states "
            f"for bucket_b_pack={bucket_b_pack}")
        R = bucket_b_pack * (1 + n_sample)
        width = 1 + n_sample

        # Row layout: prompt p owns rows [p*width : p*width+width]; row p*width is
        # clean, the next n_sample are perturbed. Same convention as
        # run_np_decode_packed (_packed_row_blocks).
        blocks = _packed_row_blocks(bucket_b_pack, n_sample)
        clean_row_idx = torch.tensor(
            [blk["clean"] for blk in blocks], dtype=torch.long, device=device)
        perturbed_row_idx = torch.tensor(
            [r for blk in blocks for r in blk["perturbed"]],
            dtype=torch.long, device=device)

        # Per-layer buffer DICTS at PACKED shapes (C-2: pinned BEFORE capture).
        # u_buf[ln]: [bucket_b_pack*n_sample, d_out] (one row per perturbed row,
        # prompt-major, aligned with perturbed_row_idx). x_buf[ln]: [bucket_b_pack,
        # d_in] (one clean-input row per prompt, captured via clean_row_idx).
        subset = {ln: self.np_modules[ln] for ln in layer_names}
        u_buf_dict, x_buf_dict = {}, {}
        for ln, wrapped in subset.items():
            w = wrapped.wrapped.weight                 # [d_out, d_in]
            assert w.is_floating_point(), (
                f"_np_capture_step_packed: layer {ln!r} weight must be floating "
                f"(got {w.dtype}) for u_buf dtype parity.")
            d_out, d_in = int(w.shape[0]), int(w.shape[1])
            u_buf_dict[ln] = torch.zeros(
                bucket_b_pack * n_sample, d_out, device=device, dtype=w.dtype)
            x_buf_dict[ln] = torch.zeros(
                bucket_b_pack, d_in, device=device, dtype=w.dtype)

        st = self._ensure_np_state()
        sigma = float(st.get("sigma", 0.0))
        # Install the EXACT dict objects the graph will pin. NEVER rebind these
        # after capture (C-2). Replay mutates u_buf_dict[ln] / x_buf_dict[ln] in
        # place; the forward reads st["u_buf"][self.name] during capture.
        st["mode"] = "perturb_all_layers"
        st["n_clean_rows"] = 1
        st["u_buf"] = u_buf_dict
        st["x_buf"] = x_buf_dict
        st["perturbed_row_idx"] = perturbed_row_idx
        st["clean_row_idx"] = clean_row_idx
        st["sigma"] = sigma

        # Persistent input buffers (fixed shape R for the graph's life).
        input_ids_buf = torch.zeros(R, dtype=torch.long, device=device)
        positions_buf = torch.zeros(R, dtype=torch.long, device=device)

        # Build the packed attn_metadata ONCE at the bucket width. Each prompt's
        # (1+N) rows carry its block_ids + token-0 seq_len (q_pos+1); max_seqlen_k
        # is frozen at the cap. Per replay (E2) we mutate the clean-row slots +
        # per-row seqused_k in place; block_table is fixed (prefix KV).
        per_row_block_ids, slot_mapping, positions, seq_lens, query_lens = (
            [], [], [], [], [])
        token0 = []
        for p, state in enumerate(prefill_states):
            block_ids = state["block_ids"]
            block_size = state["block_size"]
            prompt_len = state["prompt_len"]
            q_pos = state["kv_cursor"]
            if q_pos < prompt_len:
                q_token = state["prompt_token_ids"][q_pos]
            else:
                q_token = state["committed_tokens"][q_pos - prompt_len]
            clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
            # clean row writes KV; perturbed rows PAD(-1) -> reshape_and_cache skips.
            per_row_block_ids += [block_ids] * width
            slot_mapping += [clean_slot] + [-1] * n_sample
            positions += [q_pos] * width
            seq_lens += [q_pos + 1] * width
            query_lens += [1] * width
            token0 += [(int(q_token), int(q_pos))] * width

        attn_meta, total, meta_bufs = (
            self._np_build_attn_metadata_packed_persistent(
                per_row_block_ids, query_lens, seq_lens, slot_mapping, positions,
                max_seq_len_override=max_seq_len_cap))

        # Seed input buffers with token-0 ids/positions (per-row; prompts differ).
        ids_cpu = torch.tensor([t[0] for t in token0], dtype=torch.long)
        pos_cpu = torch.tensor([t[1] for t in token0], dtype=torch.long)
        input_ids_buf.copy_(ids_cpu.to(device))
        positions_buf.copy_(pos_cpu.to(device))

        # Warm up a few eager steps so cuBLAS workspaces / autotune settle before
        # capture (vLLM does the same before its own capture).
        for _ in range(3):
            with torch.no_grad(), set_forward_context(
                attn_meta, self.model_runner.vllm_config, num_tokens=total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE):
                _ = model(input_ids=input_ids_buf, positions=positions_buf)
        torch.cuda.synchronize()

        # Capture. Graph-pool-release gotcha (VERBATIM from _np_capture_step): a
        # prior graph is still alive (use_count>0) when we capture the next one,
        # and capturing into a pool a live graph holds trips CUDACachingAllocator's
        # "use_count > 0" assert. Release the PREVIOUS graph (freeing its pool)
        # before capturing so each graph owns its own default pool.
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

        # Post-fix the perturbed-row slots to PAD(-1) once. Only the clean rows'
        # slots (clean_row_idx) change per token; perturbed rows never write KV.
        meta_bufs["slot_mapping"][perturbed_row_idx] = -1

        return {
            "graph": graph,
            "input_ids_buf": input_ids_buf,
            "positions_buf": positions_buf,
            "hidden_buf": hidden_buf,
            "meta_bufs": meta_bufs,
            "attn_meta": attn_meta,
            "total": total,
            # C-2: the EXACT pinned objects -- replay mutates these IN PLACE.
            "u_buf": u_buf_dict,
            "x_buf": x_buf_dict,
            "clean_row_idx": clean_row_idx,
            "perturbed_row_idx": perturbed_row_idx,
            "bucket_b_pack": bucket_b_pack,
            "n_sample": n_sample,
            "layer_names": list(layer_names),
        }

    def _np_fill_u_buf_all_layers_packed(self, u_buf_dict, np_cfg, layer_names,
                                         step, slot_rollout_ids, n_sample):
        """C-6 single noise-refill site for the all-layer PACKED graph (E2).

        The all-layer analog of run_np_decode_packed's per-prompt refill loop, but
        for EVERY matched layer's u_buf at once. Each layer's packed u_buf is
        [bucket_b_pack*n_sample, d_out] prompt-major: rows [p*n_sample : (p+1)*
        n_sample] belong to slot p. We seed slot p's N rows with ITS rollout id
        slot_rollout_ids[p] and the SAME noise_seed(global_seed, step, layer,
        rollout, q) key the eager packed / serial paths use -> the bytes written
        are bit-identical to those paths (parity-by-construction, spec §3/§5.2).

        slot_rollout_ids: list (len bucket_b_pack) of the rollout id bound to each
        graph SLOT. For a FINISHED/PAD slot (output discarded) the id is still a
        valid int; the noise written is harmless (those rows are ignored), but we
        fill them anyway so the captured buffer is never read undefined -- keeping
        the replay deterministic regardless of which slots are active this token.
        """
        bucket_b_pack = len(slot_rollout_ids)
        for layer_name in layer_names:
            buf = u_buf_dict[layer_name]          # [bucket_b_pack*n_sample, d_out]
            d_out = buf.shape[1]
            for p in range(bucket_b_pack):
                rid = int(slot_rollout_ids[p])
                for q in range(n_sample):
                    seed = noise_seed(int(np_cfg["global_seed"]), int(step),
                                      layer_name, rid, q)
                    u = draw_noise(seed, (d_out,), buf.device, buf.dtype,
                                   np_cfg["sample_method"])
                    buf[p * n_sample + q].copy_(u)

    def _np_replay_step_packed(self, model, states, active_idx, n_sample,
                               layer_names, np_cfg, step, slot_rollout_ids,
                               graph_state):
        """Run ONE decode token across all bucket prompts via packed CUDA-graph
        replay (E2). The packed analog of _np_replay_step.

        states: list (len bucket_b_pack) of per-slot state dicts (from
            _np_prefill_packed); slot p is the prompt bound to graph slot p at
            capture (its KV is baked into the graph's block_table -- a prompt
            CANNOT change slots). states[p]["active"] flags whether slot p decodes
            this token.
        active_idx: indices into states of slots active THIS token (len <=
            bucket_b_pack). Inactive slots are FINISHED (hit EOS) or PAD (the
            active set is < the bucket width). All are C-4 pad-handled.
        slot_rollout_ids: rollout id bound to each slot (parity seed identity).
        graph_state: the dict from _np_capture_step_packed (graph + persistent
            input/meta buffers + per-layer u_buf/x_buf dicts + clean/perturbed row
            indices + bucket_b_pack). Mutated IN PLACE; never rebound (C-2).

        Returns [R, vocab] logits (R = bucket_b_pack*(1+n_sample)). The caller
        slices each ACTIVE slot's (1+n_sample) block; finished/pad rows are
        discarded but kept attention-well-defined (C-4)."""
        gs = graph_state
        bucket_b_pack = int(gs["bucket_b_pack"])
        assert len(states) == bucket_b_pack, (
            f"_np_replay_step_packed: {len(states)} states for bucket width "
            f"{bucket_b_pack}")
        width = 1 + n_sample
        active_set = set(int(i) for i in active_idx)

        # --- Per-slot metadata (C-4): active -> real; finished/pad -> last-valid.
        slot_states = []
        for p in range(bucket_b_pack):
            st_p = states[p]
            block_ids = st_p["block_ids"]
            block_size = st_p["block_size"]
            prompt_len = st_p["prompt_len"]
            q_pos = int(st_p["kv_cursor"])
            if q_pos < prompt_len:
                q_token = st_p["prompt_token_ids"][q_pos]
            else:
                q_token = st_p["committed_tokens"][q_pos - prompt_len]
            # LAST-VALID fallback for a pad/finished slot: the slot has always
            # decoded at least its token-0 (q_pos >= prompt_len-1 >= 0 after
            # prefill), so kv_cursor / its token are a valid in-range position even
            # for a just-finished prompt. (We never advance kv_cursor for a slot
            # that hit EOS, so kv_cursor stays the last real position -> seq_len>0.)
            is_active = p in active_set and bool(st_p["active"])
            clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
            slot_states.append({
                "active": is_active,
                "q_token": int(q_token),
                "q_pos": q_pos,
                "clean_slot": int(clean_slot),
                "last_q_token": int(q_token),
                "last_q_pos": q_pos,
            })
        meta = _packed_replay_row_meta(slot_states)

        # --- Refill persistent input + metadata buffers IN PLACE (pointers stable;
        # the graph holds these by reference). Row layout: slot p owns rows
        # [p*width : p*width+width]; row p*width is clean, the next N perturbed.
        ids_buf = gs["input_ids_buf"]
        pos_buf = gs["positions_buf"]
        mb = gs["meta_bufs"]
        sm = mb["slot_mapping"]           # [R] int64; perturbed rows fixed -1 at capture
        sl = mb["seq_lens_gpu"]           # [R] int32 seqused_k (the load-bearing tensor)
        for p in range(bucket_b_pack):
            m = meta[p]
            base = p * width
            # all width rows query the same token at the same position.
            ids_buf[base:base + width].fill_(m["q_token"])
            pos_buf[base:base + width].fill_(m["q_pos"])
            sl[base:base + width].fill_(m["seq_len"])
            # clean row's slot is m["clean_slot"] (-1 for finished/pad). Perturbed
            # rows were fixed to -1 at capture -- never write KV -- so we only set
            # the clean row's slot here.
            sm[base].fill_(m["clean_slot"])

        # --- C-6 single noise-refill site: ALL layers' u_buf, seeded per slot.
        # TIMING-ISOLATION HARNESS (NP_BENCH_SKIP_NOISE=1): skip the per-token
        # refill to measure decode cost WITHOUT the 896 host-orchestrated
        # draw_noise calls/token. The orchestrator pre-fills u_buf ONCE before the
        # loop so the buffers still hold valid (non-garbage) noise -> the forward
        # is representative, only the per-token regeneration is removed. This
        # BREAKS gradient correctness (stale noise) and is for wall-clock
        # attribution ONLY; never set it in a training run.
        if not os.environ.get("NP_BENCH_SKIP_NOISE"):
            self._np_fill_u_buf_all_layers_packed(
                gs["u_buf"], np_cfg, layer_names, step, slot_rollout_ids, n_sample)

        # --- Replay + sync + eager per-token logits over ALL rows.
        gs["graph"].replay()
        torch.cuda.synchronize()
        logits = model.compute_logits(gs["hidden_buf"])   # [R, vocab]
        return logits

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

    def _np_prefill_packed(self, model, device, list_of_prompt_ids,
                           max_new_tokens=None):
        """Prefill B_pack prompts, each into its OWN disjoint high-indexed scratch-KV
        slice. Returns a list of per-prompt state dicts (same shape as _np_prefill's
        state, plus 'active': True). Fails fast if the B_pack slices don't fit the
        GPU block pool.

        Each prompt p gets blocks_per_prompt blocks carved from the TOP of the pool
        downward and disjoint across prompts:
          prompt 0: [num_gpu_blocks - 1*bpp, num_gpu_blocks)
          prompt 1: [num_gpu_blocks - 2*bpp, num_gpu_blocks - 1*bpp)
          ...
        so no two prompts' clean rows ever write the same KV slot.

        max_new_tokens: reserve for the ACTUAL budget -- longest prompt in this
            wave plus this many generated tokens -- instead of the full
            max_model_len. Reserving max_model_len wastes ~20x the KV a
            1024-token generation needs and was what capped pack_width at 8
            (results/zo_opd.md). None keeps the old full-context reservation.

        SAFETY: the attention block table is zero-filled and only the first
        len(block_ids) entries are written, so a sequence that outgrew its slice
        would read block 0 -- silently corrupting another slot's KV instead of
        erroring. The assert below is what makes that unreachable; it must stay
        exact (>= the largest position the decode loop can ever write).
        """
        mr = self.model_runner
        block_size = int(mr.cache_config.block_size)
        num_gpu_blocks = int(mr.cache_config.num_gpu_blocks)
        max_blocks = int(
            mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)

        b_pack = len(list_of_prompt_ids)
        longest_prompt = max(len(p) for p in list_of_prompt_ids)
        if max_new_tokens is None:
            need = int(mr.max_model_len)
        else:
            need = min(longest_prompt + int(max_new_tokens),
                       int(mr.max_model_len))
        blocks_per_prompt = min((need + block_size - 1) // block_size, max_blocks)
        cap = blocks_per_prompt * block_size
        # The decode loop writes clean KV at positions [0, prompt_len + T), so the
        # slice must cover the longest prompt plus the whole generation budget.
        assert longest_prompt + int(max_new_tokens or 0) <= cap, (
            f"packed scratch KV slice too small: longest_prompt={longest_prompt} "
            f"+ max_new_tokens={max_new_tokens} > blocks_per_prompt="
            f"{blocks_per_prompt} x block_size={block_size} = {cap}. "
            f"Raise the reservation or lower max_tokens.")
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

    def _topk_store(self, logits, k):
        """Store top-k log-probs (GPU-side log_softmax over FULL vocab) + ids.

        log_softmax MUST be over the full vocab BEFORE slicing, else the
        normalizer is wrong. ids come from the clean/student row (0); ALL rows
        are gathered on those same ids -> [1+N, k] log-probs + [k] ids, both on
        CPU. Replaces the full-vocab D2H copy in the decode drivers (the scorer
        consumes top-k log-probs directly). k is clamped to the vocab size.
        """
        lp = torch.log_softmax(logits.float(), dim=-1)
        k = min(int(k), lp.shape[-1])
        ids = torch.topk(lp[0], k).indices
        return lp[:, ids].to("cpu"), ids.to("cpu")

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

    def assemble_all_layers_and_apply(self, layer_signals, L_q_steps, L_clean_steps,
                                      sigma, sample_mode, normalize, token_agg,
                                      lr, update_clip):
        """One-RPC bundle of (assemble δW) + (apply δW locally) for ALL layers.

        layer_signals: {name: {"u": u_steps, "x": x_steps}}. The shared loss
        L_q_steps/L_clean_steps (one combined loss per (token, sample), scored on
        the clean rollout) is passed ONCE and reused for every layer -- never
        copied T×L times (the C-5 bug this design avoids). We detach the shared
        L_q_steps once before the loop, batch-assemble every layer's δW with one
        GPU reduce per layer (device=self.device), then apply each in-place.

        Returns {name: ‖δW‖} (post-clip) so the trainer can log + assert >0 per layer.
        """
        L_q_dev = [lq.detach() if hasattr(lq, "detach") else torch.as_tensor(lq)
                   for lq in L_q_steps]
        dws = assemble_all_layers(L_q_dev, L_clean_steps, layer_signals,
                                  sigma=sigma, sample_mode=sample_mode,
                                  normalize=normalize, token_agg=token_agg,
                                  device=self.device)
        return {ln: float(self.apply_node_update(ln, dw, lr, update_clip))
                for ln, dw in dws.items()}


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


def assemble_all_layers(L_q_per_step, L_clean_per_step, layer_signals,
                        sigma, sample_mode, normalize, token_agg, device=None):
    """Batched assemble for ALL layers. layer_signals: {name: {"u":[...], "x":[...]}}.
    L_q/L_clean are SHARED across layers (one combined loss per (token,sample)).
    Returns {name: dW [d_out,d_in] on CPU}. Reuses assemble_layer_delta per layer."""
    return {ln: assemble_layer_delta(L_q_per_step, L_clean_per_step,
                                     sig["u"], sig["x"], sigma=sigma,
                                     sample_mode=sample_mode, normalize=normalize,
                                     token_agg=token_agg, device=device)
            for ln, sig in layer_signals.items()}


def _select_bucket(B, b_pack_buckets):
    """Pick the ONE graph bucket width for B prompts (E3). The smallest bucket
    >= B from b_pack_buckets, so all B prompts fit one captured graph and the
    leftover slots are PAD rows (C-4). Raises if B exceeds the largest bucket --
    the caller must chunk B down to <= max(b_pack_buckets) (the graphed driver
    needs a FIXED-width graph; it cannot wave a single graph over more prompts
    than it was captured for). B must be >= 1.

    Examples (buckets [2,4,8,16]): B=1->2, B=3->4, B=4->4, B=5->8, B=16->16."""
    if int(B) < 1:
        raise ValueError(f"_select_bucket: B must be >= 1 (got {B})")
    buckets = sorted(int(b) for b in b_pack_buckets)
    if not buckets:
        raise ValueError("_select_bucket: b_pack_buckets is empty")
    for b in buckets:
        if b >= int(B):
            return b
    raise ValueError(
        f"_select_bucket: B={B} exceeds the largest bucket {buckets[-1]}; the "
        f"caller must chunk B down to <= {buckets[-1]} (b_pack_buckets="
        f"{buckets}).")


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


def _packed_replay_row_meta(slot_states):
    """C-4 pad-row metadata for the packed-graph replay (pure Python, testable).

    The captured packed graph serves a FIXED set of `bucket_b_pack` prompt SLOTS;
    slot p is bound to one prompt (its KV block_ids were baked into the graph's
    block_table at capture, so a prompt CANNOT be moved to another slot -- it must
    stay in its own slot for the whole decode). Each slot owns (1+n_sample) rows:
    one clean row (writes KV) + n_sample perturbed rails (slot=-1, never write KV).

    Input `slot_states`: list (len bucket_b_pack), one dict per slot, with the
    fields needed to derive THIS token's per-slot metadata:
      active       : bool -- ACTIVE (decode this token) vs FINISHED/PAD.
      q_token      : int  -- input id queried this token.
      q_pos        : int  -- absolute position (== seq_len-1).
      clean_slot   : int  -- KV slot the clean row writes when ACTIVE.
      last_q_token : int  -- the slot's LAST VALID token id (for pad reuse).
      last_q_pos   : int  -- the slot's LAST VALID position (for pad reuse).
    (The caller fills q_token/q_pos/clean_slot from states[p] for active slots and
    last_q_* from the slot's last committed token; this helper just *selects*.)

    Per slot returns {"q_token","q_pos","seq_len","clean_slot"}:
      ACTIVE slot   -> the REAL values; clean_slot = its KV slot; seq_len=q_pos+1.
      FINISHED/PAD  -> C-4: the row's output is DISCARDED, but the kernel still
        processes all R rows, so the row must be attention-WELL-DEFINED. Reuse the
        slot's LAST VALID (q_token,q_pos) -- NEVER 0/stale -- so seq_len=q_pos+1>0
        and the position is in-range; FORCE clean_slot=-1 (a finished prompt
        writes NO KV; its clean row must not corrupt the cache). A zero seqused_k
        makes FlashAttention's KV iteration undefined (the #1 way to crash /
        corrupt the ACTIVE rows' attention) -- that is exactly what this avoids.
    """
    meta = []
    for s in slot_states:
        if s["active"]:
            q_pos = int(s["q_pos"])
            meta.append({
                "q_token": int(s["q_token"]),
                "q_pos": q_pos,
                "seq_len": q_pos + 1,
                "clean_slot": int(s["clean_slot"]),
            })
        else:
            # C-4: last-valid (never zero) seq_len/position; clean_slot forced -1.
            q_pos = int(s["last_q_pos"])
            meta.append({
                "q_token": int(s["last_q_token"]),
                "q_pos": q_pos,
                "seq_len": q_pos + 1,
                "clean_slot": -1,
            })
    return meta


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


def _pad_waves_to_pack_width(slot_pids, slot_rids, pack_width):
    """Chunk (prompt,rollout) slots into FIXED-width waves for the graphed packed
    all-layer driver. Every wave has EXACTLY `pack_width` prompts -- the final
    short wave is PADDED up to `pack_width` by REPEATING slot 0's prompt/rollout
    id. This guarantees `_select_bucket(pack_width)` picks the SAME bucket for
    every wave, so `run_np_decode_packed_graphed` captures ONE graph and never
    trips the CUDACachingAllocator "use_count>0" assert from a 2nd distinct-bucket
    capture while the first is cache-pinned (the E3 multi-bucket carry-forward).

    Returns a list of (wave_pids, wave_rids, real_count) tuples: wave_pids/
    wave_rids are length `pack_width`; real_count (<= pack_width) is how many of
    the leading slots are real -- the trainer slices outputs to [:real_count] and
    discards the padded tail. slot_pids/slot_rids must be non-empty and same len."""
    if len(slot_pids) != len(slot_rids):
        raise ValueError(
            f"_pad_waves_to_pack_width: {len(slot_pids)} pids vs "
            f"{len(slot_rids)} rids")
    if not slot_pids:
        raise ValueError("_pad_waves_to_pack_width: empty slots")
    pack_width = int(pack_width)
    if pack_width < 1:
        raise ValueError(
            f"_pad_waves_to_pack_width: pack_width must be >= 1 (got {pack_width})")
    waves = []
    for w0 in range(0, len(slot_pids), pack_width):
        wave_pids = list(slot_pids[w0:w0 + pack_width])
        wave_rids = list(slot_rids[w0:w0 + pack_width])
        real_count = len(wave_pids)
        # Pad the short final wave up to pack_width by repeating slot 0 (any valid
        # prompt/id works -- the padded tail's outputs are discarded).
        while len(wave_pids) < pack_width:
            wave_pids.append(list(slot_pids[0]) if isinstance(slot_pids[0], list)
                             else slot_pids[0])
            wave_rids.append(slot_rids[0])
        waves.append((wave_pids, wave_rids, real_count))
    return waves
