"""es_token worker extension: per-token rank-1 WEIGHT-perturbation rails inside
a fully-CUDA-graphed packed decode. See docs/plans/es_token_trainer.md.

Subclasses the NP V3 WorkerExtension and reuses its prefill / attn-metadata /
slot / EOS / commit / NCCL machinery verbatim. What changes vs NP:

  PERTURBATION  rail n at every matched linear: y += sigma * ((r_n (.) v_t)^T x)
                * (s_n (.) u_t)  -- a rank-1 WEIGHT perturbation applied without
                materializing delta_W. Signs (s_n, r_n) are FIXED buffers
                (Hadamard rails); (u_t, v_t) is ONE shared per-(slot, token)
                noise vector read as fixed views of a flat noise_buf.
  NOISE         one fused draw per (slot, token) covering ALL layers
                (vs NP's per-(layer, rail) draws -- the measured 74%-of-decode
                refill tax does not exist here by construction).
  CAPTURES      none. No x capture, no u capture. The per-token payload is the
                per-rail logprob of the clean sampled token (gather + logsumexp
                into a device buffer, ONE D2H at wave end).
  SYNC          no per-token full cuda.synchronize(); the only host read per
                token is the sampled tokens' .tolist() (set ES_FULL_SYNC=1 to
                restore the blanket sync if an ordering issue is ever suspected).
  SIGMA         a [1] device tensor multiplied IN-graph (a graph input), so the
                same captured graph serves any sigma incl. 0 -- unlike a baked
                python-float scalar, which would freeze the capture-time sigma
                into the graph.
  ASSEMBLY      chunked GEMMs from seed-regenerated noise (es_assemble_and_apply)
                -- no per-token Python reduction.
"""
import os

import torch

from verl.trainer.es_token.grad_estimator import assemble_chunk
from verl.trainer.es_token.rail_kernel import apply_rail, rail_supported
from verl.trainer.es_token.noise_kernel import fill_rademacher_rows
from verl.trainer.es_token.seeding import (
    build_noise_layout, build_seed_table, draw_token_noise, es_token_seed)
from verl.trainer.es_token.signs import build_layer_signs
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    WorkerExtension as NPWorkerExtension,
    _packed_replay_row_meta,
    _packed_row_blocks,
    _repack,
    _select_bucket,
    _unpack,
)

try:  # vLLM-internal import used by capture/eager forwards (matches NP).
    from vllm.forward_context import set_forward_context
except Exception:  # pragma: no cover - CPU unit tests don't import vLLM
    set_forward_context = None


class ESTokenLinear(torch.nn.Module):
    """Wraps a matched vLLM linear with the es_token rank-1 rail perturbation.

    Active only in mode "perturb_es". Row layout is the packed NP layout:
    perturbed rows are scattered (st["perturbed_row_idx"]), row st["clean_row_idx"][p]
    is slot p's clean row. Each perturbed row i belongs to rail
    st["es_rail_idx"][i] of slot st["es_prompt_idx"][i] and gets

        y[i] += sigma_l * ((R[rail] (.) v[slot])^T x[i]) * (S[rail] (.) u[slot])

    where (u, v) are this layer's fixed views into the flat per-slot noise_buf
    and sigma_l is a [1] device tensor (a graph input -- NOT a baked scalar).
    All operands are persistent buffers or fixed index tensors; no RNG, fixed
    shapes -> CUDA-graph-capturable.
    """

    def __init__(self, wrapped: torch.nn.Module, name: str, st_ref):
        super().__init__()
        self.wrapped = wrapped
        self.name = name
        self._st_ref = st_ref

    def forward(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        st = self._st_ref()
        if st.get("mode") != "perturb_es":
            return out
        x = args[0]
        y, bias, was_tuple = _unpack(out)

        off_u, d_out, off_v, d_in = st["es_layout"][self.name]
        nb = st["es_noise_buf"]                       # [bucket, d_total]
        sigma = st["es_sigma_buf"][self.name]         # [1] device tensor
        pri = st["perturbed_row_idx"]                 # [P]
        rail = st["es_rail_idx"]                      # [P]
        pidx = st["es_prompt_idx"]                    # [P]

        sf = st.get("es_signs_flat")
        if sf is not None and rail_supported(x, y):
            # ONE fused launch for the whole rail op (see rail_kernel.py). The
            # PyTorch form below issues ~14 kernels per layer, which at 112
            # layers made the decode CUDA-graph node-bound.
            apply_rail(x, y, nb, sf, sigma, pri, rail, pidx,
                       off_u, d_out, off_v, d_in, nb.stride(0), sf.stride(0))
            return _repack(y, bias, was_tuple)

        u = nb[:, off_u:off_u + d_out]                # [bucket, d_out] view
        v = nb[:, off_v:off_v + d_in]                 # [bucket, d_in]  view
        S, R = st["es_signs"][self.name]              # [N, d_out], [N, d_in]
        x_p = x[pri]                                  # [P, d_in]
        v_eff = R[rail] * v[pidx]                     # [P, d_in]
        alpha = (x_p * v_eff).sum(dim=-1, keepdim=True)   # [P, 1]
        u_eff = S[rail] * u[pidx]                     # [P, d_out]
        y[pri] = y[pri] + sigma * alpha * u_eff
        return _repack(y, bias, was_tuple)


class WorkerExtension(NPWorkerExtension):
    # ------------------------------------------------------------- install ---
    def install_es_layers(self, perturb_rules, n_rails, global_seed):
        """Wrap every matched linear with ESTokenLinear; build the flat-noise
        layout + fixed Hadamard sign buffers. Idempotent. Returns the resolved
        layer-name list (named_modules order -- identical on every worker, so
        the noise layout is identical everywhere)."""
        from verl.trainer.np.layer_resolve import resolve_modules

        st = self._ensure_np_state()
        model = self.model_runner.model
        device = self.model_runner.device
        names = [n for n, _ in model.named_modules()]
        matched = resolve_modules(list(perturb_rules), names,
                                  error_if_empty=True)

        self.np_modules = {}   # name kept for inherited broadcast/norm helpers
        layer_dims = []
        for layer_name in matched:
            parent = model
            *path, leaf = layer_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            child = getattr(parent, leaf)
            if isinstance(child, ESTokenLinear):
                wrapped = child
            else:
                wrapped = ESTokenLinear(child, layer_name, lambda: self.np_state)
                setattr(parent, leaf, wrapped)
            self.np_modules[layer_name] = wrapped
            w = wrapped.wrapped.weight
            assert w.is_floating_point(), (
                f"es_token: layer {layer_name!r} weight must be floating "
                f"(got {w.dtype}).")
            layer_dims.append((layer_name, int(w.shape[0]), int(w.shape[1])))

        layout, d_total = build_noise_layout(layer_dims)
        self.es_layout = layout
        self.es_d_total = int(d_total)
        self.es_n_rails = int(n_rails)
        self.es_dtype = self.np_modules[matched[0]].wrapped.weight.dtype
        self.es_signs = {}
        self.es_w_rms = {}
        # Flat [n_rails, d_total] copy of every layer's signs in the SAME layout
        # as the per-slot noise buffer, so the fused rail kernel can address a
        # layer's s/r rows by (rail, offset) without a per-layer gather.
        self.es_signs_flat = torch.empty(int(n_rails), int(d_total),
                                         device=device, dtype=self.es_dtype)
        for layer_name, d_out, d_in in layer_dims:
            self.es_signs[layer_name] = build_layer_signs(
                layer_name, int(n_rails), d_out, d_in, int(global_seed),
                self.es_dtype, device)
            off_u, _, off_v, _ = layout[layer_name]
            S_l, R_l = self.es_signs[layer_name]
            self.es_signs_flat[:, off_u:off_u + d_out] = S_l
            self.es_signs_flat[:, off_v:off_v + d_in] = R_l
        for layer_name, d_out, d_in in layer_dims:
            w = self.np_modules[layer_name].wrapped.weight
            self.es_w_rms[layer_name] = float(
                w.detach().float().pow(2).mean().sqrt().item())
        st["mode"] = "off"
        return list(matched)

    def _es_sigma_eff(self, es_cfg):
        """Per-layer effective sigma: absolute (default) or sigma*RMS(W_l)."""
        sigma = float(es_cfg["sigma"])
        if es_cfg.get("sigma_mode", "absolute") == "relative":
            return {ln: sigma * self.es_w_rms[ln] for ln in self.es_layout}
        return {ln: sigma for ln in self.es_layout}

    # --------------------------------------------------------------- noise ---
    def _es_fill_noise(self, noise_buf, es_cfg, step_t, slot_rollout_ids):
        """ONE fused draw per (slot, token) covering all layers. The only RNG
        in the decode hot loop. Bit-regenerable at assembly from
        (global_seed, t, rollout_id) on the same device/dtype.

        For the shipping "bernoulli" method this is a single Triton launch that
        writes +-1 straight into the destination dtype (noise_kernel.py); the
        seeds come from a table built once per wave. Other methods keep the
        original per-slot draw."""
        gseed = int(es_cfg["global_seed"])
        method = es_cfg["sample_method"]
        d_total = noise_buf.shape[1]
        if method == "bernoulli":
            tbl = getattr(self, "_es_seed_tbl", None)
            t = int(step_t)
            if tbl is not None and t < tbl[0].shape[0]:
                seeds_dev, seeds_host = tbl[0][t], tbl[1][t]
            else:  # eager/oracle callers that never built a table
                seeds_host = [es_token_seed(gseed, t, int(rid))
                              for rid in slot_rollout_ids]
                seeds_dev = None
            fill_rademacher_rows(noise_buf, seeds_host, seeds_dev)
            return
        for p, rid in enumerate(slot_rollout_ids):
            noise_buf[p].copy_(draw_token_noise(
                gseed, int(step_t), int(rid), d_total, noise_buf.device,
                noise_buf.dtype, method))

    # ------------------------------------------------------------- capture ---
    def _es_install_state(self, bucket, n_sample, device):
        """Allocate (or reuse) the per-bucket persistent es buffers and install
        them on np_state. Returns the runstate dict the decode loop uses.
        For the graphed path these EXACT objects are pinned by the capture --
        never rebound afterwards; refilled in place."""
        blocks = _packed_row_blocks(bucket, n_sample)
        clean_row_idx = torch.tensor([blk["clean"] for blk in blocks],
                                     dtype=torch.long, device=device)
        perturbed_row_idx = torch.tensor(
            [r for blk in blocks for r in blk["perturbed"]],
            dtype=torch.long, device=device)
        rail_idx = torch.tensor(
            [n for _ in range(bucket) for n in range(n_sample)],
            dtype=torch.long, device=device)
        prompt_idx = torch.tensor(
            [p for p in range(bucket) for _ in range(n_sample)],
            dtype=torch.long, device=device)
        noise_buf = torch.zeros(bucket, self.es_d_total, device=device,
                                dtype=self.es_dtype)
        sigma_buf = {ln: torch.zeros(1, device=device, dtype=self.es_dtype)
                     for ln in self.es_layout}
        st = self._ensure_np_state()
        st.update({
            "mode": "perturb_es",
            "es_noise_buf": noise_buf,
            "es_layout": self.es_layout,
            "es_signs": self.es_signs,
            "es_signs_flat": self.es_signs_flat,
            "es_sigma_buf": sigma_buf,
            "perturbed_row_idx": perturbed_row_idx,
            "clean_row_idx": clean_row_idx,
            "es_rail_idx": rail_idx,
            "es_prompt_idx": prompt_idx,
        })
        return {
            "noise_buf": noise_buf,
            "sigma_buf": sigma_buf,
            "clean_row_idx": clean_row_idx,
            "perturbed_row_idx": perturbed_row_idx,
            "rail_idx": rail_idx,
            "prompt_idx": prompt_idx,
            "bucket": bucket,
            "n_sample": n_sample,
        }

    def _es_capture_step_packed(self, model, device, bucket, n_sample,
                                prefill_states, max_seq_len_cap, rs):
        """Capture ONE es_token packed step forward at fixed bucket width.
        Mirrors NP's _np_capture_step_packed (persistent input/meta buffers,
        warmup, per-graph pool release, frozen max_seqlen_k at the cap) with the
        es perturbation state already installed via _es_install_state."""
        from vllm.config.compilation import CUDAGraphMode

        assert len(prefill_states) == bucket
        width = 1 + n_sample
        R = bucket * width

        input_ids_buf = torch.zeros(R, dtype=torch.long, device=device)
        positions_buf = torch.zeros(R, dtype=torch.long, device=device)

        per_row_block_ids, slot_mapping, positions, seq_lens, query_lens = (
            [], [], [], [], [])
        token0 = []
        for state in prefill_states:
            block_ids = state["block_ids"]
            block_size = state["block_size"]
            prompt_len = state["prompt_len"]
            q_pos = state["kv_cursor"]
            if q_pos < prompt_len:
                q_token = state["prompt_token_ids"][q_pos]
            else:
                q_token = state["committed_tokens"][q_pos - prompt_len]
            clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
            per_row_block_ids += [block_ids] * width
            slot_mapping += [clean_slot] + [-1] * n_sample
            positions += [q_pos] * width
            seq_lens += [q_pos + 1] * width
            query_lens += [1] * width
            token0 += [(int(q_token), int(q_pos))] * width

        attn_meta, total, meta_bufs = (
            self._np_build_attn_metadata_packed_persistent(
                per_row_block_ids, query_lens, seq_lens, slot_mapping,
                positions, max_seq_len_override=max_seq_len_cap))

        ids_cpu = torch.tensor([t[0] for t in token0], dtype=torch.long)
        pos_cpu = torch.tensor([t[1] for t in token0], dtype=torch.long)
        input_ids_buf.copy_(ids_cpu.to(device))
        positions_buf.copy_(pos_cpu.to(device))

        for _ in range(3):
            with torch.no_grad(), set_forward_context(
                attn_meta, self.model_runner.vllm_config, num_tokens=total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE):
                _ = model(input_ids=input_ids_buf, positions=positions_buf)
        torch.cuda.synchronize()

        # Per-graph pool release (verbatim NP gotcha): free the previous live
        # graph before capturing a new one, or CUDACachingAllocator asserts.
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
                hidden_buf = model(input_ids=input_ids_buf,
                                   positions=positions_buf)
        self._np_active_graph = graph

        meta_bufs["slot_mapping"][rs["perturbed_row_idx"]] = -1

        gs = dict(rs)
        gs.update({
            "graph": graph,
            "input_ids_buf": input_ids_buf,
            "positions_buf": positions_buf,
            "hidden_buf": hidden_buf,
            "meta_bufs": meta_bufs,
            "attn_meta": attn_meta,
            "total": total,
        })
        return gs

    # -------------------------------------------------------------- replay ---
    def _es_update_step_buffers(self, gs, states, n_sample):
        """Refill the persistent input + metadata buffers in place for this
        token (NP C-4 pad semantics: finished/pad slots get last-valid meta and
        clean_slot=-1). Shared by the replay and eager-oracle paths."""
        bucket = int(gs["bucket"])
        width = 1 + n_sample
        slot_states = []
        for p in range(bucket):
            st_p = states[p]
            block_ids = st_p["block_ids"]
            block_size = st_p["block_size"]
            prompt_len = st_p["prompt_len"]
            q_pos = int(st_p["kv_cursor"])
            if q_pos < prompt_len:
                q_token = st_p["prompt_token_ids"][q_pos]
            else:
                q_token = st_p["committed_tokens"][q_pos - prompt_len]
            clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
            slot_states.append({
                "active": bool(st_p["active"]),
                "q_token": int(q_token),
                "q_pos": q_pos,
                "clean_slot": int(clean_slot),
                "last_q_token": int(q_token),
                "last_q_pos": q_pos,
            })
        meta = _packed_replay_row_meta(slot_states)

        ids_buf = gs["input_ids_buf"]
        pos_buf = gs["positions_buf"]
        mb = gs["meta_bufs"]
        sm = mb["slot_mapping"]
        sl = mb["seq_lens_gpu"]
        for p in range(bucket):
            m = meta[p]
            base = p * width
            ids_buf[base:base + width].fill_(m["q_token"])
            pos_buf[base:base + width].fill_(m["q_pos"])
            sl[base:base + width].fill_(m["seq_len"])
            sm[base].fill_(m["clean_slot"])
        return meta

    def _es_replay_step_packed(self, model, states, n_sample, es_cfg, step_t,
                               slot_rollout_ids, gs):
        """One graphed decode token: in-place buffer refill + ONE fused noise
        draw per slot + replay + eager compute_logits. NO per-token full sync
        (the sampled tokens' .tolist() in the orchestrator is the only host
        read; ES_FULL_SYNC=1 restores the blanket sync for debugging)."""
        self._es_update_step_buffers(gs, states, n_sample)
        if not os.environ.get("ES_BENCH_SKIP_NOISE"):
            self._es_fill_noise(gs["noise_buf"], es_cfg, step_t,
                                slot_rollout_ids)
        gs["graph"].replay()
        if os.environ.get("ES_FULL_SYNC"):
            torch.cuda.synchronize()
        logits = model.compute_logits(gs["hidden_buf"])   # [R, vocab]
        return logits

    def _es_eager_step_packed(self, model, device, states, n_sample, es_cfg,
                              step_t, slot_rollout_ids, rs, max_seq_len_cap):
        """Eager parity oracle: SAME bucket-padded row layout, SAME es state
        and noise refill as the graphed path, but a fresh eager forward per
        token (metadata rebuilt each call)."""
        bucket = int(rs["bucket"])
        width = 1 + n_sample
        slot_states = []
        for p in range(bucket):
            st_p = states[p]
            block_ids = st_p["block_ids"]
            block_size = st_p["block_size"]
            prompt_len = st_p["prompt_len"]
            q_pos = int(st_p["kv_cursor"])
            if q_pos < prompt_len:
                q_token = st_p["prompt_token_ids"][q_pos]
            else:
                q_token = st_p["committed_tokens"][q_pos - prompt_len]
            clean_slot = self._np_slot_for_position(block_ids, block_size, q_pos)
            slot_states.append({
                "active": bool(st_p["active"]),
                "q_token": int(q_token),
                "q_pos": q_pos,
                "clean_slot": int(clean_slot),
                "last_q_token": int(q_token),
                "last_q_pos": q_pos,
            })
        meta = _packed_replay_row_meta(slot_states)

        input_ids, positions, slot_mapping, seq_lens, query_lens = (
            [], [], [], [], [])
        per_row_block_ids = []
        for p in range(bucket):
            m = meta[p]
            input_ids += [m["q_token"]] * width
            positions += [m["q_pos"]] * width
            slot_mapping += [m["clean_slot"]] + [-1] * n_sample
            seq_lens += [m["seq_len"]] * width
            query_lens += [1] * width
            per_row_block_ids += [states[p]["block_ids"]] * width

        if not os.environ.get("ES_BENCH_SKIP_NOISE"):
            self._es_fill_noise(rs["noise_buf"], es_cfg, step_t,
                                slot_rollout_ids)

        attn_meta, total = self._np_build_attn_metadata_packed(
            per_row_block_ids, query_lens, seq_lens, slot_mapping, positions)
        with torch.no_grad():
            hidden = self._np_run_forward(
                model, device, input_ids, positions, attn_meta, total)
            logits = model.compute_logits(hidden)
        return logits

    # --------------------------------------------------------- orchestrator --
    def run_es_decode_packed(self, list_of_prompt_ids, sampling_params, es_cfg,
                             rollout_ids, use_graph=True):
        """Packed es_token decode for B prompts. Returns per real prompt:
            clean_tokens[p] : list[int]
            payload[p]      : [T_p, 1+N] CPU fp32 -- each rail's logprob of the
                              clean sampled token (col 0 = clean rail)
        Bucket selection / pad slots / EOS bucket-padding follow NP V3 exactly;
        the captured graph is cached per bucket width (max_seqlen_k frozen at
        max_model_len so the cache stays valid across waves of any prompt len).
        """
        st = self._ensure_np_state()
        mr = self.model_runner
        model = mr.model
        device = mr.device
        n_sample = int(es_cfg["n_sample"])
        max_tokens = int(es_cfg["max_tokens"])
        width = 1 + n_sample

        B = len(list_of_prompt_ids)
        assert len(rollout_ids) == B
        bucket = _select_bucket(B, list(es_cfg.get("b_pack_buckets", [2, 4])))

        padded_prompt_ids = list(list_of_prompt_ids) + [
            list(list_of_prompt_ids[0]) for _ in range(bucket - B)]
        # Reserve KV for the real budget (longest prompt + max_tokens), not
        # the full max_model_len -- that 20x over-reservation was what capped
        # pack_width at 8.
        # ES_KV_FULL_RESERVE=1 restores the old full-max_model_len reservation,
        # for A/B-ing the budget-sized carving against it.
        states = self._np_prefill_packed(
            model, device, padded_prompt_ids,
            max_new_tokens=(None if os.environ.get("ES_KV_FULL_RESERVE")
                            else max_tokens))
        for p in range(B, bucket):
            states[p]["active"] = False
        slot_rollout_ids = [int(rollout_ids[p]) for p in range(B)] + [
            int(rollout_ids[0]) for _ in range(bucket - B)]

        # Frozen kernel-grid cap: max_model_len (vLLM's own decode-graph
        # practice) -- valid for every wave the cached graph will ever serve.
        max_seq_len_cap = int(mr.max_model_len)

        # Seeds for every (token, slot) of this wave, derived once instead of
        # per token inside the decode loop.
        if es_cfg["sample_method"] == "bernoulli":
            self._es_seed_tbl = build_seed_table(
                int(es_cfg["global_seed"]), max_tokens, slot_rollout_ids, device)
        else:
            self._es_seed_tbl = None

        sigma_eff = self._es_sigma_eff(es_cfg)
        if use_graph:
            if not hasattr(self, "_es_graph_by_bucket"):
                self._es_graph_by_bucket = {}
            if bucket not in self._es_graph_by_bucket:
                rs = self._es_install_state(bucket, n_sample, device)
                for ln, s in sigma_eff.items():
                    rs["sigma_buf"][ln].fill_(float(s))
                gs = self._es_capture_step_packed(
                    model, device, bucket, n_sample, states, max_seq_len_cap,
                    rs)
                self._es_graph_by_bucket[bucket] = gs
            gs = self._es_graph_by_bucket[bucket]
            # Reinstall the PINNED objects on st (harmless for the graph, needed
            # if an eager call rebound them) and set this call's sigma.
            st.update({
                "mode": "perturb_es",
                "es_noise_buf": gs["noise_buf"],
                "es_layout": self.es_layout,
                "es_signs": self.es_signs,
            "es_signs_flat": self.es_signs_flat,
                "es_sigma_buf": gs["sigma_buf"],
                "perturbed_row_idx": gs["perturbed_row_idx"],
                "clean_row_idx": gs["clean_row_idx"],
                "es_rail_idx": gs["rail_idx"],
                "es_prompt_idx": gs["prompt_idx"],
            })
            for ln, s in sigma_eff.items():
                gs["sigma_buf"][ln].fill_(float(s))
            rs = gs
        else:
            rs = self._es_install_state(bucket, n_sample, device)
            for ln, s in sigma_eff.items():
                rs["sigma_buf"][ln].fill_(float(s))

        clean_row_idx = rs["clean_row_idx"]
        payload_buf = torch.zeros(bucket * width, max_tokens, device=device,
                                  dtype=torch.float32)
        clean_tokens = [[] for _ in range(B)]
        temp = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
        top_p = float(es_cfg.get("top_p", 1.0) or 1.0)

        try:
            for t in range(max_tokens):
                active_idx = [p for p in range(B) if states[p]["active"]]
                if not active_idx:
                    break
                if use_graph:
                    logits = self._es_replay_step_packed(
                        model, states, n_sample, es_cfg, t, slot_rollout_ids,
                        rs)
                else:
                    logits = self._es_eager_step_packed(
                        model, device, states, n_sample, es_cfg, t,
                        slot_rollout_ids, rs, max_seq_len_cap)

                # Vectorized payload + clean sampling over ALL slots at once.
                logits_f = logits.float()                       # [R, vocab]
                lse = torch.logsumexp(logits_f, dim=-1)         # [R]
                clean_logits = logits_f[clean_row_idx]          # [bucket, vocab]
                if temp == 0.0:
                    next_toks = clean_logits.argmax(dim=-1)     # [bucket]
                else:
                    probs = torch.softmax(clean_logits / temp, dim=-1)
                    # top-p. Default 1.0 keeps the historical behaviour (pure
                    # multinomial over the full 151 k vocab). BP's rollout and
                    # every eval use 0.95, and at step 0 -- identical weights --
                    # that gap alone is 1397 training tokens vs 837 at eval
                    # (docs/results/zo_opd.md 12.6). Set es_token.top_p=0.95 to
                    # estimate the gradient on the distribution we score.
                    if top_p < 1.0:
                        sp_, si_ = torch.sort(probs, dim=-1, descending=True)
                        cum = sp_.cumsum(dim=-1)
                        drop = cum - sp_ > top_p
                        sp_ = sp_.masked_fill(drop, 0.0)
                        sp_ = sp_ / sp_.sum(dim=-1, keepdim=True)
                        probs = torch.zeros_like(probs).scatter_(1, si_, sp_)
                    next_toks = torch.multinomial(probs, 1)[:, 0]
                chosen = next_toks.repeat_interleave(width)     # [R]
                tok_logp = logits_f.gather(1, chosen[:, None])[:, 0] - lse
                payload_buf[:, t] = tok_logp

                toks = next_toks.tolist()   # the one host sync per token
                force_stop = es_cfg.get("force_stop_at")  # test-only: staggered
                for p in active_idx:                      # EOS gate (parity (c))
                    tok = int(toks[p])
                    clean_tokens[p].append(tok)
                    if self._np_is_eos(tok, sampling_params) or (
                            force_stop is not None
                            and len(clean_tokens[p]) >= int(force_stop[p])):
                        states[p]["active"] = False
                    else:
                        self._np_commit_clean(states[p], tok)
        finally:
            st["mode"] = "off"

        payload_cpu = payload_buf.to("cpu")
        payload = []
        for p in range(B):
            T_p = len(clean_tokens[p])
            block = payload_cpu[p * width:(p + 1) * width, :T_p]  # [1+N, T_p]
            payload.append(block.t().contiguous())                # [T_p, 1+N]
        return {"clean_tokens": clean_tokens, "payload": payload}

    # -------------------------------------------------------------- export ---
    def es_export_weights(self):
        """Return {vllm_layer_name: cpu fp32 tensor} for every perturbed layer.

        Prefers the fp32 master (the authoritative accumulator) and falls back to
        the live bf16 weight when fp32_master is off. Used by the trainer to write
        an HF checkpoint; the fused vLLM layouts (qkv_proj, gate_up_proj) are split
        back into their HF counterparts on the driver side.
        """
        out = {}
        master = getattr(self, "es_master", None) or {}
        with torch.no_grad():
            for ln in self.np_modules:
                t = master.get(ln)          # already on host when fp32_master is on
                if t is None:
                    t = self.np_modules[ln].wrapped.weight
                out[ln] = t.detach().to("cpu", torch.float32).clone()
        return out

    # ------------------------------------------------------------ assemble ---
    def es_assemble_and_apply(self, rollout_ids, t_idx, scales, es_cfg, lr,
                              update_clip=None, chunk=1024):
        """Build every matched layer's delta_W from seed-regenerated noise +
        the trainer-computed rail scales, then apply W <- W - lr*dW in place.

        rollout_ids: [M] ints -- record j's rollout id (regenerates its noise)
        t_idx:       [M] ints -- record j's token index within its rollout
        scales:      [M, N] float -- RAW rail differences (l_n - baseline_t),
                     WITHOUT the 1/sigma: the finite-difference normalization is
                     per-LAYER (1/sigma_eff[l], so sigma_mode=relative stays
                     unbiased) and is applied here, not trainer-side.
        Returns {layer: ||dW||} (1/N, 1/sigma_l and token_agg scaling included).

        Pure batched GEMM per (chunk, rail, layer): no per-token Python
        reduction (the NP 835 s assemble residual does not exist here).
        Noise is regenerated on THIS device at the SAME dtype the decode drew
        (bit-identical bytes by the seeding invariant).
        """
        device = self.model_runner.device
        dtype = self.es_dtype
        n_rails = int(self.es_n_rails)
        gseed = int(es_cfg["global_seed"])
        method = es_cfg["sample_method"]
        token_agg = es_cfg.get("token_agg", "sum")

        rollout_ids = list(rollout_ids)
        t_idx = list(t_idx)
        scales_t = torch.as_tensor(scales, dtype=torch.float32).to(device)
        M = len(rollout_ids)
        assert scales_t.shape == (M, n_rails), (
            f"scales {tuple(scales_t.shape)} != ({M}, {n_rails})")

        acc = {ln: torch.zeros(d_out, d_in, dtype=torch.float32, device=device)
               for ln, (_, d_out, _, d_in) in self.es_layout.items()}
        # NOTE: this dict is ~5.65 GB (every perturbed layer, fp32) and is the
        # single largest transient on the card; `acc[ln] = None` in the apply
        # loop below releases it layer by layer.

        noise_chunk = torch.empty(min(int(chunk), M), self.es_d_total,
                                  dtype=dtype, device=device)
        for c0 in range(0, M, int(chunk)):
            c1 = min(c0 + int(chunk), M)
            m = c1 - c0
            nc = noise_chunk[:m]
            if method == "bernoulli":
                # One launch for the whole chunk (was m x ~6 kernels).
                seeds = [es_token_seed(gseed, int(t_idx[c0 + j]),
                                       int(rollout_ids[c0 + j]))
                         for j in range(m)]
                fill_rademacher_rows(nc, seeds)
            else:
                for j in range(m):
                    nc[j].copy_(draw_token_noise(
                        gseed, int(t_idx[c0 + j]), int(rollout_ids[c0 + j]),
                        self.es_d_total, device, dtype, method))
            sc = scales_t[c0:c1]
            for ln, (off_u, d_out, off_v, d_in) in self.es_layout.items():
                u = nc[:, off_u:off_u + d_out]
                v = nc[:, off_v:off_v + d_in]
                S, R = self.es_signs[ln]
                assemble_chunk(sc, u, v, S, R, acc[ln])

        sigma_eff = self._es_sigma_eff(es_cfg)
        # vLLM holds the weights in bf16, whose ulp near |W|~0.02 is ~6e-5. A
        # step smaller than half an ulp rounds straight back to the old value,
        # so an in-place bf16 SGD step silently drops most of the update at the
        # LRs that do not diverge (measured: ~1.4% of elements move at an
        # Adam-sized 1e-6). Accumulate into an fp32 master and round once.
        fp32_master = bool(es_cfg.get("fp32_master", True))
        if fp32_master and getattr(self, "es_master", None) is None:
            self.es_master = {}
        norms = {}
        # Update-quality diagnostics (docs/results/zo_opd.md 12.4). Two numbers
        # decide whether a run can learn, and neither was logged before:
        #   footprint = RMS(lr*dW)/RMS(W) -- the per-step relative weight motion.
        #     Every ES arm in this repo that learns runs at 1.6e-2..5e-2
        #     (results/ES/es_results.md 10.4, 11.3); es_token ran at 1.4e-4.
        #   dw_cos_prev = cos(dW_t, dW_{t-1}) on a fixed coordinate sketch.
        #     The estimator is unbiased but nearly all noise, so this reads the
        #     COHERENT fraction: ~0 means the update is a random walk.
        if not hasattr(self, "_es_prev_sketch"):
            self._es_prev_sketch = {}
        SK = 100_000
        foots, coss = {}, {}
        with torch.no_grad():
            for ln, dw in acc.items():
                denom = (float(n_rails) * float(sigma_eff[ln])
                         * (float(M) if token_agg == "mean" else 1.0))
                dw.div_(denom)
                if update_clip is not None:
                    dw.clamp_(-float(update_clip), float(update_clip))
                weight = self.np_modules[ln].wrapped.weight
                if fp32_master:
                    # Held on the HOST. An fp32 copy of the 1.41 B perturbed
                    # params is 5.65 GB, which does not fit alongside the student
                    # engine, the co-located teacher and this accumulator on one
                    # 93 GB card. The round trip is ~11 GB of PCIe per step
                    # against a ~150 s step, i.e. under 1%.
                    master = self.es_master.get(ln)
                    if master is None:
                        master = weight.detach().float().cpu().clone()
                        self.es_master[ln] = master
                    master.add_(dw.to("cpu"), alpha=-float(lr))
                    weight.copy_(master.to(weight.device, weight.dtype))
                else:
                    weight.add_(dw.to(weight.dtype), alpha=-float(lr))
                norms[ln] = float(dw.norm().item())
                w_rms = float(weight.float().pow(2).mean().sqrt().item())
                if w_rms > 0:
                    foots[ln] = (float(lr) * norms[ln]
                                 / (dw.numel() ** 0.5) / w_rms)
                flat = dw.view(-1)
                stride = max(1, flat.numel() // SK)
                sk = flat[::stride][:SK].clone()
                prev = self._es_prev_sketch.get(ln)
                if prev is not None and prev.numel() == sk.numel():
                    d = sk.norm() * prev.norm()
                    if float(d) > 0:
                        coss[ln] = float((sk @ prev) / d)
                self._es_prev_sketch[ln] = sk
                acc[ln] = None  # free as we go
        torch.cuda.synchronize()
        return {"norms": norms, "footprint": foots, "dw_cos_prev": coss}
