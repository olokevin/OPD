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
from verl.trainer.es_token.rail_attn_kernel import (
    RailAttnWorkspace, pick_num_splits, rail_attention_fold, rail_attention_seq,
    rail_attention_shared)
from verl.trainer.es_token.lm_head_kernel import (
    LMHeadWorkspace, lm_head_gather_logit, lm_head_stream)
from verl.trainer.es_token.fused_rail_kernels import (
    RailArgs, advance as es_advance, fill_rademacher_rows_t, lm_tail, norm_rail,
    qkv_rail_norm_rope, silu_mul_rail)
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
try:
    from vllm._custom_ops import reshape_and_cache_flash
except Exception:  # pragma: no cover
    reshape_and_cache_flash = None


class ESRailAttention(torch.nn.Module):
    """Wraps a vLLM `Attention` module. Transparent outside es mode or when
    st["es_attn_impl"] == "rows" (the shipping path: every rail row is its own
    FA request, so the slot's KV pages are re-read once per rail). Otherwise it
    (1) writes the clean rows' K/V into the paged cache with the same op vLLM
    uses (rail rows carry slot -1 and are skipped) and (2) runs a rail-aware
    attention that reads each KV page ONCE per slot for all (1+N) rails
    (opd_profile_plan.md §18):

        "shared" -- Triton split-KV kernel, rails x GQA-group as one query tile
                    (rail_attn_kernel.rail_attention_shared)
        "fold"   -- rails folded into the head axis, stock FA3 GQA packing
        "seq"    -- one FA3 request per slot with seqlen_q = R, non-causal:
                    the fold's KV reuse with no permute copies (0902)

    Semantics are identical to "rows": rails attend the clean history including
    the clean current-token K/V; only the clean row writes KV.
    """

    def __init__(self, wrapped, name, st_ref):
        super().__init__()
        self.wrapped = wrapped
        self.name = name
        self._st_ref = st_ref

    def forward(self, query, key, value, output_shape=None):
        st = self._st_ref()
        impl = st.get("es_attn_impl", "rows")
        if st.get("mode") != "perturb_es" or impl == "rows":
            if output_shape is not None:
                return self.wrapped(query, key, value, output_shape=output_shape)
            return self.wrapped(query, key, value)
        from vllm.forward_context import get_forward_context
        attn = self.wrapped
        fc = get_forward_context()
        attn_metadata = fc.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[attn.layer_name]
        kv_cache = attn.kv_cache[fc.virtual_engine]
        key_cache, value_cache = kv_cache.unbind(0)
        H = attn.impl.num_heads
        Hkv = attn.impl.num_kv_heads
        D = attn.impl.head_size
        n_rows = query.shape[0]
        reshape_and_cache_flash(
            key.view(n_rows, Hkv, D), value.view(n_rows, Hkv, D), key_cache,
            value_cache, attn_metadata.slot_mapping, attn.impl.kv_cache_dtype,
            attn._k_scale, attn._v_scale)
        am = st["es_attn_meta"]
        out = torch.empty_like(query)
        q4 = query.view(am["bucket"], am["width"], H, D)
        o4 = out.view(am["bucket"], am["width"], H, D)
        if impl == "shared":
            rail_attention_shared(q4, key_cache, value_cache, am["bt_B"], am["sl_B"],
                                  attn.impl.scale, out=o4, ws=am["ws"],
                                  num_splits=am["num_splits"],
                                  block_n=int(os.environ.get("ES_RAIL_ATTN_BLOCK_N", 64)))
        elif impl in ("fold", "fold2"):
            rail_attention_fold(q4, key_cache, value_cache, am["bt_B"], am["sl_B"],
                                attn.impl.scale, am["max_seqlen_k"], out=o4,
                                fa_version=(2 if impl == "fold2" else 3),
                                cu_seqlens_q=am["cu_b"],
                                q_fold_buf=am["q_fold_buf"], o_fold_buf=am["o_fold_buf"])
        elif impl == "seq":
            rail_attention_seq(q4, key_cache, value_cache, am["bt_B"], am["sl_B"],
                               attn.impl.scale, am["max_seqlen_k"], out=o4,
                               cu_seqlens_q=am["cu_q"])
        else:
            raise ValueError(f"unknown es_attn_impl {impl!r}")
        if os.environ.get("ES_ATTN_CHECK"):   # debug: per-layer diff vs FA rows path
            ref = torch.zeros_like(out)
            attn.impl.forward(attn, query.view(n_rows, H, D), key.view(n_rows, Hkv, D),
                              value.view(n_rows, Hkv, D), kv_cache, attn_metadata,
                              output=ref.view(n_rows, H, D))
            d = (out.float() - ref.float()).abs()
            per_row = d.view(n_rows, -1).amax(1)
            i = int(per_row.argmax())
            print(f"[attn-check] {self.name} max|d|={float(d.max()):.4e} "
                  f"mean={float(d.mean()):.2e} ref_max={float(ref.float().abs().max()):.3f} "
                  f"worst_row={i} (slot {i // am['width']}, rail {i % am['width']}) "
                  f"sl={am['sl_B'].tolist()}", flush=True)
        return out


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
        if st.get("es_rail_impl") == "fused":
            # Zero-launch rail (fused_rail_kernels.py): the consumer of this
            # output applies the rail; it only needs the GEMM input. Stash the
            # reference (fixed address inside the captured graph).
            st["es_x"][self.name] = x
            return out
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


def _es_rail_args(st, producer):
    """RailArgs for the perturbed linear `producer`, or None if it is not
    perturbed (not in the layout) / there is no producer."""
    if producer is None:
        return None
    layout = st["es_layout"].get(producer)
    if layout is None:
        return None
    x = st["es_x"].get(producer)
    if x is None:
        raise RuntimeError(f"es fused rail: no stashed input for {producer}")
    off_u, _, off_v, d_in = layout
    return RailArgs(x, st["es_noise_buf"], st["es_signs_flat"],
                    st["es_sigma_buf"][producer], off_u, off_v, d_in, st["es_width"])


def _es_fused_active(mod):
    st = mod._es_st()
    return st.get("mode") == "perturb_es" and st.get("es_rail_impl") == "fused", st


def _es_norm_forward(self, x, residual=None):
    """Patched RMSNorm.forward: [rail of the producing linear] + residual add +
    norm in one launch (fused_rail_kernels.norm_rail). Layer 0's
    input_layernorm (no residual, no producer) stays stock."""
    on, st = _es_fused_active(self)
    if not on or residual is None:
        return self._es_orig_forward(x, residual) if residual is not None else self._es_orig_forward(x)
    norm_rail(x, residual, self.weight.data, self.variance_epsilon,
              _es_rail_args(st, self._es_producer))
    return x, residual


def _es_silu_forward(self, x):
    """Patched SiluAndMul.forward: gate_up rail + silu*mul in one launch."""
    on, st = _es_fused_active(self)
    if not on:
        return self._es_orig_forward(x)
    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device)
    silu_mul_rail(x, out, _es_rail_args(st, self._es_producer))
    return out


def _es_attn_forward(self, positions, hidden_states):
    """Patched Qwen3Attention.forward: qkv rail + q_norm + k_norm + RoPE in one
    launch, replacing three (+ the rail op). Same attention / o_proj after."""
    on, st = _es_fused_active(self)
    if not on:
        return self._es_orig_forward(positions=positions, hidden_states=hidden_states)
    qkv, _ = self.qkv_proj(hidden_states)          # ESTokenLinear: raw GEMM, stashes x
    rope = self.rotary_emb
    if rope.cos_sin_cache.dtype != qkv.dtype or rope.cos_sin_cache.device != qkv.device:
        rope._match_cos_sin_cache_dtype(qkv)
    qkv_rail_norm_rope(qkv, positions, self.q_norm.weight.data, self.k_norm.weight.data,
                       self.q_norm.variance_epsilon, rope.cos_sin_cache,
                       self.num_heads, self.num_kv_heads, self.head_dim,
                       _es_rail_args(st, self._es_producer))
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


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

        # Wrap every vLLM Attention module so es mode can route to the
        # rail-aware kernels (transparent when es_attn_impl == "rows").
        from vllm.attention.layer import Attention as _VLLMAttention
        self.es_attn_modules = {}
        for mod_name, mod in list(model.named_modules()):
            if isinstance(mod, ESRailAttention):
                self.es_attn_modules[mod_name] = mod
            elif isinstance(mod, _VLLMAttention) and not mod_name.endswith(".wrapped"):
                parent = model
                *path, leaf = mod_name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                wrapped_attn = ESRailAttention(mod, mod_name, lambda: self.np_state)
                setattr(parent, leaf, wrapped_attn)
                self.es_attn_modules[mod_name] = wrapped_attn

        self._es_install_fused_consumers(model)

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

    def _es_install_fused_consumers(self, model):
        """rail_impl="fused" (fused_rail_kernels.py): patch, per Qwen3 decoder
        layer, the forward of input_layernorm / post_attention_layernorm /
        mlp.act_fn / self_attn (and the final model.norm) with versions that
        also apply the rail of the linear whose output they consume. Instance-
        level patches: the module tree and names are untouched, and every
        patched forward is a pass-through outside fused mode. Idempotent.
        Records the set of producers covered; fused mode refuses to run if a
        perturbed linear has no patched consumer."""
        import types

        try:
            from vllm.model_executor.layers.activation import SiluAndMul
            from vllm.model_executor.layers.layernorm import RMSNorm
            from vllm.model_executor.models.qwen3 import Qwen3Attention
        except Exception:  # pragma: no cover
            self._es_fused_consumers = set()
            return
        st_ref = lambda: self.np_state
        consumers = set()

        def patch(mod, fn, producer):
            if not getattr(mod, "_es_patched", False):
                mod._es_orig_forward = type(mod).forward.__get__(mod)
                mod._es_st = st_ref
                mod._es_patched = True
                mod.forward = types.MethodType(fn, mod)
            mod._es_producer = producer
            if producer is not None:
                consumers.add(producer)

        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            self._es_fused_consumers = set()
            return
        n = 0
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if not (isinstance(attn, Qwen3Attention)
                    and isinstance(layer.input_layernorm, RMSNorm)
                    and isinstance(layer.post_attention_layernorm, RMSNorm)
                    and isinstance(layer.mlp.act_fn, SiluAndMul)
                    and getattr(attn.rotary_emb, "is_neox_style", False)
                    and attn.rotary_emb.rotary_dim == attn.head_dim
                    and attn.head_dim % 2 == 0):
                continue
            pre = f"model.layers.{i}"
            patch(layer.input_layernorm, _es_norm_forward,
                  f"model.layers.{i - 1}.mlp.down_proj" if i > 0 else None)
            patch(layer.post_attention_layernorm, _es_norm_forward, f"{pre}.self_attn.o_proj")
            patch(layer.mlp.act_fn, _es_silu_forward, f"{pre}.mlp.gate_up_proj")
            patch(attn, _es_attn_forward, f"{pre}.self_attn.qkv_proj")
            n = i + 1
        if n and isinstance(getattr(inner, "norm", None), RMSNorm):
            patch(inner.norm, _es_norm_forward, f"model.layers.{n - 1}.mlp.down_proj")
        self._es_fused_consumers = consumers

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
    def _es_install_state(self, bucket, n_sample, device, attn_impl="rows",
                          rail_impl="kernel"):
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
            "es_attn_impl": str(attn_impl),
            "es_rail_impl": str(rail_impl),
            "es_x": {},
            "es_width": 1 + int(n_sample),
        })
        return {
            "attn_impl": str(attn_impl),
            "rail_impl": str(rail_impl),
            "noise_buf": noise_buf,
            "sigma_buf": sigma_buf,
            "clean_row_idx": clean_row_idx,
            "perturbed_row_idx": perturbed_row_idx,
            "rail_idx": rail_idx,
            "prompt_idx": prompt_idx,
            "bucket": bucket,
            "n_sample": n_sample,
        }

    def _es_build_attn_meta(self, bucket, width, bt_B, sl_B, max_seq_len_cap,
                            device, attn_impl):
        """Per-slot metadata + workspaces for the rail-aware attention.
        bt_B [bucket, max_blocks] int32, sl_B [bucket] int32 (seqused_k incl.
        the current token) -- the graphed path passes persistent tensors it
        mutates in place; the eager oracle builds fresh ones per token."""
        am = dict(bucket=int(bucket), width=int(width), bt_B=bt_B, sl_B=sl_B,
                  max_seqlen_k=int(max_seq_len_cap), impl=str(attn_impl))
        if attn_impl != "rows" and getattr(self, "es_attn_modules", None):
            a0 = next(iter(self.es_attn_modules.values())).wrapped
            Hq, Hkv, D = a0.impl.num_heads, a0.impl.num_kv_heads, a0.impl.head_size
            G = Hq // Hkv
            ns = pick_num_splits(bucket, Hkv)
            am["num_splits"] = ns
            am["ws"] = RailAttnWorkspace(bucket, Hkv, width, G, ns, D, device)
            am["cu_b"] = torch.arange(bucket + 1, dtype=torch.int32, device=device)
            am["cu_q"] = am["cu_b"] * int(width)      # seq: one request of R rows per slot
            if attn_impl in ("fold", "fold2"):
                am["q_fold_buf"] = torch.empty(bucket, Hkv * width * G, D,
                                               dtype=self.es_dtype, device=device)
                am["o_fold_buf"] = torch.empty_like(am["q_fold_buf"])
            else:
                am["q_fold_buf"] = am["o_fold_buf"] = None
        return am

    def _es_refresh_kv_pages(self, gs, states):
        """Copy THIS wave's KV page ids into the graph's pinned block table.

        The table was built at capture from the first wave's prefill states,
        but _np_prefill_packed carves each wave's pages by (longest prompt +
        max_tokens), so a later wave with a different longest prompt owns
        DIFFERENT pages and the captured table would read stale KV. Refreshed
        in place (same storage the graph holds), so it is graph-safe."""
        if os.environ.get("ES_NO_KV_REFRESH"):
            return
        bt = gs["meta_bufs"]["block_table"]
        width = 1 + int(gs["n_sample"])
        bt_cpu = torch.zeros(bt.shape, dtype=bt.dtype)
        for p, s in enumerate(states):
            ids = torch.tensor(s["block_ids"], dtype=bt.dtype)
            r0 = p * width
            bt_cpu[r0:r0 + width, : ids.numel()] = ids
        bt.copy_(bt_cpu.to(bt.device))
        am = gs.get("es_attn_meta")
        if am is not None:
            am["bt_B"].copy_(bt[::width])

    def es_reset_graphs(self):
        """Drop every cached decode graph (bench sweeps capture many)."""
        import gc as _gc
        self._es_graph_by_bucket = {}
        self._np_active_graph = None
        _gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def _es_capture_step_packed(self, model, device, bucket, n_sample,
                                prefill_states, max_seq_len_cap, rs,
                                step_impl="eager", max_tokens=None, greedy=True,
                                capture=True):
        """Capture ONE es_token packed step forward at fixed bucket width.
        Mirrors NP's _np_capture_step_packed (persistent input/meta buffers,
        warmup, per-graph pool release, frozen max_seqlen_k at the cap) with the
        es perturbation state already installed via _es_install_state.

        step_impl="graph": the graph body is the WHOLE token step -- noise
        fill, forward, streaming LM head, sampling, payload write and the
        next-token state advance (fused_rail_kernels.py) -- so the decode
        loop is replay() with no per-token host work. capture=False builds
        the same pinned state without a graph (eager oracle)."""
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

        # Per-slot metadata for the rail-aware attention: the slot's block
        # table row (pinned; refreshed per wave) and a [bucket] seqused_k
        # mutated per token next to the per-row one. Pinned by the capture.
        sl_B = meta_bufs["seq_lens_gpu"][::width].clone()
        am = self._es_build_attn_meta(bucket, width, meta_bufs["block_table"][::width].contiguous(),
                                      sl_B, max_seq_len_cap, device, rs.get("attn_impl", "rows"))
        rs["es_attn_meta"] = am
        self._ensure_np_state()["es_attn_meta"] = am

        gs = dict(rs)
        gs.update({
            "input_ids_buf": input_ids_buf,
            "positions_buf": positions_buf,
            "meta_bufs": meta_bufs,
            "attn_meta": attn_meta,
            "total": total,
            "step_impl": str(step_impl),
        })
        if step_impl == "graph":
            self._es_alloc_graph_step(gs, model, device, bucket, n_sample,
                                      int(max_tokens), bool(greedy))
            meta_bufs["slot_mapping"][rs["perturbed_row_idx"]] = -1
            for _ in range(3):   # warm-up (Triton compile); each pass at t=0 --
                gs["t_cnt"].zero_()   # the step buffers are only max_tokens wide
                self._es_step_body(model, gs)
            torch.cuda.synchronize()
            if not capture:
                gs["graph"] = None
                gs["hidden_buf"] = None
                return gs
        else:
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
        # Under TP the all-reduce (custom AR / NCCL) must be captured inside
        # vLLM's own graph_capture() context on its capture stream, exactly as
        # gpu_model_runner does; TP=1 has no collective and needs neither.
        from contextlib import nullcontext
        gc_ctx = nullcontext()
        try:
            from vllm.distributed.parallel_state import get_tp_group
            from vllm.distributed.parallel_state import graph_capture as _vllm_gc
            if get_tp_group().world_size > 1:
                gc_ctx = _vllm_gc(device=device)
        except Exception:  # pragma: no cover - single-GPU / CPU tests
            pass
        if step_impl == "graph":
            with gc_ctx as gcc:
                with torch.cuda.graph(graph, stream=getattr(gcc, "stream", None)):
                    hidden_buf = self._es_step_body(model, gs)
        else:
            with torch.no_grad(), set_forward_context(
                attn_meta, self.model_runner.vllm_config, num_tokens=total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE), gc_ctx as gcc:
                with torch.cuda.graph(graph, stream=getattr(gcc, "stream", None)):
                    hidden_buf = model(input_ids=input_ids_buf,
                                       positions=positions_buf)
        self._np_active_graph = graph

        meta_bufs["slot_mapping"][rs["perturbed_row_idx"]] = -1
        gs["graph"] = graph
        gs["hidden_buf"] = hidden_buf
        return gs

    # ------------------------------------------------- in-graph token step ---
    def _es_alloc_graph_step(self, gs, model, device, bucket, n_sample, max_tokens, greedy):
        """Pinned buffers for the in-graph token step (all fixed-shape; the
        graph key carries max_tokens and greedy)."""
        width = 1 + n_sample
        R = bucket * width
        lm_head = model.lm_head
        V = int(getattr(model.logits_processor, "org_vocab_size", lm_head.weight.shape[0]))
        clean_idx = torch.full((R,), -1, dtype=torch.int32, device=device)
        clean_idx[gs["clean_row_idx"]] = torch.arange(bucket, dtype=torch.int32, device=device)
        gs.update({
            "max_tokens": int(max_tokens),
            "greedy": bool(greedy),
            "W_lm": lm_head.weight[:V],
            "clean_idx": clean_idx,
            "ws_lm": LMHeadWorkspace(R, V, bucket, 128, device),
            "argmax_buf": torch.zeros(bucket, dtype=torch.long, device=device),
            "inv_temp": torch.ones(1, dtype=torch.float32, device=device),
            "t_cnt": torch.zeros(1, dtype=torch.long, device=device),
            "active": torch.ones(bucket, dtype=torch.int32, device=device),
            "ngen": torch.zeros(bucket, dtype=torch.int32, device=device),
            "tokens_buf": torch.zeros(bucket, max_tokens, dtype=torch.long, device=device),
            "payload_buf": torch.zeros(R, max_tokens, dtype=torch.float32, device=device),
            "seed_tbl": torch.zeros(max_tokens, bucket, dtype=torch.int64, device=device),
            "force_buf": torch.full((bucket, max_tokens), -1, dtype=torch.long, device=device),
            "force_stop": torch.full((bucket,), 2 ** 62, dtype=torch.long, device=device),
            "eos_buf": torch.full((16,), -1, dtype=torch.long, device=device),
            "block_size": int(self.model_runner.cache_config.block_size),
        })

    def _es_step_body(self, model, gs):
        """ONE token step, device-only (captured as the graph body):
        noise(t) -> forward -> streaming head -> argmax/Gumbel -> payload
        tail -> advance -> t += 1. Returns the hidden buffer."""
        from vllm.config.compilation import CUDAGraphMode
        width = 1 + int(gs["n_sample"])
        fill_rademacher_rows_t(gs["noise_buf"], gs["seed_tbl"], gs["t_cnt"])
        with torch.no_grad(), set_forward_context(
                gs["attn_meta"], self.model_runner.vllm_config, num_tokens=gs["total"],
                cudagraph_runtime_mode=CUDAGraphMode.NONE):
            hidden = model(input_ids=gs["input_ids_buf"], positions=gs["positions_buf"])
        ws = gs["ws_lm"]
        clean_logits, _, _ = lm_head_stream(hidden, gs["W_lm"], gs["clean_idx"], ws=ws,
                                            compute_lse=False)
        if gs["greedy"]:
            torch.argmax(clean_logits, dim=-1, out=gs["argmax_buf"])
        else:   # Gumbel-max == multinomial(softmax(logits / T)); RNG is graph-safe
            u = torch.rand_like(clean_logits)
            torch.argmax(clean_logits * gs["inv_temp"] - torch.log(-torch.log(u)),
                         dim=-1, out=gs["argmax_buf"])
        lm_tail(hidden, gs["W_lm"], ws.mp, ws.sp, gs["argmax_buf"], gs["force_buf"],
                gs["t_cnt"], gs["payload_buf"], width)
        mb = gs["meta_bufs"]
        bt = mb["block_table"]
        es_advance(gs["argmax_buf"], gs["force_buf"], gs["active"], gs["ngen"],
                   gs["tokens_buf"], gs["input_ids_buf"], gs["positions_buf"],
                   mb["seq_lens_gpu"], gs["es_attn_meta"]["sl_B"], mb["slot_mapping"],
                   bt.as_strided((int(gs["bucket"]), bt.shape[1]),
                                 (width * bt.stride(0), bt.stride(1))),
                   gs["force_stop"], gs["eos_buf"], gs["t_cnt"], width, gs["block_size"])
        gs["t_cnt"] += 1
        return hidden

    def _es_decode_graph_step(self, model, device, states, B, bucket, n_sample,
                              es_cfg, sampling_params, slot_rollout_ids,
                              max_seq_len_cap, sigma_eff, attn_impl, rail_impl,
                              use_graph, force_tokens):
        """Decode loop for step_impl="graph": the host does replay() per token
        and reads the active mask every ES_ACTIVE_CHECK_EVERY (32) tokens;
        tokens and payload come back in ONE D2H at the end."""
        st = self._ensure_np_state()
        width = 1 + n_sample
        max_tokens = int(es_cfg["max_tokens"])
        temp = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
        greedy = temp == 0.0
        if not hasattr(self, "_es_graph_by_bucket"):
            self._es_graph_by_bucket = {}
        gkey = (bucket, n_sample, attn_impl, rail_impl, "graph", max_tokens, greedy, bool(use_graph))
        if gkey not in self._es_graph_by_bucket:
            rs = self._es_install_state(bucket, n_sample, device, attn_impl=attn_impl,
                                        rail_impl=rail_impl)
            for ln, sg in sigma_eff.items():
                rs["sigma_buf"][ln].fill_(float(sg))
            gs = self._es_capture_step_packed(
                model, device, bucket, n_sample, states, max_seq_len_cap, rs,
                step_impl="graph", max_tokens=max_tokens, greedy=greedy,
                capture=use_graph)
            self._es_graph_by_bucket[gkey] = gs
        gs = self._es_graph_by_bucket[gkey]
        self._es_refresh_kv_pages(gs, states)
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
            "es_attn_impl": attn_impl,
            "es_attn_meta": gs.get("es_attn_meta"),
            "es_rail_impl": rail_impl,
            "es_x": {},
            "es_width": width,
        })
        for ln, sg in sigma_eff.items():
            gs["sigma_buf"][ln].fill_(float(sg))
        if not greedy:
            gs["inv_temp"].fill_(1.0 / temp)

        # per-wave device state
        assert es_cfg["sample_method"] == "bernoulli", "graph step: bernoulli noise only"
        seeds_dev, _ = build_seed_table(int(es_cfg["global_seed"]), max_tokens,
                                        slot_rollout_ids, device)
        gs["seed_tbl"].copy_(seeds_dev)
        fb = gs["force_buf"]
        fb.fill_(-1)
        if force_tokens is not None:
            for p in range(min(B, len(force_tokens))):
                ft = torch.as_tensor(list(force_tokens[p])[:max_tokens], dtype=torch.long)
                if ft.numel():
                    fb[p, : ft.numel()].copy_(ft.to(device))
        fs = gs["force_stop"]
        fs.fill_(2 ** 62)
        force_stop = es_cfg.get("force_stop_at")
        if force_stop is not None:
            for p in range(min(B, len(force_stop))):
                fs[p].fill_(int(force_stop[p]))
        stop = getattr(sampling_params, "_all_stop_token_ids", None) or set()
        if not stop:
            try:
                eos = self.model_runner.model_config.hf_config.eos_token_id
                stop = {int(eos)} if isinstance(eos, int) else set(int(e) for e in (eos or []))
            except Exception:
                stop = set()
        stop = sorted(int(e) for e in stop)
        assert len(stop) <= gs["eos_buf"].numel(), stop
        gs["eos_buf"].fill_(-1)
        if stop:
            gs["eos_buf"][: len(stop)].copy_(torch.tensor(stop, dtype=torch.long, device=device))
        self._es_update_step_buffers(gs, states, n_sample)
        gs["active"].copy_(torch.tensor([1 if states[p]["active"] else 0 for p in range(bucket)],
                                        dtype=torch.int32, device=device))
        gs["t_cnt"].zero_()
        gs["ngen"].zero_()

        check_every = int(os.environ.get("ES_ACTIVE_CHECK_EVERY", 32))
        try:
            for t in range(max_tokens):
                if use_graph:
                    gs["graph"].replay()
                else:
                    self._es_step_body(model, gs)
                if (t + 1) % check_every == 0 and t + 1 < max_tokens:
                    if int(gs["active"].sum().item()) == 0:
                        break
        finally:
            st["mode"] = "off"
        tokens = gs["tokens_buf"].cpu()
        ngen = gs["ngen"].cpu().tolist()
        payload_cpu = gs["payload_buf"].cpu()
        clean_tokens, payload = [], []
        for p in range(B):
            T_p = int(ngen[p])
            clean_tokens.append(tokens[p, :T_p].tolist())
            payload.append(payload_cpu[p * width:(p + 1) * width, :T_p].t().contiguous())
        return {"clean_tokens": clean_tokens, "payload": payload}

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
        sl_B = gs["es_attn_meta"]["sl_B"] if gs.get("es_attn_meta") is not None else None
        for p in range(bucket):
            m = meta[p]
            base = p * width
            ids_buf[base:base + width].fill_(m["q_token"])
            pos_buf[base:base + width].fill_(m["q_pos"])
            sl[base:base + width].fill_(m["seq_len"])
            sm[base].fill_(m["clean_slot"])
            if sl_B is not None:
                sl_B[p].fill_(m["seq_len"])
        return meta

    def _es_replay_step_packed(self, model, states, n_sample, es_cfg, step_t,
                               slot_rollout_ids, gs):
        """One graphed decode token: in-place buffer refill + ONE fused noise
        draw per slot + replay; returns the [R, d] hidden buffer (the LM head
        runs in the orchestrator: full or streaming). NO per-token full sync
        (the sampled tokens' .tolist() in the orchestrator is the only host
        read; ES_FULL_SYNC=1 restores the blanket sync for debugging)."""
        self._es_update_step_buffers(gs, states, n_sample)
        if not os.environ.get("ES_BENCH_SKIP_NOISE"):
            self._es_fill_noise(gs["noise_buf"], es_cfg, step_t,
                                slot_rollout_ids)
        gs["graph"].replay()
        if os.environ.get("ES_FULL_SYNC"):
            torch.cuda.synchronize()
        return gs["hidden_buf"]                           # [R, d]

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
        attn_impl = rs.get("attn_impl", "rows")
        if attn_impl != "rows":
            mr = self.model_runner
            max_blocks = int(
                mr.input_batch.block_table.block_tables[0].max_num_blocks_per_req)
            bt_B = torch.zeros((bucket, max_blocks), dtype=torch.int32, device=device)
            for p in range(bucket):
                ids = states[p]["block_ids"]
                bt_B[p, : len(ids)] = torch.tensor(ids, dtype=torch.int32, device=device)
            sl_B = torch.tensor(seq_lens[::width], dtype=torch.int32, device=device)
            self._ensure_np_state()["es_attn_meta"] = self._es_build_attn_meta(
                bucket, width, bt_B, sl_B, max_seq_len_cap, device, attn_impl)
        with torch.no_grad():
            hidden = self._np_run_forward(
                model, device, input_ids, positions, attn_meta, total)
        return hidden

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
        # Kernel selection (opd_profile_plan.md Part III). Defaults = the
        # shipping path; "shared"/"fold" + "stream" are the rail-aware kernels.
        attn_impl = str(es_cfg.get("attn_impl", "rows"))
        lm_impl = str(es_cfg.get("lm_head_impl", "full"))
        rail_impl = str(es_cfg.get("rail_impl", "kernel"))
        step_impl = str(es_cfg.get("step_impl", "eager"))
        assert attn_impl in ("rows", "shared", "fold", "fold2", "seq"), attn_impl
        assert lm_impl in ("full", "stream"), lm_impl
        assert rail_impl in ("kernel", "fused"), rail_impl
        assert step_impl in ("eager", "graph"), step_impl
        if rail_impl == "fused":
            missing = set(self.es_layout) - getattr(self, "_es_fused_consumers", set())
            assert not missing, ("rail_impl=fused: perturbed linears without a fused "
                                 f"consumer (non-Qwen3 layer?): {sorted(missing)[:4]}")
        if lm_impl == "stream":
            try:
                from vllm.distributed import get_tensor_model_parallel_world_size
                tp = int(get_tensor_model_parallel_world_size())
            except Exception:
                tp = 1
            if tp > 1:   # lm_head is vocab-sharded under TP; the streaming
                lm_impl = "full"   # kernel has no gather yet -> gathered full path

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
        force_tokens = es_cfg.get("force_tokens")   # test-only teacher forcing
        if step_impl == "graph":
            top_p = float(es_cfg.get("top_p", 1.0) or 1.0)
            assert lm_impl == "stream", "step_impl=graph needs lm_head_impl=stream"
            assert top_p >= 1.0, "step_impl=graph: top-p sampling is not in-graph (use step_impl=eager)"
            return self._es_decode_graph_step(
                model, device, states, B, bucket, n_sample, es_cfg, sampling_params,
                slot_rollout_ids, max_seq_len_cap, sigma_eff, attn_impl, rail_impl,
                use_graph, force_tokens)
        if use_graph:
            if not hasattr(self, "_es_graph_by_bucket"):
                self._es_graph_by_bucket = {}
            gkey = (bucket, n_sample, attn_impl, rail_impl)
            if gkey not in self._es_graph_by_bucket:
                rs = self._es_install_state(bucket, n_sample, device,
                                            attn_impl=attn_impl, rail_impl=rail_impl)
                for ln, s in sigma_eff.items():
                    rs["sigma_buf"][ln].fill_(float(s))
                gs = self._es_capture_step_packed(
                    model, device, bucket, n_sample, states, max_seq_len_cap,
                    rs)
                self._es_graph_by_bucket[gkey] = gs
            gs = self._es_graph_by_bucket[gkey]
            # This wave's KV pages into the pinned block table (see method).
            self._es_refresh_kv_pages(gs, states)
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
                "es_attn_impl": attn_impl,
                "es_attn_meta": gs.get("es_attn_meta"),
                "es_rail_impl": rail_impl,
                "es_x": {},
                "es_width": width,
            })
            for ln, s in sigma_eff.items():
                gs["sigma_buf"][ln].fill_(float(s))
            rs = gs
        else:
            rs = self._es_install_state(bucket, n_sample, device,
                                        attn_impl=attn_impl, rail_impl=rail_impl)
            for ln, s in sigma_eff.items():
                rs["sigma_buf"][ln].fill_(float(s))

        # Streaming LM head state (lm_head_kernel.py): the lm_head weight, the
        # clean-row -> slot index and persistent (m, s) / clean-logit buffers.
        if lm_impl == "stream":
            lm_head = model.lm_head
            V = int(getattr(model.logits_processor, "org_vocab_size",
                            lm_head.weight.shape[0]))
            W_lm = lm_head.weight[:V]
            clean_idx = torch.full((bucket * width,), -1, dtype=torch.int32,
                                   device=device)
            clean_idx[rs["clean_row_idx"]] = torch.arange(
                bucket, dtype=torch.int32, device=device)
            ws_lm = LMHeadWorkspace(bucket * width, V, bucket, 128, device)
        prof_lm = os.environ.get("ES_PROFILE_LMHEAD")
        lm_ms = 0.0

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
                    hidden = self._es_replay_step_packed(
                        model, states, n_sample, es_cfg, t, slot_rollout_ids,
                        rs)
                else:
                    hidden = self._es_eager_step_packed(
                        model, device, states, n_sample, es_cfg, t,
                        slot_rollout_ids, rs, max_seq_len_cap)

                if prof_lm:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                # LM head + payload + clean sampling over ALL slots at once.
                if lm_impl == "stream":
                    # Never materialises [R, V]: clean rows' logits (fp32) for
                    # sampling + every row's LSE; the clean token's rail logit
                    # is a [R, d] gather-dot after sampling.
                    clean_logits, lse, _ = lm_head_stream(
                        hidden, W_lm, clean_idx, ws=ws_lm)      # [bucket, V], [R]
                else:
                    logits = model.compute_logits(hidden)       # [R, vocab]
                    logits_f = logits.float()                   # [R, vocab]
                    lse = torch.logsumexp(logits_f, dim=-1)     # [R]
                    clean_logits = logits_f[clean_row_idx]      # [bucket, vocab]
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
                if force_tokens is not None:   # test-only: pin the clean token
                    ft = [int(force_tokens[p][t]) if p < B and t < len(force_tokens[p])
                          else int(next_toks[p]) for p in range(bucket)]
                    next_toks = torch.tensor(ft, dtype=next_toks.dtype,
                                             device=next_toks.device)
                chosen = next_toks.repeat_interleave(width)     # [R]
                if lm_impl == "stream":
                    tok_logp = lm_head_gather_logit(hidden, W_lm, chosen) - lse
                else:
                    tok_logp = logits_f.gather(1, chosen[:, None])[:, 0] - lse
                payload_buf[:, t] = tok_logp
                if prof_lm:
                    ev1.record()
                    ev1.synchronize()
                    lm_ms += ev0.elapsed_time(ev1)

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
        if prof_lm:
            self._es_lmhead_ms = lm_ms

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
