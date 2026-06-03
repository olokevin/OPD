## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import ast
import warnings
from .polar_express import PolarExpress, FastApplyPolarExpress

@torch.compile
def jiacheng(G, steps):
    """
    Jiacheng optimized polynomials
    """
    assert len(G.shape) >= 2
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315) 
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def svd_exact_polar(G, _, cutoff=None, reverse=False):
    """
    Exact polar factorization via SVD
    """
    assert len(G.shape) >= 2
    U, Sigma, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
    if cutoff is None:
        return (U @ Vh).to(G.dtype)
    else:
        Sigma = ((Sigma / Sigma.max()) >= cutoff).to(G.dtype)  # zero out small singular values
        if reverse: Sigma = 2*Sigma - 1
        return (U @ torch.diag(Sigma) @ Vh).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        lr_adam: Optional learning rate for AdamW-updated parameters.
        If None, AdamW-updated parameters share `lr`.
        lr_embedding: Optional learning rate for embedding parameters updated by AdamW.
        If None, embedding parameters share AdamW lr.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        norm_method: Optional update normalization method for Muon-style updates.
        "shape" selects row/col normalization by update shape:
        m>n -> row, m<n -> col, m==n -> none.
        norm_layers: "all" or a list of layer-name substrings to gate normalization.
        cola_ortho_method: "default" uses CoLA LR scaling sqrt(shape[1] / shape[0]);
        "outin" uses CoLA LR scaling shape[1] / shape[0];
        "spectron" uses lr/(sigma_a + sigma_b + 1) for CoLA params;
        "spectron-rms" uses rho-based LR with CoLA factor-shape scaling;
        "spectron-scale" uses spectron denominator with CoLA factor-shape scaling.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 lr_adam=None,
                 lr_embedding=None,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 rms_scaling=True,
                 nuclear_scaling=False,
                 polar_method="Keller",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 split_heads=False,
                 nheads=None,
                 norm_method=None,
                 norm_layers="all",
                 polar_args={},
                 polar_params=None,
                 structured_ortho_method="mup",
                 cola_ortho_method="default",
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "NewtonSchultz", "Keller", "Pole") where PolE = PolarExpress
        """
        if norm_method is not None:
            norm_method = norm_method.lower()
        norm_layers = self._normalize_norm_layers(norm_layers)
        named_params = list(named_params)
        allowed_norm_methods = {None, "row", "col", "row-col", "col-row", "shape"}
        if norm_method not in allowed_norm_methods:
            raise ValueError(
                f"Unknown norm_method: {norm_method}. "
                f"Expected one of {sorted(m for m in allowed_norm_methods if m is not None)} or None."
            )
        norm_layers_for_defaults = "all" if norm_layers == "all" else tuple(norm_layers)
        structured_ortho_method = structured_ortho_method.lower()
        allowed_structured_ortho_methods = {"mup", "svd", "rms", "naive"}
        if structured_ortho_method not in allowed_structured_ortho_methods:
            raise ValueError(
                f"Unknown structured_ortho_method: {structured_ortho_method}. "
                f"Expected one of {sorted(allowed_structured_ortho_methods)}."
            )
        cola_ortho_method = cola_ortho_method.lower()
        allowed_cola_ortho_methods = {
            "default",
            "outin",
            "spectron",
            "spectron-rms",
            "spectron-scale",
        }
        if cola_ortho_method not in allowed_cola_ortho_methods:
            raise ValueError(
                f"Unknown cola_ortho_method: {cola_ortho_method}. "
                f"Expected one of {sorted(allowed_cola_ortho_methods)}."
            )
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
                norm_method=norm_method,
                norm_layers=norm_layers_for_defaults,
                structured_ortho_method=structured_ortho_method,
                cola_ortho_method=cola_ortho_method,
        )
        adamw_lr = lr if lr_adam is None else lr_adam
        embedding_lr = adamw_lr if lr_embedding is None else lr_embedding
        self._param_names_by_id = {}
        for name, p in named_params:
            self._param_names_by_id.setdefault(id(p), []).append(name)
        self._norm_layers = norm_layers_for_defaults
        self._norm_layer_tokens = (
            tuple(layer.lower() for layer in norm_layers)
            if norm_layers != "all"
            else ()
        )
        if norm_layers == "all":
            self._norm_param_ids = set(self._param_names_by_id.keys())
        else:
            self._norm_param_ids = set()
            for param_id, names in self._param_names_by_id.items():
                if any(
                    token in name.lower()
                    for token in self._norm_layer_tokens
                    for name in names
                ):
                    self._norm_param_ids.add(param_id)
        self._cola_ab_by_base_name = {}
        cola_a_by_base = {}
        cola_b_by_base = {}
        for name, p in named_params:
            if name.endswith("cola_a"):
                cola_a_by_base[name[: -len("cola_a")]] = p
            elif name.endswith("cola_b"):
                cola_b_by_base[name[: -len("cola_b")]] = p
        for base_name, cola_a in cola_a_by_base.items():
            cola_b = cola_b_by_base.get(base_name, None)
            if cola_b is None:
                continue
            self._cola_ab_by_base_name[base_name] = (cola_a, cola_b)
        self.cola_ortho_method = cola_ortho_method
        self._record_step = None
        self._active_record = {}
        self._active_momentum_record = {}
        self._active_applied_update_record = {}
        self._last_record = {}
        self._last_momentum_record = {}
        self._last_applied_update_record = {}
        
        # print("EMBED TOKENS AND LM_HEAD ARE NOT HANDLED CORRECTLY FOR MUON, THEY SHOULD BE WITH ADAMW.")
        muon_params, muon_params_names = [], []
        btt_r_params = []
        btt_g_params = []
        btt_l_params = []
        adamw_params = []
        embedding_params = []
        for name, p in named_params:
            is_embedding = any(
                token in name for token in ["embeddings", "embed_tokens", "wte", "wpe"]
            )
            is_excluded = is_embedding or ("lm_head" in name)
            if "btt_r" in name:
                btt_r_params.append(p)
            elif "btt_g" in name:
                btt_g_params.append(p)
            elif "btt_l" in name:
                btt_l_params.append(p)
            elif p.ndim == 2 and not is_excluded:
                muon_params.append(p)
                muon_params_names.append(name)
            elif is_embedding:
                embedding_params.append(p)
            else:
                adamw_params.append(p)
        muon_like_params = list(muon_params)
        muon_like_params.extend(btt_r_params)
        muon_like_params.extend(btt_g_params)
        muon_like_params.extend(btt_l_params)
        params = []
        if len(muon_like_params) > 0:
            params.append({"params": muon_like_params, "lr": lr})
        if len(adamw_params) > 0:
            params.append({"params": adamw_params, "lr": adamw_lr})
        if len(embedding_params) > 0:
            params.append({"params": embedding_params, "lr": embedding_lr})
        if split_heads or (nheads is not None):
            warnings.warn(
                "[Muon] split_heads/nheads are deprecated and ignored; "
                "Muon now runs matrix-only updates without split-head logic.",
                stacklevel=2,
            )
        super().__init__(params, defaults)
        self._warned_uncategorized_param_ids = set()
        self._warned_cola_lr_fallback_pair_keys = set()
        self._embedding_param_ids = {id(p) for p in embedding_params}
        
        # Sort parameters into those for which we will use Muon, and those for which we will not.
        # Muon applies to matrix-shaped (2D) non-excluded parameters in muon_params.
        for p, p_name in zip(muon_params, muon_params_names):
            self.state[p]["use_muon"] = True
            self.state[p]["apply_muon_norm"] = id(p) in self._norm_param_ids
            if p_name.endswith("cola_a"):
                self.state[p]["is_cola_a"] = True
                self.state[p]["cola_pair_key"] = p_name[: -len("cola_a")]
            elif p_name.endswith("cola_b"):
                self.state[p]["is_cola_b"] = True
                self.state[p]["cola_pair_key"] = p_name[: -len("cola_b")]
            elif p_name.endswith("cola_g"):
                self.state[p]["is_cola_g"] = True
                self.state[p]["cola_pair_key"] = p_name[: -len("cola_g")]
        for p in btt_r_params:
            self.state[p]["use_muon"] = False
            self.state[p]["apply_muon_norm"] = id(p) in self._norm_param_ids
            self.state[p]["is_btt_r"] = True
        for p in btt_g_params:
            self.state[p]["use_muon"] = False
            self.state[p]["apply_muon_norm"] = id(p) in self._norm_param_ids
            self.state[p]["is_btt_r"] = True
        for p in btt_l_params:
            self.state[p]["use_muon"] = False
            self.state[p]["apply_muon_norm"] = id(p) in self._norm_param_ids
            self.state[p]["is_btt_l"] = True

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False
            self.state[p]["apply_muon_norm"] = False
        for p in embedding_params:
            # Do not use Muon for parameters in embedding_params
            self.state[p]["use_muon"] = False
            self.state[p]["apply_muon_norm"] = False

        self._initialize_update_buckets()

        print("[Muon] Parameters updated by Muon (standard 2-D matrices):")
        self._print_param_names(self._flatten_params(self._bucket_muon_matrix_by_group))
        print("[Muon] Parameters updated by Muon CoLA:")
        self._print_param_names(
            self._flatten_params(self._bucket_muon_cola_a_by_group)
            + self._flatten_params(self._bucket_muon_cola_b_by_group)
            + self._flatten_params(self._bucket_muon_cola_g_by_group)
        )
        print("[Muon] Parameters updated by Muon BTT:")
        self._print_param_names(self._flatten_params(self._bucket_btt_by_group))
        print("[Muon] Parameters updated by AdamW:")
        self._print_param_names(self._flatten_params(self._bucket_adamw_by_group))
        print("[Muon] Embedding parameters updated by AdamW:")
        self._print_param_names(self._flatten_params(self._bucket_embedding_adamw_by_group))

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)
        self._vmap_status = None

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress 
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "svd-exact":
            return partial(svd_exact_polar, cutoff=polar_args.get("svd_cutoff", None), reverse=polar_args.get("svd_reverse", False))
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(
        self,
        lr,
        rms_scaling,
        nuclear_scaling,
        param_shape,
        grad,
        grad_sign,
        btt_scale=None,
    ):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            ### original 
            scale *= math.sqrt(fan_out / fan_in)
            ### Jornan
            # scale *= math.sqrt(max(1, fan_out / fan_in))
            # Match with Kimi
            # scale *= math.sqrt(max(fan_out/fan_in, fan_in/fan_out))
            ### Kimi (not use)
            # scale *= 0.2 * math.sqrt(max(fan_out,fan_in))
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        if btt_scale is not None:
            scale *= btt_scale
        return lr * scale

    def adjust_lr_for_cola_default(
        self,
        lr,
        rms_scaling,
        nuclear_scaling,
        p,
        grad,
        grad_sign,
    ):
        scale = 1.0
        if rms_scaling:
            rows, cols = p.shape[:2]
            # CoLA factor shapes:
            # cola_a / cola_g: (in_features, rank) -> sqrt(rank / in_features)
            # cola_b: (rank, out_features) -> sqrt(out_features / rank)
            scale *= math.sqrt(cols / rows)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def adjust_lr_for_cola_outin(
        self,
        lr,
        rms_scaling,
        nuclear_scaling,
        p,
        grad,
        grad_sign,
    ):
        scale = 1.0
        if rms_scaling:
            rows, cols = p.shape[:2]
            # CoLA factor shapes:
            # cola_a / cola_g: (in_features, rank) -> rank / in_features
            # cola_b: (rank, out_features) -> out_features / rank
            scale *= cols / rows
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def _normalize_u(self, u, norm_method):
        if norm_method is None:
            return u
        eps = 1e-7
        if norm_method == "row":
            return u / (u.norm(dim=-1, keepdim=True) + eps)
        if norm_method == "col":
            return u / (u.norm(dim=-2, keepdim=True) + eps)
        if norm_method == "shape":
            m, n = u.shape[-2], u.shape[-1]
            if m > n:
                return u / (u.norm(dim=-1, keepdim=True) + eps)
            if m < n:
                return u / (u.norm(dim=-2, keepdim=True) + eps)
            return u
        if norm_method == "row-col":
            u = u / (u.norm(dim=-1, keepdim=True) + eps)
            return u / (u.norm(dim=-2, keepdim=True) + eps)
        if norm_method == "col-row":
            u = u / (u.norm(dim=-2, keepdim=True) + eps)
            return u / (u.norm(dim=-1, keepdim=True) + eps)
        raise ValueError(f"Unknown norm_method: {norm_method}")

    def _normalize_norm_layers(self, norm_layers):
        if norm_layers is None:
            return "all"
        if isinstance(norm_layers, str):
            stripped = norm_layers.strip()
            if stripped.lower() == "all":
                return "all"
            if stripped == "":
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        "norm_layers string list must be parseable as a Python list of strings."
                    ) from exc
                if not isinstance(parsed, (list, tuple, set)):
                    raise ValueError(
                        "norm_layers string list must be parseable as a Python list of strings."
                    )
                return self._normalize_norm_layers(parsed)
            return [layer.strip() for layer in stripped.split(",") if layer.strip()]
        if isinstance(norm_layers, (list, tuple, set)):
            layers = []
            for layer in norm_layers:
                if not isinstance(layer, str):
                    raise ValueError(
                        "norm_layers must be 'all' or a sequence of strings."
                    )
                stripped = layer.strip()
                if stripped:
                    layers.append(stripped)
            return layers
        raise ValueError("norm_layers must be 'all' or a sequence of strings.")

    def _maybe_normalize_u(self, p, u, group):
        if group["norm_method"] is None:
            return u
        state = self.state[p]
        if not state.get("apply_muon_norm", False):
            return u
        return self._normalize_u(u, group["norm_method"])

    def _apply_polar_per_matrix(self, mats, steps):
        # mats: (K, M, N)
        if hasattr(torch, "vmap"):
            if self._vmap_status is None:
                try:
                    _ = torch.vmap(lambda x: self.polar_factorizer(x, steps))(mats[:1])
                    self._vmap_status = "vmap"
                    print("[Muon] BTT polar: using vmap")
                except Exception:
                    self._vmap_status = "loop"
                    print("[Muon] BTT polar: vmap failed, falling back to loop")
            if self._vmap_status == "vmap":
                return torch.vmap(lambda x: self.polar_factorizer(x, steps))(mats)
        outs = []
        for i in range(mats.shape[0]):
            outs.append(self.polar_factorizer(mats[i], steps))
        return torch.stack(outs, dim=0)

    def _adamw_update(self, p, g, group):
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["weight_decay"]
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["moment1"] = torch.zeros_like(g)
            state["moment2"] = torch.zeros_like(g)
        state["step"] += 1
        step = state["step"]
        buf1 = state["moment1"]
        buf2 = state["moment2"]
        buf1.lerp_(g, 1 - beta1)
        buf2.lerp_(g.square(), 1 - beta2)

        g = buf1 / (eps + buf2.sqrt())

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        scale = bias_correction1 / bias_correction2**0.5
        p.data.mul_(1 - lr * weight_decay)
        p.data.add_(g, alpha=-lr / scale)

    def _spectral_norm_power_iteration(self, p, n_iters):
        w = p.data
        if w.ndim == 0:
            return w.abs().to(torch.float32)
        if w.ndim == 1:
            return w.norm().to(torch.float32)
        if w.ndim > 2:
            w = w.reshape(w.shape[0], -1)
        w = w.to(torch.float32)
        _, n = w.shape
        eps = 1e-12
        state = self.state[p]
        v = state.get("spectron_v", None)
        if (v is None) or (v.numel() != n) or (v.device != w.device):
            v = torch.randn(n, device=w.device, dtype=w.dtype)
        else:
            v = v.to(device=w.device, dtype=w.dtype)
        v = v / (v.norm() + eps)

        n_iters = max(1, int(n_iters))
        u = None
        for _ in range(n_iters):
            u = w @ v
            u = u / (u.norm() + eps)
            v = w.t() @ u
            v = v / (v.norm() + eps)
        sigma = torch.dot(u, w @ v).abs()
        state["spectron_v"] = v.detach()
        return sigma

    def adjust_lr_for_cola_spectron(self, lr, p, ns_steps, pair_lr_cache):
        pair_key = self.state[p].get("cola_pair_key", None)
        if pair_key is None:
            return float(lr)

        cached_lr = pair_lr_cache.get(pair_key, None)
        if cached_lr is not None:
            return cached_lr

        cola_ab = self._cola_ab_by_base_name.get(pair_key, None)
        if cola_ab is None:
            adjusted_lr = float(lr)
        else:
            cola_a, cola_b = cola_ab
            sigma_a = float(self._spectral_norm_power_iteration(cola_a, ns_steps).item())
            sigma_b = float(self._spectral_norm_power_iteration(cola_b, ns_steps).item())
            adjusted_lr = float(lr) / (sigma_a + sigma_b + 1.0)
        pair_lr_cache[pair_key] = adjusted_lr
        return adjusted_lr

    def _warn_cola_lr_fallback(self, pair_key, reason, method_name):
        warn_key = (method_name, pair_key)
        if warn_key in self._warned_cola_lr_fallback_pair_keys:
            return
        warnings.warn(
            f"[Muon] CoLA {method_name} LR fallback to base lr for pair {pair_key}: {reason}",
            stacklevel=2,
        )
        self._warned_cola_lr_fallback_pair_keys.add(warn_key)

    def adjust_lr_for_cola_spectron_rms(self, lr, p, ns_steps, pair_lr_cache):
        pair_key = self.state[p].get("cola_pair_key", None)
        if pair_key is None:
            return float(lr)

        cached_lr_by_type = pair_lr_cache.get(pair_key, None)
        if cached_lr_by_type is None:
            cola_ab = self._cola_ab_by_base_name.get(pair_key, None)
            if cola_ab is None:
                cached_lr_by_type = {"cola_a_like": float(lr), "cola_b": float(lr)}
            else:
                cola_a, cola_b = cola_ab
                if cola_a.ndim != 2 or cola_b.ndim != 2:
                    self._warn_cola_lr_fallback(
                        pair_key,
                        "spectron-rms requires 2-D CoLA factors",
                        "spectron-rms",
                    )
                    cached_lr_by_type = {"cola_a_like": float(lr), "cola_b": float(lr)}
                else:
                    r_from_a, n = cola_a.shape
                    m, r_from_b = cola_b.shape
                    if r_from_a != r_from_b:
                        self._warn_cola_lr_fallback(
                            pair_key,
                            "spectron-rms requires cola_a.shape[0] == cola_b.shape[1]",
                            "spectron-rms",
                        )
                        cached_lr_by_type = {
                            "cola_a_like": float(lr),
                            "cola_b": float(lr),
                        }
                    else:
                        r = float(r_from_a)
                        m = float(m)
                        n = float(n)
                        sigma_a = float(
                            self._spectral_norm_power_iteration(cola_a, ns_steps).item()
                        )
                        sigma_b = float(
                            self._spectral_norm_power_iteration(cola_b, ns_steps).item()
                        )
                        eps = 1e-12
                        denom = (
                            math.sqrt(max(r / max(m, eps), 0.0)) * sigma_b
                            + math.sqrt(max(n / max(r, eps), 0.0)) * sigma_a
                            + 1.0
                        )
                        rho = 1.0 / max(denom, eps)
                        cached_lr_by_type = {
                            "cola_a_like": float(lr) * rho * math.sqrt(max(r / max(n, eps), 0.0)),
                            "cola_b": float(lr) * rho * math.sqrt(max(m / max(r, eps), 0.0)),
                        }
            pair_lr_cache[pair_key] = cached_lr_by_type

        state = self.state[p]
        if state.get("is_cola_b", False):
            return cached_lr_by_type["cola_b"]
        return cached_lr_by_type["cola_a_like"]

    def adjust_lr_for_cola_spectron_scale(self, lr, p, ns_steps, pair_lr_cache):
        pair_key = self.state[p].get("cola_pair_key", None)
        if pair_key is None:
            return float(lr)

        cached_lr_by_type = pair_lr_cache.get(pair_key, None)
        if cached_lr_by_type is None:
            cola_ab = self._cola_ab_by_base_name.get(pair_key, None)
            if cola_ab is None:
                cached_lr_by_type = {"cola_a_like": float(lr), "cola_b": float(lr)}
            else:
                cola_a, cola_b = cola_ab
                if cola_a.ndim != 2 or cola_b.ndim != 2:
                    self._warn_cola_lr_fallback(
                        pair_key,
                        "spectron-scale requires 2-D CoLA factors",
                        "spectron-scale",
                    )
                    cached_lr_by_type = {"cola_a_like": float(lr), "cola_b": float(lr)}
                else:
                    in_features, r_from_a = cola_a.shape
                    r_from_b, out_features = cola_b.shape
                    if r_from_a != r_from_b:
                        self._warn_cola_lr_fallback(
                            pair_key,
                            "spectron-scale requires cola_a.shape[1] == cola_b.shape[0]",
                            "spectron-scale",
                        )
                        cached_lr_by_type = {
                            "cola_a_like": float(lr),
                            "cola_b": float(lr),
                        }
                    else:
                        sigma_a = float(
                            self._spectral_norm_power_iteration(cola_a, ns_steps).item()
                        )
                        sigma_b = float(
                            self._spectral_norm_power_iteration(cola_b, ns_steps).item()
                        )
                        eps = 1e-12
                        rank = float(r_from_a)
                        in_features = float(in_features)
                        out_features = float(out_features)
                        rho = 1.0 / max(sigma_a + sigma_b + 1.0, eps)
                        # rho = max(1.0 / max(sigma_a + sigma_b + 1.0, eps), 0.1)
                        spectron_lr = float(lr) * rho
                        cached_lr_by_type = {
                            "cola_a_like": spectron_lr
                            * math.sqrt(max(rank / max(in_features, eps), 0.0)),
                            "cola_b": spectron_lr
                            * math.sqrt(max(out_features / max(rank, eps), 0.0)),
                        }
            pair_lr_cache[pair_key] = cached_lr_by_type

        state = self.state[p]
        if state.get("is_cola_b", False):
            return cached_lr_by_type["cola_b"]
        return cached_lr_by_type["cola_a_like"]

    def _warn_uncategorized_param(self, p, reason):
        param_id = id(p)
        if param_id in self._warned_uncategorized_param_ids:
            return
        names = self._param_names_by_id.get(param_id, [])
        name = names[0] if len(names) > 0 else "<unnamed>"
        warnings.warn(
            f"[Muon] Falling back to AdamW for param {name} "
            f"shape={tuple(p.shape)}: {reason}",
            stacklevel=2,
        )
        self._warned_uncategorized_param_ids.add(param_id)

    def _append_adamw_bucket(self, p, adamw_bucket, embedding_adamw_bucket):
        if id(p) in self._embedding_param_ids:
            embedding_adamw_bucket.append(p)
        else:
            adamw_bucket.append(p)

    def _initialize_update_buckets(self):
        self._bucket_muon_matrix_by_group = []
        self._bucket_muon_cola_a_by_group = []
        self._bucket_muon_cola_b_by_group = []
        self._bucket_muon_cola_g_by_group = []
        self._bucket_btt_by_group = []
        self._bucket_adamw_by_group = []
        self._bucket_embedding_adamw_by_group = []

        for group in self.param_groups:
            muon_matrix_params = []
            muon_cola_a_params = []
            muon_cola_b_params = []
            muon_cola_g_params = []
            btt_params = []
            adamw_params = []
            embedding_adamw_params = []

            for p in group["params"]:
                state = self.state[p]
                is_btt = state.get("is_btt_r", False) or state.get("is_btt_l", False)
                is_cola_a = state.get("is_cola_a", False)
                is_cola_b = state.get("is_cola_b", False)
                is_cola_g = state.get("is_cola_g", False)
                is_cola = is_cola_a or is_cola_b or is_cola_g

                if is_btt:
                    btt_params.append(p)
                    continue

                if is_cola:
                    if state.get("use_muon", False) and p.ndim == 2:
                        if is_cola_a:
                            muon_cola_a_params.append(p)
                        elif is_cola_b:
                            muon_cola_b_params.append(p)
                        else:
                            muon_cola_g_params.append(p)
                    else:
                        self._warn_uncategorized_param(
                            p,
                            "CoLA parameter did not meet matrix Muon criteria; using AdamW fallback",
                        )
                        state["use_muon"] = False
                        self._append_adamw_bucket(p, adamw_params, embedding_adamw_params)
                    continue

                if state.get("use_muon", False) and p.ndim == 2:
                    muon_matrix_params.append(p)
                    continue

                if state.get("use_muon", False):
                    self._warn_uncategorized_param(
                        p,
                        "Muon parameter was not matrix-shaped and was routed to AdamW fallback",
                    )
                    state["use_muon"] = False
                self._append_adamw_bucket(p, adamw_params, embedding_adamw_params)

            self._bucket_muon_matrix_by_group.append(muon_matrix_params)
            self._bucket_muon_cola_a_by_group.append(muon_cola_a_params)
            self._bucket_muon_cola_b_by_group.append(muon_cola_b_params)
            self._bucket_muon_cola_g_by_group.append(muon_cola_g_params)
            self._bucket_btt_by_group.append(btt_params)
            self._bucket_adamw_by_group.append(adamw_params)
            self._bucket_embedding_adamw_by_group.append(embedding_adamw_params)

    def _flatten_params(self, params_by_group):
        flat = []
        for params in params_by_group:
            flat.extend(params)
        return flat

    def _print_param_names(self, params):
        printed_ids = set()
        for p in params:
            param_id = id(p)
            if param_id in printed_ids:
                continue
            printed_ids.add(param_id)
            names = self._param_names_by_id.get(param_id, [])
            if len(names) == 0:
                print(f"  <unnamed:{param_id}>")
                continue
            for name in names:
                print(f"  {name}")

    def _muon_update_params(self, params, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]

        for p in params:
            g = p.grad
            if g is None:
                continue
            if g.ndim != 2:
                self._warn_uncategorized_param(
                    p, "_muon_update_params expects matrix-shaped gradients"
                )
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if group["nesterov"]:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

            u = self.polar_factorizer(g, group["ns_steps"])

            adjusted_lr = self.adjust_lr_for_muon(
                lr,
                group["rms_scaling"],
                group["nuclear_scaling"],
                p.shape,
                g.bfloat16(),
                u,
            )
            if isinstance(adjusted_lr, torch.Tensor):
                adjusted_lr = float(adjusted_lr.item())

            p.data.mul_(1 - lr * weight_decay)

            u = self._maybe_normalize_u(p, u, group)
            if self._record_step is not None:
                param_names = self._param_names_by_id.get(id(p), [])
                if len(param_names) > 0:
                    u_cpu = u.detach().cpu()
                    applied_update_cpu = u.detach().mul(float(adjusted_lr)).cpu()
                    buf_cpu = buf.detach().cpu() if p.ndim == 2 else None
                    for name in param_names:
                        if p.ndim == 2:
                            self._active_record[name] = u_cpu
                            self._active_applied_update_record[name] = applied_update_cpu
                            self._active_momentum_record[name] = buf_cpu
            p.data.add_(u, alpha=-adjusted_lr)

    def _cola_muon_update_params(self, cola_a_params, cola_b_params, cola_g_params, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        cola_params = cola_a_params + cola_b_params + cola_g_params
        if len(cola_params) == 0:
            return
        cola_ortho_method = self.cola_ortho_method
        pair_lr_cache = {}

        def _update_one_param(p):
            g = p.grad
            if g is None:
                return
            if g.ndim != 2:
                self._warn_uncategorized_param(
                    p, "_cola_muon_update_params expects matrix-shaped gradients"
                )
                return

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if group["nesterov"]:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

            u = self.polar_factorizer(g, group["ns_steps"])

            if cola_ortho_method == "spectron":
                adjusted_lr = self.adjust_lr_for_cola_spectron(
                    lr=lr,
                    p=p,
                    ns_steps=group["ns_steps"],
                    pair_lr_cache=pair_lr_cache,
                )
            elif cola_ortho_method == "spectron-rms":
                adjusted_lr = self.adjust_lr_for_cola_spectron_rms(
                    lr=lr,
                    p=p,
                    ns_steps=group["ns_steps"],
                    pair_lr_cache=pair_lr_cache,
                )
            elif cola_ortho_method == "spectron-scale":
                adjusted_lr = self.adjust_lr_for_cola_spectron_scale(
                    lr=lr,
                    p=p,
                    ns_steps=group["ns_steps"],
                    pair_lr_cache=pair_lr_cache,
                )
            elif cola_ortho_method == "outin":
                adjusted_lr = self.adjust_lr_for_cola_outin(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p,
                    g.bfloat16(),
                    u,
                )
            else:
                adjusted_lr = self.adjust_lr_for_cola_default(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p,
                    g.bfloat16(),
                    u,
                )
                if isinstance(adjusted_lr, torch.Tensor):
                    adjusted_lr = float(adjusted_lr.item())
            p.data.mul_(1 - lr * weight_decay)

            u = self._maybe_normalize_u(p, u, group)
            if self._record_step is not None:
                param_names = self._param_names_by_id.get(id(p), [])
                if len(param_names) > 0:
                    u_cpu = u.detach().cpu()
                    applied_update_cpu = u.detach().mul(float(adjusted_lr)).cpu()
                    buf_cpu = buf.detach().cpu() if p.ndim == 2 else None
                    for name in param_names:
                        if p.ndim == 2:
                            self._active_record[name] = u_cpu
                            self._active_applied_update_record[name] = applied_update_cpu
                            self._active_momentum_record[name] = buf_cpu
            p.data.add_(u, alpha=-adjusted_lr)

        for p in cola_a_params:
            _update_one_param(p)
        for p in cola_b_params:
            _update_one_param(p)
        for p in cola_g_params:
            _update_one_param(p)

    def begin_record_step(self, step):
        self._record_step = int(step)
        self._active_record = {}
        self._active_momentum_record = {}
        self._active_applied_update_record = {}

    def end_record_step(self):
        self._record_step = None
        self._last_record = self._active_record
        self._last_momentum_record = self._active_momentum_record
        self._last_applied_update_record = self._active_applied_update_record
        self._active_record = {}
        self._active_momentum_record = {}
        self._active_applied_update_record = {}

    def consume_last_record(self):
        out = self._last_record
        self._last_record = {}
        self._last_momentum_record = {}
        self._last_applied_update_record = {}
        return out

    def consume_last_record_details(self):
        out = {
            "muon_u": self._last_record,
            "momentum_buffer": self._last_momentum_record,
            "applied_update": self._last_applied_update_record,
        }
        self._last_record = {}
        self._last_momentum_record = {}
        self._last_applied_update_record = {}
        return out

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group_idx, group in enumerate(self.param_groups):
            muon_matrix_params = self._bucket_muon_matrix_by_group[group_idx]
            muon_cola_a_params = self._bucket_muon_cola_a_by_group[group_idx]
            muon_cola_b_params = self._bucket_muon_cola_b_by_group[group_idx]
            muon_cola_g_params = self._bucket_muon_cola_g_by_group[group_idx]
            btt_params = self._bucket_btt_by_group[group_idx]
            adamw_params = self._bucket_adamw_by_group[group_idx]
            embedding_adamw_params = self._bucket_embedding_adamw_by_group[group_idx]

            ############################
            #           Muon           #
            ############################
            
            self._muon_update_params(muon_matrix_params, group)

            ############################
            #           Cola Muon           #
            ############################

            self._cola_muon_update_params(
                muon_cola_a_params,
                muon_cola_b_params,
                muon_cola_g_params,
                group,
            )
                
            ############################
            #         BTT Muon         #
            ############################

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in btt_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if state.get("is_btt_r", False):
                    method = group["structured_ortho_method"]
                    if g.dim() == 4:
                        # Old layout: (m, n, r, b)
                        m, n, r, b = g.shape
                        if method == "mup":
                            mats = g.permute(1, 0, 2, 3).reshape(n, m * r, b)
                            u = (
                                self._apply_polar_per_matrix(mats, group["ns_steps"])
                                .reshape(n, m, r, b)
                                .permute(1, 0, 2, 3)
                            )
                            btt_scale = math.sqrt(m * r / b)
                        elif method == "svd":
                            mats = g.reshape(-1, r, b)
                            u = self._apply_polar_per_matrix(
                                mats, group["ns_steps"]
                            ).reshape(m, n, r, b)
                            btt_scale = math.sqrt(r / b / n)  
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * r, n * b),
                                group["ns_steps"],
                            ).reshape(m, r, n, b).transpose(1, 2)
                            btt_scale = math.sqrt((m * r) / (n * b))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    elif g.dim() == 3:
                        # Canonical new layout: (n, b, m*r)
                        n, b, mr = g.shape
                        if method == "mup":
                            u = self._apply_polar_per_matrix(
                                g, group["ns_steps"]
                            )
                            btt_scale = math.sqrt(mr / b)
                        elif method == "svd":
                            rank = getattr(p, "btt_rank", None)
                            m = getattr(p, "btt_m", None)
                            if (
                                not isinstance(rank, int)
                                or rank <= 0
                                or not isinstance(m, int)
                                or m <= 0
                                or (m * rank) != mr
                            ):
                                raise ValueError(
                                    "BTT svd update expects btt_r metadata "
                                    "(btt_rank, btt_m) with m * rank == g.shape[2]."
                                )

                            mats = g.permute(0, 2, 1).reshape(n, m, rank, b)
                            mats = mats.reshape(n * m, rank, b)
                            u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                            u = u.reshape(n, m, rank, b).reshape(n, mr, b)
                            u = u.permute(0, 2, 1)
                            btt_scale = math.sqrt(rank / b)
                        elif method == "rms":
                            rank = getattr(p, "btt_rank", None)
                            m = getattr(p, "btt_m", None)
                            if (
                                not isinstance(rank, int)
                                or rank <= 0
                                or not isinstance(m, int)
                                or m <= 0
                                or (m * rank) != mr
                            ):
                                raise ValueError(
                                    "BTT rms update expects btt_r metadata "
                                    "(btt_rank, btt_m) with m * rank == g.shape[2]."
                                )

                            mats = g.permute(0, 2, 1).reshape(n, m, rank, b)
                            mats = mats.reshape(n * m, rank, b)
                            u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                            u = u.reshape(n, m, rank, b).reshape(n, mr, b)
                            u = u.permute(0, 2, 1)
                            btt_scale = math.sqrt(rank / b)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.permute(2, 0, 1).reshape(mr, n * b),
                                group["ns_steps"],
                            ).reshape(mr, n, b).permute(1, 2, 0)
                            btt_scale = math.sqrt(mr / (n * b))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    else:
                        self._adamw_update(p, g, group)
                        continue
                elif state.get("is_btt_l", False):
                    method = group["structured_ortho_method"]
                    if g.dim() == 4:
                        # Old layout: (m, n, a, r)
                        m, n, a, r = g.shape
                        if method == "mup":
                            mats = g.permute(0, 2, 1, 3).reshape(m, a, n * r)
                            u = (
                                self._apply_polar_per_matrix(mats, group["ns_steps"])
                                .reshape(m, a, n, r)
                                .permute(0, 2, 1, 3)
                            )
                            btt_scale = math.sqrt(a / (n * r))
                        elif method == "svd":
                            mats = g.reshape(-1, a, r)
                            u = self._apply_polar_per_matrix(
                                mats, group["ns_steps"]
                            ).reshape(m, n, a, r)
                            btt_scale = math.sqrt(a / r / n)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * a, n * r),
                                group["ns_steps"],
                            ).reshape(m, a, n, r).transpose(1, 2)
                            btt_scale = math.sqrt((m * a) / (n * r))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    elif g.dim() == 3:
                        # Canonical new layout: (m, n*r, a)
                        m, nr, a = g.shape
                        if method == "mup":
                            u = self._apply_polar_per_matrix(
                                g, group["ns_steps"]
                            )
                            btt_scale = math.sqrt(a / nr)
                        elif method == "svd":
                            rank = getattr(p, "btt_rank", None)
                            n = getattr(p, "btt_n", None)
                            if (
                                not isinstance(rank, int)
                                or rank <= 0
                                or not isinstance(n, int)
                                or n <= 0
                                or (n * rank) != nr
                            ):
                                raise ValueError(
                                    "BTT svd update expects btt_l metadata "
                                    "(btt_rank, btt_n) with n * rank == g.shape[1]."
                                )

                            mats = g.reshape(m, n, rank, a).permute(0, 1, 3, 2)
                            mats = mats.reshape(m * n, a, rank)
                            u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                            u = u.reshape(m, n, a, rank).permute(0, 1, 3, 2)
                            u = u.reshape(m, nr, a)
                            btt_scale = math.sqrt(a / rank)
                        elif method == "rms":
                            rank = getattr(p, "btt_rank", None)
                            n = getattr(p, "btt_n", None)
                            if (
                                not isinstance(rank, int)
                                or rank <= 0
                                or not isinstance(n, int)
                                or n <= 0
                                or (n * rank) != nr
                            ):
                                raise ValueError(
                                    "BTT rms update expects btt_l metadata "
                                    "(btt_rank, btt_n) with n * rank == g.shape[1]."
                                )

                            mats = g.reshape(m, n, rank, a).permute(0, 1, 3, 2)
                            mats = mats.reshape(m * n, a, rank)
                            u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                            u = u.reshape(m, n, a, rank).permute(0, 1, 3, 2)
                            u = u.reshape(m, nr, a)
                            btt_scale = math.sqrt(a / rank / n)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * a, nr),
                                group["ns_steps"],
                            ).reshape(m, a, nr).transpose(1, 2)
                            btt_scale = math.sqrt((m * a) / nr)
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    else:
                        self._adamw_update(p, g, group)
                        continue
                else:
                    # Unknown BTT tensor layout; fall back to AdamW for safety.
                    self._adamw_update(p, g, group)
                    continue

                # adjusted_lr = self.adjust_lr_for_muon(
                #     lr,
                #     False,
                #     False,
                #     (1, 1),
                #     g.bfloat16(),
                #     u,
                #     btt_scale=btt_scale,
                # )
                
                adjusted_lr = lr * btt_scale

                p.data.mul_(1 - lr * weight_decay)
                u = self._maybe_normalize_u(p, u, group)
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                self._adamw_update(p, g, group)
            for p in embedding_adamw_params:
                g = p.grad
                if g is None:
                    continue
                self._adamw_update(p, g, group)
                    
        return loss
