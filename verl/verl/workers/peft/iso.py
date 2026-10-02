"""ISO adapters: fixed-spectrum BP training (arXiv:2607.19331).

The ES side of this thread lives in
``verl/workers/rollout/vllm_rollout/es_worker_extension.py`` and is documented in
``docs/results/ES/es_results.md`` §10.  This module is the **first-order (BP)**
side, exposed to verl's FSDP actor through ``peft.mode``.

Paper-faithful modes (ISO-Optimizer, §4.3 of the paper)
-------------------------------------------------------
``iso``      every target linear is ``W = U S0 V^T`` (thin SVD of W0, ``S0`` frozen).
             The base optimizer (AdamW) steps ``U`` and ``V`` directly -- their
             autograd gradients are exactly Eq. 34, ``G_U = G_W V S0`` and
             ``G_V = G_W^T U S0`` -- and after every step both factors are mapped back
             onto the Stiefel manifold with the polar retraction (Eq. 30/36),
             computed in fp64.  ``sigma(W) == S0`` up to floating-point error.
``isobtt``   the same thing on fura's block-wise SVD: ``W[:, blk_j] = U_j S_j V_j^T``
             per input block (blocks from ``_closest_factor_pair``, as in the ES
             ``isobtt`` / fura factoring), ``S_j`` frozen, ``U_j, V_j`` trained and
             retracted.  Each *block's* spectrum is fixed; the global one is not.

Two implementation details that do not change the maths:

* **Delta storage.**  ``U = U0 + dU`` with ``U0`` frozen and ``dU`` the trainable
  tensor (same for V), and the forward is ``W0 + (U S0 V^T - U0 S0 V0^T)`` with the
  frozen original ``W0``.  With weight decay 0 (the paper's setting, enforced in
  ``retract_iso``) AdamW is shift-invariant, so the trajectory is identical to
  stepping ``U`` itself.  What it buys: under FSDP bf16 mixed precision the forward
  sees ``W0`` exactly plus a small correction, so step 0 is the base model
  bit-for-bit and the actor's weights match the bf16 weights vLLM samples from;
  stepping full ``U`` in bf16 would put ~1 ulp of noise on every weight.
* **Retraction by Newton-Schulz.**  The paper uses an fp64 SVD polar.  ``polar(X)``
  is unique for full-rank X, and the fp64 Newton-Schulz iteration converges to the
  same matrix (max diff 4e-14 on a 9728x2560 frame) in 2 iterations, ~40x faster
  than the SVD; the SVD remains as the fallback if NS does not converge.

Legacy Cayley modes (ES-vs-BP thread, Aug 2026)
-----------------------------------------------
These were called ``iso`` / ``isobtt`` until 2026-10-02 and are kept so the BP runs
in ``docs/results/ES/es_results.md`` stay reproducible.  They use the identity
``F(W0) = {C_L W0 C_R^T : C_L in O(m), C_R in O(n)}`` and parameterise each ``C``
as ``Cay(Omega) = (I - Omega/2)^-1 (I + Omega/2)`` of a trainable skew ``Omega``,
so the constraint holds for any optimizer output and no retraction is needed.

``iso_cayley``     ``W_eff = C_L W0 C_R^T`` with ``C`` block-diagonal (block ``b``) in a
                   fixed random basis -- a strict subset of the paper's family.
``isobtt_cayley``  block-wise right rotation ``W[:, blk_j] = W0[:, blk_j] C_j``.
``isobtt_mix``     ``isobtt_cayley`` plus an orthogonal input mixer ``M in O(n_blk)``
                   acting on the block-slices; ``M (x) I_b`` is orthogonal, so the
                   *global* spectrum of W stays exactly fixed.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.workers.peft.base import PEFTAdapter

ISO_MODES = ("iso", "isobtt")
ISO_CAYLEY_MODES = ("iso_cayley", "isobtt_cayley", "isobtt_mix")

_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP = ("gate_proj", "up_proj", "down_proj")
_SKIP = ("lm_head", "embed_tokens")


def _closest_factor_pair(n: int):
    """(n_blocks, block_size) with n_blocks*block_size == n, block ~ sqrt(n).

    Identical to ``es_worker_extension._es_closest_factor_pair`` so the BP and ES
    runs factor every layer the same way.
    """
    root = int(n ** 0.5)
    for p in range(root, 0, -1):
        if n % p == 0:
            return p, n // p
    return 1, n


def _block_size(dim: int, requested: int) -> int:
    """Largest b <= requested dividing dim."""
    b = min(int(requested), int(dim))
    while b > 1 and dim % b:
        b -= 1
    return b


def _skew(w: torch.Tensor) -> torch.Tensor:
    return 0.5 * (w - w.transpose(-1, -2))


def _cayley(w: torch.Tensor) -> torch.Tensor:
    """Cay(skew(w)) -- exactly orthogonal for any w, and I at w=0.

    Always evaluated in fp32: the *solve* has to be accurate, and Cayley of a
    bf16-rounded skew is still an exactly orthogonal matrix (rounding changes
    which rotation you get, not whether it is one).
    """
    a = 0.5 * _skew(w.float())
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype).expand_as(a)
    return torch.linalg.solve(eye - a, eye + a)


def _blk(z: torch.Tensor, perm: torch.Tensor, inv: torch.Tensor, c: torch.Tensor,
         transpose: bool) -> torch.Tensor:
    """z @ P^T blkdiag(C) P  (or its transpose), over the last dim."""
    lead = z.shape[:-1]
    nb, b, _ = c.shape
    zb = z.index_select(-1, perm).view(*lead, nb, b)
    if transpose:
        out = torch.einsum("...jb,jcb->...jc", zb, c)
    else:
        out = torch.einsum("...jb,jbc->...jc", zb, c)
    return out.reshape(*lead, nb * b).index_select(-1, inv)


class IsoLinear(nn.Module):
    """W_eff = C_L W0 C_R^T, both factors block-diagonal Cayley rotations.

    W0 is kept as a *frozen nn.Parameter* (not a buffer) so FSDP shards it and so
    every tensor in the module carries the actor's dtype -- FSDP1 flattens each
    unit into one FlatParameter and rejects mixed dtypes.
    """

    def __init__(self, lin: nn.Linear, block: int, gen: torch.Generator):
        super().__init__()
        w = lin.weight.data
        out_f, in_f = w.shape
        bl, br = _block_size(out_f, block), _block_size(in_f, block)
        self.bl, self.br, self.out_f, self.in_f = bl, br, out_f, in_f
        self.weight = nn.Parameter(w, requires_grad=False)
        self.bias = (nn.Parameter(lin.bias.data, requires_grad=False)
                     if lin.bias is not None else None)
        for dim, tag in ((out_f, "l"), (in_f, "r")):
            p = torch.randperm(dim, generator=gen).to(w.device)
            inv = torch.empty_like(p)
            inv[p] = torch.arange(dim, device=p.device)
            self.register_buffer(f"perm_{tag}", p, persistent=True)
            self.register_buffer(f"inv_{tag}", inv, persistent=True)
        self.omega_l = nn.Parameter(torch.zeros(out_f // bl, bl, bl, dtype=w.dtype, device=w.device))
        self.omega_r = nn.Parameter(torch.zeros(in_f // br, br, br, dtype=w.dtype, device=w.device))

    def forward(self, x):
        # y = x C_R W0^T C_L^T  ==  x @ (C_L W0 C_R^T)^T
        u = _blk(x, self.perm_r, self.inv_r, _cayley(self.omega_r).to(x.dtype), False)
        v = F.linear(u, self.weight.to(x.dtype), None)
        y = _blk(v, self.perm_l, self.inv_l, _cayley(self.omega_l).to(x.dtype), True)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        """Dense W_eff = C_L W0 C_R^T in fp32 (cast by the caller).

        Must reproduce `forward` exactly, so mind the sides: `_blk(., transpose=True)`
        is right-multiplication by C^T, hence C_L W == (W^T C_L^T)^T.
        """
        w = _blk(self.weight.float(), self.perm_r, self.inv_r, _cayley(self.omega_r), True)
        w = _blk(w.t().contiguous(), self.perm_l, self.inv_l, _cayley(self.omega_l), True)
        return w.t().contiguous()

    def trainable_numel(self):
        stored = self.omega_l.numel() + self.omega_r.numel()
        eff = (self.omega_l.shape[0] * self.bl * (self.bl - 1)
               + self.omega_r.shape[0] * self.br * (self.br - 1)) // 2
        return stored, eff


class IsoBTTLinear(nn.Module):
    """Block-wise right rotation: W[:, blk_j] = W0[:, blk_j] C_j, C_j in O(b).

    This is the ES ``isobtt`` family written in its simplest equivalent form.  The
    ES worker builds it as ``A_j Cay(.) R_j`` from a per-block SVD; right-multiplying
    the *original* block by an orthogonal matrix spans the same set
    (``sigma(W0_j C_j) = sigma(W0_j)`` either way) while avoiding the SVD entirely --
    so there is no frozen ``A``/``R0`` to store and, unlike the SVD form, the
    identity init reproduces the pretrained weight **bit-exactly** instead of at the
    1.6e-3 bf16 reconstruction floor.

    ``mix=True`` additionally learns ``M in O(n_blk)`` acting on the block-slices,
    i.e. ``W_eff = W_btt (M (x) I_b)``.  ``M (x) I_b`` is orthogonal, so the *global*
    spectrum of W stays exactly fixed and the arm stays inside ``F(W0)``.
    """

    def __init__(self, lin: nn.Linear, mix: bool):
        super().__init__()
        w = lin.weight.data
        out_f, in_f = w.shape
        n_blk, b = _closest_factor_pair(in_f)
        self.n_blk, self.b, self.out_f, self.in_f = n_blk, b, out_f, in_f
        self.weight = nn.Parameter(w, requires_grad=False)
        self.bias = (nn.Parameter(lin.bias.data, requires_grad=False)
                     if lin.bias is not None else None)
        self.omega = nn.Parameter(torch.zeros(n_blk, b, b, dtype=w.dtype, device=w.device))
        self.omega_m = (nn.Parameter(torch.zeros(n_blk, n_blk, dtype=w.dtype, device=w.device))
                        if mix else None)

    def _rot(self, dtype):
        c = _cayley(self.omega).to(dtype)
        m = _cayley(self.omega_m.unsqueeze(0)).squeeze(0).to(dtype) if self.omega_m is not None else None
        return c, m

    def forward(self, x):
        lead = x.shape[:-1]
        xb = x.reshape(*lead, self.n_blk, self.b)
        c, m = self._rot(x.dtype)
        if m is not None:
            xb = torch.einsum("ij,...jb->...ib", m, xb)
        u = torch.einsum("...jb,jcb->...jc", xb, c).reshape(*lead, self.in_f)
        return F.linear(u, self.weight.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        c, m = self._rot(torch.float32)
        w0 = self.weight.float().reshape(self.out_f, self.n_blk, self.b)
        wb = torch.einsum("ojc,jcb->ojb", w0, c)
        if m is not None:
            wb = torch.einsum("ojb,jk->okb", wb, m)
        return wb.reshape(self.out_f, self.in_f).contiguous()

    def trainable_numel(self):
        stored = self.omega.numel() + (self.omega_m.numel() if self.omega_m is not None else 0)
        eff = self.n_blk * self.b * (self.b - 1) // 2
        if self.omega_m is not None:
            eff += self.n_blk * (self.n_blk - 1) // 2
        return stored, eff


class IsoAdapter(PEFTAdapter):
    mode = "iso"

    def __init__(self, peft_cfg, model_config=None, teacher_model_path: Optional[str] = None):
        super().__init__(peft_cfg, model_config=model_config)
        self.mode = peft_cfg.mode
        self._converted: list[str] = []

    def _targets(self):
        tm = self.peft_cfg.target_modules
        if isinstance(tm, str):
            if tm == "attn":
                return _ATTN
            if tm == "mlp":
                return _MLP
            return _ATTN + _MLP
        return tuple(tm)

    def apply(self, model, *, tokenizer, calib_loader_builder):
        cfg = self.peft_cfg.iso
        block = int(cfg.block_size)
        gen = torch.Generator().manual_seed(int(cfg.seed))
        targets = self._targets()

        for p in model.parameters():
            p.requires_grad_(False)

        replace = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or any(s in name for s in _SKIP):
                continue
            if not name.endswith(targets):
                continue
            replace.append((name, mod))

        stored = eff = 0
        for name, mod in replace:
            if self.mode == "iso_cayley":
                new = IsoLinear(mod, block, gen)
            else:
                new = IsoBTTLinear(mod, mix=(self.mode == "isobtt_mix"))
            new = new.to(mod.weight.device)
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], new)
            s, e = new.trainable_numel()
            stored += s
            eff += e
            self._converted.append(name)

        for n, p in model.named_parameters():
            p.requires_grad_(n.endswith(("omega", "omega_l", "omega_r", "omega_m")))

        # The embeddings are frozen, so the hidden states entering each checkpointed
        # decoder block carry no grad_fn and torch.utils.checkpoint returns a detached
        # output ("None of the inputs have requires_grad=True") -- the loss then has no
        # graph at all and backward() fails. Same fix LoRA and BlockTT apply.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        total = sum(p.numel() for p in model.parameters())
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[ISO] mode={self.mode} converted={len(self._converted)} linears | "
                  f"trainable stored={stored:,} ({100*stored/max(total,1):.3f}% of {total:,}) | "
                  f"effective manifold dim={eff:,} ({100*eff/max(total,1):.3f}%)", flush=True)
        return model

    def export_for_vllm(self, fsdp_module):
        """Dense weights for the rollout engine: materialise every ISO module.

        Two constraints the obvious implementation gets wrong:

        1. **Clone.** The caller runs this inside ``FSDP.summon_full_params(...)``, so
           ``param.detach()`` returns a *view* into the temporarily gathered flat
           parameter. FSDP frees that storage on context exit, and vLLM then reads
           tensors whose storage has been resized to 0 (``setStorage: ... out of
           bounds for storage of size 0``). Everything handed back must own its
           memory. (BlockTT never hit this: its production runs used FSDP2, where
           ``summon_full_params`` raises and the caller falls back to a direct call.)
        2. **bf16.** The actor module is fp32 (see MODEL_DTYPE), but vLLM stores bf16.
           Exporting fp32 would hold a second full-precision copy of the model at the
           worst possible moment; casting here is what vLLM would do anyway.
        """
        def _clean(n: str) -> str:
            return n.replace("_fsdp_wrapped_module.", "")

        def _emit(t):
            dt = torch.bfloat16 if t.dtype == torch.float32 else t.dtype
            return t.detach().to(dt).clone()

        out, prefixes = {}, []
        for name, mod in fsdp_module.named_modules():
            if not isinstance(mod, (IsoLinear, IsoBTTLinear)):
                continue
            prefixes.append(name + ".")
            out[_clean(f"{name}.weight")] = _emit(mod.materialize())
            if getattr(mod, "bias", None) is not None:
                out[_clean(f"{name}.bias")] = _emit(mod.bias)
        prefixes = tuple(prefixes)
        for name, param in fsdp_module.named_parameters():
            if prefixes and name.startswith(prefixes):
                continue
            out[_clean(name)] = _emit(param)
        return out

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        """Fold the rotations back into dense nn.Linear weights, then save."""
        os.makedirs(out_dir, exist_ok=True)
        for name, mod in list(fsdp_module.named_modules()):
            if not isinstance(mod, (IsoLinear, IsoBTTLinear)):
                continue
            w = mod.materialize()
            lin = nn.Linear(w.shape[1], w.shape[0], bias=getattr(mod, "bias", None) is not None,
                            device=w.device, dtype=mod.weight.dtype)
            lin.weight.data.copy_(w.to(mod.weight.dtype))
            if getattr(mod, "bias", None) is not None:
                lin.bias.data.copy_(mod.bias)
            parent = fsdp_module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else fsdp_module
            setattr(parent, name.rsplit(".", 1)[-1], lin)
        fsdp_module.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        return {
            "mode": self.mode,
            "target_modules": self.peft_cfg.target_modules,
            "iso": {"block_size": self.peft_cfg.iso.block_size, "seed": self.peft_cfg.iso.seed},
            "converted": len(self._converted),
        }


# ----------------------------------------------------------------------------------
# Paper-faithful ISO-Optimizer: modes `iso` / `isobtt`
# ----------------------------------------------------------------------------------

_FRAME_TRAINABLE = ("iso_du", "iso_dv")


@torch.no_grad()
def _polar(x: torch.Tensor, tol: float = 1e-13, max_iter: int = 20) -> torch.Tensor:
    """polar(X) = P Q^T for the thin SVD X = P S Q^T (paper Eq. 30), batched, fp64.

    Newton-Schulz ``X <- X (3I - X^T X) / 2`` keeps the singular vectors and drives
    every singular value to 1, so it converges to exactly this matrix whenever
    ``||X^T X - I|| < 1`` -- always true right after one small optimizer step.
    """
    x = x.double()
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    for _ in range(max_iter):
        g = x.mT @ x
        err = (g - eye).abs().amax()
        if err < tol:
            return x
        if err >= 1:
            break
        x = x @ (1.5 * eye - 0.5 * g)
    p, _, qh = torch.linalg.svd(x, full_matrices=False)
    return p @ qh


class IsoFrameLinear(nn.Module):
    """``W = W0 + (U S0 V^T - U0 S0 V0^T)`` per input block, U/V trained + retracted.

    ``n_blk == 1`` is the paper's ISO (one thin SVD of the whole matrix);
    ``n_blk > 1`` is ``isobtt`` (one thin SVD per input block of width ``b``).
    Factor tensors are batched over blocks: ``U0, dU: (n_blk, m, q)``,
    ``V0, dV: (n_blk, b, q)``, ``S0: (n_blk, q)`` with ``q = min(m, b)``.

    Every tensor is an ``nn.Parameter`` in the actor's dtype (frozen ones with
    ``requires_grad=False``) so FSDP shards all of them and gathers them together.
    """

    def __init__(self, lin: nn.Linear, n_blk: int):
        super().__init__()
        w = lin.weight.data
        out_f, in_f = w.shape
        assert in_f % n_blk == 0, (in_f, n_blk)
        b = in_f // n_blk
        self.n_blk, self.b, self.out_f, self.in_f = n_blk, b, out_f, in_f
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else w.device
        wb = w.to(dev, torch.float64).reshape(out_f, n_blk, b).permute(1, 0, 2)
        u, s, vh = torch.linalg.svd(wb, full_matrices=False)
        dt, wd = w.dtype, w.device

        def _p(t, grad=False):
            return nn.Parameter(t.to(wd, dt).contiguous(), requires_grad=grad)

        self.weight = nn.Parameter(w, requires_grad=False)
        self.bias = _p(lin.bias.data) if lin.bias is not None else None
        self.iso_u0 = _p(u)
        self.iso_s = _p(s)
        self.iso_v0 = _p(vh.mT)
        self.iso_du = _p(torch.zeros_like(u), grad=True)
        self.iso_dv = _p(torch.zeros_like(vh.mT), grad=True)

    def _delta(self, dtype) -> torch.Tensor:
        """U S0 V^T - U0 S0 V0^T  ==  dU S0 V^T + U0 S0 dV^T, as an (out, in) matrix."""
        s = self.iso_s.to(dtype).unsqueeze(-2)
        u0, du = self.iso_u0.to(dtype), self.iso_du.to(dtype)
        v0, dv = self.iso_v0.to(dtype), self.iso_dv.to(dtype)
        d = torch.bmm(du * s, (v0 + dv).mT) + torch.bmm(u0 * s, dv.mT)
        return d.permute(1, 0, 2).reshape(self.out_f, self.in_f)

    def forward(self, x):
        w = self.weight.to(x.dtype) + self._delta(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        """Dense W in fp32, computed in fp64 (cast by the caller)."""
        return (self.weight.double() + self._delta(torch.float64)).float()

    @torch.no_grad()
    def retract(self) -> None:
        """U <- polar(U0 + dU), V <- polar(V0 + dV) (paper Eq. 36); stored back as deltas."""
        for p0, dp in ((self.iso_u0, self.iso_du), (self.iso_v0, self.iso_dv)):
            x0 = p0.double()
            dp.copy_((_polar(x0 + dp.double()) - x0).to(dp.dtype))

    def trainable_numel(self):
        return self.iso_du.numel() + self.iso_dv.numel()


def _fsdp_units(root: nn.Module):
    """Group every submodule under the FSDP unit that owns its parameters.

    Returns ``[(unit, [(name, module), ...]), ...]``; ``unit`` is None when ``root`` is
    not FSDP-wrapped.  Lets callers gather one unit (one decoder layer) at a time:
    gathering the whole frame-parameterised model at once is ~3.5x the dense size.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # torch < 2.6
        FSDPModule = ()
    groups: dict = {}

    def walk(mod, name, unit):
        if FSDPModule and isinstance(mod, FSDPModule):
            raise NotImplementedError("peft.mode iso/isobtt supports FSDP1 (actor.strategy=fsdp) only")
        if isinstance(mod, FSDP):
            unit = mod
        groups.setdefault(unit, []).append((name, mod))
        for cn, c in mod.named_children():
            walk(c, f"{name}.{cn}" if name else cn, unit)

    walk(root, "", None)
    return list(groups.items())


def _summon(unit, writeback: bool):
    import contextlib
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if unit is None:
        return contextlib.nullcontext()
    return FSDP.summon_full_params(unit, recurse=False, writeback=writeback)


@torch.no_grad()
def retract_iso(module: nn.Module, optimizer=None) -> bool:
    """Polar-retract every ``IsoFrameLinear`` after an optimizer step. No-op otherwise.

    Called from ``dp_actor._optimizer_step``.  Gathers one FSDP unit at a time with
    writeback; every rank computes the (deterministic, cheap) retraction of the
    gathered unit and writes back its own shard.
    """
    groups = module.__dict__.get("_iso_frame_groups")
    if groups is None:
        groups = [(u, [m for _, m in mods if isinstance(m, IsoFrameLinear)])
                  for u, mods in _fsdp_units(module)]
        groups = [(u, ms) for u, ms in groups if ms]
        module.__dict__["_iso_frame_groups"] = groups
    if not groups:
        return False
    if optimizer is not None and any(g.get("weight_decay", 0) != 0 for g in optimizer.param_groups):
        # The paper trains with weight decay 0 (App. H).  With delta storage a nonzero
        # decay would shrink U - U0 instead of U, i.e. a different optimizer.
        raise ValueError("peft.mode iso/isobtt requires actor.optim.weight_decay=0")
    for unit, mods in groups:
        with _summon(unit, writeback=True):
            for m in mods:
                m.retract()
    return True


def _clean(n: str) -> str:
    return n.replace("_fsdp_wrapped_module.", "").replace("_fsdp_wrapped_module", "")


@torch.no_grad()
def _dense_state_dict(root: nn.Module, dtype=torch.bfloat16) -> dict:
    """HF-named dense weights, materialised one FSDP unit at a time."""
    out, seen = {}, set()
    for unit, mods in _fsdp_units(root):
        with _summon(unit, writeback=False):
            for name, mod in mods:
                if isinstance(mod, IsoFrameLinear):
                    out[_clean(f"{name}.weight")] = mod.materialize().to(dtype)
                    if mod.bias is not None:
                        out[_clean(f"{name}.bias")] = mod.bias.detach().to(dtype).clone()
                    continue
                for pn, p in mod._parameters.items():
                    if p is None or id(p) in seen or "flat_param" in pn:
                        continue
                    seen.add(id(p))
                    t = p.detach()
                    out[_clean(f"{name}.{pn}" if name else pn)] = (
                        t.to(dtype) if t.is_floating_point() else t).clone()
    return out


def _convert_frames(model: nn.Module, mode: str, targets) -> tuple[list[str], int]:
    replace = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and not any(s in n for s in _SKIP) and n.endswith(targets)]
    names, stored = [], 0
    for name, mod in replace:
        n_blk = 1 if mode == "iso" else _closest_factor_pair(mod.in_features)[0]
        new = IsoFrameLinear(mod, n_blk)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], new)
        names.append(name)
        stored += new.trainable_numel()
    return names, stored


def _set_frame_trainable(model: nn.Module, train_others: bool) -> None:
    frozen = set()
    for m in model.modules():
        if isinstance(m, IsoFrameLinear):
            frozen |= {id(m.weight), id(m.iso_u0), id(m.iso_v0), id(m.iso_s)}
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(_FRAME_TRAINABLE) or (train_others and id(p) not in frozen))


class IsoFrameAdapter(IsoAdapter):
    """Modes ``iso`` / ``isobtt``: the paper's ISO-Optimizer (see module docstring).

    Everything that is not a converted linear (embeddings / tied LM head, norms,
    biases) is trained by the plain base optimizer when ``peft.iso.train_others``
    (default), so ISO-AdamW vs AdamW differ only in how the projection matrices are
    parameterised.  The paper does not say how it treats these tensors.
    """

    # The worker must hand export_for_vllm the FSDP root, not wrap it in a whole-model
    # summon_full_params: we gather unit by unit.
    export_needs_fsdp_root = True

    def apply(self, model, *, tokenizer, calib_loader_builder):
        dtypes = {p.dtype for p in model.parameters()}
        if dtypes != {torch.float32}:
            # The trainable deltas are their own optimizer master copy; in bf16 a
            # lr~1e-6 step on U would round away.
            raise ValueError(f"peft.mode={self.mode} needs fp32 actor params (MODEL_DTYPE=fp32); got {dtypes}")
        self._converted, stored = _convert_frames(model, self.mode, self._targets())
        _set_frame_trainable(model, bool(self.peft_cfg.iso.train_others))
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[ISO] mode={self.mode} converted={len(self._converted)} linears | "
                  f"frame deltas={stored:,} | trainable total={trainable:,} | "
                  f"stored total={total:,} | train_others={self.peft_cfg.iso.train_others}", flush=True)
        return model

    def export_for_vllm(self, fsdp_module):
        return _dense_state_dict(fsdp_module)

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        """Dense bf16 HF checkpoint (rank 0 writes; every rank joins the gathers)."""
        import torch.distributed as dist
        sd = _dense_state_dict(fsdp_module)
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        from huggingface_hub import save_torch_state_dict
        os.makedirs(out_dir, exist_ok=True)
        save_torch_state_dict({k: v.cpu() for k, v in sd.items()}, out_dir)
        inner = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
        inner.config.save_pretrained(out_dir)
        if getattr(inner, "generation_config", None) is not None:
            inner.generation_config.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        return {
            "mode": self.mode,
            "target_modules": self.peft_cfg.target_modules,
            "iso": {"train_others": bool(self.peft_cfg.iso.train_others)},
            "converted": len(self._converted),
        }

    @classmethod
    def rebuild_from_meta(cls, model, meta):
        """Resume: rebuild the same module topology; the checkpoint load fills the values."""
        tm = meta.get("target_modules", "all")
        targets = (_ATTN if tm == "attn" else _MLP if tm == "mlp"
                   else _ATTN + _MLP if isinstance(tm, str) else tuple(tm))
        _convert_frames(model, meta["mode"], targets)
        _set_frame_trainable(model, bool(meta.get("iso", {}).get("train_others", True)))
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model
