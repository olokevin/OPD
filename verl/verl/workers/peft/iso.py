"""ISO adapters: fixed-spectrum BP training (arXiv:2607.19331).

The ES side of this thread lives in
``verl/workers/rollout/vllm_rollout/es_worker_extension.py`` and is documented in
``docs/results/ES/es_results.md`` §10.  This module is the **first-order (BP)**
counterpart, exposing the *same* three parameterisations to verl's FSDP actor so
that ES-vs-BP is a controlled comparison.

Recap of the geometry.  ISO constrains every 2-D weight to the fixed-spectrum
family ``F(W0) = {U S0 V^T}``.  Because ``O(m)`` acts transitively on the Stiefel
manifold, that family is exactly the **bi-orthogonal orbit**

    F(W0) = { C_L W0 C_R^T : C_L in O(m), C_R in O(n) },

so feasibility needs no SVD and no retraction: any bi-orthogonal transform of W is
in the family and preserves sigma(W) *exactly*.  ISO's own optimizer instead steps
``U, V`` freely and projects back with an fp64 polar retraction every step, which
is affordable for one gradient step but not for anything else.

For BP we get the constraint for free by **parameterising the orthogonal factors
themselves**: every ``C`` is ``Cay(Omega)`` for a trainable skew ``Omega``, where

    Cay(X) := (I - X/2)^-1 (I + X/2)

is exactly orthogonal for any skew X.  Consequences:

  * ``Omega = 0`` at init  =>  ``C = I``  =>  the step-0 forward is the pretrained
    model bit-for-bit (mode ``iso``) or at the bf16 BTT-reconstruction floor
    (``isobtt*``), so every arm starts from the same place.
  * the constraint holds for *any* value the optimizer produces -- plain AdamW and
    plain FSDP work unchanged, with no Riemannian optimizer, no retraction step and
    no projection.  There is nothing to drift off, so the ``_iso_recondition``
    machinery the ES trainer needs has no analogue here.
  * ``Omega`` is stored as a full square matrix and skew-symmetrised in the
    forward; the symmetric half sits in the kernel of the map and receives exactly
    zero gradient, so the *effective* trainable dimension is ``b(b-1)/2`` per
    block, half the stored count.  Both are reported.

Modes
-----
``iso``         ``W_eff = C_L W0 C_R^T`` with ``C`` block-diagonal (block ``b``) in a
                fixed random basis.  W0 stays frozen and is never materialised
                during training: the forward is ``x -> x C_R -> W0 -> C_L^T``, two
                cheap block-diagonal matmuls around the untouched base linear.
``isobtt``      block-wise SVD ``W[:, blk_j] = A_j R_j`` with ``A_j = U_j diag(S_j)``
                frozen and ``R_j = Cay(Omega_j) R0_j`` in ``O(b)`` trained.  Each
                *block's* spectrum is fixed exactly.
``isobtt_mix``  ``isobtt`` plus an orthogonal input mixer ``M in O(n_blk)`` applied to
                the block-slices before the contraction, relaxing the block-locality
                of Remark 5.2 (cf. the free-``M`` ablation in
                lora-without-regret ``docs/exp_results/lift_commonsense.md``).  As a
                full-input operator ``M (x) I_b`` is orthogonal, so the *global*
                spectrum of W stays exactly fixed and the arm remains inside
                ``F(W0)`` -- unlike a free ``M``, which would leave the family.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.workers.peft.base import PEFTAdapter

ISO_MODES = ("iso", "isobtt", "isobtt_mix")

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
            if self.mode == "iso":
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
