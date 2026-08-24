"""Numerical checks for the ISO **BP** (first-order) parameterisations.

Companion to scripts/es/test_iso_es.py (which covers the ES side). Verifies that
the autograd modules in verl/workers/peft/iso.py (a) start exactly at the
pretrained model, (b) keep the spectrum fixed for *any* optimizer step, and
(c) export a dense weight that matches their own forward.

  python3 scripts/es/test_iso_bp.py [--device cuda]
"""
import argparse
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/yequan/Project/compression/OPD/verl")
from verl.workers.config.peft import PEFTConfig                      # noqa: E402
from verl.workers.peft.iso import (                                  # noqa: E402
    IsoAdapter, IsoBTTLinear, IsoLinear, _cayley, _closest_factor_pair, _skew,
)


def rel(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    return bool(ok)


def tiny_model(dev, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=2, vocab_size=1024,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).to(dev, dtype)
    # random init is near-zero for some layers; make the weights generic
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 2:
                p.normal_(0, 0.02)
    return m.eval()


def spectra(mod):
    """Singular values of the module's effective weight, in fp64."""
    w = mod.materialize().double()
    if isinstance(mod, IsoBTTLinear) and mod.omega_m is None:
        n, b = mod.n_blk, mod.b
        return torch.linalg.svdvals(w.reshape(w.shape[0], n, b).permute(1, 0, 2)).flatten()
    return torch.linalg.svdvals(w)


def unit_checks(dev):
    ok = True
    torch.manual_seed(0)
    w = torch.randn(6, 64, 64, device=dev)
    c0 = _cayley(torch.zeros(6, 64, 64, device=dev))
    eye = torch.eye(64, device=dev).expand(6, 64, 64)
    ok &= check("Cay(0) == I", torch.allclose(c0, eye, atol=1e-6))
    ok &= check("skew(w) is skew", (_skew(w) + _skew(w).transpose(-1, -2)).abs().max().item() < 1e-6)
    for s in (1e-4, 1e-2, 1.0, 10.0):
        c = _cayley(w * s)
        e = (torch.bmm(c, c.transpose(1, 2)) - eye).abs().max().item()
        ok &= check(f"Cay orthogonal at scale {s}", e < 1e-4, f"max|CC^T-I|={e:.2e}")
    # gradient flows, and only through the skew part.  NB: reduce with a random
    # weighting, not .sum() -- d(sum Cay)/dw is identically zero at any w because
    # summing over a skew generator cancels, which would look like a dead gradient.
    p = torch.zeros(2, 8, 8, device=dev, requires_grad=True)
    torch.manual_seed(1)
    (_cayley(p) * torch.randn(2, 8, 8, device=dev)).sum().backward()
    g = p.grad
    ok &= check("d/dw Cay is nonzero", g.abs().max().item() > 0, f"max|g|={g.abs().max():.2e}")
    ok &= check("gradient is skew (symmetric half is in the kernel)",
                (g + g.transpose(-1, -2)).abs().max().item() < 1e-6)
    return ok


def run_mode(mode, dev, dtype):
    ok = True
    model = tiny_model(dev, dtype)
    x = torch.randint(0, 1024, (2, 16), device=dev)
    with torch.no_grad():
        ref = model(x).logits.float().clone()
    base_w = {n: m.weight.data.clone() for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    cfg = PEFTConfig(mode=mode, target_modules="all")
    cfg.iso.block_size = 64
    adapter = IsoAdapter(cfg, model_config=None)
    model = adapter.apply(model, tokenizer=None, calib_loader_builder=None)
    iso_mods = {n: m for n, m in model.named_modules() if isinstance(m, (IsoLinear, IsoBTTLinear))}
    ok &= check("modules converted", len(iso_mods) == 14, f"n={len(iso_mods)}")

    # (a) identity init reproduces the pretrained model.  `iso` stores W0 verbatim
    # so this is bit-exact; `isobtt*` rebuild W from a per-block SVD whose frozen A
    # is stored in the model dtype, so in bf16 they sit on the 1.6e-3 ULP floor
    # (measured 1.66e-3/layer here, 1.56e-3 on real Qwen weights -- ES thread §6),
    # which depth amplifies in the logits.  Assert on the per-layer weight error,
    # which is the quantity that is actually under our control.
    with torch.no_grad():
        got = model(x).logits.float()
    werr = max(rel(m.materialize().float(), base_w[n].float()) for n, m in iso_mods.items())
    ok &= check("identity init reproduces base weights", werr == 0.0, f"max rel={werr:.2e}")
    ltol = 1e-6
    ok &= check("identity init reproduces base logits", rel(got, ref) <= ltol,
                f"rel={rel(got, ref):.2e} (tol {ltol:.0e})")

    # (b) materialize() agrees with the module's own forward -- checked at NON-ZERO
    # omega.  At omega=0 every rotation is the identity, so a left/right or
    # transpose mix-up in materialize() is invisible; that is exactly the bug this
    # check exists for (the vLLM rollout weight would silently disagree with the
    # trained policy).
    torch.manual_seed(7)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.normal_(0, 0.3)
    worst = 0.0
    for nm, mod in iso_mods.items():
        z = torch.randn(4, mod.in_f, device=dev, dtype=dtype)
        with torch.no_grad():
            fwd = mod(z)
            mat = torch.nn.functional.linear(
                z, mod.materialize().to(dtype),
                None if mod.bias is None else mod.bias.to(dtype))
        worst = max(worst, rel(fwd, mat))
    ok &= check("materialize() == forward() at random omega",
                worst < (1e-5 if dtype == torch.float32 else 2e-2), f"max rel={worst:.2e}")
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()

    # (c) only the omegas train
    tr = [n for n, p in model.named_parameters() if p.requires_grad]
    ok &= check("only omega* are trainable",
                all(n.endswith(("omega", "omega_l", "omega_r", "omega_m")) for n in tr) and tr,
                f"n_trainable_tensors={len(tr)}")

    # (d) THE claim: after a real optimizer step the spectrum is unchanged
    s0 = {n: spectra(m) for n, m in iso_mods.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-2)
    for _ in range(5):
        opt.zero_grad()
        model(x, labels=x).loss.backward()
        gn = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        opt.step()
    ok &= check("gradients reach the omegas", gn > 0, f"max|grad|={gn:.2e}")
    worst_d = worst_move = 0.0
    for n, m in iso_mods.items():
        s = spectra(m)
        worst_d = max(worst_d, ((s - s0[n]).norm() / s0[n].norm()).item())
        worst_move = max(worst_move, rel(m.materialize().float(), base_w[n].float()))
    ok &= check("weights actually moved", worst_move > 1e-3, f"max ||dW||/||W||={worst_move:.2e}")
    ok &= check("spectrum fixed after 5 AdamW steps", worst_d < 1e-5, f"max ||dsigma||/||sigma||={worst_d:.2e}")

    if mode == "isobtt_mix":
        m = next(iter(iso_mods.values()))
        mm = _cayley(m.omega_m.unsqueeze(0)).squeeze(0)
        e = (mm @ mm.T - torch.eye(mm.shape[0], device=dev)).abs().max().item()
        ok &= check("input mixer M is orthogonal", e < 1e-5, f"max|MM^T-I|={e:.2e}")
        ok &= check("M moved away from identity",
                    (mm - torch.eye(mm.shape[0], device=dev)).abs().max().item() > 1e-4)

    stored = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable stored {stored:,} / {total:,} = {100*stored/total:.2f}%")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print("=== unit checks ===")
    allok = unit_checks(args.device)
    for mode in ("iso", "isobtt", "isobtt_mix"):
        for dtype in (torch.float32, torch.bfloat16):
            print(f"\n=== mode {mode} ({dtype}) ===")
            allok &= run_mode(mode, args.device, dtype)
    print("\n" + ("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
