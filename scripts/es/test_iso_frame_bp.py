"""Checks for the paper-faithful ISO-Optimizer BP modes (peft.mode iso / isobtt).

Single process (default):
  python3 scripts/es/test_iso_frame_bp.py
    * step 0 is the base model bit-for-bit
    * frame gradients are the paper's Eq. 34 (G_U = G_W V S0, G_V = G_W^T U S0)
    * after AdamW + retraction: U, V orthonormal and sigma(W) == S0 (per block for isobtt)
    * delta storage == stepping U, V themselves (fp64 reference, SVD polar), 5 steps
    * weight_decay != 0 is refused

FSDP1 (2 ranks, use_orig_params=True, bf16 mixed precision, one unit per layer):
  torchrun --nproc_per_node 2 scripts/es/test_iso_frame_bp.py --fsdp
    * unit-by-unit export == the unwrapped model's dense weights (fp32 and bf16 MP)
    * fp32: FSDP step + retract_iso == the same step in a single process
    * bf16 MP: frames stay orthonormal and sigma(W) == S0 after training steps
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/yequan/Project/compression/OPD/verl")
from verl.workers.config.peft import PEFTConfig                       # noqa: E402
from verl.workers.peft.iso import (                                   # noqa: E402
    IsoFrameAdapter, IsoFrameLinear, _dense_state_dict, _fsdp_units, _polar, _summon, retract_iso,
)

OK = True


def check(tag, ok, detail=""):
    global OK
    OK &= bool(ok)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}", flush=True)


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      tie_word_embeddings=True, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).float()


def adapt(model, mode):
    cfg = PEFTConfig(mode=mode)
    return IsoFrameAdapter(cfg).apply(model, tokenizer=None, calib_loader_builder=lambda: None)


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (2, 24), generator=g)


def loss_fn(model, ids):
    return model(input_ids=ids, labels=ids).loss


def single_process(mode, dev):
    print(f"\n=== {mode} ===")
    base = tiny_model().to(dev)
    ids = batch().to(dev)
    with torch.no_grad():
        ref_logits = base(input_ids=ids).logits
    model = adapt(copy.deepcopy(base), mode).to(dev)
    with torch.no_grad():
        out = model(input_ids=ids).logits
    check("step 0 == base model bit-for-bit", torch.equal(out, ref_logits))

    # Eq. 34: compare with the dense model's weight gradient.
    base.zero_grad()
    loss_fn(base, ids).backward()
    model.zero_grad()
    loss_fn(model, ids).backward()
    worst = 0.0
    for name, m in model.named_modules():
        if not isinstance(m, IsoFrameLinear):
            continue
        gw = base.get_submodule(name).weight.grad.double()
        gwb = gw.reshape(m.out_f, m.n_blk, m.b).permute(1, 0, 2)          # (nb, m, b)
        s = m.iso_s.double().unsqueeze(-2)
        gu = torch.bmm(gwb, m.iso_v0.double()) * s
        gv = torch.bmm(gwb.mT, m.iso_u0.double()) * s
        for a, b in ((m.iso_du.grad.double(), gu), (m.iso_dv.grad.double(), gv)):
            worst = max(worst, ((a - b).norm() / b.norm().clamp_min(1e-30)).item())
    check("frame grads == Eq. 34", worst < 1e-4, f"(max rel err {worst:.1e})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    for _ in range(3):
        opt.zero_grad()
        loss_fn(model, ids).backward()
        opt.step()
        retract_iso(model, opt)
    orth, spec, moved = 0.0, 0.0, 0.0
    for name, m in model.named_modules():
        if not isinstance(m, IsoFrameLinear):
            continue
        for p0, dp in ((m.iso_u0, m.iso_du), (m.iso_v0, m.iso_dv)):
            x = p0.double() + dp.double()
            eye = torch.eye(x.shape[-1], device=dev, dtype=x.dtype)
            orth = max(orth, (x.mT @ x - eye).abs().max().item())
        w = m.materialize().double()
        sv = torch.linalg.svdvals(w.reshape(m.out_f, m.n_blk, m.b).permute(1, 0, 2))
        spec = max(spec, ((sv - m.iso_s.double()).abs().max() / m.iso_s.double().max()).item())
        moved = max(moved, ((w - m.weight.double()).norm() / m.weight.double().norm()).item())
    check("U, V orthonormal after retraction", orth < 1e-6, f"(max |X^T X - I| {orth:.1e})")
    check("sigma(W) == S0 after 3 steps", spec < 1e-6, f"(max rel {spec:.1e}, weights moved {moved:.1e})")
    check("weights actually moved", moved > 1e-4)

    bad = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    try:
        retract_iso(model, bad)
        check("weight_decay != 0 refused", False)
    except ValueError:
        check("weight_decay != 0 refused", True)


def delta_equals_direct(mode, dev, steps=5, lr=1e-3):
    """Delta storage vs the paper's literal form: AdamW on U, V + fp64 SVD polar."""
    torch.manual_seed(3)
    m, n = 48, 40
    lin = nn.Linear(n, m, bias=False).to(dev).float()
    x = torch.randn(64, n, device=dev)
    tgt = torch.randn(64, m, device=dev)
    mod = IsoFrameLinear(lin, 1 if mode == "iso" else 4).to(dev)
    nb, b = mod.n_blk, mod.b
    u = nn.Parameter(mod.iso_u0.detach().double().clone())
    v = nn.Parameter(mod.iso_v0.detach().double().clone())
    s = mod.iso_s.detach().double()
    opt_ref = torch.optim.AdamW([u, v], lr=lr, weight_decay=0)
    opt = torch.optim.AdamW([mod.iso_du, mod.iso_dv], lr=lr, weight_decay=0)
    w0 = mod.weight.detach().double()
    # The fp32 rounding of U0/V0 makes U0 S V0^T differ from W0 by ~1e-7; the
    # delta form keeps that constant residual, so compare against W0 + (W - W_init).
    w_init = torch.bmm(u.detach() * s.unsqueeze(-2), v.detach().mT).permute(1, 0, 2).reshape(m, n)
    for _ in range(steps):
        opt_ref.zero_grad()
        w = torch.bmm(u * s.unsqueeze(-2), v.mT).permute(1, 0, 2).reshape(m, n)
        ((x.double() @ w.T - tgt.double()) ** 2).mean().backward()
        opt_ref.step()
        with torch.no_grad():
            for p in (u, v):
                pp, _, qh = torch.linalg.svd(p, full_matrices=False)
                p.copy_(pp @ qh)
        opt.zero_grad()
        ((mod(x) - tgt) ** 2).mean().backward()
        opt.step()
        mod.retract()
    with torch.no_grad():
        w_ref = w0 + torch.bmm(u * s.unsqueeze(-2), v.mT).permute(1, 0, 2).reshape(m, n) - w_init
        err = ((mod.materialize().double() - w_ref).norm() / (w_ref - w0).norm()).item()
    check(f"{mode}: delta storage == literal AdamW-on-(U,V) + SVD polar, {steps} steps", err < 1e-3,
          f"(rel err of the update {err:.1e})")


def polar_check(dev):
    x = torch.linalg.qr(torch.randn(3, 200, 50, device=dev, dtype=torch.float64))[0]
    x = x + 1e-3 * torch.randn_like(x)
    p, _, qh = torch.linalg.svd(x, full_matrices=False)
    err = (_polar(x) - p @ qh).abs().max().item()
    check("Newton-Schulz polar == SVD polar", err < 1e-12, f"(max diff {err:.1e})")


def fsdp_check(mode, bf16):
    """bf16=False: exact comparison with a single-process run (fp32 everywhere).
    bf16=True: the production setting (bf16 mixed precision) -- step-0 export is exact
    and after training steps the frames are still orthonormal with sigma(W) == S0."""
    import torch.distributed as dist
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    rank = dist.get_rank()
    dev = torch.device("cuda", torch.cuda.current_device())
    tag = "bf16 mixed precision" if bf16 else "fp32"
    if rank == 0:
        print(f"\n=== FSDP1 x{dist.get_world_size()}: {mode}, {tag} ===", flush=True)
    model = adapt(tiny_model().to(dev), mode)
    ref = copy.deepcopy(model)
    mp = (MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
          if bf16 else None)
    wrapped = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                                   transformer_layer_cls={Qwen3DecoderLayer}),
                   device_id=dev, use_orig_params=True, sync_module_states=True, mixed_precision=mp)
    init = _dense_state_dict(ref, dtype=torch.float32)
    got = _dense_state_dict(wrapped, dtype=torch.float32)
    err = max(((got[k] - init[k]).norm() / init[k].norm()).item() for k in init)
    check("unit-by-unit export == unwrapped dense weights", set(got) == set(init) and err == 0,
          f"(max rel err {err:.1e})")

    ids = batch().to(dev)
    opt = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    opt_ref = torch.optim.AdamW([p for p in ref.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
    for _ in range(2):
        opt.zero_grad()
        loss_fn(wrapped, ids).backward()
        opt.step()
        retract_iso(wrapped, opt)
        opt_ref.zero_grad()
        loss_fn(ref, ids).backward()
        opt_ref.step()
        retract_iso(ref, opt_ref)
    got = _dense_state_dict(wrapped, dtype=torch.float32)
    if not bf16:
        want = _dense_state_dict(ref, dtype=torch.float32)
        err = max(((got[k] - want[k]).norm() / (want[k] - init[k]).norm().clamp_min(1e-30)).item()
                  for k in want if (want[k] - init[k]).norm() > 0)
        check("FSDP step + retract_iso == single-process step", err < 1e-5,
              f"(max rel err of the update {err:.1e})")
        return
    orth, spec = 0.0, 0.0
    for unit, mods in _fsdp_units(wrapped):
        with _summon(unit, writeback=False):
            for _, m in mods:
                if not isinstance(m, IsoFrameLinear):
                    continue
                for p0, dp in ((m.iso_u0, m.iso_du), (m.iso_v0, m.iso_dv)):
                    x = p0.double() + dp.double()
                    eye = torch.eye(x.shape[-1], device=dev, dtype=x.dtype)
                    orth = max(orth, (x.mT @ x - eye).abs().max().item())
                sv = torch.linalg.svdvals(m.materialize().double().reshape(m.out_f, m.n_blk, m.b).permute(1, 0, 2))
                spec = max(spec, ((sv - m.iso_s.double()).abs().max() / m.iso_s.double().max()).item())
    moved = max(((got[k] - init[k]).norm() / init[k].norm()).item() for k in init)
    check("frames orthonormal after FSDP steps", orth < 1e-6, f"(max |X^T X - I| {orth:.1e})")
    check("sigma(W) == S0 after FSDP steps", spec < 1e-6, f"(max rel {spec:.1e}, weights moved {moved:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp", action="store_true")
    args = ap.parse_args()
    if args.fsdp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        for mode in ("iso", "isobtt"):
            fsdp_check(mode, bf16=False)
            fsdp_check(mode, bf16=True)
        dist.destroy_process_group()
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        polar_check(dev)
        for mode in ("iso", "isobtt"):
            single_process(mode, dev)
            delta_equals_direct(mode, dev)
    if int(os.environ.get("RANK", "0")) == 0:
        print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"), flush=True)
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
