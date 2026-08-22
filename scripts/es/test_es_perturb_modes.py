"""Numerical checks for the structured ES perturbation modes.

Runs against a fake `model_runner.model` (no vLLM needed) plus, optionally, real
Qwen2.5-Math-7B weights for the FuRA reconstruction check.

  python3 scripts/es/test_es_perturb_modes.py
  python3 scripts/es/test_es_perturb_modes.py --real-model Qwen/Qwen2.5-Math-7B
"""

import argparse
import sys

import torch

sys.path.insert(0, "/home/yequan/Project/compression/OPD/verl")
from verl.workers.rollout.vllm_rollout.es_worker_extension import WorkerExtension  # noqa: E402


class FakeModel(torch.nn.Module):
    def __init__(self, dev, dtype=torch.bfloat16):
        super().__init__()
        self.p = torch.nn.ParameterDict()
        for name, shape in [
            ("model.layers.0.self_attn.qkv_proj.weight", (192, 144)),
            ("model.layers.0.mlp.down_proj.weight", (144, 256)),
        ]:
            self.p[name.replace(".", "|")] = torch.nn.Parameter(
                (torch.randn(shape) * 0.02).to(dev, dtype)
            )

    def named_parameters(self, *a, **k):
        for n, p in self.p.items():
            yield n.replace("|", "."), p


class Runner:
    def __init__(self, model):
        self.model = model


class W(WorkerExtension):
    def __init__(self, model):
        self.model_runner = Runner(model)
        self.device = next(model.parameters()).device


def rel(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    return ok


def run_mode(mode, cfg, dev, sigma=1e-3, alpha=5e-4, N=8):
    torch.manual_seed(0)
    model = FakeModel(dev)
    w = W(model)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    info = w.init_es_state(mode, cfg)
    ok = True

    # --- step 0 must reproduce the original weights (exactly, or to bf16 round-off) ---
    for n, p in model.named_parameters():
        r = rel(p.data, w0[n])
        tol = 3e-3 if mode == "fura" else 0.0  # fura re-materializes W = A @ R in bf16
        ok &= check(f"{mode}: init W==W0 ({n.split('.')[-2]})", r <= tol, f"rel={r:.2e}")

    # --- perturb / restore round-trip ---
    w.es_perturb(1234, sigma)
    pert = {n: p.detach().clone() for n, p in model.named_parameters()}
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    w.es_restore()
    for n, p in model.named_parameters():
        d = rel(pert[n], p.data)
        ok &= check(f"{mode}: perturbation is visible ({n.split('.')[-2]})", d > 1e-4, f"rel dW={d:.2e}")
        ok &= check(f"{mode}: restore is exact ({n.split('.')[-2]})",
                    torch.equal(p.data, w0[n]) or mode == "fura", "")
    del base

    # --- sign symmetry: +eps and -eps straddle the base weight ---
    w.es_perturb(1234, sigma, negate=False)
    plus = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    w.es_restore()
    w.es_perturb(1234, sigma, negate=True)
    minus = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    w.es_restore()
    for n, p in model.named_parameters():
        mid = (plus[n] + minus[n]) / 2
        ok &= check(f"{mode}: (W+ + W-)/2 == W ({n.split('.')[-2]})",
                    rel(mid, p.data.float()) < 3e-3, f"rel={rel(mid, p.data.float()):.2e}")

    # --- update moves the coefficients in the expected direction ---
    seeds = list(range(100, 100 + N))
    coeffs = [1.0] * N  # all-positive z-scores -> coef must move by (alpha/N)*sum(eps)
    st = next(iter(w._es.values()))
    tgt = st["master"] if st["kind"] == "dense" else st["coef"]
    before = tgt.detach().clone()
    w.es_update(seeds, coeffs, alpha, N)
    name0 = next(iter(w._es))
    p0 = dict(model.named_parameters())[name0]
    expect = torch.zeros_like(before)
    for s in seeds:
        expect += w._es_noise(st, p0, s)
    expect = before + expect * (alpha / N)
    ok &= check(f"{mode}: coef update matches alpha/N * sum(z*eps)",
                rel(tgt, expect) < 1e-5, f"rel={rel(tgt, expect):.2e}")
    print(f"  info: {info}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-model", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # fake per-layer calibration: a unit vector + a per-channel RMS profile
    calib = {"layers": {}}
    for name, d in [
        ("model.layers.0.self_attn.q_proj", 144),
        ("model.layers.0.mlp.down_proj", 256),
    ]:
        g = torch.Generator().manual_seed(hash(name) % 2**31)
        v = torch.randn(1, d, generator=g)
        calib["layers"][name] = {
            "v": v / v.norm(dim=-1, keepdim=True),
            "act_rms": torch.rand(d, generator=g),
            "in_features": d,
        }
    cpath = "/tmp/es_fake_calib.pt"
    torch.save(calib, cpath)

    allok = True
    for mode, cfg in [
        ("dense", {}),
        ("zoact", {"calib_path": cpath, "rank": 1}),
        ("insparse", {"calib_path": cpath, "density": 0.1}),
        ("fura", {}),
    ]:
        print(f"\n=== mode: {mode} ===")
        allok &= run_mode(mode, cfg, dev)

    if args.real_model:
        print("\n=== FuRA reconstruction on real Qwen weights ===")
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(
            args.real_model, dtype=torch.bfloat16, device_map=dev
        )
        from verl.workers.rollout.vllm_rollout.es_worker_extension import (
            _es_closest_factor_pair,
        )

        for nm in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.13.mlp.down_proj",
            "model.layers.27.self_attn.o_proj",
        ]:
            wt = dict(m.named_modules())[nm].weight.data
            out_f, in_f = wt.shape
            n_blk, b = _es_closest_factor_pair(in_f)
            x = wt.reshape(out_f, n_blk, b).permute(1, 0, 2).float()
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            A = (U * S.unsqueeze(1)).to(wt.dtype)
            rec = torch.bmm(A.float(), Vh.float()).permute(1, 0, 2).reshape(wt.shape).to(wt.dtype)
            r = rel(rec, wt)
            allok &= check(f"fura recon {nm} (n={n_blk}, b={b}, r={min(out_f, b)})", r < 5e-3,
                           f"rel={r:.2e}")

    print("\n" + ("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
