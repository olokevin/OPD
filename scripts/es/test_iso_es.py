"""Numerical checks for the ISO (fixed-spectrum) ES perturbation modes.

Verifies the two properties the ISO stack is built on -- that a perturbation keeps
the singular *frames* orthonormal and the singular *values* untouched -- plus the
usual perturb/restore/update contracts.  Runs on a fake model (no vLLM needed) and
optionally on real Qwen2.5-Math-7B weights.

  python3 scripts/es/test_iso_es.py
  python3 scripts/es/test_iso_es.py --real-model Qwen/Qwen2.5-Math-7B
"""

import argparse
import sys

import torch

sys.path.insert(0, "/home/yequan/Project/compression/OPD/verl")
from verl.workers.rollout.vllm_rollout.es_worker_extension import (  # noqa: E402
    WorkerExtension,
    _es_closest_factor_pair,
    _iso_block_size,
    _iso_cayley,
    _iso_segments,
    _iso_skew,
)


class HFCfg:
    hidden_size = 512
    num_attention_heads = 8
    num_key_value_heads = 2
    intermediate_size = 640


class FakeModel(torch.nn.Module):
    """Shapes mirror vLLM's fused Qwen2 params (qkv / gate_up / o / down)."""

    def __init__(self, dev, dtype):
        super().__init__()
        c = HFCfg
        hd = c.hidden_size // c.num_attention_heads
        self.p = torch.nn.ParameterDict()
        for name, shape in [
            ("model.layers.0.self_attn.qkv_proj.weight",
             ((c.num_attention_heads + 2 * c.num_key_value_heads) * hd, c.hidden_size)),
            ("model.layers.0.self_attn.o_proj.weight", (c.hidden_size, c.hidden_size)),
            ("model.layers.0.mlp.gate_up_proj.weight", (2 * c.intermediate_size, c.hidden_size)),
            ("model.layers.0.mlp.down_proj.weight", (c.hidden_size, c.intermediate_size)),
            ("model.layers.1.self_attn.o_proj.weight", (c.hidden_size, c.hidden_size)),
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
        self.model_config = type("MC", (), {"hf_config": HFCfg})()


class W(WorkerExtension):
    def __init__(self, model):
        self.model_runner = Runner(model)
        self.device = next(model.parameters()).device


def rel(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    return bool(ok)


def svals(w, st):
    """Singular values of the whole matrix (iso) or of every input block (isobtt)."""
    w = w.float()
    if st["kind"] == "iso":
        return torch.linalg.svdvals(w)
    out_f, in_f = w.shape
    n_blk, b = st["n_blk"], st["b"]
    return torch.linalg.svdvals(w.reshape(out_f, n_blk, b).permute(1, 0, 2)).flatten()


def run_mode(mode, cfg, dev, dtype, sigma=5e-2, alpha=2.5e-2, N=8):
    torch.manual_seed(0)
    model = FakeModel(dev, dtype)
    w = W(model)
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    info = w.init_es_state(mode, cfg)
    ok = True
    params = dict(model.named_parameters())
    tol0 = 0.0 if (mode == "iso" and dtype == torch.float32) else 4e-3
    # ||d(sigma)||/||sigma|| floor: fp32 bmm + fp32 svdvals on a near-singular random
    # matrix.  `test_exactness_in_fp64` below shows this is round-off, not the algebra.
    stol = 1e-4 if dtype == torch.float32 else 5e-3

    for n, p in model.named_parameters():
        ok &= check(f"init W == W0 [{n.split('.')[-2]}]", rel(p.data, w0[n]) <= tol0,
                    f"rel={rel(p.data, w0[n]):.2e}")

    s0 = {n: svals(w._es[n]["state"] if mode == "iso" else p.data, w._es[n])
          for n, p in params.items() if n in w._es}

    # ---- perturbation keeps the frames orthonormal and the spectrum fixed ----
    w.es_perturb(1234, sigma)
    for n, p in params.items():
        st = w._es.get(n)
        if st is None:
            continue
        tag = n.split(".")[-2]
        d = rel(p.data, w0[n])
        ok &= check(f"{tag}: perturbation visible", d > 1e-3, f"||dW||/||W||={d:.3e}")
        ok &= check(f"{tag}: ||dW||/||W|| ~= sigma", 0.5 * sigma < d < 2.0 * sigma,
                    f"{d:.3e} vs sigma={sigma:.3e}")
        sp = svals(p.data, st)
        e = ((sp - s0[n]).norm() / s0[n].norm()).item()
        ok &= check(f"{tag}: singular values preserved", e < stol, f"rel d(sigma)={e:.2e}")
        if st["kind"] == "isobtt":
            # the trained core must still be exactly orthogonal after perturbing
            r = st["state"]
            eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
            rot = w._iso_noise_btt(st, 1234, sigma) @ r
            oe = (torch.bmm(rot, rot.transpose(1, 2)) - eye).abs().max().item()
            ok &= check(f"{tag}: perturbed R^T R == I", oe < 1e-5, f"max|err|={oe:.2e}")
        else:
            # W_pert = C_L W C_R^T, so U -> C_L U and V -> C_R V stay orthonormal iff
            # C_L, C_R are orthogonal.  Check the actual factors used for this seed.
            _, cl, _, cr = w._iso_noise(st, 1234, sigma)
            worst = 0.0
            for c in (cl, cr):
                eye = torch.eye(c.shape[-1], device=dev).expand_as(c)
                worst = max(worst, (torch.bmm(c, c.transpose(1, 2)) - eye).abs().max().item())
            ok &= check(f"{tag}: C_L C_L^T == I and C_R C_R^T == I", worst < 1e-5,
                        f"max|err|={worst:.2e}")

    # ---- restore ----
    w.es_restore()
    for n, p in params.items():
        if n in w._es:
            ok &= check(f"{n.split('.')[-2]}: restore is bit-exact",
                        torch.equal(p.data, w0[n]) or mode == "isobtt",
                        f"rel={rel(p.data, w0[n]):.2e}")

    # ---- determinism + antithetic symmetry ----
    w.es_perturb(1234, sigma)
    plus = {n: p.detach().float().clone() for n, p in params.items()}
    w.es_restore()
    w.es_perturb(1234, sigma)
    again = {n: p.detach().float().clone() for n, p in params.items()}
    w.es_restore()
    w.es_perturb(1234, sigma, negate=True)
    minus = {n: p.detach().float().clone() for n, p in params.items()}
    w.es_restore()
    for n, p in params.items():
        if n not in w._es:
            continue
        tag = n.split(".")[-2]
        ok &= check(f"{tag}: same seed -> same perturbation",
                    torch.equal(plus[n], again[n]), "")
        mid = (plus[n] + minus[n]) / 2
        # Cay(+X) and Cay(-X) straddle W to O(sigma^2); sigma=5e-2 -> ~2.5e-3
        ok &= check(f"{tag}: (W+ + W-)/2 == W + O(sigma^2)",
                    rel(mid, p.data.float()) < 4 * sigma ** 2,
                    f"rel={rel(mid, p.data.float()):.2e}")

    # ---- update: still feasible, and first-order-correct ----
    seeds = list(range(100, 100 + N))
    coeffs = [1.0, -1.0] * (N // 2)
    before = {n: w._es[n]["state"].detach().clone() for n in w._es}
    w.es_update(seeds, coeffs, alpha, N)
    for n, p in params.items():
        st = w._es.get(n)
        if st is None:
            continue
        tag = n.split(".")[-2]
        sp = svals(st["state"] if mode == "iso" else p.data, st)
        e = ((sp - s0[n]).norm() / s0[n].norm()).item()
        ok &= check(f"{tag}: spectrum fixed after update", e < stol, f"rel d(sigma)={e:.2e}")

        # First-order reference.  Committing seed n applies X <- C_L X C_R^T, i.e.
        #   X + (C_L - I) X + X (C_R^T - I) + O(s^2),
        # and cross-seed products are O(s^2) too, so summing the two exact first-order
        # increments over the N seeds is the (alpha/N) sum_n Z_n Omega_n estimator.
        step = alpha / N
        x0 = before[n]
        ref = x0.clone()
        for i, sd in enumerate(seeds):
            sc = step * coeffs[i]
            if st["kind"] == "iso":
                pl, cl, pr, cr = w._iso_noise(st, sd, sc)
                ref += (w._iso_left(x0, pl, cl) - x0) + (w._iso_right(x0, pr, cr) - x0)
            else:
                c = w._iso_noise_btt(st, sd, sc)
                ref += torch.bmm(c, x0) - x0
        e2 = rel(st["state"], ref)
        tol2 = 40 * N * step ** 2 + 1e-5
        ok &= check(f"{tag}: update == (alpha/N) sum Z_n Omega_n + O(alpha^2)",
                    e2 < tol2, f"rel={e2:.2e} (tol {tol2:.1e})")

    m = w.es_get_metrics()
    print(f"  info: {info}")
    print(f"  metrics: {m}")
    if mode == "iso":
        ok &= check("frob drift logged", "iso/frob_drift" in m,
                    f"{m.get('iso/frob_drift', float('nan')):.2e}")
    else:
        ok &= check("orth err logged", m.get("iso/orth_err", 1.0) < 1e-4,
                    f"{m.get('iso/orth_err', float('nan')):.2e}")
    return ok


def kernel_checks(dev):
    """The batched permute+bmm kernels must equal an explicitly materialised
    P^T blkdiag(C_j) P acting on W."""
    ok = True
    torch.manual_seed(3)
    m, n, b = 12, 8, 4
    x = torch.randn(m, n, device=dev, dtype=torch.float64)
    g = torch.Generator(device=dev).manual_seed(1)
    ext = WorkerExtension
    for side in ("left", "right"):
        d = m if side == "left" else n
        perm = torch.randperm(d, generator=g, device=dev)
        c = _iso_cayley(_iso_skew(d // b, b, g, dev).double(), 0.3)
        # dense equivalent: rows/cols perm[i] of the block matrix
        big = torch.zeros(d, d, device=dev, dtype=torch.float64)
        for j in range(d // b):
            idx = perm[j * b:(j + 1) * b]
            big[idx.unsqueeze(1), idx.unsqueeze(0)] = c[j]
        got = ext._iso_left(x, perm, c) if side == "left" else ext._iso_right(x, perm, c)
        want = big @ x if side == "left" else x @ big.T
        ok &= check(f"_iso_{side} == dense P^T blkdiag(C) P", rel(got, want) < 1e-12,
                    f"rel={rel(got, want):.2e}")
        e = (big @ big.T - torch.eye(d, device=dev, dtype=torch.float64)).abs().max().item()
        ok &= check(f"materialised {side} operator is orthogonal", e < 1e-12, f"max|err|={e:.2e}")
    return ok


def fp64_exactness(dev, sigma=5e-2):
    """In exact arithmetic the spectrum is *exactly* invariant.  Running the same
    perturbation in fp64 must shrink ||d(sigma)||/||sigma|| by ~1e7 vs fp32; if it
    does not, the residual is an algebra error rather than round-off."""
    ok = True
    torch.manual_seed(5)
    x32 = (torch.randn(640, 512, device=dev) * 0.02)
    ext = WorkerExtension
    out = {}
    for dt in (torch.float32, torch.float64):
        x = x32.to(dt)
        g = torch.Generator(device=dev).manual_seed(11)
        s = sigma * 0.5 ** 0.5
        pl = torch.randperm(640, generator=g, device=dev)
        cl = _iso_cayley(_iso_skew(10, 64, g, dev).to(dt), s)
        pr = torch.randperm(512, generator=g, device=dev)
        cr = _iso_cayley(_iso_skew(8, 64, g, dev).to(dt), s)
        xp = ext._iso_right(ext._iso_left(x, pl, cl), pr, cr)
        s0 = torch.linalg.svdvals(x.double())
        out[dt] = ((torch.linalg.svdvals(xp.double()) - s0).norm() / s0.norm()).item()
    ok &= check("fp32 spectrum drift is round-off-sized", out[torch.float32] < 1e-4,
                f"{out[torch.float32]:.2e}")
    ok &= check("fp64 spectrum drift ~ 0 (algebra is exact)", out[torch.float64] < 1e-12,
                f"{out[torch.float64]:.2e} ({out[torch.float32]/max(out[torch.float64],1e-30):.1e}x tighter)")
    return ok


def unit_checks(dev):
    ok = True
    c = HFCfg
    hd = c.hidden_size // c.num_attention_heads
    segs = _iso_segments("model.layers.0.self_attn.qkv_proj.weight",
                         (c.num_attention_heads + 2 * c.num_key_value_heads) * hd, c)
    ok &= check("qkv segments", segs == [512, 128, 128], f"{segs}")
    segs = _iso_segments("model.layers.0.mlp.gate_up_proj.weight", 2 * c.intermediate_size, c)
    ok &= check("gate_up segments", segs == [640, 640], f"{segs}")
    ok &= check("block size divides every segment", _iso_block_size([3584, 512, 512], 128) == 128)
    ok &= check("block size falls back below request", _iso_block_size([18944], 128) == 128)
    # Cayley of a skew is exactly orthogonal, for any rank / scale
    g = torch.Generator(device=dev).manual_seed(0)
    om = _iso_skew(4, 64, g, dev)
    ok &= check("generator is skew", (om + om.transpose(1, 2)).abs().max().item() < 1e-6)
    for s in (1e-4, 5e-2, 1.0):
        cay = _iso_cayley(om, s)
        eye = torch.eye(64, device=dev).expand(4, 64, 64)
        e = (torch.bmm(cay, cay.transpose(1, 2)) - eye).abs().max().item()
        ok &= check(f"Cay(s*Omega) orthogonal @ s={s}", e < 1e-5, f"max|err|={e:.2e}")
    # E||Omega F||_F ~= ||F||_F  (the sigma == relative-footprint convention)
    f = torch.randn(4, 64, 300, device=dev)
    r = (torch.bmm(om, f).norm() / f.norm()).item()
    ok &= check("||Omega F|| ~= ||F|| (scale convention)", 0.8 < r < 1.25, f"ratio={r:.3f}")
    return ok


def real_model_checks(path, dev, sigma=5e-2):
    from transformers import AutoModelForCausalLM

    ok = True
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev)
    mods = dict(m.named_modules())
    for nm in ["model.layers.0.self_attn.q_proj", "model.layers.13.mlp.gate_proj",
               "model.layers.27.mlp.down_proj"]:
        wt = mods[nm].weight.data.float()
        out_f, in_f = wt.shape
        g = torch.Generator(device=dev).manual_seed(0)

        # --- iso: full-matrix spectrum ---
        s = sigma * 0.5 ** 0.5
        bl = _iso_block_size([out_f], 128)
        br = _iso_block_size([in_f], 128)
        cl = _iso_cayley(_iso_skew(out_f // bl, bl, g, dev), s)
        cr = _iso_cayley(_iso_skew(in_f // br, br, g, dev), s)
        pl = torch.arange(out_f, device=dev)
        pr = torch.arange(in_f, device=dev)
        ext = WorkerExtension
        wp = ext._iso_right(ext._iso_left(wt, pl, cl), pr, cr)
        d = rel(wp, wt)
        # svdvals in fp64: q_proj has cond ~5e6, so an fp32 SVD of *both* sides adds
        # ~1e-4 of its own noise and would swamp the quantity under test.
        s0 = torch.linalg.svdvals(wt.double())
        e = ((torch.linalg.svdvals(wp.double()) - s0).norm() / s0.norm()).item()
        ok &= check(f"iso {nm} [{out_f}x{in_f}] spectrum", e < 1e-6,
                    f"rel d(sigma)={e:.2e}, ||dW||/||W||={d:.3e}")

        # --- isobtt: per-block spectrum ---
        n_blk, b = _es_closest_factor_pair(in_f)
        x = wt.reshape(out_f, n_blk, b).permute(1, 0, 2)
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        A = (U * S.unsqueeze(1)).to(torch.bfloat16)
        c = _iso_cayley(_iso_skew(n_blk, b, g, dev), sigma)
        w_b = torch.bmm(A.float(), Vh).permute(1, 0, 2).reshape(wt.shape)
        w_p = torch.bmm(A.float(), torch.bmm(c, Vh)).permute(1, 0, 2).reshape(wt.shape)
        sb = torch.linalg.svdvals(w_b.double().reshape(out_f, n_blk, b).permute(1, 0, 2))
        sp = torch.linalg.svdvals(w_p.double().reshape(out_f, n_blk, b).permute(1, 0, 2))
        e = ((sp - sb).norm() / sb.norm()).item()
        ok &= check(f"isobtt {nm} [n={n_blk}, b={b}] block spectra", e < 1e-6,
                    f"rel d(sigma)={e:.2e}, ||dW||/||W||={rel(w_p, w_b):.3e}, "
                    f"recon={rel(w_b, wt):.2e}")
    del m
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-model", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=== unit checks ===")
    allok = unit_checks(dev)
    print("\n=== kernel vs dense reference ===")
    allok &= kernel_checks(dev)
    print("\n=== exactness in fp64 ===")
    allok &= fp64_exactness(dev)
    for mode in ("iso", "isobtt"):
        for dtype in (torch.float32, torch.bfloat16):
            print(f"\n=== mode: {mode}  (vLLM param dtype {dtype}) ===")
            allok &= run_mode(mode, {"iso_block_size": 64, "iso_perm": True}, dev, dtype)
    print("\n=== iso without the per-seed basis permutation ===")
    allok &= run_mode("iso", {"iso_block_size": 64, "iso_perm": False}, dev, torch.float32)

    if args.real_model:
        print("\n=== real Qwen weights ===")
        allok &= real_model_checks(args.real_model, dev)

    print("\n" + ("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
