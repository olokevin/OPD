"""Numerical gate for the `fura_zoact` / `lora_zoact` ES modes, on real Qwen2.5-Math-7B
weights.  Same stub harness as `test_lora_es.py`; defaults to CPU so it can run while the
GPUs are training.

    python3 scripts/es/test_zoact_hybrid_es.py [--device cpu|cuda] [--rank 1]
"""
import argparse, os, sys, torch

REPO = "/home/yequan/Project/compression/OPD"
sys.path.insert(0, os.path.join(REPO, "verl"))
from verl.workers.rollout.vllm_rollout.es_worker_extension import _es_target  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "scripts/es"))
from materialize_es_ckpt import load_fused, _Materializer, DEFAULT_BASE  # noqa: E402
from transformers import AutoConfig  # noqa: E402

CALIB = os.path.join(REPO, "datasets/es_math/calib_qwen2p5_math_7b.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--rank", type=int, default=1)
ap.add_argument("--sigma", type=float, default=1e-3)
ap.add_argument("--modes", default="fura_zoact,lora_zoact")
args = ap.parse_args()

dev, r, sig = args.device, args.rank, args.sigma
cfg = AutoConfig.from_pretrained(DEFAULT_BASE)
fused, _ = load_fused(DEFAULT_BASE, dev, cfg)
W0 = {k: v.clone() for k, v in fused.items()}
den = sum(float(W0[k].float().norm()) ** 2 for k in W0) ** 0.5


def fro(a, b):
    return sum(float((a[k].float() - b[k].float()).norm()) ** 2 for k in a) ** 0.5


_ALL = (
    ("fura_zoact", {"calib_path": CALIB, "rank": r}),
    ("lora_zoact", {"calib_path": CALIB, "lora_rank": r, "lora_scale": 1.0}),
)
for mode, mcfg in [m for m in _ALL if m[0] in args.modes.split(",")]:
    mz = _Materializer(fused, cfg)
    info = mz.init_es_state(mode, mcfg)
    print(f"\n=== {mode} r={r}: {info}")

    # 1) step 0.  lora_zoact is exact (B=0); fura_zoact writes the BTT reconstruction,
    #    so it lands at the bf16 factorisation floor, exactly like `fura`.
    mz.es_restore()
    Wr = {k: fused[k].clone() for k in W0}
    rel = fro(Wr, W0) / den
    ok = (rel == 0.0) if mode == "lora_zoact" else (rel < 3e-3)
    print(f"  [1] step 0 vs base: ||W - W_base||/||W|| = {rel:.3e}   {'PASS' if ok else 'FAIL'}")

    # 2) perturb -> restore is bit-exact (W is recomputed, never add-then-subtract)
    mz.es_perturb(1234, sig)
    moved = fro(fused, Wr) / den
    mz.es_restore()
    back = max(float((fused[k] - Wr[k]).abs().max()) for k in W0)
    print(f"  [2] perturb moved W ({moved:.3e} rel); restore -> max|dW| = {back:.3e}   "
          f"{'PASS' if back == 0 else 'FAIL'}")

    # 3) antithetic symmetry: (W+ + W-)/2 == W to bf16 round-off
    mz.es_perturb(1234, sig, negate=False); Wp = {k: fused[k].clone() for k in W0}
    mz.es_perturb(1234, sig, negate=True);  Wm = {k: fused[k].clone() for k in W0}
    mz.es_restore()
    sym = max(float(((Wp[k].float() + Wm[k].float()) / 2 - Wr[k].float()).abs().max()) for k in W0)
    print(f"  [3] (W+ + W-)/2 - W = {sym:.3e} (bf16 ULP ~1.6e-4)")

    # 4) relative weight-space footprint at this sigma
    print(f"  [4] ||dW||_F/||W||_F at sigma={sig:g} = {moved:.3e}   "
          f"(dense 5.0e-2, zoact 4.2e-3, fura 4.0e-3, lora r44 3.25e-3)")

    # 5) the perturbation really lives in the calibrated input subspace
    calib = torch.load(CALIB, map_location=dev, weights_only=False)["layers"]
    from verl.workers.rollout.vllm_rollout.es_worker_extension import _es_calib_key
    worst = 0.0
    for name, st in list(mz._es.items())[:8]:
        d = (Wp[name].float() - Wr[name].float())                     # (out, in)
        v = calib[_es_calib_key(name)]["v"][:r].to(dev, torch.float32)  # (r, in)
        if st["kind"] == "fura_zoact":
            n_blk, b = st["n_blk"], st["b"]
            d = d.reshape(-1, n_blk, b).permute(1, 0, 2)               # (n, out, b)
            vv = v.reshape(r, n_blk, b).permute(1, 0, 2)               # (n, r, b)
            q = torch.linalg.qr(vv.transpose(1, 2))[0]                 # (n, b, r) orthonormal
            resid = d - torch.bmm(torch.bmm(d, q), q.transpose(1, 2))
        else:
            q = torch.linalg.qr(v.T)[0]                                # (in, r)
            resid = d - (d @ q) @ q.T
        worst = max(worst, float(resid.norm() / d.norm().clamp_min(1e-30)))
    label = ("row space == calibrated subspace" if st["kind"] == "fura_zoact"
             else "row space drifts off the calibrated subspace (A is trained)")
    print(f"  [5] off-subspace mass of dW (8 layers): {worst:.3e}   <- {label}")

    # 6) an ES update moves the coefficients
    tgt = _es_target(next(iter(mz._es.values()))).clone()
    mz.es_update([11, 22], [1.0, -1.0], alpha=5e-3, population_size=2)
    st0 = next(iter(mz._es.values()))
    dmax = float((_es_target(st0) - tgt).abs().max())
    print(f"  [6] es_update moved coefficients: max|dC| = {dmax:.3e}   {'PASS' if dmax > 0 else 'FAIL'}")

    for k in fused:
        fused[k].copy_(W0[k])
    del mz
