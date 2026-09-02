"""Weight-space footprint ||dW||_F / ||W||_F of every ES perturbation mode, at sigma=1e-3,
on the REAL Qwen2.5-Math-7B weights.

Why this exists: the numbers in es_results.md section 6 (dense 5.0e-2, zoact 4.2e-3, fura
4.0e-3) were read off `test_es_perturb_modes.py`, which runs on a 192x144 / 144x256 *fake*
model.  zoact's footprint scales as 1/sqrt(in_features) and fura's as sqrt(block_size), so
those numbers do not carry over to a 7B model -- while section 15.3's LoRA numbers were
measured on the real one.  Everything on this page is measured the same way.

    python3 scripts/es/measure_es_footprint.py [--device cpu] [--sigma 1e-3]
"""
import argparse, os, sys, torch

REPO = "/home/yequan/Project/compression/OPD"
sys.path.insert(0, os.path.join(REPO, "verl"))
sys.path.insert(0, os.path.join(REPO, "scripts/es"))
from materialize_es_ckpt import load_fused, _Materializer, DEFAULT_BASE  # noqa: E402
from transformers import AutoConfig  # noqa: E402

CALIB = os.path.join(REPO, "datasets/es_math/calib_qwen2p5_math_7b.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--sigma", type=float, default=1e-3)
ap.add_argument("--modes", default="")
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(DEFAULT_BASE)
fused, _ = load_fused(DEFAULT_BASE, a.device, cfg)
W0 = {k: v.clone() for k, v in fused.items()}

MODES = [
    ("dense", {}),
    ("zoact", {"calib_path": CALIB, "rank": 1}),
    ("insparse", {"calib_path": CALIB, "density": 0.01}),
    ("fura", {}),
    ("lora", {"lora_rank": 1, "lora_scale": 1.0}),
    ("lora", {"lora_rank": 44, "lora_scale": 1.0}),
    ("fura_zoact", {"calib_path": CALIB, "rank": 1}),
    ("lora_zoact", {"calib_path": CALIB, "lora_rank": 1, "lora_scale": 1.0}),
    ("lora_zoact", {"calib_path": CALIB, "lora_rank": 44, "lora_scale": 1.0}),
]
if a.modes:
    keep = set(a.modes.split(","))
    MODES = [m for m in MODES if m[0] in keep]

print(f"{'mode':<14} {'r':>3}  {'coeffs':>13}  {'||dW||/||W|| (targeted)':>23}  {'(all params)':>13}"
      f"  {'sigma for 5.0e-2':>17}")
for mode, mcfg in MODES:
    mz = _Materializer(fused, cfg)
    info = mz.init_es_state(mode, mcfg)
    tgt = list(mz._es)                       # the parameters this mode actually touches
    mz.es_restore()
    Wr = {k: fused[k].clone() for k in W0}
    mz.es_perturb(1234, a.sigma)

    def relnorm(keys):
        num = sum(float((fused[k].float() - Wr[k].float()).norm()) ** 2 for k in keys) ** 0.5
        den = sum(float(Wr[k].float().norm()) ** 2 for k in keys) ** 0.5
        return num / den

    f_t, f_a = relnorm(tgt), relnorm(list(W0))
    per_sigma = f_t / a.sigma
    r = mcfg.get("rank", mcfg.get("lora_rank", "-"))
    print(f"{mode:<14} {str(r):>3}  {info['coef_params']:>13,}  {f_t:>23.3e}  {f_a:>13.3e}"
          f"  {5.0e-2 / per_sigma:>17.4g}")
    for k in fused:
        fused[k].copy_(W0[k])
    del mz, Wr
