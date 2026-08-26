"""Numerical gate for the `lora` ES mode, on real Qwen2.5-Math-7B weights (CPU/GPU stub)."""
import os, sys, types, torch
REPO = "/home/yequan/Project/compression/OPD"
sys.path.insert(0, os.path.join(REPO, "verl"))
from verl.workers.rollout.vllm_rollout.es_worker_extension import StructuredESMixin, _es_target
sys.path.insert(0, os.path.join(REPO, "scripts/es"))
from materialize_es_ckpt import load_fused, _Materializer, DEFAULT_BASE
from transformers import AutoConfig

dev = "cuda"
cfg = AutoConfig.from_pretrained(DEFAULT_BASE)
fused, recipe = load_fused(DEFAULT_BASE, dev, cfg)
W0 = {k: v.clone() for k, v in fused.items()}

for rank in (1, 44):
    mz = _Materializer(fused, cfg)
    info = mz.init_es_state("lora", {"lora_rank": rank, "lora_scale": 1.0})
    print(f"\n=== rank {rank}: {info}")

    # 1) zero-init B => W == W_base bit-exactly
    mz.es_restore()
    md = max(float((fused[k] - W0[k]).abs().max()) for k in W0)
    print(f"  [1] identity at init (B=0): max|W - W_base| = {md:.3e}   {'PASS' if md == 0 else 'FAIL'}")

    # 2) perturb -> restore is bit-exact
    mz.es_perturb(1234, 1e-3)
    pert = max(float((fused[k] - W0[k]).abs().max()) for k in W0)
    mz.es_restore()
    md2 = max(float((fused[k] - W0[k]).abs().max()) for k in W0)
    print(f"  [2] perturb moved W (max {pert:.3e}); restore -> {md2:.3e}   {'PASS' if md2 == 0 else 'FAIL'}")

    # 3) (W+ + W-)/2 == W  to bf16 round-off
    mz.es_perturb(1234, 1e-3, negate=False); Wp = {k: fused[k].clone() for k in W0}
    mz.es_perturb(1234, 1e-3, negate=True);  Wm = {k: fused[k].clone() for k in W0}
    mz.es_restore()
    sym = max(float(((Wp[k].float() + Wm[k].float()) / 2 - W0[k].float()).abs().max()) for k in W0)
    print(f"  [3] (W+ + W-)/2 - W = {sym:.3e} (bf16 ULP ~1.6e-4)")

    # 4) relative weight-space footprint at sigma=1e-3
    num = sum(float((Wp[k].float() - W0[k].float()).norm()) ** 2 for k in W0) ** 0.5
    den = sum(float(W0[k].float().norm()) ** 2 for k in W0) ** 0.5
    print(f"  [4] ||dW||_F/||W||_F at sigma=1e-3 = {num/den:.3e}   (dense 5.0e-2, zoact 4.2e-3, fura 4.0e-3)")

    # 5) an ES update moves the coefficients as (alpha/N) sum Z_n eps_n
    tgt = _es_target(next(iter(mz._es.values()))).clone()
    mz.es_update([11, 22], [1.0, -1.0], alpha=5e-3, population_size=2)
    st = next(iter(mz._es.values()))
    moved = float((_es_target(st) - tgt).abs().max())
    print(f"  [5] es_update moved coefficients: max|dC| = {moved:.3e}   {'PASS' if moved > 0 else 'FAIL'}")
    for k in fused: fused[k].copy_(W0[k])
    del mz; torch.cuda.empty_cache()
