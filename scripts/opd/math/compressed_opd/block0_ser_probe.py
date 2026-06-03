"""Block 0 — Steering-Energy-Retention (SER) probe (TRACER gating experiment).

Tests the TRACER central thesis: structured low-rank compression collapses
reasoning because it *selectively erodes a low-variance, high-leverage steering
subspace*, not because it degrades the model uniformly.

Procedure
---------
1. Load dense Qwen3-4B (non-thinking base); collect OpenThought3 mixed stats
   (attn input cov + down_proj C_sigma) ONCE, exactly as layer_sensitivity.py.
2. Extract dense steering vectors u~^m_{l,c} (difference-of-means, input space)
   + attribution importance I^m_{l,c} via compress.steering.
3. Build two compressed conditions and, for each, measure per-module how much the
   weight's action along the steering directions is preserved vs random controls:
     (a) SINGLE-LAYER  : one attn block + one mlp triplet at retain 0.5 (harmless)
     (b) ALL-LAYER     : every attn+mlp at retain 0.36 (the 0% collapse)
   Metrics per (layer, module, behavior):
     - attn q/k/v/o (SVDCompressedLinear): cos(W_d u~, W_c u~) and energy ratio
       ||W_c u~||^2 / ||W_d u~||^2, with a RANDOM low-variance direction control.
     - mlp triplet (Nystrom -> shape change): functional SER over the whole MLP:
       cos(MLP_d(u~), MLP_c(u~)) and ||MLP_c(u~)||^2/||MLP_d(u~)||^2, random control.
4. Variance-vs-leverage check: per module, the steering directions' input variance
   vs their attribution importance (premise: low variance, high leverage).

Prediction
----------
SER ~ 1 for single-layer; for all-layer SER on the STEERING subspace drops far
more than on RANDOM directions of comparable variance, and worsens with depth.
Falsifier: steering subspace decays no worse than random -> thesis wrong.

Usage (one free GPU):
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    python scripts/opd/math/compressed_opd/block0_ser_probe.py \
      --layers 35,33,30,20,15,5,2,0 --out results/block0/ser_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "verl"))

from compress.compress_model import MethodSpec  # noqa: E402
from compress.calibration import collect_mixed_statistics  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.svd.svd_linear import SVDCompressedLinear  # noqa: E402
from compress.structured.nystrom import nystrom_compress_model  # noqa: E402
from compress.steering import extract_steering_vectors, SteeringConfig  # noqa: E402

# Reuse the exact calibration plumbing proven in layer_sensitivity.py.
from layer_sensitivity import (  # noqa: E402
    build_openthought3_loader, attn_linear_names, mlp_down_names, parse_layers,
)

ATTN_METHOD = "svd_llm_v2"
MLP_METHOD = "nystrom"
MIXED_SPEC = MethodSpec(attn=ATTN_METHOD, mlp=MLP_METHOD)


def _load_conversations(tokenizer, n: int):
    """Load n OpenThought3 conversations as raw message lists."""
    path = REPO_ROOT / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if msgs:
                out.append(msgs)
            if len(out) >= n:
                break
    if not out:
        raise ValueError(f"No conversations from {path}")
    return out


def _module_by_name(model, name):
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p)
    return mod


def _effective_attn_weight(module) -> torch.Tensor:
    """Dense-equivalent weight (d_out,d_in) for a dense or SVD-compressed linear."""
    if isinstance(module, SVDCompressedLinear):
        return module.materialize_dense_weight()
    return module.weight.data


@torch.no_grad()
def _attn_ser(dense_mod, comp_mod, u_tilde, randoms, device):
    """cos + energy ratio of W along u~ (dense vs compressed); plus random controls."""
    Wd = _effective_attn_weight(dense_mod).to(device=device, dtype=torch.float32)
    Wc = _effective_attn_weight(comp_mod).to(device=device, dtype=torch.float32)
    u = u_tilde.to(device=device, dtype=torch.float32)

    def _one(v):
        yd = Wd @ v
        yc = Wc @ v
        nd = yd.norm()
        cos = float(torch.dot(yd, yc) / (nd * yc.norm() + 1e-12))
        energy = float((yc.norm() ** 2) / (nd ** 2 + 1e-12))
        return cos, energy

    cos_s, en_s = _one(u)
    rand_cos, rand_en = [], []
    for r in randoms:
        c, e = _one(r.to(device=device, dtype=torch.float32))
        rand_cos.append(c); rand_en.append(e)
    return {
        "steer_cos": cos_s, "steer_energy": en_s,
        "rand_cos_mean": sum(rand_cos) / len(rand_cos),
        "rand_energy_mean": sum(rand_en) / len(rand_en),
    }


@torch.no_grad()
def _mlp_local_ser(dense_mlp, comp_mlp, x0, u_tilde, randoms, device, eps):
    """Local directional SER (review fix): directional derivative at real baseline x0.

    For a nonlinear MLP f, MLP(u~) is off-manifold/meaningless. Instead probe the
    *local* action along u~ at the dense baseline activation x0:
        Df(x0; v) = f(x0 + eps*v) - f(x0)
    and compare dense vs compressed: cos(Df_dense, Df_comp), energy ratio.
    eps is scaled so eps*||u~|| is a small fraction of ||x0||.
    """
    x0 = x0.to(device=device, dtype=torch.float32)
    u = u_tilde.to(device=device, dtype=torch.float32)

    def _df(mlp, v):
        vhat = v / (v.norm() + 1e-12)
        step = eps * x0.norm() * vhat
        return (mlp((x0 + step).unsqueeze(0)) - mlp(x0.unsqueeze(0))).squeeze(0)

    def _one(v):
        dd = _df(dense_mlp, v)
        dc = _df(comp_mlp, v)
        nd = dd.norm()
        cos = float(torch.dot(dd, dc) / (nd * dc.norm() + 1e-12))
        energy = float((dc.norm() ** 2) / (nd ** 2 + 1e-12))
        return cos, energy

    cos_s, en_s = _one(u)
    rand_cos, rand_en = [], []
    for r in randoms:
        c, e = _one(r.to(device=device, dtype=torch.float32))
        rand_cos.append(c); rand_en.append(e)
    return {
        "steer_cos": cos_s, "steer_energy": en_s,
        "rand_cos_mean": sum(rand_cos) / len(rand_cos),
        "rand_energy_mean": sum(rand_en) / len(rand_en),
    }


def _variance_matched_random_dirs(reference, cov, k, seed):
    """k random directions with the SAME covariance-induced variance as `reference`.

    Review fix: the null must be "random direction of equal VARIANCE" under the
    module input covariance C, not equal L2 norm. We draw unit Gaussian directions
    q, then rescale so q^T C q == u^T C u (variance-matched), keeping a random
    direction. Falls back to L2-norm matching if C is unavailable.
    """
    g = torch.Generator().manual_seed(seed)
    d = reference.shape[0]
    dirs = []
    if cov is not None:
        C = cov.to(dtype=torch.float32)
        uhat = reference / (reference.norm() + 1e-12)
        target_var = float(uhat @ (C @ uhat))  # u^T C u / ||u||^2
        for _ in range(k):
            q = torch.randn(d, generator=g, dtype=torch.float32)
            q = q / (q.norm() + 1e-12)
            qv = float(q @ (C @ q))
            scale = (target_var / (qv + 1e-12)) ** 0.5
            dirs.append(q * scale * reference.norm())  # match variance, keep dir random
        return dirs
    # fallback: L2-norm match
    target_norm = reference.norm()
    for _ in range(k):
        r = torch.randn(d, generator=g, dtype=torch.float32)
        dirs.append(r * (target_norm / (r.norm() + 1e-12)))
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-tag", default="qwen3-4b-base")
    ap.add_argument("--layers", default="35,33,30,20,15,5,2,0",
                    help="layers to report SER on (subset for speed)")
    ap.add_argument("--single-layer-target", type=int, default=20,
                    help="layer compressed in the single-layer (harmless) condition; "
                         "default 20 — validated harmless at retain 0.5 in the "
                         "per-layer sensitivity sweep (docs/results/compressed_opd.md)")
    ap.add_argument("--single-layer-ratio", type=float, default=0.8)
    ap.add_argument("--all-layer-ratio", type=float, default=0.8)
    ap.add_argument("--skip-last-layers", type=int, default=1,
                    help="leave the top-N decoder layers DENSE in the all-layer condition")
    ap.add_argument("--mlp-eps", type=float, default=1e-2,
                    help="relative step for the local MLP directional-derivative probe")
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048)
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--steer-num-convs", type=int, default=128)
    ap.add_argument("--steer-max-length", type=int, default=2048)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-importance", action="store_true",
                    help="skip attribution patching (faster sanity runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    base_cpu.eval()
    n_layers = base_cpu.config.num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    # ensure the single-layer harmless target is also measured
    if args.single_layer_target not in layers:
        layers = [args.single_layer_target] + layers

    attn_by_layer = attn_linear_names(base_cpu)
    down_by_layer = mlp_down_names(base_cpu)

    # ---- live GPU dense model + mixed stats (collected once) ---- #
    model = copy.deepcopy(base_cpu).to(device)
    logger.info("Collecting OpenThought3 mixed statistics (one pass)...")
    calib_loader = build_openthought3_loader(
        tokenizer, num_seqs=args.calib_num_seqs,
        max_length=args.calib_max_length, batch_size=args.calib_batch_size,
    )
    stats = collect_mixed_statistics(
        model, calib_loader, MIXED_SPEC, device=device, skip_layers=("lm_head",),
    )
    del calib_loader
    torch.cuda.empty_cache()

    # ---- steering vectors from the DENSE model ---- #
    logger.info("Extracting steering vectors (difference-of-means) from dense model...")
    convs = _load_conversations(tokenizer, args.steer_num_convs)
    # collect input covariance only for the modules we report SER on (attn linears +
    # gate_proj for the MLP control basis) — keeps memory bounded.
    cov_modules = []
    for li in layers:
        cov_modules.extend(attn_by_layer.get(li, []))
        cov_modules.append(f"model.layers.{li}.mlp.gate_proj")
    scfg = SteeringConfig(
        max_length=args.steer_max_length, device=device, dtype=torch.float32,
        compute_importance=not args.no_importance,
        cov_modules=tuple(cov_modules), collect_mlp_baseline=True,
    )
    steer = extract_steering_vectors(
        model, tokenizer, convs,
        skip_layers=("lm_head", "embed_tokens"), cfg=scfg,
    )
    vectors = steer["vectors"]
    importance = steer["importance"]
    dir_var = steer["dir_var"]
    steer_cov = steer["cov"]
    mlp_x0 = steer["mlp_x0"]
    logger.info(f"Steering vectors on {len(vectors)} modules; "
                f"behaviors with hits: {sorted(steer['n_pos'].keys())[:6]}...")

    # snapshot dense effective weights for the reported layers BEFORE compressing
    dense_attn_W = {}
    dense_mlp = {}
    for li in layers:
        for nm in attn_by_layer.get(li, []):
            dense_attn_W[nm] = _effective_attn_weight(_module_by_name(model, nm)).cpu().clone()
        mlp_parent = f"model.layers.{li}.mlp"
        try:
            dense_mlp[li] = copy.deepcopy(_module_by_name(model, mlp_parent)).to(device).float()
        except AttributeError:
            pass

    results = {
        "model": args.model, "model_tag": args.model_tag, "n_layers": n_layers,
        "layers": layers, "config": vars(args),
        "single_layer_ratio": args.single_layer_ratio,
        "all_layer_ratio": args.all_layer_ratio,
        "conditions": {}, "variance_vs_leverage": [],
    }

    # ---- variance-vs-leverage check (per reported module/behavior) ---- #
    # Premise: steering directions are LOW directional-variance (u^T C u/||u||^2)
    # but HIGH attribution importance. We also report the mean diagonal of C as the
    # module's typical per-dim variance scale, so dir_var can be read relative to it.
    for li in layers:
        for nm in attn_by_layer.get(li, []):
            bd = vectors.get(nm, {})
            C = steer_cov.get(nm)
            mean_diag_var = float(C.diagonal().mean()) if C is not None else None
            for b, u in bd.items():
                results["variance_vs_leverage"].append({
                    "layer": li, "module": nm, "behavior": b,
                    "dir_var": dir_var.get(nm, {}).get(b),       # u^T C u / ||u||^2
                    "mean_diag_var": mean_diag_var,              # avg per-dim var (ref scale)
                    "importance": importance.get(nm, {}).get(b),
                })

    def _build_compressed(condition: str):
        """Return a fresh compressed GPU model for the given condition."""
        m = copy.deepcopy(base_cpu).to(device)
        if condition == "single_layer":
            li = args.single_layer_target  # validated-harmless layer (review fix)
            attn_names = attn_by_layer[li]
            subset = {n: stats[n].clone() for n in attn_names if n in stats}
            svd_llm_v2_compress_model(
                m, subset, compression_ratio=args.single_layer_ratio,
                skip_layers=("lm_head",), device=device, objective="forward",
            )
            down = down_by_layer[li]
            nystrom_compress_model(
                m, {down: stats[down].clone()}, sparsity=1.0 - args.single_layer_ratio,
                skip_layers=("lm_head",), device=device,
            )
            return m, [li]
        elif condition == "all_layer":
            protect = set(range(n_layers - args.skip_last_layers, n_layers)) \
                if args.skip_last_layers > 0 else set()

            def _li(name):
                try:
                    return int(name.split(".layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    return None

            attn_subset = {n: s.clone() for n, s in stats.items()
                           if ".self_attn." in n and _li(n) not in protect}
            svd_llm_v2_compress_model(
                m, attn_subset, compression_ratio=args.all_layer_ratio,
                skip_layers=("lm_head",), device=device, objective="forward",
            )
            down_subset = {n: s.clone() for n, s in stats.items()
                           if n.endswith(".down_proj") and _li(n) not in protect}
            nystrom_compress_model(
                m, down_subset, sparsity=1.0 - args.all_layer_ratio,
                skip_layers=("lm_head",), device=device,
            )
            return m, layers
        raise ValueError(condition)

    for condition in ["single_layer", "all_layer"]:
        logger.info(f"=== condition: {condition} ===")
        t0 = time.time()
        comp_model, measured_layers = _build_compressed(condition)
        cells = []
        for li in measured_layers:
            # attn modules
            for nm in attn_by_layer.get(li, []):
                bd = vectors.get(nm, {})
                if not bd or nm not in dense_attn_W:
                    continue
                comp_mod = _module_by_name(comp_model, nm)
                # build a dense module wrapper holding the snapshotted dense W
                Wd = dense_attn_W[nm]
                dense_lin = torch.nn.Linear(Wd.shape[1], Wd.shape[0], bias=False)
                dense_lin.weight.data = Wd.clone()
                dense_lin = dense_lin.to(device).float()
                Cnm = steer_cov.get(nm)  # input cov for variance-matched control
                for b, u in bd.items():
                    randoms = _variance_matched_random_dirs(
                        u, Cnm, args.n_random, args.random_seed + li)
                    m_ = _attn_ser(dense_lin, comp_mod, u, randoms, device)
                    m_.update({"layer": li, "module": nm, "kind": "attn",
                               "behavior": b, "importance": importance.get(nm, {}).get(b),
                               "dir_var": dir_var.get(nm, {}).get(b)})
                    cells.append(m_)
                del dense_lin
            # mlp LOCAL directional-derivative probe at the dense baseline x0 (review fix)
            mlp_parent = f"model.layers.{li}.mlp"
            if li in dense_mlp and mlp_parent in mlp_x0:
                gate_nm = f"{mlp_parent}.gate_proj"
                bd = vectors.get(gate_nm, {})
                Cgate = steer_cov.get(gate_nm)
                x0 = mlp_x0[mlp_parent]
                comp_mlp = _module_by_name(comp_model, mlp_parent).to(device).float()
                for b, u in bd.items():
                    randoms = _variance_matched_random_dirs(
                        u, Cgate, args.n_random, args.random_seed + 1000 + li)
                    m_ = _mlp_local_ser(dense_mlp[li], comp_mlp, x0, u, randoms,
                                        device, args.mlp_eps)
                    m_.update({"layer": li, "module": mlp_parent, "kind": "mlp",
                               "behavior": b,
                               "importance": importance.get(gate_nm, {}).get(b),
                               "dir_var": dir_var.get(gate_nm, {}).get(b)})
                    cells.append(m_)
        results["conditions"][condition] = cells
        logger.info(f"  {condition}: {len(cells)} cells in {time.time()-t0:.0f}s")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        del comp_model
        torch.cuda.empty_cache()

    # ---- summary verdict ---- #
    def _agg(cells):
        import statistics as st
        if not cells:
            return {}
        return {
            "steer_cos_median": st.median(c["steer_cos"] for c in cells),
            "rand_cos_median": st.median(c["rand_cos_mean"] for c in cells),
            "steer_energy_median": st.median(c["steer_energy"] for c in cells),
            "rand_energy_median": st.median(c["rand_energy_mean"] for c in cells),
            "n": len(cells),
        }
    summary = {cond: _agg(cells) for cond, cells in results["conditions"].items()}
    results["summary"] = summary
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("==== SER SUMMARY ====")
    for cond, agg in summary.items():
        logger.info(f"  {cond}: {agg}")
    al = summary.get("all_layer", {})
    if al:
        gap = al.get("rand_cos_median", 0) - al.get("steer_cos_median", 0)
        logger.info(f"  all_layer steering-vs-random cos gap = {gap:+.4f} "
                    f"(thesis predicts > 0: steering eroded MORE than random)")
    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
