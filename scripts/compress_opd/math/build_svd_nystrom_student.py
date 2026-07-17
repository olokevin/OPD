"""Stage 1 of compress→OPD: reasoning-aware structured compression of
Qwen3-4B (non-thinking) at **retain ratio 0.7**, then save as an HF model dir
the verl OPD trainer can load as the actor.

Method (from docs/wiki/reasoning_aware_compress_calib.md, §3 D0 / §6 entry
points — the forward-only "svd_v2 attn + nystrom mlp" cell):

  - self_attn (q/k/v/o_proj): **SVD-LLM-V2** input-whitened low-rank
    factorization (`svd_llm_v2_compress_model`, objective="forward").
  - mlp (gate/up/down_proj): **Nystrom** neuron subsampling
    (`nystrom_compress_model`).
  - Operating point: retain 0.7, **last decoder layer left dense**
    (`--skip-last-layers 1`).

Calibration follows the **2026-06-04 productionized default** (sequence-reweighted,
full-length OpenThought3 math traces — `compress_common.build_calib_loader` with
calib="openthought3", length="full" + `collect_*` default reweight="sequence").
That format study (wiki §5) showed +5..+16pp MATH-500 over the legacy
token/2048-window scheme at this exact method.

Why this stage at all: the previous compressed_opd thread
(docs/results/compressed_opd.md) collapsed to 0% MATH at the aggressive 0.36
retain ratio with one-shot SVD+Nystrom. The wiki shows the same forward-only
method gets **66%** at retain 0.7 with the legacy 2048-window calib and **71%**
with the new sequence/full default — well above the looping cliff (r* ≈ 0.65).
So 0.7 is the safest aggressive operating point for a compress-then-OPD attempt.

Output:
  - HF model dir at --save-dir, loadable by `AutoModelForCausalLM.from_pretrained`
    (verl actor) and `AutoTokenizer.from_pretrained` (verl tokenizer).
  - Metrics JSON at --metrics-json: {params, c4_ppl, math500_acc, ...}.

Usage (single H100, in the `verl` conda env):
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/yequan/huggingface PYTHONPATH=src:verl \
    /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/compress_opd/math/build_svd_nystrom_student.py \
        --save-dir /data/yequan/compress_opd/students/svd_nystrom_r0.7 \
        --metrics-json scripts/compress_opd/math/results/svd_nystrom_r0.7_pre_opd.json
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "verl"))
# Reuse the A/B/D mechanism-fix drivers' shared eval/calib helpers — they bake
# in the operating-point protocol (last-layer skip, MATH-500 + C4 PPL contract,
# OpenThought3 sequence/full calibration default).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "reasoning_aware_compress"))

import compress_common as cc  # noqa: E402

from compress.calibration import (  # noqa: E402
    collect_both_covariances_from_loader,
    collect_covariances_from_loader,
    collect_mixed_statistics,
    collect_nystrom_combined_statistics,
)
from compress.compress_model import MethodSpec  # noqa: E402
from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model  # noqa: E402
from compress.svd.svd_linear import SVDCompressedLinear  # noqa: E402
from compress.structured.nystrom import (  # noqa: E402
    nystrom_compress_model,
    nystrom_combined_compress_model,
)
import torch.nn as nn  # noqa: E402


def merge_svd_to_dense(model) -> int:
    """Materialize every SVDCompressedLinear (attn q/k/v/o_proj after SVD-V2)
    back into a STANDARD nn.Linear by computing the exact dense weight.

    SVDCompressedLinear.forward is  y = (x @ V_r) [* svd_s] @ U_r + bias, with
    V_r:(d_in, rank), U_r:(rank, d_out) -> equivalent to nn.Linear with
    W.T = V_r @ (diag(svd_s) @) U_r, i.e. W = (V_r @ Us).T of shape (d_out, d_in).
    This is an EXACT reparametrization (identical forward), so the model's
    function — and therefore its eval — is unchanged; it just becomes a stock
    dense checkpoint that AutoModelForCausalLM/verl can load. Returns #merged."""
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, SVDCompressedLinear)]
    for name, mod in targets:
        V_r = mod.V_r.data.float()                 # (d_in, rank)
        U_r = mod.U_r.data.float()                 # (rank, d_out)
        if getattr(mod, "svd_s", None) is not None:
            U_r = mod.svd_s.data.float().unsqueeze(1) * U_r
        W = (V_r @ U_r).t().contiguous()           # (d_out, d_in)
        d_out, d_in = W.shape
        lin = nn.Linear(d_in, d_out, bias=mod.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(W.to(mod.U_r.dtype))
            if mod.bias is not None:
                lin.bias.copy_(mod.bias.data)
        lin = lin.to(device=mod.U_r.device, dtype=mod.U_r.dtype)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[-1], lin)
    return len(targets)


def patch_uniform_intermediate(model) -> int:
    """Set config.intermediate_size to the (uniform) post-Nystrom MLP width so a
    stock Qwen3ForCausalLM rebuilds with the right shape on reload. Requires the
    compression to be uniform across layers (build with --skip-last-layers 0)."""
    sizes = {m.gate_proj.weight.shape[0] for m in model.modules()
             if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj")}
    if len(sizes) != 1:
        raise ValueError(
            f"non-uniform MLP intermediate sizes {sorted(sizes)} cannot be saved as "
            f"a stock Qwen3 (one config.intermediate_size). Rebuild with "
            f"--skip-last-layers 0 for a stock-dense save.")
    k = sizes.pop()
    model.config.intermediate_size = int(k)
    return int(k)


def zero_pad_mlps(model) -> "tuple[int, int]":
    """Zero-pad every shrunk MLP up to config.intermediate_size so the model is a
    STOCK Qwen3 (uniform width) with the padded columns EXACTLY zero. Unlike
    patch_uniform_intermediate (which shrinks the config and requires uniform/
    skip_last=0 compression), this KEEPS the last layer dense (skip_last=1) by
    padding the shrunk layers instead. The padding is preserved (frozen at zero)
    during OPD via verl's SPARSEGPT_PRESERVE_MASK=1, so the model stays effectively
    compressed while remaining vLLM/from_pretrained-loadable. Returns (n_padded, I)."""
    import torch.nn as nn
    I = int(model.config.intermediate_size)
    n_padded = 0
    for m in model.modules():
        if not (hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj")):
            continue
        w = m.gate_proj.weight.shape[0]
        if w >= I:
            continue  # already full width (e.g. the protected last layer)
        H_in = m.gate_proj.weight.shape[1]      # hidden size (gate/up in_features)
        H_out = m.down_proj.weight.shape[0]     # hidden size (down out_features)
        dev, dt = m.gate_proj.weight.device, m.gate_proj.weight.dtype
        for name in ("gate_proj", "up_proj"):   # [w, H] -> [I, H], zero rows w:I
            old = getattr(m, name)
            new = nn.Linear(H_in, I, bias=old.bias is not None, device=dev, dtype=dt)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:w].copy_(old.weight)
                if old.bias is not None:
                    new.bias.zero_(); new.bias[:w].copy_(old.bias)
            setattr(m, name, new)
        old = m.down_proj                        # [H, w] -> [H, I], zero cols w:I
        new = nn.Linear(I, H_out, bias=old.bias is not None, device=dev, dtype=dt)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :w].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        m.down_proj = new
        n_padded += 1
    return n_padded, I


def compress_svd_nystrom(model, attn_fwd_cov, mlp_stats, *, ratio, device, protect,
                         objective="forward", attn_bwd_cov=None):
    """SVD-V2 on self_attn + Nystrom on mlp; layers in `protect` (last-N decoder
    layers) are left dense by dropping their statistics (a layer with no entry
    is silently skipped by the compress core — same pattern as every A/B/D
    driver and `ratio_sweep_trace.py`).

    objective="forward" (wiki D0, repo default — input-whitening only):
        - attn: svd_llm_v2_compress_model(..., objective="forward")
        - mlp:  nystrom_compress_model(forward C_sigma only)
        Expects attn_fwd_cov = forward attn input covariances,
                mlp_stats    = forward Nystrom C_sigma dict.

    objective="combined" (wiki D2 — bilateral / OBD-LLM-style):
        - attn: svd_llm_v2_compress_model(..., objective="combined", backward_covariances=...)
        - mlp:  nystrom_combined_compress_model(joint K = Cf^{1/2} Cb Cf^{1/2})
        Expects attn_fwd_cov = forward attn input cov, attn_bwd_cov = backward
                attn output-grad cov, mlp_stats = dict[down_proj -> (C_f, C_b)]
                from collect_nystrom_combined_statistics.
    """
    attn_fwd = cc.drop_protected_stats(
        {k: v for k, v in attn_fwd_cov.items() if ".self_attn." in k}, protect)
    mlp = cc.drop_protected_stats(mlp_stats, protect)

    logger.disable("compress")
    try:
        if objective == "forward":
            nystrom_compress_model(
                model, {k: v.clone() for k, v in mlp.items()},
                sparsity=1.0 - ratio, skip_layers=("lm_head",), device=device,
            )
            svd_llm_v2_compress_model(
                model, {k: v.clone() for k, v in attn_fwd.items()},
                compression_ratio=ratio, skip_layers=("lm_head",),
                device=device, objective="forward",
            )
        elif objective == "combined":
            if attn_bwd_cov is None:
                raise ValueError("objective='combined' requires attn_bwd_cov")
            attn_bwd = cc.drop_protected_stats(
                {k: v for k, v in attn_bwd_cov.items() if ".self_attn." in k}, protect)
            # mlp_stats here is the {down_proj -> (C_f, C_b)} dict from
            # collect_nystrom_combined_statistics; clone the pair tensors.
            mlp_cloned = {k: (cf.clone(), cb.clone()) for k, (cf, cb) in mlp.items()}
            nystrom_combined_compress_model(
                model, mlp_cloned,
                sparsity=1.0 - ratio, skip_layers=("lm_head",), device=device,
            )
            svd_llm_v2_compress_model(
                model, {k: v.clone() for k, v in attn_fwd.items()},
                compression_ratio=ratio, skip_layers=("lm_head",),
                device=device, objective="combined",
                backward_covariances={k: v.clone() for k, v in attn_bwd.items()},
            )
        else:
            raise ValueError(f"objective must be 'forward' or 'combined', got {objective!r}")
    finally:
        logger.enable("compress")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cc.DEFAULT_MODEL,
                    help="HF model id or local path (default Qwen/Qwen3-4B)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(cc.DTYPE_MAP))
    ap.add_argument("--ratio", type=float, default=0.7,
                    help="retain ratio (wiki best safe aggressive op = 0.7)")
    ap.add_argument("--skip-last-layers", type=int, default=1,
                    help="leave the top-N decoder layers DENSE (protocol: 1)")
    ap.add_argument("--objective", default="forward",
                    choices=["forward", "combined"],
                    help="forward = wiki D0 (input-whitening only). combined = "
                         "wiki D2 (bilateral, fwd + CE-backward whitening) for "
                         "both attn (svd_llm_v2 objective=combined) and mlp "
                         "(nystrom_combined joint K = Cf^{1/2} Cb Cf^{1/2}).")
    # Calibration knobs — defaults are the 2026-06-04 productionized recipe.
    ap.add_argument("--calib", default="openthought3",
                    choices=["openthought3", "c4"])
    ap.add_argument("--calib-num-seqs", type=int, default=128)
    ap.add_argument("--calib-max-length", type=int, default=2048,
                    help="only used by the legacy window2048 escape hatch")
    ap.add_argument("--calib-batch-size", type=int, default=2)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--calib-length", default="full",
                    choices=["full", "lt2048", "window2048"],
                    help="full = new sequence/full default (wiki §5); window2048 "
                         "reproduces the pre-2026-06-04 baseline")
    ap.add_argument("--calib-reweight", default="sequence",
                    choices=["sequence", "token"],
                    help="sequence = new default (every conversation weighted "
                         "equally); token = legacy, pair with window2048")
    # Eval contract — matches every other A/B/D driver.
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--math-limit", type=int, default=200,
                    help="MATH-500 problems to grade (200 = headline contract)")
    ap.add_argument("--math-max-new-tokens", type=int, default=2048)
    ap.add_argument("--math-batch-size", type=int, default=16)
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true",
                    help="skip C4 PPL (the only metric needing internet/C4) — "
                         "use on offline compute nodes")
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--save-stock-dense", action="store_true",
                    help="before save, merge SVD-V2 attn factors (U_r/V_r) into "
                         "dense nn.Linear weights and patch config.intermediate_size "
                         "to the uniform Nystrom width, so the checkpoint loads as a "
                         "stock Qwen3ForCausalLM (required for the verl actor). Use "
                         "with --skip-last-layers 0 so the MLP width is uniform.")
    ap.add_argument("--save-zero-padded", action="store_true",
                    help="before save, merge SVD-V2 attn factors to dense AND zero-pad "
                         "the shrunk MLPs up to config.intermediate_size, keeping the "
                         "last layer dense (skip_last=1). Loads as a stock Qwen3 (vLLM/"
                         "verl); the zero padding is frozen during OPD via "
                         "SPARSEGPT_PRESERVE_MASK=1, so it stays effectively compressed. "
                         "This is the SAME compression as the compress_sft run.")
    ap.add_argument("--save-dir", required=True,
                    help="HF model dir for the compressed student (verl actor input)")
    ap.add_argument("--metrics-json", default=None)
    args = ap.parse_args()

    dtype = cc.DTYPE_MAP[args.dtype]
    device = args.device
    tokenizer = cc.load_tokenizer(args.model)

    logger.info(f"Loading base model {args.model} (CPU master copy)...")
    base_cpu = cc.load_model(args.model, dtype, "cpu")
    base_cpu.eval()
    protect = cc.protected_layer_set(base_cpu, args.skip_last_layers)
    n_layers = cc.num_hidden_layers(base_cpu)
    logger.info(f"skip_last_layers={args.skip_last_layers}: protecting decoder "
                f"layers {sorted(protect)} of {n_layers} (left dense)")

    t0 = time.time()

    # ---- statistics: calibration covariances ------------------------------- #
    # Both collectors default reweight="sequence" (2026-06-04 productionized).
    # `build_calib_loader` defaults length="full" so we get the new mask-aware
    # full-conversation loader instead of the legacy 2048-window packer.
    #
    # objective=forward:  forward attn input cov + MLP Nystrom forward C_sigma.
    # objective=combined: both fwd+bwd attn covariances (CE backward) +
    #                     MLP joint Nystrom (C_f, C_b) statistics.
    logger.info(
        f"Collecting calibration statistics (objective={args.objective} "
        f"calib={args.calib} length={args.calib_length} "
        f"reweight={args.calib_reweight} n_seqs={args.calib_num_seqs})..."
    )
    m0 = copy.deepcopy(base_cpu).to(device)

    def _calib():
        # cap = DROP (never truncate) traces longer than it. forward caps at 16384;
        # combined runs a full-length backward (cc forces bs=1 on the full path) —
        # caps at 10240 (16384 and a real 12386-token trace OOM an 80GB GPU).
        msl = 10240 if args.objective == "combined" else 16384
        return cc.build_calib_loader(
            args.calib, tokenizer, num_seqs=args.calib_num_seqs,
            max_length=args.calib_max_length, batch_size=args.calib_batch_size,
            seed=args.calib_seed, length=args.calib_length, max_seq_len=msl,
        )

    attn_bwd_cov = None
    if args.objective == "forward":
        # MLP Nystrom forward stats (only down_proj input cov, C_sigma).
        mlp_stats = collect_mixed_statistics(
            m0, _calib(), MethodSpec(attn=None, mlp="nystrom"),
            device=device, skip_layers=("lm_head",),
            reweight=args.calib_reweight,
        )
        # Attention forward input covariance (drives SVD-V2 whitening).
        attn_fwd_cov = collect_covariances_from_loader(
            m0, _calib(), device=device, skip_layers=("lm_head",),
            reweight=args.calib_reweight,
        )
    else:  # combined: fwd+bwd attn covariances + joint Nystrom (C_f, C_b)
        # MLP joint Nystrom statistics (forward C_f + backward C_b from
        # down_proj input gradient). Uses default CE-on-shifted-self-labels
        # backward (loss_fn=None).
        mlp_stats = collect_nystrom_combined_statistics(
            m0, _calib(), device=device, skip_layers=("lm_head",),
            reweight=args.calib_reweight,
        )
        # Bilateral attention covariances (forward input + CE-backward
        # output-grad), one pass.
        attn_fwd_cov, attn_bwd_cov = collect_both_covariances_from_loader(
            m0, _calib(), device=device, skip_layers=("lm_head",),
            reweight=args.calib_reweight,
        )
    del m0
    torch.cuda.empty_cache()

    # ---- compress ----------------------------------------------------------- #
    logger.info(
        f"Compressing: SVD-V2 (attn, objective={args.objective}) + "
        f"{'Nystrom-combined' if args.objective == 'combined' else 'Nystrom'} (mlp) "
        f"@ retain {args.ratio}"
    )
    model = copy.deepcopy(base_cpu).to(device)
    compress_svd_nystrom(
        model, attn_fwd_cov, mlp_stats,
        ratio=args.ratio, device=device, protect=protect,
        objective=args.objective, attn_bwd_cov=attn_bwd_cov,
    )
    del base_cpu
    torch.cuda.empty_cache()

    total, nonzero = cc.count_params(model)
    logger.info(f"Params: total={total / 1e9:.3f}B  nonzero={nonzero / 1e9:.3f}B "
                f"(zero rate {(1 - nonzero / total) * 100:.1f}%)")

    # ---- eval (MATH-500 + C4 PPL — the standing contract) ------------------- #
    metrics = cc.eval_cell(
        model, tokenizer, device=device, math_limit=args.math_limit,
        math_max_new_tokens=args.math_max_new_tokens,
        math_batch_size=args.math_batch_size, ppl_seqlen=args.ppl_seqlen,
        skip_math=args.skip_math, skip_ppl=args.skip_ppl,
    )
    macc = "-" if metrics["math500_acc"] is None else f"{metrics['math500_acc'] * 100:.2f}%"
    cppl = "-" if metrics["c4_ppl"] is None else f"{metrics['c4_ppl']:.4f}"
    logger.info(f"PRE-OPD: nz={metrics['params_nonzero_B']:.3f}B "
                f"c4_ppl={cppl} math={macc}")

    # ---- save compressed HF dir (verl actor input) -------------------------- #
    if not args.skip_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_stock_dense:
            # SVD-V2 leaves attn as factorized SVDCompressedLinear (U_r/V_r) and
            # Nystrom shrinks the MLP width — neither is a stock Qwen3 layer, so a
            # naive save is NOT loadable by AutoModelForCausalLM. Merge the attn
            # factors back to dense nn.Linear (exact, no quality change) and patch
            # config.intermediate_size to the uniform Nystrom width.
            n_merged = merge_svd_to_dense(model)
            k = patch_uniform_intermediate(model)
            logger.info(f"stock-dense save: merged {n_merged} SVD attn modules; "
                        f"config.intermediate_size -> {k}")
        elif args.save_zero_padded:
            # Keep the last layer dense (skip_last=1): merge attn to dense, then
            # zero-pad the shrunk MLPs up to config.intermediate_size so the model is
            # a stock Qwen3 with frozen-zero padding (SPARSEGPT_PRESERVE_MASK=1 in OPD).
            n_merged = merge_svd_to_dense(model)
            n_pad, I = zero_pad_mlps(model)
            logger.info(f"zero-padded save: merged {n_merged} SVD attn modules; "
                        f"zero-padded {n_pad} shrunk MLPs -> uniform {I}")
        logger.info(f"Saving HF weights -> {save_dir}")
        model.save_pretrained(str(save_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(save_dir))

    method = ("svd_v2_attn+nystrom_combined_mlp" if args.objective == "combined"
              else "svd_v2_attn+nystrom_mlp")
    out = {
        "config": vars(args),
        "method": method,
        "objective": args.objective,
        "retain_ratio": args.ratio,
        "protected_layers": sorted(protect),
        "elapsed_s": time.time() - t0,
        **metrics,
    }
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote metrics -> {args.metrics_json}")
    print("==== METRICS ====")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
