"""C4-PPL comparison: forward-only Nystrom vs joint (combined) Nystrom MLP compression.

Compresses Llama-3-8B's MLP layers to a target retained-parameter fraction using
C4 calibration, under two methods:

  * ``nystrom``          — MoDeGPT Type-I forward-only kernel C_f = Z^T Z
  * ``nystrom_combined`` — joint fwd+bwd kernel K_joint = Cf^{1/2} Cb Cf^{1/2} + lam*I
                           (docs/plans/nystrom_combined.md)

and reports C4 validation perplexity for the dense baseline and both compressed
models. Attention layers are left dense in both cases so the comparison isolates
the MLP-core selection/reconstruction.

Run (verl env, single free GPU):

    CUDA_VISIBLE_DEVICES=1 \
    /home/yequan/miniconda3/envs/verl/bin/python \
        scripts/opd/math/compressed_opd/nystrom_combined_c4ppl.py \
        --retain 0.6 --n-calib 128 --seqlen 2048
"""

import argparse
import sys

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")

from compress.compress_model import compress_model_with_loader
from compress.loaders import build_c4_calib_loader
from compress.ppl_eval import evaluate_model_ppl

LLAMA3_8B = (
    "/data/yequan/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/"
    "snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
)


def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=LLAMA3_8B)
    ap.add_argument("--retain", type=float, default=0.6,
                    help="Fraction of MLP hidden dim retained (0.6 = 60%).")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--calib-seed", type=int, default=3)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-dense", action="store_true",
                    help="Skip the dense baseline PPL (faster re-runs).")
    ap.add_argument("--methods", default="nystrom,nystrom_combined",
                    help="Comma-separated subset of {nystrom,nystrom_combined} to run.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="Enable gradient checkpointing during the backward "
                         "calibration pass (bounds memory for nystrom_combined).")
    args = ap.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Building C4 calib loader: n={args.n_calib}, seqlen={args.seqlen}")
    calib_loader = build_c4_calib_loader(
        tokenizer, num_seqs=args.n_calib, max_length=args.seqlen,
        batch_size=args.batch_size, seed=args.calib_seed,
    )
    # Materialize batches once so both methods see identical calibration data.
    calib_batches = list(calib_loader)
    logger.info(f"Materialized {len(calib_batches)} calib batches")

    results = {}

    # --- Dense baseline -------------------------------------------------
    if not args.skip_dense:
        logger.info("=== Dense baseline ===")
        model = load_model(args.model_path, device)
        ppl = evaluate_model_ppl(
            model, tokenizer, seqlen=args.ppl_seqlen, datasets=("c4",), device=device,
        )
        results["dense"] = ppl["c4"]
        logger.info(f"dense C4 PPL = {ppl['c4']:.4f}")
        del model
        torch.cuda.empty_cache()

    # --- Compressed models ---------------------------------------------
    for method in methods:
        logger.info(f"=== {method} (retain {args.retain:.0%} MLP) ===")
        model = load_model(args.model_path, device)
        if args.grad_checkpointing and method == "nystrom_combined":
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            logger.info("Gradient checkpointing enabled for backward calib pass")
        spec = f'{{"attn": null, "mlp": "{method}"}}'
        compress_model_with_loader(
            model,
            calib_batches,
            compression_ratio=args.retain,
            method=spec,
            device=device,
            skip_layers=("lm_head",),
        )
        if args.grad_checkpointing and method == "nystrom_combined":
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
        model.to(device)
        model.eval()
        ppl = evaluate_model_ppl(
            model, tokenizer, seqlen=args.ppl_seqlen, datasets=("c4",), device=device,
        )
        results[method] = ppl["c4"]
        logger.info(f"{method} C4 PPL = {ppl['c4']:.4f}")
        del model
        torch.cuda.empty_cache()

    # --- Report ---------------------------------------------------------
    logger.info("================ SUMMARY ================")
    logger.info(f"Llama-3-8B, MLP retain={args.retain:.0%}, C4 calib "
                f"(n={args.n_calib}, seqlen={args.seqlen}), C4-val PPL:")
    for k in ("dense", "nystrom", "nystrom_combined"):
        if k in results:
            logger.info(f"  {k:18s}: {results[k]:.4f}")
    if "nystrom" in results and "nystrom_combined" in results:
        gap = results["nystrom_combined"] - results["nystrom"]
        rel = gap / results["nystrom"] * 100.0
        logger.info(f"  gap (combined - nystrom): {gap:+.4f} ({rel:+.2f}%)")

    print("RESULTS_JSON", {k: round(v, 4) for k, v in results.items()})


if __name__ == "__main__":
    main()
