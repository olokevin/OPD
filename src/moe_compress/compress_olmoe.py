"""Compress OLMoE experts with one method, materialize a reloadable HF checkpoint,
and (optionally) eval it training-free. Mirrors scripts/compress_sft/build_svd_nystrom_student.py
but (a) targets EXPERTS only (attn + router frozen) and (b) dispatches to the
6-method registry in moe_compress.methods.

Run (verl env — OLMoE experts are per-Linear here; no fused-3D blocker):
  PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface CUDA_VISIBLE_DEVICES=1 \\
    /home/yequan/miniconda3/envs/verl/bin/python -m moe_compress.compress_olmoe \\
      --method svd_llm_v2 --retain 0.75 --seed 0 \\
      --save-dir /data/yequan/moe_compress/ckpts/svd_llm_v2_r0.75_s0 \\
      --metrics-json /data/yequan/moe_compress/metrics/svd_llm_v2_r0.75_s0.json --eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "verl"))

from moe_compress import methods as M           # noqa: E402
from moe_compress.budget import expert_param_stats, compute_budget, log_budget  # noqa: E402
from moe_compress.calib import build_standard_calib_loader, calib_coverage  # noqa: E402

MODEL_DEFAULT = "allenai/OLMoE-1B-7B-0924-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--method", required=True, choices=sorted(M.REGISTRY))
    ap.add_argument("--retain", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-corpus", default="openthoughts", choices=["openthoughts", "c4"])
    ap.add_argument("--calib-seqs", type=int, default=256)
    ap.add_argument("--calib-len", type=int, default=2048)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    logger.info(f"=== compress {args.method} retain={args.retain} seed={args.seed} ===")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    baseline = expert_param_stats(model)
    baseline["top_k"] = model.config.num_experts_per_tok
    logger.info(f"baseline expert params: {baseline['total']:,} "
                f"({baseline['total']/1e9:.3f}B, {baseline['per_layer_experts'][0]} experts/layer)")

    calib = build_standard_calib_loader(
        tok, corpus=args.calib_corpus, num_seqs=args.calib_seqs,
        max_len=args.calib_len, batch_size=4,
    )

    # routing-coverage sanity (per-expert methods need every expert routed >=1)
    if args.method in ("reap_drop", "slimqwen_merge", "hcsmoe_merge", "svd_llm_v2", "sparsegpt"):
        cov = calib_coverage(model, calib, device=args.device)
        logger.info(f"calib coverage: min_tokens/expert={cov['min']} "
                    f"dead={cov['n_dead']}/{cov['n_experts']} total_tokens={cov['total_tokens']:,}")
        if cov["n_dead"] > 0:
            logger.warning(f"{cov['n_dead']} experts NEVER routed in calib — "
                           f"their stats are undefined (increase --calib-seqs)")

    M.get(args.method)(model, retain=args.retain, calib_loader=calib,
                       seed=args.seed, device=args.device)

    # Sync config to the surviving expert geometry so the checkpoint reloads.
    # drop/merge change expert COUNT; weight-approx (svd) changes intermediate_size.
    # Both must be uniform across layers/experts for a single HF config to reload.
    counts = [len(model.model.layers[li].mlp.experts)
              for li in range(model.config.num_hidden_layers)]
    if len(set(counts)) == 1 and counts[0] != model.config.num_experts:
        new_ne = counts[0]
        new_topk = min(model.config.num_experts_per_tok, new_ne)
        logger.info(f"config sync: num_experts {model.config.num_experts}->{new_ne}, "
                    f"num_experts_per_tok {model.config.num_experts_per_tok}->{new_topk}")
        model.config.num_experts = new_ne
        model.config.num_experts_per_tok = new_topk
    elif len(set(counts)) > 1:
        logger.warning(f"non-uniform expert counts across layers: {sorted(set(counts))} "
                       f"— config.num_experts left as-is (reload may fail)")
    # intermediate_size: read every expert's gate_proj out-dim; sync if uniform.
    inters = set()
    for li in range(model.config.num_hidden_layers):
        for ex in model.model.layers[li].mlp.experts:
            inters.add(ex.gate_proj.out_features)
    if len(inters) == 1 and next(iter(inters)) != model.config.intermediate_size:
        new_inter = next(iter(inters))
        logger.info(f"config sync: intermediate_size {model.config.intermediate_size}->{new_inter}")
        model.config.intermediate_size = new_inter
    elif len(inters) > 1:
        logger.warning(f"non-uniform expert intermediate_size {sorted(inters)} "
                       f"— config.intermediate_size left as-is (reload may fail)")

    budget = compute_budget(model, baseline)
    # MoBE writes dense-but-low-rank weights, so nonzero-count budget reads ~1.0;
    # override with the achieved FACTOR retain it stashed on the model.
    if hasattr(model, "_mobe_factor_retain"):
        fr = float(model._mobe_factor_retain)
        budget["storage_retain"] = round(fr, 4)
        budget["active_retain"] = round(fr, 4)
        budget["budget_note"] = "factor-param retain (low-rank A@B factors), not nonzero count"
    log_budget(budget, tag=args.method)

    metrics = {
        "method": args.method, "family": M.FAMILIES[args.method],
        "retain_target": args.retain, "seed": args.seed,
        "calib": {"corpus": args.calib_corpus, "seqs": args.calib_seqs, "len": args.calib_len},
        "budget": budget, "compress_sec": round(time.time() - t0, 1),
    }

    if args.eval:
        from moe_compress.eval_tasks import eval_all
        elim = args.eval_limit if args.eval_limit and args.eval_limit > 0 else None
        metrics["step0"] = eval_all(model, tok, args.device, limit=elim)

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir, safe_serialization=True)
        tok.save_pretrained(args.save_dir)
        logger.info(f"saved compressed HF checkpoint -> {args.save_dir}")
        # reload sanity: ensure it loads back as a valid OLMoE
        del model
        torch.cuda.empty_cache()
        rl = AutoModelForCausalLM.from_pretrained(args.save_dir, dtype=torch.bfloat16,
                                                  trust_remote_code=True, low_cpu_mem_usage=True)
        rb = compute_budget(rl, baseline)
        if "budget_note" in budget:  # MoBE: on-disk weights are dense (low-rank), so
            # nonzero-count budget can't verify it; just confirm the model reloads.
            metrics["reload_ok"] = True
            logger.info(f"reload sanity: ok=True (MoBE stores dense low-rank weights; "
                        f"factor budget = {budget['storage_retain']}, reloaded shape OK)")
        else:
            metrics["reload_ok"] = abs(rb["storage_retain"] - budget["storage_retain"]) < 1e-3
            logger.info(f"reload sanity: ok={metrics['reload_ok']} "
                        f"(storage_retain {rb['storage_retain']})")

    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"metrics -> {args.metrics_json}")
    logger.info(f"=== done in {time.time()-t0:.0f}s ===\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
