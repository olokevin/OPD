"""Standalone smoke test for the svd_nystrom in-process compress path, bypassing the
full SFT/DeepSpeed loop. Validates: triplet discovery, attn SVD + MLP/expert Nystrom,
config.intermediate_size update, trainability flags, and the CompressSaveCallback
materialize+save+reload roundtrip.

Run in the `sft` conda env:
  CUDA_VISIBLE_DEVICES=5 HF_HOME=/data/yequan/huggingface \\
    /home/yequan/miniconda3/envs/sft/bin/python \\
    LlamaFactory && python scripts/.../_smoke_svd_nystrom.py --model Qwen/Qwen3-4B-Base \\
      --objective forward --calib-num-seqs 8
(this script adds LlamaFactory/src to sys.path itself)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LlamaFactory" / "src"))
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--objective", choices=["forward", "combined"], default="forward")
    ap.add_argument("--calib-num-seqs", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--skip-last-layers", type=int, default=1)
    ap.add_argument("--save", action="store_true",
                    help="also test the materialize+save+reload roundtrip")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.model.compress_setup import init_compress_model
    from compress.structured.nystrom import find_mlp_triplets
    from compress.svd.svd_linear import SVDCompressedLinear

    # The 2GB OpenThought3 jsonl lives only in the main checkout's working tree
    # (gitignored). Use the absolute path the SFT yamls also point at.
    calib = Path(
        "/home/yequan/Project/compression/OPD/datasets/OpenThought3-Qwen3-4B/data/train.jsonl"
    )
    calib_mode = "svd_v2_combined" if args.objective == "combined" else "svd_v2"

    cfg0 = AutoConfig.from_pretrained(args.model)
    orig_inter = cfg0.intermediate_size
    print(f"[smoke] model={args.model} orig intermediate_size={orig_inter} "
          f"layers={cfg0.num_hidden_layers}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pre_triplets = find_mlp_triplets(model, ("lm_head",))
    print(f"[smoke] find_mlp_triplets BEFORE compression: {len(pre_triplets)} triplets")

    fa = FinetuningArguments(
        finetuning_type="svd_nystrom", calib_mode=calib_mode,
        compression_ratio=args.ratio, skip_last_layers=args.skip_last_layers,
        calib_source="traces", calib_traces_path=str(calib),
        calib_num_seqs=args.calib_num_seqs,
    )

    class _MA:  # minimal model_args shim
        model_name_or_path = args.model
        trust_remote_code = True
    model_args = _MA()

    model = init_compress_model(cfg0, model, model_args, fa, is_trainable=True)

    # --- invariant checks ---
    n_svd = sum(1 for m in model.modules() if isinstance(m, SVDCompressedLinear))
    svd_trainable = all(
        m.U_r.requires_grad and m.V_r.requires_grad
        for m in model.modules() if isinstance(m, SVDCompressedLinear)
    )
    post_triplets = find_mlp_triplets(model, ("lm_head",))
    # a compressed gate_proj out_features should equal ceil(ratio*orig_inter)
    import math
    exp_k = math.ceil(args.ratio * orig_inter)
    mods = dict(model.named_modules())
    widths = {int(mods[p].gate_proj.weight.shape[0]) for (p, _g, _u, _d) in post_triplets}
    mlp_trainable = all(
        mods[p].gate_proj.weight.requires_grad and mods[p].down_proj.weight.requires_grad
        for (p, _g, _u, _d) in post_triplets
    )
    new_inter = model.config.intermediate_size

    print(f"[smoke] SVD attn modules: {n_svd} (trainable={svd_trainable})")
    print(f"[smoke] post triplets: {len(post_triplets)}  widths={sorted(widths)} "
          f"(expected k={exp_k})  mlp_trainable={mlp_trainable}")
    print(f"[smoke] config.intermediate_size {orig_inter} -> {new_inter}")

    assert n_svd > 0, "no SVDCompressedLinear attn modules created"
    assert svd_trainable, "SVD factors not trainable"
    assert mlp_trainable, "Nystrom MLP not trainable"
    assert widths == {exp_k}, f"non-uniform/unexpected MLP width {widths} != {{{exp_k}}}"
    assert new_inter == exp_k, f"config.intermediate_size {new_inter} != {exp_k}"

    if args.save:
        from llamafactory.train.callbacks import _build_materialized_state_dict
        sd = _build_materialized_state_dict(model)
        with tempfile.TemporaryDirectory() as d:
            from transformers import AutoModelForCausalLM as AM
            peer = AM.from_config(model.config)
            missing, unexpected = peer.load_state_dict(sd, strict=False)
            # tied lm_head + non-persistent rotary buffers are expected and harmless
            # (HF recomputes rotary_emb.inv_freq; lm_head ties to embed_tokens).
            def _ignorable(k):
                return ("lm_head" in k or "rotary_emb" in k or "inv_freq" in k)
            real_missing = [k for k in missing if not _ignorable(k)]
            real_unexpected = [k for k in unexpected if not _ignorable(k)]
            print(f"[smoke] reload: {len(real_missing)} missing, "
                  f"{len(real_unexpected)} unexpected (ignoring lm_head/rotary)")
            assert not real_missing, f"missing keys on reload: {real_missing[:5]}"
            assert not real_unexpected, f"unexpected keys: {real_unexpected[:5]}"
            peer.save_pretrained(d)
            print(f"[smoke] save_pretrained OK -> {d} (files: "
                  f"{sorted(p.name for p in Path(d).iterdir())[:6]})")

    print("[smoke] ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
