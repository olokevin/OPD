"""Rebuild a full HF checkpoint (and delta statistics) from an ES coefficient blob.

The ES trainer only stores the *coefficients* it optimises (`es_coef_best.pt`), not a
model: for the structured modes the frozen half of the factorisation (`A`, the calib
basis `v`, the sparse index set) is re-derivable from the base weights, and for
`dense`/`iso` the blob already is the full master.  To evaluate a finished ES arm on
anything other than the in-trainer MATH-500 loop we need the weights back.

Reconstruction reuses the **exact** code path the training worker used
(`StructuredESMixin.init_es_state` + `es_restore`) against a stub that presents the
base model in vLLM's *fused* layout (`qkv_proj`, `gate_up_proj`), so nothing about the
factorisation, the block sizes or the SVD convention can drift.  The fused weights are
then un-fused back into HF names and written out as a normal checkpoint.

Usage:
    python scripts/es/materialize_es_ckpt.py \
        --coef /data/yequan/es/ES-q2p5-7b/<run>/es_train_*/es_coef_best.pt \
        --out  /data/yequan/es/materialized/<run> \
        [--stats-only]
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "verl"))

from verl.workers.rollout.vllm_rollout.es_worker_extension import (  # noqa: E402
    StructuredESMixin,
    _es_target,
)

DEFAULT_BASE = "/data/yequan/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e"


# --------------------------------------------------------------------------- stub
class _StubModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())


class _Materializer(StructuredESMixin):
    """The training worker's ES state machine, without vLLM."""

    def __init__(self, params, hf_cfg):
        self.model_runner = SimpleNamespace(
            model=_StubModel(params),
            model_config=SimpleNamespace(hf_config=hf_cfg),
        )


# ------------------------------------------------------------------- fuse / unfuse
def load_fused(base_dir, device, hf_cfg):
    """HF state dict -> vLLM's fused parameter layout (+ the recipe to invert it)."""
    from safetensors.torch import load_file

    sd = {}
    index = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        sd.update(load_file(os.path.join(base_dir, shard)))

    hd = hf_cfg.hidden_size // hf_cfg.num_attention_heads
    q, kv = hf_cfg.num_attention_heads * hd, hf_cfg.num_key_value_heads * hd
    inter = hf_cfg.intermediate_size

    fused, recipe = {}, []          # recipe: (fused_name, [(hf_name, size), ...])
    for name, t in sd.items():
        t = t.to(device)
        if ".self_attn.q_proj." in name or ".mlp.gate_proj." in name:
            continue                # folded into the fused tensor below
        if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
            continue
        if ".mlp.up_proj." in name:
            continue
        fused[name] = t
        recipe.append((name, [(name, t.shape[0])]))

    for i in range(hf_cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for suffix, parts, sizes in (
            (".self_attn.qkv_proj", ["q_proj", "k_proj", "v_proj"], [q, kv, kv]),
            (".mlp.gate_up_proj", ["gate_proj", "up_proj"], [inter, inter]),
        ):
            stem = "self_attn" if "self_attn" in suffix else "mlp"
            for kind in ("weight", "bias"):
                srcs = [f"{p}.{stem}.{x}.{kind}" for x in parts]
                if not all(s in sd for s in srcs):
                    continue
                fused[f"{p}{suffix}.{kind}"] = torch.cat(
                    [sd[s].to(device) for s in srcs], dim=0
                )
                recipe.append((f"{p}{suffix}.{kind}", list(zip(srcs, sizes))))
    return fused, recipe


def unfuse(fused, recipe):
    out = {}
    for fname, parts in recipe:
        t = fused[fname]
        if len(parts) == 1:
            out[parts[0][0]] = t
            continue
        off = 0
        for hf_name, size in parts:
            out[hf_name] = t[off : off + size].contiguous()
            off += size
        assert off == t.shape[0], (fname, off, t.shape)
    return out


# ------------------------------------------------------------------------- deltas
_GROUPS = (
    ("q_proj", "Q"), ("k_proj", "K"), ("v_proj", "V"), ("o_proj", "WO"),
    ("gate_proj", "MLP"), ("up_proj", "MLP"), ("down_proj", "MLP"),
    ("layernorm", "LayerNorm"), ("model.norm", "LayerNorm"),
    ("embed_tokens", "Embed"), ("lm_head", "Embed"),
)


def _group(name):
    for key, g in _GROUPS:
        if key in name:
            return g
    return "other"


def delta_stats(base_hf, new_hf, tau=1e-6):
    """Frobenius drift + update sparsity, in the paper's (arXiv:2601.20861) terms."""
    per_tensor, num, den, n_below, n_zero, n_tot = {}, 0.0, 0.0, 0, 0, 0
    for name, w0 in base_hf.items():
        w1 = new_hf[name]
        d = w1.float() - w0.to(w1.device).float()
        fro_d = float(d.norm())
        fro_0 = float(w0.float().norm())
        below = int((d.abs() < tau).sum())
        # bf16 has ~1.6e-4 ULP at |w|~0.02, so `sparsity(tau=1e-6)` mostly counts
        # coordinates that moved less than one ULP.  `exact_zero` is the honest
        # "this coordinate did not change at all in the stored dtype" fraction.
        zero = int((d == 0).sum())
        per_tensor[name] = {
            "group": _group(name),
            "layer": int(name.split(".")[2]) if name.startswith("model.layers.") else -1,
            "numel": d.numel(),
            "fro_delta": fro_d,
            "fro_base": fro_0,
            "rel": fro_d / max(fro_0, 1e-12),
            "sparsity": below / d.numel(),
            "exact_zero": zero / d.numel(),
        }
        num += fro_d ** 2
        den += fro_0 ** 2
        n_below += below
        n_zero += zero
        n_tot += d.numel()
        del d
    return {
        "tau": tau,
        "global_fro_delta": num ** 0.5,
        "global_fro_base": den ** 0.5,
        "global_rel": (num ** 0.5) / (den ** 0.5),
        "global_sparsity": n_below / n_tot,
        "global_exact_zero": n_zero / n_tot,
        "per_tensor": per_tensor,
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--tau", type=float, default=1e-6)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    hf_cfg = AutoConfig.from_pretrained(args.base)
    print(f"[mat] loading base {args.base}", flush=True)
    fused, recipe = load_fused(args.base, args.device, hf_cfg)
    # keep the reference copy on CPU: `dense`/`iso` allocate a full fp32 master on GPU
    base_hf = {k: v.to("cpu").clone() for k, v in unfuse(fused, recipe).items()}

    blob = torch.load(args.coef, map_location="cpu", weights_only=False)
    mode, cfg = blob["mode"], dict(blob.get("cfg") or {})
    print(f"[mat] mode={mode} cfg={cfg} layers={len(blob['layers'])}", flush=True)

    mz = _Materializer(fused, hf_cfg)
    info = mz.init_es_state(mode, cfg)
    print(f"[mat] {info}", flush=True)

    missing = set(mz._es) - set(blob["layers"])
    extra = set(blob["layers"]) - set(mz._es)
    assert not missing and not extra, f"state mismatch: missing={list(missing)[:3]} extra={list(extra)[:3]}"

    for name, st in mz._es.items():
        tgt = _es_target(st)
        tgt.copy_(blob["layers"][name].to(tgt.device, tgt.dtype))
    del blob
    mz.es_restore()                                  # W <- P(coef), the trained weights

    new_hf = unfuse(fused, recipe)
    os.makedirs(args.out, exist_ok=True)

    print("[mat] computing delta statistics", flush=True)
    stats = delta_stats(base_hf, new_hf, tau=args.tau)
    stats["mode"], stats["coef"] = mode, args.coef
    with open(os.path.join(args.out, "delta_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[mat] ||dW||_F/||W||_F = {stats['global_rel']:.4e}   "
          f"sparsity(<{args.tau:g}) = {100*stats['global_sparsity']:.2f}%   "
          f"untouched = {100*stats['global_exact_zero']:.2f}%", flush=True)

    if args.stats_only:
        return

    from safetensors.torch import save_file

    print(f"[mat] writing HF checkpoint -> {args.out}", flush=True)
    sd = {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in new_hf.items()}
    save_file(sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    hf_cfg.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    gen_cfg = os.path.join(args.base, "generation_config.json")
    if os.path.exists(gen_cfg):
        import shutil
        shutil.copy(gen_cfg, args.out)
    print("[mat] done", flush=True)


if __name__ == "__main__":
    main()
