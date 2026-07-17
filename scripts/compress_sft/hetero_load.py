"""Load a compress_sft `-merged` checkpoint whose per-layer MLP width is
HETEROGENEOUS and therefore NOT loadable via vanilla
``AutoModelForCausalLM.from_pretrained``.

With ``skip_last_layers=1`` the svd_nystrom recipe leaves the last decoder layer
fully dense while Nystrom-shrinks every other layer's MLP, so the saved checkpoint
has e.g. layers 0..N-2 at ``gate/up/down_proj`` width ``round(ratio*I)`` and the
last layer at the full ``I`` — but ``config.intermediate_size`` is a single scalar
(``I``). ``from_pretrained`` builds every MLP at ``I`` and then fails with a shape
mismatch on the shrunk layers.

``load_compressed_merged`` instantiates the architecture from config, resizes each
decoder layer's ``mlp.{gate,up,down}_proj`` ``nn.Linear`` to the checkpoint's actual
shapes, then loads the state dict. (lm_head is tied to embed_tokens, so its weight is
legitimately absent from the checkpoint.)
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


def _load_state_dict(model_dir: str) -> "dict[str, torch.Tensor]":
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if shards:
        from safetensors.torch import load_file
        sd: "dict[str, torch.Tensor]" = {}
        for s in shards:
            sd.update(load_file(s))
        return sd
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"no .safetensors or pytorch_model.bin in {model_dir}")


def is_heterogeneous(model_dir: str) -> bool:
    """True if any decoder-layer MLP width differs from config.intermediate_size."""
    cfg = AutoConfig.from_pretrained(model_dir)
    sd = _load_state_dict(model_dir)
    for k, v in sd.items():
        if k.endswith("mlp.gate_proj.weight") and v.shape[0] != cfg.intermediate_size:
            return True
    return False


def load_compressed_merged(model_dir: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Rebuild + load a (possibly heterogeneous) compressed merged checkpoint.

    Falls back to plain from_pretrained when the checkpoint is homogeneous.
    """
    if not is_heterogeneous(model_dir):
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()

    cfg = AutoConfig.from_pretrained(model_dir)
    sd = _load_state_dict(model_dir)
    model = AutoModelForCausalLM.from_config(cfg)  # random init, full-width MLPs

    layers = model.model.layers
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            w_key = f"model.layers.{i}.mlp.{name}.weight"
            if w_key not in sd:
                continue
            out_f, in_f = sd[w_key].shape
            lin = getattr(mlp, name)
            if tuple(lin.weight.shape) != (out_f, in_f):
                setattr(mlp, name, nn.Linear(in_f, out_f, bias=lin.bias is not None))

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied -> legitimately absent
    if missing or unexpected:
        raise RuntimeError(
            f"hetero load mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    return model.to(device=device, dtype=dtype).eval()
