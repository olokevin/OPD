"""LoRA adapter — delegates to peft.get_peft_model.

Returns None from export_for_vllm so the sharding manager keeps using the
existing collect_lora_params + TensorLoRARequest path.
"""
from __future__ import annotations

import os
from typing import Optional

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from verl.workers.peft.base import PEFTAdapter


_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def _vllm_max_lora_rank(rank: int) -> int:
    for r in _VLLM_LORA_RANKS:
        if rank <= r:
            return r
    raise ValueError(f"lora rank {rank} exceeds vLLM max {_VLLM_LORA_RANKS[-1]}")


def _target_modules_to_peft(spec):
    if isinstance(spec, str):
        if spec == "all":
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        if spec == "mlp":
            return ["gate_proj", "up_proj", "down_proj"]
        if spec == "attn":
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        # Pass through literals like "all-linear" — peft handles them.
        return spec
    return list(spec)


class LoRAAdapter(PEFTAdapter):
    mode = "lora"

    def __init__(self, peft_cfg, model_config=None):
        super().__init__(peft_cfg, model_config=model_config)
        self._peft_model: Optional[PeftModel] = None

    def apply(self, model, *, tokenizer, calib_loader_builder):
        adapter_path = self.peft_cfg.lora.adapter_path
        if adapter_path is not None:
            model.enable_input_require_grads()
            peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            pc = peft_model.peft_config["default"]
            if isinstance(pc.task_type, str):
                pc.task_type = TaskType.CAUSAL_LM
        else:
            model.enable_input_require_grads()
            lora_kwargs = dict(
                task_type=TaskType.CAUSAL_LM,
                r=self.peft_cfg.lora.rank,
                lora_alpha=self.peft_cfg.lora.alpha,
                lora_dropout=self.peft_cfg.lora.dropout,
                bias=self.peft_cfg.lora.bias,
                target_modules=_target_modules_to_peft(self.peft_cfg.target_modules),
            )
            peft_model = get_peft_model(model, LoraConfig(**lora_kwargs))
        self._peft_model = peft_model
        return peft_model

    def export_for_vllm(self, fsdp_module):
        # Fall through to verl's existing collect_lora_params + TensorLoRARequest path.
        return None

    def vllm_engine_kwargs(self):
        rank = self.peft_cfg.lora.rank
        if rank <= 0:
            return {}
        return {
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": _vllm_max_lora_rank(rank),
        }

    def peft_config(self):
        if self._peft_model is None:
            return None
        return self._peft_model.peft_config.get("default")

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # fsdp_module here is the PeftModel (or the unwrapped one if Task 7 already
        # extracted it from FSDP). Either way, save_pretrained writes the adapter.
        fsdp_module.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        return {
            "mode": "lora",
            "target_modules": self.peft_cfg.target_modules,
            "lora": {
                "rank": self.peft_cfg.lora.rank,
                "alpha": self.peft_cfg.lora.alpha,
                "dropout": self.peft_cfg.lora.dropout,
                "bias": self.peft_cfg.lora.bias,
            },
        }
