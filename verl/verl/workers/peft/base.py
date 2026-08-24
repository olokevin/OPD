"""PEFTAdapter ABC + NullAdapter.

Adapters wrap one PEFT mode (LoRA / QLoRA / BlockTT / SVD / none) behind a
uniform interface so the actor worker and sharding manager have a single
dispatch point.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch.nn as nn


class PEFTAdapter(ABC):
    mode: str = "abstract"

    def __init__(self, peft_cfg, model_config=None):
        self.peft_cfg = peft_cfg
        self.model_config = model_config

    @classmethod
    def from_config(
        cls,
        peft_cfg,
        *,
        model_config,
        teacher_model_path: Optional[str] = None,
    ) -> "PEFTAdapter":
        # Importing here avoids a circular import at module-load time.
        from verl.workers.peft.lora import LoRAAdapter
        from verl.workers.peft.qlora import QLoRAAdapter
        from verl.workers.peft.blocktt import BlockTTAdapter
        from verl.workers.peft.svd import SVDAdapter
        from verl.workers.peft.iso import IsoAdapter

        registry = {
            "none": NullAdapter,
            "lora": LoRAAdapter,
            "qlora": QLoRAAdapter,
            "blocktt": BlockTTAdapter,
            "svd": SVDAdapter,
            "iso": IsoAdapter,
            "isobtt": IsoAdapter,
            "isobtt_mix": IsoAdapter,
        }
        cls_ = registry[peft_cfg.mode]
        # Only BTT / SVD adapters know about teacher_model_path; the others
        # don't accept the kwarg, so fall back to the legacy signature.
        try:
            return cls_(
                peft_cfg,
                model_config=model_config,
                teacher_model_path=teacher_model_path,
            )
        except TypeError:
            return cls_(peft_cfg, model_config=model_config)

    def needs_calibration(self) -> bool:
        return False

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        *,
        tokenizer,
        calib_loader_builder: Callable[[], Any],
    ) -> nn.Module: ...

    def export_for_vllm(self, fsdp_module) -> Optional[dict]:
        return None

    def vllm_engine_kwargs(self) -> dict:
        return {}

    def peft_config(self):
        return None

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # Default: assume HF model directly.
        fsdp_module.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        return {"mode": self.mode, "target_modules": self.peft_cfg.target_modules}

    @classmethod
    def rebuild_from_meta(cls, model: nn.Module, meta: dict) -> nn.Module:
        return model


class NullAdapter(PEFTAdapter):
    mode = "none"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        return model
