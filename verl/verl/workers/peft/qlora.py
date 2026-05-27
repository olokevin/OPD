"""QLoRA adapter: reload base in 4-bit (bnb) then apply LoRA on top."""
from __future__ import annotations

import torch

from verl.workers.peft.lora import LoRAAdapter


_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class QLoRAAdapter(LoRAAdapter):
    mode = "qlora"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        # Reload the base in 4-bit using BitsAndBytesConfig, discarding the
        # full-precision model that fsdp_workers loaded.
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if self.model_config is None:
            raise ValueError(
                "QLoRAAdapter.apply requires model_config (set when PEFTAdapter.from_config "
                "is called) to know where to reload the base model from"
            )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.peft_cfg.qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_DTYPE[self.peft_cfg.qlora.bnb_4bit_compute_dtype],
            bnb_4bit_use_double_quant=self.peft_cfg.qlora.bnb_4bit_use_double_quant,
        )
        local_path = getattr(self.model_config, "local_path", None) or self.model_config.path
        base_4bit = AutoModelForCausalLM.from_pretrained(
            local_path,
            quantization_config=bnb_cfg,
            torch_dtype=_DTYPE[self.peft_cfg.qlora.bnb_4bit_compute_dtype],
            trust_remote_code=getattr(self.model_config, "trust_remote_code", False),
        )
        # Delegate to LoRAAdapter.apply for the LoRA wrap.
        return super().apply(base_4bit, tokenizer=tokenizer, calib_loader_builder=calib_loader_builder)

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        super().save_pretrained(fsdp_module, out_dir)
        # Record the original base path so eval can reload bf16 base + adapter.
        import os
        base_path = getattr(self.model_config, "path", None) if self.model_config else None
        if base_path is not None:
            with open(os.path.join(out_dir, "base_model_path.txt"), "w") as f:
                f.write(base_path)

    def topology_meta(self) -> dict:
        meta = super().topology_meta()
        meta["mode"] = "qlora"
        meta["qlora"] = {
            "bnb_4bit_quant_type": self.peft_cfg.qlora.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.peft_cfg.qlora.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.peft_cfg.qlora.bnb_4bit_use_double_quant,
        }
        return meta
