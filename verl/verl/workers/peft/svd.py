"""SVD adapter: plain (per-layer SVD decomposition) and calibrated (svd_v2 /
svd_v2_combined via compress.apply_calibrated_svd)."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import torch

from verl.workers.peft.base import PEFTAdapter


class SVDAdapter(PEFTAdapter):
    mode = "svd"

    def __init__(self, peft_cfg, model_config=None, teacher_model_path: Optional[str] = None):
        super().__init__(peft_cfg, model_config=model_config)
        self._is_calibrated = peft_cfg.calib.mode != "none"
        # Path to teacher LM for calib.loss=="opd"; loaded lazily in apply().
        self._teacher_model_path: Optional[str] = teacher_model_path

    def needs_calibration(self) -> bool:
        return self._is_calibrated

    def _trainable_type(self) -> str:
        tm = self.peft_cfg.target_modules
        return tm if (isinstance(tm, str) and tm in {"all", "mlp", "attn"}) else "all"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        from compress.integration import (
            apply_calibrated_svd,
            configure_compress_svd_trainability,
            convert_linear_to_svd_compress,
            get_svd_target_module_names,
        )

        sd = self.peft_cfg.svd
        include_names = get_svd_target_module_names(self._trainable_type())
        if self._is_calibrated:
            args = SimpleNamespace(
                train_mode="svd",
                trainable_type=self._trainable_type(),
                train_position=sd.train_position,
                s_merged_to=sd.s_merged_to,
                compression_ratio=sd.compression_ratio,
                calib_mode=self.peft_cfg.calib.mode,
                calib_source=self.peft_cfg.calib.source,
                calib_traces_path=self.peft_cfg.calib.traces_path,
                calib_num_seqs=self.peft_cfg.calib.num_seqs,
                calib_max_length=self.peft_cfg.calib.max_length,
                calib_seed=self.peft_cfg.calib.seed,
                calib_batch_size=self.peft_cfg.calib.batch_size,
            )
            args.calib_loss = self.peft_cfg.calib.loss
            if self.peft_cfg.calib.loss == "opd":
                args.calib_top_k = self.peft_cfg.calib.top_k
                args.calib_top_k_strategy = self.peft_cfg.calib.top_k_strategy
                args.calib_reward_weight_mode = self.peft_cfg.calib.reward_weight_mode
                args.calib_temperature = self.peft_cfg.calib.temperature
                args.calib_teacher_temperature = self.peft_cfg.calib.teacher_temperature
            calib_loader = calib_loader_builder()
            if calib_loader is None:
                raise RuntimeError("SVDAdapter calibration requires a non-None calib_loader.")
            device = "cuda" if torch.cuda.is_available() else None
            if self.peft_cfg.calib.loss == "opd":
                if not self._teacher_model_path:
                    raise RuntimeError(
                        "SVDAdapter with calib.loss='opd' requires teacher_model_path "
                        "(typically reward_model.model.path / REWARD_MODEL_PATH)"
                    )
                from transformers import AutoModelForCausalLM
                tdtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                teacher = AutoModelForCausalLM.from_pretrained(
                    self._teacher_model_path, torch_dtype=tdtype,
                )
                teacher = teacher.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
                for p in teacher.parameters():
                    p.requires_grad_(False)
                teacher.eval()
                args.teacher_model = teacher
            model = apply_calibrated_svd(model, args, calib_loader=calib_loader,
                                         device=device, hyphen_style=True)
        else:
            convert_linear_to_svd_compress(
                model,
                include_names=include_names,
                s_merged_to=sd.s_merged_to,
                train_position=sd.train_position,
            )
            configure_compress_svd_trainability(
                model,
                train_position=sd.train_position,
            )
        return model

    @torch.no_grad()
    def export_for_vllm(self, fsdp_module):
        from compress.integration import SVDCompressedLinear

        out = {}
        for name, module in fsdp_module.named_modules():
            if not isinstance(module, SVDCompressedLinear):
                continue
            out[f"{name}.weight"] = module.materialize_dense_weight()
            if module.bias is not None:
                out[f"{name}.bias"] = module.bias.detach()
        return out

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        from compress.integration import materialize_svd_to_linear

        os.makedirs(out_dir, exist_ok=True)
        materialize_svd_to_linear(fsdp_module)
        fsdp_module.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        sd = self.peft_cfg.svd
        return {
            "mode": "svd",
            "target_modules": self.peft_cfg.target_modules,
            "svd": {
                "train_position": sd.train_position,
                "s_merged_to": sd.s_merged_to,
                "compression_ratio": sd.compression_ratio,
            },
            "calib": {"mode": self.peft_cfg.calib.mode, "source": self.peft_cfg.calib.source},
        }
