"""BlockTT adapter: plain (SVD/QR init), calibrated (v2 / twosteps / ...),
and qfura (NF4-quantized frozen core via QBTTLinear)."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Optional

import torch

from compress.integration import (
    BTTLinear,
    apply_calibrated_btt,
    configure_compress_btt_trainability,
    convert_and_quantize_linear_to_qbtt_streaming,
    convert_linear_to_btt_compress,
    get_blocktt_target_module_names,
    materialize_calibrated_btt_to_linear,
    materialize_calibrated_btt_weights,
    resolve_blocktt_decomp_modes,
)

from verl.workers.peft.base import PEFTAdapter


class BlockTTAdapter(PEFTAdapter):
    mode = "blocktt"

    def __init__(self, peft_cfg, model_config=None):
        super().__init__(peft_cfg, model_config=model_config)
        self._topology_payload: Optional[dict] = None
        self._is_calibrated: bool = peft_cfg.calib.mode != "none"
        self._is_qfura: bool = peft_cfg.blocktt.qfura.enabled

    def needs_calibration(self) -> bool:
        return self._is_calibrated

    def _trainable_type(self) -> str:
        tm = self.peft_cfg.target_modules
        if isinstance(tm, str) and tm in {"all", "mlp", "attn"}:
            return tm
        return "all"

    def _build_compress_args(self):
        bt = self.peft_cfg.blocktt
        return SimpleNamespace(
            train_mode="blocktt",
            trainable_type=self._trainable_type(),
            decomp_mode=bt.decomp_mode,
            blocktt_rank=bt.rank,
            convert_mode=bt.convert_mode,
            train_position=bt.train_position,
            s_merged_to=bt.s_merged_to,
            blocktt_factorize_by_head=bt.factorize_by_head,
            no_train_bias=not bt.train_bias,
            calib_mode=self.peft_cfg.calib.mode,
            calib_source=self.peft_cfg.calib.source,
            calib_traces_path=self.peft_cfg.calib.traces_path,
            calib_num_seqs=self.peft_cfg.calib.num_seqs,
            calib_max_length=self.peft_cfg.calib.max_length,
            calib_seed=self.peft_cfg.calib.seed,
            calib_batch_size=self.peft_cfg.calib.batch_size,
        )

    def apply(self, model, *, tokenizer, calib_loader_builder):
        bt = self.peft_cfg.blocktt
        include_names = get_blocktt_target_module_names(self._trainable_type())
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            bt.decomp_mode,
            include_names=include_names,
            default_mode="input_one_block",
        )
        args = self._build_compress_args()
        args.decomp_mode = decomp_mode
        args.blocktt_module_decomp_modes = module_decomp_modes

        if self._is_calibrated:
            calib_loader = calib_loader_builder()
            if calib_loader is None:
                raise RuntimeError(
                    "BlockTTAdapter calibration mode requires a non-None calib_loader; "
                    "check peft.calib.* and that calib_loader_builder was passed."
                )
            device = "cuda" if torch.cuda.is_available() else None
            model, stats = apply_calibrated_btt(model, args, calib_loader=calib_loader,
                                                device=device, hyphen_style=True)
            self._topology_payload = {"calib_stats": stats}
        else:
            convert_linear_to_btt_compress(
                model,
                target_module_names=include_names,
                decomp_mode=decomp_mode,
                rank=bt.rank,
                convert_mode=bt.convert_mode,
                factorize_by_head=bt.factorize_by_head,
                module_decomp_modes=module_decomp_modes,
            )
            configure_compress_btt_trainability(
                model,
                train_position=bt.train_position,
                s_merged_to=bt.s_merged_to,
                train_bias=bt.train_bias,
            )

        if self._is_qfura:
            convert_and_quantize_linear_to_qbtt_streaming(model)

        # Record minimal topology used by save / resume.
        from compress.topology import export_btt_topology
        self._topology_payload = self._topology_payload or {}
        self._topology_payload["btt_topology"] = export_btt_topology(model)
        return model

    @torch.no_grad()
    def export_for_vllm(self, fsdp_module):
        return {k: v for k, v in materialize_calibrated_btt_weights(fsdp_module)}

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # materialize all BTT/QBTT factors back into nn.Linear weights, then save_pretrained.
        materialize_calibrated_btt_to_linear(fsdp_module)
        fsdp_module.save_pretrained(out_dir)

    def topology_meta(self) -> dict:
        bt = self.peft_cfg.blocktt
        meta = {
            "mode": "blocktt",
            "target_modules": self.peft_cfg.target_modules,
            "blocktt": {
                "decomp_mode": bt.decomp_mode,
                "rank": bt.rank,
                "convert_mode": bt.convert_mode,
                "train_position": bt.train_position,
                "s_merged_to": bt.s_merged_to,
                "factorize_by_head": bt.factorize_by_head,
                "train_bias": bt.train_bias,
                "normalize_after_update": bt.normalize_after_update,
                "qfura": bt.qfura.enabled,
            },
            "calib": {
                "mode": self.peft_cfg.calib.mode,
                "source": self.peft_cfg.calib.source,
            },
        }
        if self._topology_payload and "btt_topology" in self._topology_payload:
            meta["compress_topology_path"] = "btt_topology.json"
        return meta

    def write_compress_sidecar(self, out_dir: str) -> None:
        """Called by fsdp_workers.save_checkpoint on first save to persist
        btt_topology.json under <ckpt>/compress/."""
        if not self._topology_payload or "btt_topology" not in self._topology_payload:
            return
        os.makedirs(os.path.join(out_dir, "compress"), exist_ok=True)
        with open(os.path.join(out_dir, "compress", "btt_topology.json"), "w") as f:
            json.dump(self._topology_payload["btt_topology"], f)

    @classmethod
    def rebuild_from_meta(cls, model, meta: dict):
        from compress.topology import rebuild_btt_from_topology
        topology_path = meta.get("_resolved_topology_path")
        if topology_path is None:
            raise RuntimeError("BlockTT rebuild_from_meta requires _resolved_topology_path "
                               "to be injected (typically by fsdp_workers).")
        with open(topology_path) as f:
            topology = json.load(f)
        rebuild_btt_from_topology(model, topology)
        return model
