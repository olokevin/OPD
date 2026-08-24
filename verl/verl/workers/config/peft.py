"""Unified PEFT config for actor (and optionally critic).

See docs/superpowers/specs/2026-05-26-verl-peft-blocktt-svd-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf


VALID_MODES = ("none", "lora", "qlora", "blocktt", "svd", "iso", "isobtt", "isobtt_mix")
VALID_CALIB_MODES = (
    "none", "v2", "v2_bp", "v2_combined", "twosteps", "svd_v2", "svd_v2_combined",
)
VALID_CALIB_SOURCES = ("c4", "traces", "training_data")


@dataclass
class LoRAConfig:
    rank: int = 0
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    adapter_path: Optional[str] = None
    exclude_modules: Optional[Union[str, list[str]]] = None


@dataclass
class QLoRAConfig:
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class BlockTTQfuraConfig:
    enabled: bool = False


@dataclass
class BlockTTConfig:
    decomp_mode: str = "input_one_block"
    rank: Union[str, int, float] = "full"
    convert_mode: str = "svd"
    train_position: str = "small"
    s_merged_to: str = "frozen"
    factorize_by_head: bool = True
    train_bias: bool = True
    normalize_after_update: bool = False
    qfura: BlockTTQfuraConfig = field(default_factory=BlockTTQfuraConfig)


@dataclass
class SVDConfig:
    train_position: str = "output"
    s_merged_to: str = "frozen"
    compression_ratio: float = 1.0


@dataclass
class IsoConfig:
    """Fixed-spectrum (ISO) BP modes. See verl/workers/peft/iso.py."""
    # Block size of the Cayley rotation generators (mode `iso` only; the
    # `isobtt*` modes take their block size from the layer's own factorisation).
    block_size: int = 128
    # Seed for the fixed random block basis (mode `iso` only).
    seed: int = 0


@dataclass
class CalibConfig:
    mode: str = "none"
    source: str = "c4"
    traces_path: Optional[str] = None
    num_seqs: int = 128
    max_length: int = 2048
    batch_size: int = 8
    seed: int = 3
    cpu_offload: bool = False
    # Gradient-source for v2_combined / svd_v2_combined modes.
    # "ce" — standard cross-entropy (default). "opd" — token-level on-policy
    # distillation against a teacher model on device.
    loss: str = "ce"
    top_k: int = 16                         # OPD: Top-K size for student's selected ids
    top_k_strategy: str = "only_stu"        # only currently supported value
    reward_weight_mode: str = "student_p"   # student_p | teacher_p | none
    temperature: float = 1.0
    teacher_temperature: float = 1.0


@dataclass
class PEFTConfig:
    mode: str = "none"
    target_modules: Union[str, list[str]] = "all"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    blocktt: BlockTTConfig = field(default_factory=BlockTTConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    iso: IsoConfig = field(default_factory=IsoConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"peft.mode must be one of {VALID_MODES}; got {self.mode!r}")
        if self.calib.mode not in VALID_CALIB_MODES:
            raise ValueError(
                f"peft.calib.mode must be one of {VALID_CALIB_MODES}; got {self.calib.mode!r}"
            )
        if self.calib.source not in VALID_CALIB_SOURCES:
            raise ValueError(
                f"peft.calib.source must be one of {VALID_CALIB_SOURCES}; "
                f"got {self.calib.source!r}"
            )
        if self.mode == "qlora" and self.lora.rank <= 0:
            raise ValueError("qlora requires peft.lora.rank > 0")
        if self.mode in {"iso", "isobtt", "isobtt_mix"} and self.iso.block_size < 2:
            raise ValueError("peft.iso.block_size must be >= 2")
        if self.calib.mode != "none" and self.mode not in {"blocktt", "svd"}:
            raise ValueError(
                f"peft.calib.mode={self.calib.mode!r} requires peft.mode in {{blocktt, svd}}; "
                f"got peft.mode={self.mode!r}"
            )
        if (
            self.mode == "blocktt"
            and self.calib.mode != "none"
            and not self.calib.mode.startswith("svd_")
        ):
            rank = self.blocktt.rank
            if isinstance(rank, int) and not isinstance(rank, bool):
                raise ValueError(
                    "integer peft.blocktt.rank is only valid when calib.mode=none; "
                    "for calibrated BTT pass 'full' or a float in (0, 1]"
                )
        if self.calib.mode == "traces" or (
            self.calib.mode != "none" and self.calib.source == "traces"
        ):
            if not self.calib.traces_path:
                raise ValueError("peft.calib.traces_path is required when calib.source=traces")
        if self.calib.loss not in {"ce", "opd"}:
            raise ValueError(
                f"peft.calib.loss must be 'ce' or 'opd'; got {self.calib.loss!r}"
            )
        if self.calib.loss == "opd" and self.calib.mode == "none":
            raise ValueError(
                "peft.calib.loss='opd' requires peft.calib.mode to be set "
                "(e.g. v2_combined)"
            )

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> "PEFTConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, DictConfig):
            raw = OmegaConf.to_container(cfg, resolve=True)
        else:
            raw = dict(cfg)
        sub_specs = {
            "lora": LoRAConfig, "qlora": QLoRAConfig,
            "blocktt": BlockTTConfig, "svd": SVDConfig, "iso": IsoConfig,
            "calib": CalibConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, val in raw.items():
            if key in sub_specs:
                sub_raw = dict(val) if val is not None else {}
                if key == "blocktt" and "qfura" in sub_raw:
                    qf = sub_raw.pop("qfura")
                    sub_raw["qfura"] = BlockTTQfuraConfig(**dict(qf))
                kwargs[key] = sub_specs[key](**sub_raw)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    @classmethod
    def legacy_shim(cls, *, peft_cfg: Any, model_cfg: Any) -> "PEFTConfig":
        """If model_cfg.lora_rank > 0 and peft_cfg.mode == "none", populate
        peft_cfg with lora fields. Otherwise return peft_cfg unchanged."""
        peft = cls.from_omegaconf(peft_cfg)
        if peft.mode != "none":
            return peft
        model_raw = (
            OmegaConf.to_container(model_cfg, resolve=True)
            if isinstance(model_cfg, DictConfig)
            else dict(model_cfg)
        )
        legacy_rank = int(model_raw.get("lora_rank", 0) or 0)
        legacy_adapter = model_raw.get("lora_adapter_path")
        if legacy_rank <= 0 and legacy_adapter is None:
            return peft
        peft.mode = "lora"
        peft.lora.rank = legacy_rank
        peft.lora.alpha = int(model_raw.get("lora_alpha", peft.lora.alpha))
        peft.lora.adapter_path = legacy_adapter
        tm = model_raw.get("target_modules")
        if tm is not None:
            peft.target_modules = tm
        exclude = model_raw.get("exclude_modules")
        if exclude is not None:
            peft.lora.exclude_modules = (
                exclude if isinstance(exclude, str) else list(exclude)
            )
        peft.__post_init__()
        return peft
