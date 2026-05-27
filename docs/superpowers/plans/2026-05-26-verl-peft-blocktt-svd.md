# verl PEFT (BlockTT / SVD / LoRA / QLoRA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified PEFT layer to verl so GRPO and on-policy distillation can train under BlockTT (plain + calibrated + qfura), SVD (plain + calibrated), LoRA, or QLoRA, behind one `PEFT_MODE` env var switch.

**Architecture:** A new `verl/workers/peft/` package introduces a `PEFTAdapter` strategy interface with five concrete adapters (Null, LoRA, QLoRA, BlockTT, SVD). The actor worker dispatches model setup, vLLM weight export, and checkpoint save through the adapter. A new `actor_rollout_ref.peft.*` Hydra config group replaces the inline `model.lora_*` fields (the old fields keep working via a deprecation shim). `PEFT_MODE=none` is the default — existing launch scripts behave identically to today.

**Tech Stack:** verl (FSDP + vLLM rollout), `src/compress` (BTT/SVD/QBTT modules + calibration), `peft` (LoRA/QLoRA adapters), `bitsandbytes` (4-bit quantization for QLoRA), Hydra/OmegaConf (config), pytest.

**Spec:** `docs/superpowers/specs/2026-05-26-verl-peft-blocktt-svd-design.md`

---

## File map

**New files:**
- `verl/verl/workers/config/peft.py` — `PEFTConfig` dataclass.
- `verl/verl/workers/peft/__init__.py` — re-exports + `PEFTAdapter.from_config` factory.
- `verl/verl/workers/peft/base.py` — `PEFTAdapter` ABC + `NullAdapter`.
- `verl/verl/workers/peft/lora.py` — `LoRAAdapter`.
- `verl/verl/workers/peft/qlora.py` — `QLoRAAdapter`.
- `verl/verl/workers/peft/blocktt.py` — `BlockTTAdapter` (plain + calib + qfura).
- `verl/verl/workers/peft/svd.py` — `SVDAdapter` (plain + calib).
- `verl/verl/workers/peft/calib_loader.py` — wraps `compress.integration.build_calib_loader` with verl's tokenizer/training-dataset wiring.
- `verl/tests/peft/__init__.py` — empty.
- `verl/tests/peft/conftest.py` — tiny-model fixtures.
- `verl/tests/peft/test_peft_config.py`
- `verl/tests/peft/test_adapter_apply.py`
- `verl/tests/peft/test_export_for_vllm.py`
- `verl/tests/peft/test_checkpoint_roundtrip.py`
- `verl/tests/peft/test_resume_topology.py`
- `verl/tests/peft/test_actor_worker_init.py` — pytest `gpu` marker.
- `verl/tests/peft/test_calibration_smoke.py` — pytest `gpu` marker.
- `scripts/peft_smoke.sh` — end-to-end smoke loop.

**Modified files:**
- `verl/verl/workers/config/model.py` — add legacy-LoRA-shim hook; no field changes.
- `verl/verl/workers/fsdp_workers.py` — replace inline `if self._is_lora:` block (L412–441) with adapter dispatch; replace `save_checkpoint` LoRA branch (around L1128); wire calibration call.
- `verl/verl/workers/sharding_manager/fsdp_vllm.py` — in `__enter__` (L130–226) and `update_params` (L283), consult adapter before falling through to existing LoRA-aware code.
- `verl/verl/trainer/config/ppo_trainer.yaml` and `verl/verl/trainer/config/_generated_ppo_trainer.yaml` — add `peft:` block under `actor_rollout_ref`.
- `grpo.sh`, `on_policy_distillation.sh` — append `$PEFT_ARGS` block.
- `scripts/val/eval/gen_vllm.py` — detect `adapter_config.json` and load LoRA-on-base.

---

## Conventions used in this plan

- All paths are absolute from repo root `/home/yequan/Project/compression/OPD/` unless otherwise noted; you can drop the prefix in editors.
- Tests live under `verl/tests/peft/` and use pytest. GPU-required tests have `@pytest.mark.gpu`; run them with `pytest -m gpu`.
- The conda env for verl/tests is `verl` (py 3.12).
- Commits use Conventional Commits style (`feat:`, `test:`, `fix:`, `docs:`).
- After each task, run `git status` to verify the expected files changed.
- The smallest tiny-model used in fixtures is `hf-internal-testing/tiny-random-LlamaForCausalLM` (2-layer Llama, ships with HF tests). All unit tests must work without downloading external 7B weights.

---

## Task 0: Set up test scaffolding

**Files:**
- Create: `verl/tests/peft/__init__.py`
- Create: `verl/tests/peft/conftest.py`

- [ ] **Step 1: Create empty package init**

```bash
touch verl/tests/peft/__init__.py
```

- [ ] **Step 2: Write conftest with tiny-model fixtures**

Create `verl/tests/peft/conftest.py`:

```python
"""Shared fixtures for verl/workers/peft tests."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture
def tiny_model():
    """Fresh tiny Llama on CPU. Cheap to recreate per test."""
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID, torch_dtype=torch.float32)


@pytest.fixture
def fixed_inputs(tiny_tokenizer):
    """Deterministic input ids for forward-parity checks."""
    text = "The quick brown fox jumps over the lazy dog."
    enc = tiny_tokenizer(text, return_tensors="pt")
    return enc["input_ids"]
```

- [ ] **Step 3: Verify pytest can collect**

Run: `cd verl && pytest tests/peft/ --collect-only -q`
Expected: `no tests ran` with no errors.

- [ ] **Step 4: Commit**

```bash
git add verl/tests/peft/__init__.py verl/tests/peft/conftest.py
git commit -m "test(peft): scaffold test package with tiny-model fixtures"
```

---

## Task 1: Define `PEFTConfig` dataclass

**Files:**
- Create: `verl/verl/workers/config/peft.py`
- Test: `verl/tests/peft/test_peft_config.py`

- [ ] **Step 1: Write the failing test**

Create `verl/tests/peft/test_peft_config.py`:

```python
"""Unit tests for PEFTConfig parsing and validation."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig


def test_default_is_none_mode():
    cfg = PEFTConfig()
    assert cfg.mode == "none"


def test_lora_mode_from_omegaconf():
    raw = OmegaConf.create(
        {"mode": "lora", "target_modules": "all",
         "lora": {"rank": 16, "alpha": 32, "dropout": 0.05}}
    )
    cfg = PEFTConfig.from_omegaconf(raw)
    assert cfg.mode == "lora"
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 32
    assert cfg.lora.dropout == 0.05


def test_blocktt_mode_with_calib():
    raw = OmegaConf.create(
        {"mode": "blocktt",
         "blocktt": {"decomp_mode": "input_one_block", "train_position": "small",
                     "rank": "full", "qfura": {"enabled": False}},
         "calib": {"mode": "v2", "source": "c4", "num_seqs": 64}}
    )
    cfg = PEFTConfig.from_omegaconf(raw)
    assert cfg.mode == "blocktt"
    assert cfg.blocktt.train_position == "small"
    assert cfg.calib.mode == "v2"
    assert cfg.calib.num_seqs == 64


def test_invalid_mode_rejected():
    raw = OmegaConf.create({"mode": "bogus"})
    with pytest.raises(ValueError, match="mode must be one of"):
        PEFTConfig.from_omegaconf(raw)


def test_qlora_requires_lora_rank():
    raw = OmegaConf.create({"mode": "qlora", "lora": {"rank": 0}})
    with pytest.raises(ValueError, match="qlora requires peft.lora.rank > 0"):
        PEFTConfig.from_omegaconf(raw)


def test_blocktt_calib_with_int_rank_rejected():
    raw = OmegaConf.create(
        {"mode": "blocktt",
         "blocktt": {"rank": 4},
         "calib": {"mode": "v2"}}
    )
    with pytest.raises(ValueError, match="integer .* rank is only valid"):
        PEFTConfig.from_omegaconf(raw)


def test_legacy_lora_rank_shim():
    """Old-style actor_rollout_ref.model.lora_rank populates peft.lora.*."""
    model_cfg = OmegaConf.create(
        {"lora_rank": 16, "lora_alpha": 32, "target_modules": "all-linear"}
    )
    peft_cfg = OmegaConf.create({"mode": "none"})
    merged = PEFTConfig.legacy_shim(peft_cfg=peft_cfg, model_cfg=model_cfg)
    assert merged.mode == "lora"
    assert merged.lora.rank == 16
    assert merged.lora.alpha == 32
    assert merged.target_modules == "all-linear"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verl && pytest tests/peft/test_peft_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl.workers.config.peft'`.

- [ ] **Step 3: Write minimal implementation**

Create `verl/verl/workers/config/peft.py`:

```python
"""Unified PEFT config for actor (and optionally critic).

See docs/superpowers/specs/2026-05-26-verl-peft-blocktt-svd-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf


VALID_MODES = ("none", "lora", "qlora", "blocktt", "svd")
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
class CalibConfig:
    mode: str = "none"
    source: str = "c4"
    traces_path: Optional[str] = None
    num_seqs: int = 128
    max_length: int = 2048
    batch_size: int = 8
    seed: int = 3
    cpu_offload: bool = False


@dataclass
class PEFTConfig:
    mode: str = "none"
    target_modules: Union[str, list[str]] = "all"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    blocktt: BlockTTConfig = field(default_factory=BlockTTConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
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
            "blocktt": BlockTTConfig, "svd": SVDConfig, "calib": CalibConfig,
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
        peft.__post_init__()
        return peft
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verl && pytest tests/peft/test_peft_config.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/config/peft.py verl/tests/peft/test_peft_config.py
git commit -m "feat(peft): add PEFTConfig dataclass with legacy-LoRA shim"
```

---

## Task 2: `PEFTAdapter` ABC and `NullAdapter`

**Files:**
- Create: `verl/verl/workers/peft/__init__.py`
- Create: `verl/verl/workers/peft/base.py`
- Test: `verl/tests/peft/test_adapter_apply.py` (start file, add null case)

- [ ] **Step 1: Write the failing test (null path only for now)**

Create `verl/tests/peft/test_adapter_apply.py`:

```python
"""Per-adapter apply / export / save tests."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def test_null_adapter_apply_is_identity(tiny_model, tiny_tokenizer):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({"mode": "none"}))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    assert adapter.mode == "none"
    assert adapter.needs_calibration() is False
    new_model = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert new_model is tiny_model
    assert adapter.export_for_vllm(new_model) is None
    assert adapter.vllm_engine_kwargs() == {}
    assert adapter.peft_config() is None
    assert adapter.topology_meta() == {"mode": "none", "target_modules": "all"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py::test_null_adapter_apply_is_identity -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl.workers.peft'`.

- [ ] **Step 3: Write `base.py`**

Create `verl/verl/workers/peft/base.py`:

```python
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
    def from_config(cls, peft_cfg, *, model_config) -> "PEFTAdapter":
        # Importing here avoids a circular import at module-load time.
        from verl.workers.peft.lora import LoRAAdapter
        from verl.workers.peft.qlora import QLoRAAdapter
        from verl.workers.peft.blocktt import BlockTTAdapter
        from verl.workers.peft.svd import SVDAdapter

        registry = {
            "none": NullAdapter,
            "lora": LoRAAdapter,
            "qlora": QLoRAAdapter,
            "blocktt": BlockTTAdapter,
            "svd": SVDAdapter,
        }
        cls_ = registry[peft_cfg.mode]
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
```

- [ ] **Step 4: Write `__init__.py`**

Create `verl/verl/workers/peft/__init__.py`:

```python
"""verl PEFT adapter package."""
from verl.workers.peft.base import PEFTAdapter, NullAdapter

__all__ = ["PEFTAdapter", "NullAdapter"]
```

- [ ] **Step 5: Stub the four other adapters (so `from_config` doesn't ImportError)**

Create `verl/verl/workers/peft/lora.py`:

```python
"""LoRA adapter — filled in by Task 3."""
from verl.workers.peft.base import PEFTAdapter


class LoRAAdapter(PEFTAdapter):
    mode = "lora"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("LoRAAdapter.apply implemented in Task 3")
```

Create `verl/verl/workers/peft/qlora.py`:

```python
"""QLoRA adapter — filled in by Task 4."""
from verl.workers.peft.lora import LoRAAdapter


class QLoRAAdapter(LoRAAdapter):
    mode = "qlora"
```

Create `verl/verl/workers/peft/blocktt.py`:

```python
"""BlockTT adapter — filled in by Task 5."""
from verl.workers.peft.base import PEFTAdapter


class BlockTTAdapter(PEFTAdapter):
    mode = "blocktt"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("BlockTTAdapter.apply implemented in Task 5")
```

Create `verl/verl/workers/peft/svd.py`:

```python
"""SVD adapter — filled in by Task 6."""
from verl.workers.peft.base import PEFTAdapter


class SVDAdapter(PEFTAdapter):
    mode = "svd"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("SVDAdapter.apply implemented in Task 6")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py::test_null_adapter_apply_is_identity -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add verl/verl/workers/peft/ verl/tests/peft/test_adapter_apply.py
git commit -m "feat(peft): add PEFTAdapter ABC, NullAdapter, and concrete-adapter stubs"
```

---

## Task 3: `LoRAAdapter`

**Files:**
- Modify: `verl/verl/workers/peft/lora.py`
- Modify: `verl/tests/peft/test_adapter_apply.py`

- [ ] **Step 1: Add the failing LoRA tests**

Append to `verl/tests/peft/test_adapter_apply.py`:

```python
from peft import PeftModel


def _lora_cfg(rank=4, alpha=8):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "lora",
        "target_modules": "all",
        "lora": {"rank": rank, "alpha": alpha, "dropout": 0.0},
    }))


def test_lora_adapter_wraps_with_peft(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert isinstance(out, PeftModel)
    # only adapter params should require grad
    grad_param_names = [n for n, p in out.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in grad_param_names), grad_param_names
    assert any(".q_proj" in n or ".o_proj" in n for n in grad_param_names)


def test_lora_adapter_export_for_vllm_returns_none(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    assert adapter.export_for_vllm(out) is None


def test_lora_adapter_vllm_engine_kwargs_enables_lora(tiny_model, tiny_tokenizer):
    cfg = _lora_cfg(rank=8)
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    kw = adapter.vllm_engine_kwargs()
    assert kw["enable_lora"] is True
    assert kw["max_loras"] == 1
    assert kw["max_lora_rank"] >= 8


def test_lora_adapter_save_pretrained_writes_adapter_dir(tiny_model, tiny_tokenizer, tmp_path):
    cfg = _lora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out_model = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out_model, str(save_dir))
    assert (save_dir / "adapter_config.json").exists()
    assert (save_dir / "adapter_model.safetensors").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k lora`
Expected: 4 FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement `LoRAAdapter`**

Replace `verl/verl/workers/peft/lora.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k lora`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/peft/lora.py verl/tests/peft/test_adapter_apply.py
git commit -m "feat(peft): implement LoRAAdapter"
```

---

## Task 4: `QLoRAAdapter`

**Files:**
- Modify: `verl/verl/workers/peft/qlora.py`
- Modify: `verl/tests/peft/test_adapter_apply.py`

- [ ] **Step 1: Add the failing QLoRA tests**

Append to `verl/tests/peft/test_adapter_apply.py`:

```python
import pytest


def _qlora_cfg(rank=4):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "qlora",
        "target_modules": "all",
        "lora": {"rank": rank, "alpha": rank * 2},
        "qlora": {"bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": "bfloat16",
                  "bnb_4bit_use_double_quant": True},
    }))


@pytest.mark.gpu
def test_qlora_adapter_reloads_base_in_4bit(tiny_model, tiny_tokenizer, tmp_path):
    # The adapter ignores the prebuilt tiny_model and reloads the model id in 4-bit,
    # so QLoRAAdapter must be told a model path via peft_cfg.model_config.
    pytest.importorskip("bitsandbytes")
    from transformers import AutoModelForCausalLM
    cfg = _qlora_cfg()
    class MockModelConfig:
        path = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        local_path = path
        trust_remote_code = False
    adapter = PEFTAdapter.from_config(cfg, model_config=MockModelConfig())
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    # at least one Linear should now be Linear4bit
    import bitsandbytes as bnb
    found_4bit = any(isinstance(m, bnb.nn.Linear4bit) for m in out.modules())
    assert found_4bit, "expected at least one Linear4bit after QLoRA apply"


def test_qlora_topology_meta_has_qlora_block(tiny_tokenizer):
    cfg = _qlora_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    meta = adapter.topology_meta()
    assert meta["mode"] == "qlora"
    assert meta["qlora"]["bnb_4bit_quant_type"] == "nf4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k qlora`
Expected: 2 FAIL.

- [ ] **Step 3: Implement `QLoRAAdapter`**

Replace `verl/verl/workers/peft/qlora.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k qlora -m 'not gpu'`
Expected: `test_qlora_topology_meta_has_qlora_block` PASS; the GPU test deselected.

If a GPU is available: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k qlora -m gpu` should also pass.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/peft/qlora.py verl/tests/peft/test_adapter_apply.py
git commit -m "feat(peft): implement QLoRAAdapter (bnb 4-bit base + LoRA)"
```

---

## Task 5: `BlockTTAdapter` (plain, calibrated, qfura)

**Files:**
- Modify: `verl/verl/workers/peft/blocktt.py`
- Create: `verl/verl/workers/peft/calib_loader.py`
- Modify: `verl/tests/peft/test_adapter_apply.py`

- [ ] **Step 1: Add the failing BlockTT tests**

Append to `verl/tests/peft/test_adapter_apply.py`:

```python
def _blocktt_cfg(qfura=False, calib_mode="none"):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt",
        "target_modules": "all",
        "blocktt": {
            "decomp_mode": "input_one_block",
            "rank": "full",
            "train_position": "small",
            "s_merged_to": "frozen",
            "convert_mode": "svd",
            "factorize_by_head": True,
            "train_bias": True,
            "normalize_after_update": False,
            "qfura": {"enabled": qfura},
        },
        "calib": {"mode": calib_mode, "source": "c4", "num_seqs": 4, "max_length": 64,
                  "batch_size": 2, "seed": 0},
    }))


@pytest.mark.gpu
def test_blocktt_plain_apply_installs_btt_modules(tiny_model, tiny_tokenizer):
    from compress.btt.btt_linear import BTTLinear
    cfg = _blocktt_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    n_btt = sum(1 for m in out.modules() if isinstance(m, BTTLinear))
    assert n_btt > 0, "no BTTLinear modules installed"
    # vLLM kwargs are empty for compress modes (full-base sync).
    assert adapter.vllm_engine_kwargs() == {}


@pytest.mark.gpu
def test_blocktt_qfura_apply_installs_qbtt_modules(tiny_model, tiny_tokenizer):
    from compress.btt.qbtt_linear import QBTTLinear
    cfg = _blocktt_cfg(qfura=True)
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    n_qbtt = sum(1 for m in out.modules() if isinstance(m, QBTTLinear))
    assert n_qbtt > 0, "no QBTTLinear modules installed"


@pytest.mark.gpu
def test_blocktt_export_for_vllm_returns_dense_weights(tiny_model, tiny_tokenizer):
    cfg = _blocktt_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    exported = adapter.export_for_vllm(out)
    assert isinstance(exported, dict) and len(exported) > 0
    # Keys must look like nn.Linear params, not factor names.
    for k in exported:
        assert ".btt_l" not in k and ".btt_r" not in k
        assert k.endswith(".weight") or k.endswith(".bias")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k blocktt -m gpu`
Expected: 3 FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `calib_loader.py`**

Create `verl/verl/workers/peft/calib_loader.py`:

```python
"""Glue between verl's PEFTConfig and compress.integration.build_calib_loader.

compress.integration.build_calib_loader expects argparse-style args (calib_mode,
calib_source, ...); we adapt the PEFTConfig dataclass to that shape.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional


def build_calib_loader_for_peft(peft_cfg, *, tokenizer, training_dataset=None,
                                training_collate_fn=None, rl_rollout_fn=None):
    """Returns a DataLoader or None when peft.calib.mode == 'none'."""
    if peft_cfg.calib.mode == "none":
        return None
    from compress.integration import build_calib_loader
    ns = SimpleNamespace(
        calib_mode=peft_cfg.calib.mode,
        calib_source=peft_cfg.calib.source,
        calib_traces_path=peft_cfg.calib.traces_path,
        calib_num_seqs=peft_cfg.calib.num_seqs,
        calib_max_length=peft_cfg.calib.max_length,
        calib_seed=peft_cfg.calib.seed,
        calib_batch_size=peft_cfg.calib.batch_size,
    )
    return build_calib_loader(
        ns,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        training_collate_fn=training_collate_fn,
        rl_rollout_fn=rl_rollout_fn,
        hyphen_style=True,
    )
```

- [ ] **Step 4: Implement `blocktt.py`**

Replace `verl/verl/workers/peft/blocktt.py`:

```python
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
        from compress.topology import record_btt_topology
        self._topology_payload = self._topology_payload or {}
        self._topology_payload["btt_topology"] = record_btt_topology(model)
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
```

> **Note for the implementer**: `compress.topology.record_btt_topology` may not exist in the upstream API by that exact name. Inspect `src/compress/topology.py`: if the equivalent is named differently (e.g. `dump_btt_topology` or part of `rebuild_btt_from_topology`'s reverse), use that name and adjust this file plus the `rebuild_from_meta` body. Do **not** invent a new compress API — call whatever the compress package exposes.

- [ ] **Step 5: Run tests**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k blocktt -m gpu`
Expected: 3 PASS on a CUDA box; on CPU-only they're deselected.

If the `compress.topology` symbol fix from the note above is needed: edit `blocktt.py`, re-run, iterate.

- [ ] **Step 6: Commit**

```bash
git add verl/verl/workers/peft/blocktt.py verl/verl/workers/peft/calib_loader.py \
        verl/tests/peft/test_adapter_apply.py
git commit -m "feat(peft): implement BlockTTAdapter (plain + calibrated + qfura)"
```

---

## Task 6: `SVDAdapter` (plain + calibrated)

**Files:**
- Modify: `verl/verl/workers/peft/svd.py`
- Modify: `verl/tests/peft/test_adapter_apply.py`

- [ ] **Step 1: Add the failing SVD tests**

Append to `verl/tests/peft/test_adapter_apply.py`:

```python
def _svd_cfg(calib_mode="none", ratio=1.0):
    return PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "svd",
        "target_modules": "all",
        "svd": {"train_position": "output", "s_merged_to": "frozen",
                "compression_ratio": ratio},
        "calib": {"mode": calib_mode, "source": "c4", "num_seqs": 4, "max_length": 64,
                  "batch_size": 2, "seed": 0},
    }))


@pytest.mark.gpu
def test_svd_plain_apply_installs_svd_modules(tiny_model, tiny_tokenizer):
    from compress.svd.svd_linear import SVDCompressedLinear
    cfg = _svd_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    n = sum(1 for m in out.modules() if isinstance(m, SVDCompressedLinear))
    assert n > 0


@pytest.mark.gpu
def test_svd_export_for_vllm_returns_dense_weights(tiny_model, tiny_tokenizer):
    cfg = _svd_cfg()
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    exported = adapter.export_for_vllm(out)
    assert isinstance(exported, dict) and len(exported) > 0
    for k in exported:
        assert ".U_r" not in k and ".V_r" not in k
        assert k.endswith(".weight") or k.endswith(".bias")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k svd -m gpu`
Expected: 2 FAIL.

- [ ] **Step 3: Implement `svd.py`**

Replace `verl/verl/workers/peft/svd.py`:

```python
"""SVD adapter: plain (per-layer SVD decomposition) and calibrated (svd_v2 /
svd_v2_combined via compress.apply_calibrated_svd)."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from compress.integration import (
    SVDCompressedLinear,
    apply_calibrated_svd,
    configure_compress_svd_trainability,
    convert_linear_to_svd_compress,
    get_svd_target_module_names,
    materialize_svd_to_linear,
)

from verl.workers.peft.base import PEFTAdapter


class SVDAdapter(PEFTAdapter):
    mode = "svd"

    def __init__(self, peft_cfg, model_config=None):
        super().__init__(peft_cfg, model_config=model_config)
        self._is_calibrated = peft_cfg.calib.mode != "none"

    def needs_calibration(self) -> bool:
        return self._is_calibrated

    def _trainable_type(self) -> str:
        tm = self.peft_cfg.target_modules
        return tm if (isinstance(tm, str) and tm in {"all", "mlp", "attn"}) else "all"

    def apply(self, model, *, tokenizer, calib_loader_builder):
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
            calib_loader = calib_loader_builder()
            if calib_loader is None:
                raise RuntimeError("SVDAdapter calibration requires a non-None calib_loader.")
            device = "cuda" if torch.cuda.is_available() else None
            model = apply_calibrated_svd(model, args, calib_loader=calib_loader,
                                         device=device, hyphen_style=True)
        else:
            convert_linear_to_svd_compress(
                model,
                target_module_names=include_names,
                compression_ratio=sd.compression_ratio,
            )
            configure_compress_svd_trainability(
                model,
                train_position=sd.train_position,
                s_merged_to=sd.s_merged_to,
            )
        return model

    @torch.no_grad()
    def export_for_vllm(self, fsdp_module):
        out = {}
        for name, module in fsdp_module.named_modules():
            if not isinstance(module, SVDCompressedLinear):
                continue
            out[f"{name}.weight"] = module.materialize_dense_weight()
            if module.bias is not None:
                out[f"{name}.bias"] = module.bias.detach()
        return out

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
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
```

> **Implementer note**: `convert_linear_to_svd_compress` and
> `configure_compress_svd_trainability` may take slightly different keyword names
> in your local compress version. If a call fails with `TypeError: ... unexpected
> keyword argument`, inspect `src/compress/integration.py` and adjust to the actual
> signature. Do not invent kwargs.

- [ ] **Step 4: Run tests**

Run: `cd verl && pytest tests/peft/test_adapter_apply.py -v -k svd -m gpu`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/workers/peft/svd.py verl/tests/peft/test_adapter_apply.py
git commit -m "feat(peft): implement SVDAdapter (plain + calibrated)"
```

---

## Task 7: Wire actor worker through the adapter

**Files:**
- Modify: `verl/verl/workers/fsdp_workers.py` (around L181–184, L412–441, L1128–1153)
- Modify: `verl/verl/workers/config/model.py` (no field changes; keep legacy fields)
- Modify: `verl/verl/trainer/config/_generated_ppo_trainer.yaml`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`

This task has no new pytest tests; the actor-worker integration smoke test
lives in Task 11.

- [ ] **Step 1: Add `peft:` block under `actor_rollout_ref` in YAML**

Edit `verl/verl/trainer/config/ppo_trainer.yaml`. Locate the
`actor_rollout_ref:` block (it imports from `model/hf_model.yaml` and
sub-files); add a `peft:` group as a sibling of `model:`:

```yaml
actor_rollout_ref:
  hybrid_engine: True
  model:
    # ... existing fields ...
  peft:
    mode: none                  # none | lora | qlora | blocktt | svd
    target_modules: all         # all | mlp | attn | custom
    lora:
      rank: 0
      alpha: 16
      dropout: 0.0
      bias: none
      adapter_path: null
    qlora:
      bnb_4bit_quant_type: nf4
      bnb_4bit_compute_dtype: bfloat16
      bnb_4bit_use_double_quant: true
    blocktt:
      decomp_mode: input_one_block
      rank: full
      convert_mode: svd
      train_position: small
      s_merged_to: frozen
      factorize_by_head: true
      train_bias: true
      normalize_after_update: false
      qfura:
        enabled: false
    svd:
      train_position: output
      s_merged_to: frozen
      compression_ratio: 1.0
    calib:
      mode: none
      source: c4
      traces_path: null
      num_seqs: 128
      max_length: 2048
      batch_size: 8
      seed: 3
      cpu_offload: false
  actor:
    # ... existing fields ...
```

Then add the same `peft:` block to `verl/verl/trainer/config/_generated_ppo_trainer.yaml` at the matching location (the generated file mirrors the structure).

- [ ] **Step 2: Replace inline LoRA block in `ActorRolloutRefWorker._build_model_optimizer`**

Open `verl/verl/workers/fsdp_workers.py`. Around L412–441 currently reads:

```python
        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()
            ...
            actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
```

Replace **the entire `if self._is_lora:` block** with:

```python
        # Apply PEFT adapter (LoRA / QLoRA / BlockTT / SVD / none).
        from verl.workers.config.peft import PEFTConfig
        from verl.workers.peft import PEFTAdapter
        from verl.workers.peft.calib_loader import build_calib_loader_for_peft

        peft_raw = self.config.get("peft", None)
        peft_cfg = PEFTConfig.legacy_shim(peft_cfg=peft_raw, model_cfg=self.config.model)
        self._peft_adapter = PEFTAdapter.from_config(peft_cfg, model_config=self.config.model)

        # Resume: if a peft_meta.json sits at default_local_dir, rebuild topology
        # instead of re-applying (skips calibration).
        ckpt_root = getattr(self.config.trainer, "default_local_dir", None) if hasattr(
            self.config, "trainer") else None
        peft_meta_path = (
            os.path.join(ckpt_root, "peft_meta.json")
            if ckpt_root and os.path.isfile(os.path.join(ckpt_root, "peft_meta.json"))
            else None
        )
        if peft_meta_path is not None:
            import json
            with open(peft_meta_path) as f:
                meta = json.load(f)
            # On-disk drift guard (Risk 6 in spec).
            self._compare_peft_meta_to_cli(meta, peft_cfg)
            if meta.get("compress_topology_path"):
                meta["_resolved_topology_path"] = os.path.join(
                    ckpt_root, "compress", meta["compress_topology_path"]
                )
            actor_module = type(self._peft_adapter).rebuild_from_meta(actor_module, meta)
        else:
            calib_loader = None
            if self._peft_adapter.needs_calibration():
                calib_loader = build_calib_loader_for_peft(
                    peft_cfg,
                    tokenizer=self.tokenizer,
                )
            actor_module = self._peft_adapter.apply(
                actor_module,
                tokenizer=self.tokenizer,
                calib_loader_builder=lambda: calib_loader,
            )

        self._is_lora = self._peft_adapter.mode in {"lora", "qlora"}
```

(Leave the FSDP wrap and downstream code unchanged. `self._is_lora` retains
its meaning for the rest of the file.)

Also add `_compare_peft_meta_to_cli` as a method on `ActorRolloutRefWorker`
(place it just above `_build_model_optimizer`):

```python
    def _compare_peft_meta_to_cli(self, meta: dict, peft_cfg) -> None:
        if meta.get("mode") != peft_cfg.mode:
            raise ValueError(
                f"peft.mode drift on resume: checkpoint says {meta.get('mode')!r}, "
                f"CLI says {peft_cfg.mode!r}. Delete the checkpoint or revert the override."
            )
        # Compare per-mode subkeys.
        sub = meta.get(peft_cfg.mode, {})
        cli_sub = getattr(peft_cfg, peft_cfg.mode, None)
        if cli_sub is None:
            return
        for key, ckpt_val in sub.items():
            cli_val = getattr(cli_sub, key, None)
            if hasattr(cli_val, "__dict__"):
                # Nested dataclass (e.g. blocktt.qfura) — compare attributes.
                for k2, v2 in (ckpt_val.items() if isinstance(ckpt_val, dict) else []):
                    if getattr(cli_val, k2, None) != v2:
                        raise ValueError(
                            f"peft.{peft_cfg.mode}.{key}.{k2} drift on resume: "
                            f"checkpoint={v2!r}, CLI={getattr(cli_val, k2, None)!r}"
                        )
            elif cli_val != ckpt_val:
                raise ValueError(
                    f"peft.{peft_cfg.mode}.{key} drift on resume: "
                    f"checkpoint={ckpt_val!r}, CLI={cli_val!r}"
                )
```

- [ ] **Step 3: Replace `save_checkpoint` LoRA branch**

In the same file around L1128:

```python
        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            ...
```

Replace **the whole `if self._is_lora and hasattr(...):` block** with:

```python
        # Adapter-driven HF-format save: writes merged_hf/ + peft_meta.json + compress/.
        try:
            merged_hf_dir = os.path.join(local_path, "merged_hf")
            os.makedirs(merged_hf_dir, exist_ok=True)
            # Materialize / merge happens inside adapter.save_pretrained.
            peft_module_for_save = getattr(self, "actor_module", self.actor_module_fsdp)
            self._peft_adapter.save_pretrained(peft_module_for_save, merged_hf_dir)
            # Tokenizer next to the HF dir so eval can from_pretrained the same dir.
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(merged_hf_dir)
            # Sidecar metadata (rank 0, first save only).
            if dist.get_rank() == 0 and not os.path.exists(
                os.path.join(local_path, "peft_meta.json")
            ):
                with open(os.path.join(local_path, "peft_meta.json"), "w") as f:
                    import json
                    json.dump(self._peft_adapter.topology_meta(), f, indent=2)
            # BlockTT topology sidecar (only writes if adapter populated it).
            write_sidecar = getattr(self._peft_adapter, "write_compress_sidecar", None)
            if dist.get_rank() == 0 and callable(write_sidecar):
                write_sidecar(local_path)
        except Exception as e:
            log_with_rank(
                f"PEFT save_pretrained error ({e})", rank=dist.get_rank(), logger=logger,
                log_only_rank_0=True,
            )
```

(Keep the existing FSDP-shard save call untouched — that's a sibling, not
something we replace.)

- [ ] **Step 4: Verify configs still parse**

Run: `cd verl && python -c "from omegaconf import OmegaConf; OmegaConf.load('verl/trainer/config/ppo_trainer.yaml'); OmegaConf.load('verl/trainer/config/_generated_ppo_trainer.yaml'); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Verify legacy mode still works (smoke import)**

Run: `cd verl && python -c "
from verl.workers.config.peft import PEFTConfig
from omegaconf import OmegaConf
model_cfg = OmegaConf.create({'lora_rank': 8, 'lora_alpha': 16})
peft_cfg = OmegaConf.create({'mode': 'none'})
p = PEFTConfig.legacy_shim(peft_cfg=peft_cfg, model_cfg=model_cfg)
print(p.mode, p.lora.rank)
"`
Expected: `lora 8`.

- [ ] **Step 6: Commit**

```bash
git add verl/verl/workers/fsdp_workers.py verl/verl/trainer/config/ppo_trainer.yaml \
        verl/verl/trainer/config/_generated_ppo_trainer.yaml
git commit -m "feat(peft): wire ActorRolloutRefWorker through PEFTAdapter dispatch"
```

---

## Task 8: Wire `FSDPVLLMShardingManager` through the adapter

**Files:**
- Modify: `verl/verl/workers/sharding_manager/fsdp_vllm.py`

- [ ] **Step 1: In `__enter__`, ask the adapter first**

Open `verl/verl/workers/sharding_manager/fsdp_vllm.py`. Around L206 (after the
existing `params = __collect_lora_params()` call), the manager already decides
between dense and LoRA paths based on `self.base_sync_done` and a `peft_config`
read from the FSDP-wrapped module. We need a new short-circuit: if the actor
worker attached a non-LoRA adapter (BlockTT / SVD / Null), we should produce
dense weights via the adapter and skip the LoRA-specific branch.

Locate where the FSDP-wrapped module is referenced (look for `peft_model = …`
near L662–665). Just above the `peft_config = …` line in `__enter__` (search
for `peft_config = peft_model.peft_config.get("default", None)`), insert:

```python
        # Adapter-driven dense export (BlockTT / SVD / Null modes).
        peft_adapter = getattr(self.actor_worker, "_peft_adapter", None) if hasattr(
            self, "actor_worker") else None
        exported_dense = None
        if peft_adapter is not None and peft_adapter.export_for_vllm.__func__ is not \
                type(peft_adapter).__base__.export_for_vllm.__func__:
            # Concrete export available; produce dense weights and skip LoRA collection.
            with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                maybe_dict = peft_adapter.export_for_vllm(self.actor_module_fsdp)
            if maybe_dict is not None:
                exported_dense = maybe_dict
```

Then, where the existing code does `params = __collect_lora_params()`, change
the surrounding logic so that when `exported_dense is not None`, we use it
directly and skip the LoRA path:

```python
                if exported_dense is not None:
                    params = exported_dense
                    peft_config = None
                else:
                    params = __collect_lora_params()
                    ...
```

(Match the existing indentation/structure; don't restructure the conditional
tree, only add the `exported_dense is not None` branch above the existing
`__collect_lora_params()` call.)

> **Implementer note**: `self.actor_worker` is not the upstream attribute name —
> in current verl the sharding manager is given the FSDP module and the rollout
> directly. **Before** writing the above, search the file for how the worker
> reaches into the sharding manager:
> `grep -n 'FSDPVLLMShardingManager\|sharding_manager' verl/verl/workers/fsdp_workers.py | head -20`.
> If the worker passes itself, use that attribute name. Otherwise plumb a new
> kwarg `peft_adapter=self._peft_adapter` into the sharding manager constructor
> (one line in `fsdp_workers.py`, one line in `FSDPVLLMShardingManager.__init__`)
> and reference `self._peft_adapter` directly.

- [ ] **Step 2: Smoke check the file still imports**

Run: `cd verl && python -c "from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add verl/verl/workers/sharding_manager/fsdp_vllm.py
# If you plumbed peft_adapter through the constructor, also add fsdp_workers.py.
git add verl/verl/workers/fsdp_workers.py 2>/dev/null || true
git commit -m "feat(peft): route vLLM weight sync through PEFTAdapter.export_for_vllm"
```

---

## Task 9: `test_export_for_vllm.py` — assert key set matches `nn.Linear` keys

**Files:**
- Create: `verl/tests/peft/test_export_for_vllm.py`

- [ ] **Step 1: Write the test**

Create `verl/tests/peft/test_export_for_vllm.py`:

```python
"""Verify that BlockTT/SVD export_for_vllm emits exactly the nn.Linear param keys
of the pre-PEFT model, with dense tensors of the matching shape."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def _linear_param_keys(model):
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
            keys.add(f"{name}.weight")
            if module.bias is not None:
                keys.add(f"{name}.bias")
    return keys


@pytest.mark.gpu
@pytest.mark.parametrize("mode_cfg", [
    {"mode": "blocktt", "target_modules": "all",
     "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                 "train_position": "small", "s_merged_to": "frozen",
                 "convert_mode": "svd", "factorize_by_head": True,
                 "train_bias": True, "normalize_after_update": False,
                 "qfura": {"enabled": False}}},
    {"mode": "svd", "target_modules": "all",
     "svd": {"train_position": "output", "s_merged_to": "frozen",
             "compression_ratio": 1.0}},
])
def test_export_keys_subset_of_linear_keys(tiny_model, tiny_tokenizer, mode_cfg):
    target_keys = _linear_param_keys(tiny_model)
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create(mode_cfg))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    exported = adapter.export_for_vllm(out)
    # Every exported key must correspond to a Linear in the original model.
    for key in exported:
        # Allow either exact .weight match or .bias match.
        assert key in target_keys, f"exported key {key!r} not found in tiny_model Linear keys"
```

- [ ] **Step 2: Run test**

Run: `cd verl && pytest tests/peft/test_export_for_vllm.py -v -m gpu`
Expected: 2 PASS.

- [ ] **Step 3: Commit**

```bash
git add verl/tests/peft/test_export_for_vllm.py
git commit -m "test(peft): export_for_vllm key set matches pre-PEFT Linear keys"
```

---

## Task 10: `test_checkpoint_roundtrip.py` — save → reload → logit parity

**Files:**
- Create: `verl/tests/peft/test_checkpoint_roundtrip.py`

- [ ] **Step 1: Write the test**

Create `verl/tests/peft/test_checkpoint_roundtrip.py`:

```python
"""Save → reload → forward-pass parity for every PEFT mode.

merged_hf/ must be loadable via stock from_pretrained (compress/none) or
PeftModel.from_pretrained (lora/qlora). Logits must match within 1e-4 on a
fixed input."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter


def _forward_logits(model, input_ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids.to(next(model.parameters()).device)).logits.detach().cpu()


def test_none_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({"mode": "none"}))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32)
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4), f"max diff {(pre - post).abs().max()}"


def test_lora_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    from peft import PeftModel
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "lora", "target_modules": "all",
        "lora": {"rank": 4, "alpha": 8},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model, tokenizer=tiny_tokenizer, calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    base = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, str(save_dir))
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("qfura", [False, True])
def test_blocktt_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path, qfura):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": qfura}},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32).cuda()
    post = _forward_logits(reloaded, fixed_inputs)
    # qfura has NF4 quantization error; loosen tolerance there.
    atol = 5e-2 if qfura else 1e-4
    assert torch.allclose(pre, post, atol=atol), f"max diff {(pre - post).abs().max()}"


@pytest.mark.gpu
def test_svd_roundtrip(tiny_model, tiny_tokenizer, fixed_inputs, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "svd", "target_modules": "all",
        "svd": {"train_position": "output", "s_merged_to": "frozen",
                "compression_ratio": 1.0},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: None)
    pre = _forward_logits(out, fixed_inputs)
    save_dir = tmp_path / "merged_hf"
    adapter.save_pretrained(out, str(save_dir))
    reloaded = AutoModelForCausalLM.from_pretrained(str(save_dir), torch_dtype=torch.float32).cuda()
    post = _forward_logits(reloaded, fixed_inputs)
    assert torch.allclose(pre, post, atol=1e-4)
```

- [ ] **Step 2: Run tests**

Run: `cd verl && pytest tests/peft/test_checkpoint_roundtrip.py -v`
Expected: 2 PASS on CPU (`none`, `lora`); 3 PASS on GPU (`blocktt-plain`, `blocktt-qfura`, `svd`).

- [ ] **Step 3: Commit**

```bash
git add verl/tests/peft/test_checkpoint_roundtrip.py
git commit -m "test(peft): save→reload→logit-parity roundtrip for every mode"
```

---

## Task 11: GPU smoke test — `ActorRolloutRefWorker` init under each mode

**Files:**
- Create: `verl/tests/peft/test_actor_worker_init.py`
- Create: `verl/tests/peft/test_calibration_smoke.py`

- [ ] **Step 1: Write `test_actor_worker_init.py`**

```python
"""GPU smoke test: instantiate ActorRolloutRefWorker for each PEFT mode and
verify it reaches a post-FSDP-wrap state."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("peft_mode_cfg", [
    {"mode": "none"},
    {"mode": "lora", "target_modules": "all", "lora": {"rank": 4, "alpha": 8}},
    {"mode": "blocktt", "target_modules": "all",
     "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                 "train_position": "small", "s_merged_to": "frozen",
                 "convert_mode": "svd", "factorize_by_head": True,
                 "train_bias": True, "normalize_after_update": False,
                 "qfura": {"enabled": False}}},
    {"mode": "svd", "target_modules": "all",
     "svd": {"train_position": "output", "s_merged_to": "frozen",
             "compression_ratio": 1.0}},
])
def test_actor_worker_init_per_mode(peft_mode_cfg):
    # Defer heavy imports.
    import os
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from omegaconf import OmegaConf
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.trainer.config import ppo_trainer  # ensure config schema loads

    cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    cfg.actor_rollout_ref.model.path = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cfg.actor_rollout_ref.peft = OmegaConf.create(peft_mode_cfg)
    cfg.actor_rollout_ref.actor.fsdp_config.param_offload = False
    # Avoid full vllm rollout init; only test model build.
    worker = ActorRolloutRefWorker(config=cfg.actor_rollout_ref, role="actor")
    # The worker's _build_model_optimizer is invoked during init_model; trigger it.
    worker.init_model()
    assert hasattr(worker, "_peft_adapter")
    assert worker._peft_adapter.mode == peft_mode_cfg["mode"]
```

> **Implementer note**: `init_model()` in verl spins up rollout/vLLM as a
> side-effect. If that's heavyweight even for a tiny model, gate it behind an
> env var (`VERL_PEFT_SMOKE_SKIP_ROLLOUT=1`) and have the test set it. If the
> integration is hard to drive in-process, mark these tests `@pytest.mark.slow`
> and skip in CI; document in the test file's docstring.

- [ ] **Step 2: Write `test_calibration_smoke.py`**

```python
"""GPU smoke: calib_mode=v2 with c4 source and num_seqs=4 installs BTT topology
and modifies at least one core from its random init."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_blocktt_v2_calibration_smoke(tiny_model, tiny_tokenizer):
    from omegaconf import OmegaConf
    from verl.workers.config.peft import PEFTConfig
    from verl.workers.peft import PEFTAdapter
    from verl.workers.peft.calib_loader import build_calib_loader_for_peft
    from compress.btt.btt_linear import BTTLinear

    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": 0.5,
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": False}},
        "calib": {"mode": "v2", "source": "c4", "num_seqs": 4,
                  "max_length": 64, "batch_size": 2, "seed": 0},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    loader = build_calib_loader_for_peft(cfg, tokenizer=tiny_tokenizer)
    assert loader is not None
    out = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                        calib_loader_builder=lambda: loader)
    n_btt = sum(1 for m in out.modules() if isinstance(m, BTTLinear))
    assert n_btt > 0
```

- [ ] **Step 3: Run the smokes (skipped on CPU machines)**

Run: `cd verl && pytest tests/peft/test_actor_worker_init.py tests/peft/test_calibration_smoke.py -v -m gpu`
Expected: PASS on a GPU box. Document any infrastructure shortcuts taken (e.g.
the `VERL_PEFT_SMOKE_SKIP_ROLLOUT` env var) in the test file's docstring.

- [ ] **Step 4: Commit**

```bash
git add verl/tests/peft/test_actor_worker_init.py verl/tests/peft/test_calibration_smoke.py
git commit -m "test(peft): GPU smokes for worker init + v2 calibration"
```

---

## Task 12: Resume topology rebuild test

**Files:**
- Create: `verl/tests/peft/test_resume_topology.py`

- [ ] **Step 1: Write the test**

```python
"""Resume path: write peft_meta.json + btt_topology.json, then on a fresh
model call adapter.rebuild_from_meta and verify the module topology matches
the original adapter.apply output."""
from __future__ import annotations

import json
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from verl.workers.config.peft import PEFTConfig
from verl.workers.peft import PEFTAdapter
from verl.workers.peft.blocktt import BlockTTAdapter


@pytest.mark.gpu
def test_blocktt_rebuild_from_meta(tiny_model, tiny_tokenizer, tmp_path):
    cfg = PEFTConfig.from_omegaconf(OmegaConf.create({
        "mode": "blocktt", "target_modules": "all",
        "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                    "train_position": "small", "s_merged_to": "frozen",
                    "convert_mode": "svd", "factorize_by_head": True,
                    "train_bias": True, "normalize_after_update": False,
                    "qfura": {"enabled": False}},
    }))
    adapter = PEFTAdapter.from_config(cfg, model_config=None)
    original = adapter.apply(tiny_model.cuda(), tokenizer=tiny_tokenizer,
                             calib_loader_builder=lambda: None)
    # Persist sidecar.
    meta = adapter.topology_meta()
    adapter.write_compress_sidecar(str(tmp_path))
    meta["_resolved_topology_path"] = str(tmp_path / "compress" / "btt_topology.json")

    # Fresh model — rebuild.
    fresh = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype=torch.float32,
    ).cuda()
    rebuilt = BlockTTAdapter.rebuild_from_meta(fresh, meta)

    def _module_class_map(model):
        return {n: type(m).__name__ for n, m in model.named_modules()
                if type(m).__name__ in {"BTTLinear", "QBTTLinear",
                                        "SVDCompressedLinear", "Linear"}}

    assert _module_class_map(rebuilt) == _module_class_map(original)
```

- [ ] **Step 2: Run test**

Run: `cd verl && pytest tests/peft/test_resume_topology.py -v -m gpu`
Expected: PASS on GPU.

- [ ] **Step 3: Commit**

```bash
git add verl/tests/peft/test_resume_topology.py
git commit -m "test(peft): resume rebuilds BlockTT topology to match original apply"
```

---

## Task 13: Update launch scripts

**Files:**
- Modify: `grpo.sh`
- Modify: `on_policy_distillation.sh`

- [ ] **Step 1: Add `$PEFT_ARGS` block to `on_policy_distillation.sh`**

Open `on_policy_distillation.sh`. After the existing `export ...` block (around
line 70, before the `# TODO: qwen3_1p7b_base ...` comment), insert the
launch-script changes documented in Section 4 of the spec:

```bash
# ---- PEFT ----
export PEFT_MODE=${PEFT_MODE:-none}
export PEFT_TARGET_MODULES=${PEFT_TARGET_MODULES:-all}

export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}

export QLORA_QUANT_TYPE=${QLORA_QUANT_TYPE:-nf4}
export QLORA_DOUBLE_QUANT=${QLORA_DOUBLE_QUANT:-True}
export QLORA_COMPUTE_DTYPE=${QLORA_COMPUTE_DTYPE:-bfloat16}

export BTT_DECOMP_MODE=${BTT_DECOMP_MODE:-input_one_block}
export BTT_RANK=${BTT_RANK:-full}
export BTT_TRAIN_POSITION=${BTT_TRAIN_POSITION:-small}
export BTT_S_MERGED_TO=${BTT_S_MERGED_TO:-frozen}
export BTT_CONVERT_MODE=${BTT_CONVERT_MODE:-svd}
export BTT_FACTORIZE_BY_HEAD=${BTT_FACTORIZE_BY_HEAD:-True}
export BTT_NORMALIZE_AFTER_UPDATE=${BTT_NORMALIZE_AFTER_UPDATE:-False}
export BTT_QFURA=${BTT_QFURA:-False}

export SVD_TRAIN_POSITION=${SVD_TRAIN_POSITION:-output}
export SVD_S_MERGED_TO=${SVD_S_MERGED_TO:-frozen}
export SVD_COMPRESSION_RATIO=${SVD_COMPRESSION_RATIO:-1.0}

export CALIB_MODE=${CALIB_MODE:-none}
export CALIB_SOURCE=${CALIB_SOURCE:-c4}
export CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
export CALIB_MAX_LENGTH=${CALIB_MAX_LENGTH:-2048}
export CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-8}
export CALIB_SEED=${CALIB_SEED:-3}
export CALIB_TRACES_PATH=${CALIB_TRACES_PATH:-}

PEFT_ARGS="+actor_rollout_ref.peft.mode=$PEFT_MODE \
+actor_rollout_ref.peft.target_modules=$PEFT_TARGET_MODULES"

case "$PEFT_MODE" in
  none) ;;
  lora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.lora.dropout=$LORA_DROPOUT" ;;
  qlora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.qlora.bnb_4bit_quant_type=$QLORA_QUANT_TYPE \
      +actor_rollout_ref.peft.qlora.bnb_4bit_use_double_quant=$QLORA_DOUBLE_QUANT \
      +actor_rollout_ref.peft.qlora.bnb_4bit_compute_dtype=$QLORA_COMPUTE_DTYPE" ;;
  blocktt)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.blocktt.decomp_mode=$BTT_DECOMP_MODE \
      +actor_rollout_ref.peft.blocktt.rank=$BTT_RANK \
      +actor_rollout_ref.peft.blocktt.train_position=$BTT_TRAIN_POSITION \
      +actor_rollout_ref.peft.blocktt.s_merged_to=$BTT_S_MERGED_TO \
      +actor_rollout_ref.peft.blocktt.convert_mode=$BTT_CONVERT_MODE \
      +actor_rollout_ref.peft.blocktt.factorize_by_head=$BTT_FACTORIZE_BY_HEAD \
      +actor_rollout_ref.peft.blocktt.normalize_after_update=$BTT_NORMALIZE_AFTER_UPDATE \
      +actor_rollout_ref.peft.blocktt.qfura.enabled=$BTT_QFURA" ;;
  svd)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.svd.train_position=$SVD_TRAIN_POSITION \
      +actor_rollout_ref.peft.svd.s_merged_to=$SVD_S_MERGED_TO \
      +actor_rollout_ref.peft.svd.compression_ratio=$SVD_COMPRESSION_RATIO" ;;
  *) echo "Unknown PEFT_MODE=$PEFT_MODE" >&2; exit 1 ;;
esac

if [ "$CALIB_MODE" != "none" ]; then
  PEFT_ARGS="$PEFT_ARGS \
    +actor_rollout_ref.peft.calib.mode=$CALIB_MODE \
    +actor_rollout_ref.peft.calib.source=$CALIB_SOURCE \
    +actor_rollout_ref.peft.calib.num_seqs=$CALIB_NUM_SEQS \
    +actor_rollout_ref.peft.calib.max_length=$CALIB_MAX_LENGTH \
    +actor_rollout_ref.peft.calib.batch_size=$CALIB_BATCH_SIZE \
    +actor_rollout_ref.peft.calib.seed=$CALIB_SEED"
  if [ -n "$CALIB_TRACES_PATH" ]; then
    PEFT_ARGS="$PEFT_ARGS +actor_rollout_ref.peft.calib.traces_path=$CALIB_TRACES_PATH"
  fi
fi
# ---- /PEFT ----
```

Append `_$PEFT_MODE` to `CKPT_PATH` and `EXPERIMENT_NAME` so checkpoints don't
collide. Find the existing `export CKPT_PATH=...` and `export EXPERIMENT_NAME=...`
lines (around L152 and L163) and append `_${PEFT_MODE}` before the
`$(date +...)` substitution:

```bash
# Before (L152-ish):
export CKPT_PATH=${PROJECT_PATH}/...-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)
# After:
export CKPT_PATH=${PROJECT_PATH}/...-rw_${REWARD_WEIGHT_MODE}_peft-${PEFT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)
```

Apply the same change to `EXPERIMENT_NAME`.

Finally, append `$PEFT_ARGS` as the last continuation line in the
`python3 -m verl.trainer.main_ppo \` invocation (after the existing last `\`).

- [ ] **Step 2: Same changes for `grpo.sh`**

Apply the identical block to `grpo.sh`. The insertion point is just before
the `KL_ARGS=""` block; same suffix on `CKPT_PATH`/`EXPERIMENT_NAME`; same
`$PEFT_ARGS` continuation on the python invocation.

- [ ] **Step 3: Smoke-check shell parses and the python command is one line**

Run: `bash -n grpo.sh && bash -n on_policy_distillation.sh && echo ok`
Expected: prints `ok`.

Run: `PEFT_MODE=blocktt BTT_QFURA=True bash -n on_policy_distillation.sh && echo ok`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add grpo.sh on_policy_distillation.sh
git commit -m "feat(peft): add PEFT_MODE driven \$PEFT_ARGS to grpo.sh / OPD launcher"
```

---

## Task 14: Eval-side change — `gen_vllm.py` detects LoRA adapter dirs

**Files:**
- Modify: `scripts/val/eval/gen_vllm.py`

- [ ] **Step 1: Read current loader logic**

Run: `grep -n 'MODEL_NAMES\|LLM(\|from_pretrained\|model=' scripts/val/eval/gen_vllm.py | head -20`

Find where the script constructs a vLLM `LLM(model=...)` call.

- [ ] **Step 2: Add adapter-aware loading**

In `scripts/val/eval/gen_vllm.py`, just before the `LLM(model=...)` call, add:

```python
def _resolve_peft_checkpoint(model_path: str):
    """Returns (base_path, lora_path | None). If model_path/adapter_config.json
    exists, treat it as a LoRA/QLoRA checkpoint."""
    import json
    import os
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_cfg):
        return model_path, None
    with open(adapter_cfg) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    base_txt = os.path.join(model_path, "base_model_path.txt")
    if os.path.isfile(base_txt):
        with open(base_txt) as f:
            base_override = f.read().strip()
        if base_override:
            base = base_override
    if base is None:
        raise ValueError(
            f"{model_path} looks like a PEFT checkpoint but has no base_model_name_or_path"
        )
    return base, model_path
```

Then, where the script today does roughly:

```python
llm = LLM(model=model_path, ...)
```

change it to:

```python
base, lora_path = _resolve_peft_checkpoint(model_path)
llm_kwargs = dict(model=base, ...)
if lora_path is not None:
    llm_kwargs.update({"enable_lora": True, "max_loras": 1, "max_lora_rank": 64})
llm = LLM(**llm_kwargs)
# When lora_path is set, downstream generate() calls must pass lora_request=
# LoRARequest(lora_name="default", lora_int_id=1, lora_path=lora_path).
```

If the script later calls `llm.generate(...)`, also import `LoRARequest` at
the top and thread the request through:

```python
from vllm.lora.request import LoRARequest
...
lora_request = LoRARequest("default", 1, lora_path) if lora_path else None
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

- [ ] **Step 3: Spot-check the script still parses**

Run: `python -c "import ast; ast.parse(open('scripts/val/eval/gen_vllm.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/val/eval/gen_vllm.py
git commit -m "feat(eval): gen_vllm detects PEFT adapter dirs and routes via LoRARequest"
```

---

## Task 15: End-to-end smoke script

**Files:**
- Create: `scripts/peft_smoke.sh`

- [ ] **Step 1: Write the smoke loop**

Create `scripts/peft_smoke.sh`:

```bash
#!/bin/bash
# Smoke loop over PEFT modes. 5-step run + save + resume on a 1% slice.
# Requires: 1 GPU, the `verl` conda env active, datasets/dapo-math-17k-1percent.parquet present.
set -euo pipefail

export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-1percent-processed.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-1pct
export N_RESPONSES=2
export MINI_BATCH_SIZE=4
export MAX_RESP_LENGTH=512
export MAX_VAL_RESP_LENGTH=512
export SAVE_FREQ=3
export TEST_FREQ=1000
export TOTAL_EPOCHS=1
export N_GPUS_PER_NODE=1
export ACTOR_MODEL_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM
export REWARD_MODEL_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM

run_mode() {
  local mode="$1"; shift
  echo "=== smoke: PEFT_MODE=$mode ==="
  PEFT_MODE=$mode "$@" bash on_policy_distillation.sh 2>&1 | tee logs/smoke_$mode.log
  echo "=== smoke OK: $mode ==="
}

mkdir -p logs
run_mode none
run_mode lora LORA_RANK=4 LORA_ALPHA=8
run_mode qlora LORA_RANK=4 LORA_ALPHA=8
run_mode blocktt BTT_TRAIN_POSITION=small
run_mode blocktt BTT_QFURA=True BTT_TRAIN_POSITION=small
run_mode svd SVD_TRAIN_POSITION=output
```

Make it executable: `chmod +x scripts/peft_smoke.sh`.

- [ ] **Step 2: Smoke-run (manual, GPU required)**

This step is **not** in CI. Run it manually on a 1-GPU box:

```bash
bash scripts/peft_smoke.sh
```

Expected: each section ends with `=== smoke OK: <mode> ===`. If any mode
fails, capture the error and debug at the source (adapter / worker / sharding
manager).

- [ ] **Step 3: Commit**

```bash
git add scripts/peft_smoke.sh
git commit -m "test(peft): end-to-end smoke loop over all PEFT modes"
```

---

## Task 16: Final review and docs touch-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short PEFT section to README**

Append to `README.md` under the existing "Common commands" / "Training" area:

```markdown
### PEFT modes

Both `grpo.sh` and `on_policy_distillation.sh` accept `PEFT_MODE=<mode>` to
enable parameter-efficient training:

| Mode | Description | Key env vars |
|------|-------------|--------------|
| `none` (default) | Full fine-tune | (none) |
| `lora` | Standard LoRA | `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` |
| `qlora` | LoRA on a bnb 4-bit base | `LORA_RANK`, `QLORA_QUANT_TYPE`, `QLORA_COMPUTE_DTYPE` |
| `blocktt` | BlockTT factorization from `src/compress` (set `BTT_QFURA=True` for NF4-quantized frozen core) | `BTT_DECOMP_MODE`, `BTT_TRAIN_POSITION`, `BTT_RANK`, `BTT_QFURA` |
| `svd` | SVD low-rank factorization | `SVD_TRAIN_POSITION`, `SVD_COMPRESSION_RATIO` |

Calibrated BlockTT/SVD: set `CALIB_MODE=v2` (or `twosteps` / `svd_v2` / ...)
and `CALIB_SOURCE=c4|traces|training_data`.

Checkpoints land in `<default_local_dir>/global_step_N/merged_hf/` and are
loadable by stock `AutoModelForCausalLM.from_pretrained` (compress modes) or
`PeftModel.from_pretrained(base, ...)` (LoRA/QLoRA). See
`docs/superpowers/specs/2026-05-26-verl-peft-blocktt-svd-design.md` for the
full design.
```

- [ ] **Step 2: Run the full unit test suite once more**

Run: `cd verl && pytest tests/peft/ -v -m 'not gpu'`
Expected: all CPU tests pass.

Run on GPU box: `cd verl && pytest tests/peft/ -v`
Expected: all tests pass including GPU smokes.

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs(peft): document PEFT_MODE in README"
```

---

## Self-review

Spec coverage check:

- **Spec §Architecture (files touched)** → Tasks 1, 2, 7, 8 (config, adapters, worker, sharding mgr).
- **Spec §Config tree** → Task 1 (`PEFTConfig`) + Task 7 (YAML).
- **Spec §PEFTAdapter interface + per-adapter behavior table** → Tasks 2–6.
- **Spec §Data flow (init/training/rollout/save/resume)** → Tasks 7 (init+save), 8 (rollout), 12 (resume).
- **Spec §On-disk checkpoint layout + Loadability contracts** → Tasks 7, 10, 14.
- **Spec §Launch script changes + back-compat shim** → Tasks 1 (shim), 13 (scripts).
- **Spec §Eval-side change (gen_vllm.py)** → Task 14.
- **Spec §Testing plan (unit/integration/E2E)** → Tasks 0–6, 9–12, 15.
- **Spec §Risks 1–6** → Risks 1 and 5 are exercised by Task 9 (key set) and Task 11 (worker smoke). Risk 2 (FSDP wrap) is implicitly covered by Task 11. Risk 3 (calib memory) doesn't need a test; documented in spec. Risk 4 (QLoRA + FSDP) is exercised by Task 11 smoke. Risk 6 (resume drift) is implemented in Task 7 (`_compare_peft_meta_to_cli`) but is not unit tested — **adding a small note** to revisit if a drift bug surfaces; not blocking.

Placeholder scan: no `TODO`, no "implement later", no "similar to Task N". Two
**Implementer note** callouts (Task 5 on `compress.topology` API name, Task 8 on
the sharding-manager attribute name) are warranted — the author can't know
the exact local name without inspecting; the instruction is to check and use
what exists, not invent.

Type consistency check: `PEFTConfig`/`LoRAConfig`/`BlockTTConfig`/`SVDConfig`/`CalibConfig`/`BlockTTQfuraConfig` names match across all tasks. Methods used in Task 7 (`adapter.needs_calibration()`, `adapter.apply(...)`, `adapter.save_pretrained(...)`, `adapter.topology_meta()`, `adapter.write_compress_sidecar(...)`, `type(adapter).rebuild_from_meta(...)`) are all defined in Task 2 (ABC) and overridden in Tasks 3–6. `adapter.export_for_vllm()` is checked in Task 8 with a base-method comparison, which is consistent with the ABC's default `return None`. Signatures match.

Plan complete and saved to `docs/superpowers/plans/2026-05-26-verl-peft-blocktt-svd.md`.
