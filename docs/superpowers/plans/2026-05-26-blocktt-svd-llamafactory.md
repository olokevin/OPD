# BlockTT & SVD in LlamaFactory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `finetuning_type: blocktt` and `finetuning_type: svd` to LlamaFactory's SFT pipeline so the existing `src/compress` BlockTT/SVD finetuning is drivable from a YAML.

**Architecture:** New `CompressArguments` mixin extends `FinetuningArguments`; `init_adapter()` dispatches into a new `model/compress_setup.py` that lazy-imports `src/compress`; two `TrainerCallback`s in `train/callbacks.py` (per-step BTT-core normalization + merged-HF checkpoint export) wired in `train/tuner.py`; three example YAMLs under `examples/train_blocktt/`.

**Tech Stack:** Python 3.11 (`sft` conda env), HuggingFace Transformers/Trainer, DeepSpeed ZeRO-2, PyTorch, `src/compress` (`compress.integration` public API).

**Spec:** `docs/superpowers/specs/2026-05-26-blocktt-svd-llamafactory-design.md`

---

## Conventions used across this plan

- All paths are relative to repo root `/home/yequan/Project/compression/OPD/`.
- Wherever the LlamaFactory YAML keys differ from `run_rl.py`'s `--hyphen-flag` names, the dataclass uses **underscore** names and we pass `hyphen_style=False` into `compress.integration` validators.
- `_compress_setup` log calls follow LlamaFactory's `logger.info_rank0(...)` convention (imported from `llamafactory.extras.logging`).
- All pytest commands assume `cd LlamaFactory/ && conda activate sft`.

## File structure

| File | Role |
|---|---|
| `LlamaFactory/src/llamafactory/hparams/finetuning_args.py` | **Modify**. Add `CompressArguments` mixin (top of file, alphabetically near `GaloreArguments`). Extend `FinetuningArguments` mixin list. Extend `finetuning_type` Literal. Update `__post_init__` validators. |
| `LlamaFactory/src/llamafactory/model/compress_setup.py` | **Create**. Sole owner of `from compress.integration import ...`. Lazy `sys.path` injection. Plain + calibrated init paths. |
| `LlamaFactory/src/llamafactory/model/adapter.py` | **Modify**. Add a new `elif finetuning_args.finetuning_type in {"blocktt","svd"}:` branch to `init_adapter()` that calls `init_compress_model`. |
| `LlamaFactory/src/llamafactory/train/callbacks.py` | **Modify**. Add `CompressNormalizeCallback` and `CompressSaveCallback` (with private `_materialize_and_save` helper) at end of file. |
| `LlamaFactory/src/llamafactory/train/tuner.py` | **Modify**. Append the two new callbacks to the callbacks list when `finetuning_type in {blocktt,svd}`. |
| `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_sft.yaml` | **Create**. Plain BTT recipe. |
| `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_calibrated_sft.yaml` | **Create**. Calibrated BTT recipe. |
| `LlamaFactory/examples/train_blocktt/qwen3_base_svd_sft.yaml` | **Create**. Plain SVD recipe. |
| `LlamaFactory/examples/train_blocktt/README.md` | **Create**. Knob reference. |
| `LlamaFactory/tests/llamafactory/hparams/test_compress_args.py` | **Create**. Validator unit tests. |
| `LlamaFactory/tests/llamafactory/model/test_compress_setup.py` | **Create**. Path-injection + dispatch unit tests. |
| `LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py` | **Create**. Materialize helper tests on a tiny model. |

---

## Task 1: Skeleton for `CompressArguments` mixin (failing test first)

**Files:**
- Create: `LlamaFactory/tests/llamafactory/hparams/test_compress_args.py`
- Modify: `LlamaFactory/src/llamafactory/hparams/finetuning_args.py` (add stub mixin)

This task establishes the dataclass surface only. Validators come in later tasks.

- [ ] **Step 1: Write the failing test**

Create `LlamaFactory/tests/llamafactory/hparams/test_compress_args.py`:

```python
"""Unit tests for CompressArguments mixin on FinetuningArguments."""
from llamafactory.hparams.finetuning_args import FinetuningArguments


def _make(**overrides):
    return FinetuningArguments(**overrides)


def test_default_compress_fields_present():
    fa = _make()
    # Shared defaults
    assert fa.trainable_type == "all"
    assert fa.train_position is None
    assert fa.s_merged_to is None
    # BTT-only defaults
    assert fa.decomp_mode == "input_one_block"
    assert fa.blocktt_rank == "full"
    assert fa.convert_mode == "svd"
    assert fa.train_bias is True
    assert fa.blocktt_normalize_after_update is False
    assert fa.blocktt_factorize_by_head is True
    # Calib defaults (mirroring add_calibrated_btt_args)
    assert fa.calib_mode == "none"
    assert fa.calib_source == "c4"
    assert fa.calib_num_seqs == 128
    assert fa.calib_max_length == 2048
    assert fa.calib_seed == 3
    assert fa.calib_batch_size == 8
    assert fa.calib_traces_path is None
    assert fa.compression_ratio == 1.0


def test_finetuning_type_blocktt_and_svd_accepted():
    _make(finetuning_type="blocktt")
    _make(finetuning_type="svd")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/hparams/test_compress_args.py -v
```

Expected: both tests FAIL with `AttributeError` (attributes don't exist) and `AssertionError` (Literal doesn't accept `blocktt`/`svd`).

- [ ] **Step 3: Add the mixin (no validators yet)**

In `LlamaFactory/src/llamafactory/hparams/finetuning_args.py`, immediately above the `@dataclass class FinetuningArguments(` line (search for `class FinetuningArguments(`), add:

```python
@dataclass
class CompressArguments:
    """BlockTT / SVD compression-aware finetuning knobs (see src/compress)."""

    # Shared (blocktt + svd)
    trainable_type: Literal["all", "mlp", "attn"] = field(
        default="all",
        metadata={"help": "Compress target modules: all | mlp | attn."},
    )
    train_position: Optional[Literal["output", "input", "small", "large", "both"]] = field(
        default=None,
        metadata={"help": "Trainable side: svd uses output|input|both, blocktt uses small|large|both."},
    )
    s_merged_to: Optional[Literal[
        "frozen", "trainable", "output", "input",
        "split", "keep_frozen", "keep_trainable",
    ]] = field(
        default=None,
        metadata={"help": "Where to merge SVD S during init."},
    )

    # BlockTT-only
    decomp_mode: str = field(
        default="input_one_block",
        metadata={"help": "BlockTT decomp mode (scalar or dict literal)."},
    )
    blocktt_rank: str = field(
        default="full",
        metadata={"help": "BTT rank: 'full' or positive integer string."},
    )
    convert_mode: Literal["svd", "qr"] = field(
        default="svd",
        metadata={"help": "Per-block decomposition for BlockTT init: svd | qr."},
    )
    train_bias: bool = field(
        default=True,
        metadata={"help": "Train BTT biases (mirror of --no-train-bias inverted)."},
    )
    blocktt_normalize_after_update: bool = field(
        default=False,
        metadata={"help": "Normalize trainable BTT cores after each optimizer step."},
    )
    blocktt_factorize_by_head: bool = field(
        default=True,
        metadata={"help": "Align attention BTT blocks with head structure."},
    )

    # Calibrated init (1:1 with add_calibrated_btt_args, hyphen_style=False)
    calib_mode: Literal[
        "none", "v2", "v2_bp", "v2_combined",
        "twosteps", "svd_v2", "svd_v2_combined",
    ] = field(
        default="none",
        metadata={"help": "Calibrated init mode. 'none' = plain conversion."},
    )
    calib_source: Literal["c4", "traces", "training_data"] = field(
        default="c4",
        metadata={"help": "Calibration data source."},
    )
    calib_num_seqs: int = field(
        default=128,
        metadata={"help": "Number of calibration sequences."},
    )
    calib_max_length: int = field(
        default=2048,
        metadata={"help": "Max token length per calibration sample."},
    )
    calib_seed: int = field(
        default=3,
        metadata={"help": "RNG seed for calibration sampling."},
    )
    calib_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for calibration DataLoader."},
    )
    calib_traces_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to traces JSONL when calib_source=traces."},
    )
    compression_ratio: float = field(
        default=1.0,
        metadata={"help": "SVD compression ratio in (0, 1] for svd_v2 modes."},
    )
```

Then update the `FinetuningArguments` mixin list — at the existing `class FinetuningArguments(` declaration, append `CompressArguments` to the parent tuple:

```python
@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
    CompressArguments,    # <-- ADD THIS LINE
):
```

Then extend the `finetuning_type` Literal — find the existing line:

```python
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
```

Replace with:

```python
    finetuning_type: Literal["lora", "oft", "freeze", "full", "blocktt", "svd"] = field(
```

And update the assert inside `__post_init__` — find:

```python
        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
```

Replace with:

```python
        assert self.finetuning_type in [
            "lora", "oft", "freeze", "full", "blocktt", "svd",
        ], "Invalid fine-tuning method."
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/hparams/test_compress_args.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/hparams/finetuning_args.py \
        LlamaFactory/tests/llamafactory/hparams/test_compress_args.py
git commit -m "feat(llamafactory): add CompressArguments mixin and finetuning_type literals"
```

---

## Task 2: Validators in `FinetuningArguments.__post_init__`

**Files:**
- Modify: `LlamaFactory/src/llamafactory/hparams/finetuning_args.py:443` area (`__post_init__`)
- Modify: `LlamaFactory/tests/llamafactory/hparams/test_compress_args.py` (add tests)

- [ ] **Step 1: Add the failing validator tests**

Append to `LlamaFactory/tests/llamafactory/hparams/test_compress_args.py`:

```python
import pytest


def test_blocktt_default_train_position_small():
    fa = _make(finetuning_type="blocktt")
    assert fa.train_position == "small"


def test_svd_default_train_position_output():
    fa = _make(finetuning_type="svd")
    assert fa.train_position == "output"


def test_compress_default_s_merged_to_frozen():
    assert _make(finetuning_type="blocktt").s_merged_to == "frozen"
    assert _make(finetuning_type="svd").s_merged_to == "frozen"


def test_blocktt_train_position_rejects_svd_values():
    with pytest.raises(ValueError, match="train_position"):
        _make(finetuning_type="blocktt", train_position="output")


def test_svd_train_position_rejects_blocktt_values():
    with pytest.raises(ValueError, match="train_position"):
        _make(finetuning_type="svd", train_position="small")


def test_blocktt_both_with_frozen_s_merged_rejected():
    with pytest.raises(ValueError, match="train_position.*both"):
        _make(finetuning_type="blocktt", train_position="both", s_merged_to="frozen")


def test_blocktt_qr_with_s_merged_to_warns(caplog):
    # warn-and-ignore: object constructs, s_merged_to gets cleared
    fa = _make(finetuning_type="blocktt", convert_mode="qr", s_merged_to="output")
    assert fa.s_merged_to is None
    assert any("convert_mode=qr" in r.message for r in caplog.records)


def test_calib_mode_requires_compress_finetuning():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="full", calib_mode="v2")


def test_svd_calib_mode_rejects_blocktt_method():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="blocktt", calib_mode="svd_v2")


def test_btt_calib_mode_rejects_svd_method():
    with pytest.raises(ValueError, match="calib_mode"):
        _make(finetuning_type="svd", calib_mode="v2")


def test_blocktt_rank_must_parse():
    with pytest.raises(ValueError, match="blocktt_rank"):
        _make(finetuning_type="blocktt", blocktt_rank="notanumber")
    _make(finetuning_type="blocktt", blocktt_rank="full")  # OK
    _make(finetuning_type="blocktt", blocktt_rank="16")    # OK


def test_compress_rejects_galore_apollo_badam():
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="blocktt", use_galore=True)
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="svd", use_apollo=True)
    with pytest.raises(ValueError, match="GaLore|APOLLO|BAdam"):
        _make(finetuning_type="blocktt", use_badam=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/hparams/test_compress_args.py -v
```

Expected: 11 new tests FAIL (defaults not applied, no rejections happen). Original 2 still pass.

- [ ] **Step 3: Implement the validators**

In `LlamaFactory/src/llamafactory/hparams/finetuning_args.py`, inside `FinetuningArguments.__post_init__` (after the existing assert that we updated in Task 1, and after the existing `int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1` block), append:

```python
        # ---------------------------------------------------------------
        # BlockTT / SVD validators (CompressArguments)
        # ---------------------------------------------------------------
        if self.finetuning_type in ("blocktt", "svd"):
            # Reject ZeRO-3 (custom BTT/SVD layers don't survive param sharding)
            ds = ""
            for attr in ("deepspeed",):
                v = getattr(self, attr, None)
                if isinstance(v, str):
                    ds = v
            if "z3" in ds or "zero3" in ds:
                raise ValueError(
                    f"finetuning_type={self.finetuning_type!r} does not support "
                    f"DeepSpeed ZeRO-3 (got deepspeed={ds!r})."
                )

            # Reject co-use with GaLore / APOLLO / BAdam
            if self.use_galore or self.use_apollo or self.use_badam:
                raise ValueError(
                    "Cannot use GaLore, APOLLO or BAdam with "
                    f"finetuning_type={self.finetuning_type!r}."
                )

            # Default train_position
            if self.train_position is None:
                self.train_position = "small" if self.finetuning_type == "blocktt" else "output"

            # train_position whitelists
            if self.finetuning_type == "blocktt" and self.train_position not in (
                "small", "large", "both",
            ):
                raise ValueError(
                    "blocktt train_position must be one of small|large|both "
                    f"(got {self.train_position!r})."
                )
            if self.finetuning_type == "svd" and self.train_position not in (
                "output", "input", "both",
            ):
                raise ValueError(
                    "svd train_position must be one of output|input|both "
                    f"(got {self.train_position!r})."
                )

            # Default s_merged_to (skip QR which has no S to merge)
            if self.s_merged_to is None and not (
                self.finetuning_type == "blocktt" and self.convert_mode == "qr"
            ):
                self.s_merged_to = "frozen"

            # blocktt + train_position=both + s_merged_to in {frozen,trainable} is invalid
            if (
                self.finetuning_type == "blocktt"
                and self.train_position == "both"
                and self.s_merged_to in ("frozen", "trainable")
            ):
                raise ValueError(
                    "blocktt train_position=both is incompatible with "
                    f"s_merged_to={self.s_merged_to!r}; pick output|input|split."
                )

            # blocktt convert_mode=qr ignores s_merged_to (warn-and-clear)
            if (
                self.finetuning_type == "blocktt"
                and self.convert_mode == "qr"
                and self.s_merged_to is not None
            ):
                import logging
                logging.getLogger(__name__).warning(
                    "convert_mode=qr has no singular values; ignoring "
                    f"s_merged_to={self.s_merged_to!r}.",
                )
                self.s_merged_to = None

            # blocktt_rank parseable
            if self.blocktt_rank != "full":
                try:
                    rank_int = int(self.blocktt_rank)
                except ValueError as exc:
                    raise ValueError(
                        f"blocktt_rank must be 'full' or a positive integer string "
                        f"(got {self.blocktt_rank!r})."
                    ) from exc
                if rank_int <= 0:
                    raise ValueError(
                        f"blocktt_rank must be > 0 (got {rank_int})."
                    )

        # calib_mode is only meaningful when finetuning_type ∈ {blocktt, svd}
        if self.calib_mode != "none":
            if self.finetuning_type not in ("blocktt", "svd"):
                raise ValueError(
                    f"calib_mode={self.calib_mode!r} only valid with "
                    "finetuning_type blocktt or svd."
                )
            if self.calib_mode.startswith("svd_") and self.finetuning_type != "svd":
                raise ValueError(
                    f"calib_mode={self.calib_mode!r} (an SVD mode) requires "
                    "finetuning_type=svd."
                )
            if (not self.calib_mode.startswith("svd_")) and self.finetuning_type != "blocktt":
                raise ValueError(
                    f"calib_mode={self.calib_mode!r} (a BTT mode) requires "
                    "finetuning_type=blocktt."
                )
```

Also add the imports needed at the top of `finetuning_args.py` — confirm `Optional` is imported (it is via `from typing import` already if PEP 604 not in use; if file uses `int | None`-style hints, keep using `Optional` for explicit annotations to match existing mixins).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/hparams/test_compress_args.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/hparams/finetuning_args.py \
        LlamaFactory/tests/llamafactory/hparams/test_compress_args.py
git commit -m "feat(llamafactory): CompressArguments validators in __post_init__"
```

---

## Task 3: `model/compress_setup.py` skeleton with lazy sys.path injection

**Files:**
- Create: `LlamaFactory/src/llamafactory/model/compress_setup.py`
- Create: `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`

This task only wires up the lazy import and the dispatch entry-point. Conversion logic comes in Task 4.

- [ ] **Step 1: Write the failing test**

Create `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`:

```python
"""Unit tests for compress_setup lazy import and dispatch."""
import sys
import pathlib

import pytest


def test_ensure_compress_on_path_idempotent():
    from llamafactory.model import compress_setup
    # Drop any pre-existing entry
    src_dir = compress_setup._repo_src_dir()
    while str(src_dir) in sys.path:
        sys.path.remove(str(src_dir))
    compress_setup._ensure_compress_on_path()
    assert str(src_dir) in sys.path
    # Idempotent
    n = sys.path.count(str(src_dir))
    compress_setup._ensure_compress_on_path()
    assert sys.path.count(str(src_dir)) == n


def test_repo_src_dir_resolves_to_opd_src():
    from llamafactory.model import compress_setup
    src = compress_setup._repo_src_dir()
    assert src.name == "src"
    # ".../OPD/src" → parent is the repo root, which must contain LlamaFactory
    assert (src.parent / "LlamaFactory").exists()


def test_init_compress_model_skips_when_not_trainable():
    from llamafactory.model import compress_setup
    sentinel = object()
    out = compress_setup.init_compress_model(
        config=None, model=sentinel, model_args=None,
        finetuning_args=_FakeFA(finetuning_type="blocktt", calib_mode="none"),
        is_trainable=False,
    )
    assert out is sentinel


class _FakeFA:
    """Minimal FinetuningArguments stand-in for unit tests."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `compress_setup.py` skeleton**

Create `LlamaFactory/src/llamafactory/model/compress_setup.py`:

```python
"""Lazy entry point into ``src/compress`` for LlamaFactory's BlockTT/SVD
finetuning_type. This is the only LlamaFactory module that imports
``compress.integration``."""
from __future__ import annotations

import pathlib
import sys
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from ..extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel
    from ..hparams.finetuning_args import FinetuningArguments
    from ..hparams.model_args import ModelArguments

logger = get_logger(__name__)


def _repo_src_dir() -> pathlib.Path:
    """Return the path to ``<repo>/src`` so the ``compress`` package is importable."""
    # compress_setup.py is at LlamaFactory/src/llamafactory/model/compress_setup.py
    # parents: [0]=model, [1]=llamafactory, [2]=src, [3]=LlamaFactory, [4]=<repo>
    return pathlib.Path(__file__).resolve().parents[4] / "src"


def _ensure_compress_on_path() -> None:
    """Lazily prepend ``<repo>/src`` to ``sys.path``. Idempotent."""
    src_dir = str(_repo_src_dir())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info_rank0(f"compress_setup: added {src_dir} to sys.path")


def init_compress_model(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    """Apply BlockTT / SVD conversion + trainability to ``model``.

    For inference (``is_trainable=False``) the function is a no-op: trained
    checkpoints are saved as plain dense HF weights (see CompressSaveCallback),
    so the inference path uses the standard full-model loader.
    """
    if not is_trainable:
        return model

    _ensure_compress_on_path()

    # Conversion logic lands in Task 4. For now, raise an explicit error so
    # accidentally hitting this path during testing is loud rather than silent.
    raise NotImplementedError(
        "init_compress_model: conversion path not implemented yet; "
        "see Task 4 of the implementation plan."
    )


def _to_namespace(finetuning_args: "FinetuningArguments") -> Namespace:
    """Build a ``run_rl.py``-style argparse.Namespace from ``finetuning_args``.

    ``compress.integration`` helpers (``validate_calibrated_btt_args``,
    ``build_calib_loader``, ``apply_calibrated_btt``/``apply_calibrated_svd``)
    expect attribute access in the same shape as ``run_rl.py``'s argparse
    namespace, with ``hyphen_style=False`` for underscore field names.
    """
    fa = finetuning_args
    ns = Namespace()
    # train_mode: compress.integration.validate_calibrated_btt_args reads
    # this when validating calib_mode (see integration.py line ~135).
    ns.train_mode = fa.finetuning_type            # "blocktt" or "svd"
    # Compress knobs (underscore names; hyphen_style=False)
    for attr in (
        "trainable_type", "train_position", "s_merged_to",
        "decomp_mode", "blocktt_rank", "convert_mode", "train_bias",
        "blocktt_normalize_after_update", "blocktt_factorize_by_head",
        "calib_mode", "calib_source", "calib_num_seqs", "calib_max_length",
        "calib_seed", "calib_batch_size", "calib_traces_path", "compression_ratio",
    ):
        setattr(ns, attr, getattr(fa, attr))
    return ns
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/model/compress_setup.py \
        LlamaFactory/tests/llamafactory/model/test_compress_setup.py
git commit -m "feat(llamafactory): compress_setup skeleton with lazy sys.path injection"
```

---

## Task 4: Plain BlockTT + SVD conversion in `init_compress_model`

**Files:**
- Modify: `LlamaFactory/src/llamafactory/model/compress_setup.py`
- Modify: `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`

- [ ] **Step 1: Write the failing test (tiny model, plain BTT)**

Append to `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`:

```python
import torch
import torch.nn as nn


def _tiny_qwen_like_model():
    """Build a minimal Qwen-shaped nn.Module with the linear submodules
    compress.integration looks for (q/k/v/o/gate/up/down)."""
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.k_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)
            self.o_proj = nn.Linear(32, 32, bias=False)
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)
        def forward(self, x):
            x = self.q_proj(x) + self.k_proj(x) + self.v_proj(x) + self.o_proj(x)
            return self.down_proj(self.gate_proj(x).relu() * self.up_proj(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])
        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return x
    return Model()


def test_plain_blocktt_converts_linear_modules():
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    from compress.integration import BTTLinear

    model = _tiny_qwen_like_model()
    fa = _FakeFA(
        finetuning_type="blocktt", calib_mode="none",
        trainable_type="all", train_position="small",
        s_merged_to="frozen", decomp_mode="input_one_block",
        blocktt_rank="full", convert_mode="svd", train_bias=True,
        blocktt_normalize_after_update=False, blocktt_factorize_by_head=True,
    )
    out = compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )

    btt_count = sum(1 for m in out.modules() if isinstance(m, BTTLinear))
    assert btt_count > 0, "expected some Linear modules converted to BTTLinear"

    trainable = [n for n, p in out.named_parameters() if p.requires_grad]
    assert any(".btt_l" in n or ".btt_r" in n for n in trainable), trainable


def test_plain_svd_converts_linear_modules():
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    from compress.integration import SVDCompressedLinear

    model = _tiny_qwen_like_model()
    fa = _FakeFA(
        finetuning_type="svd", calib_mode="none",
        trainable_type="all", train_position="output",
        s_merged_to="frozen", decomp_mode="input_one_block",
        blocktt_rank="full", convert_mode="svd", train_bias=True,
        blocktt_normalize_after_update=False, blocktt_factorize_by_head=True,
    )
    out = compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )

    svd_count = sum(1 for m in out.modules() if isinstance(m, SVDCompressedLinear))
    assert svd_count > 0

    trainable = [n for n, p in out.named_parameters() if p.requires_grad]
    assert any(".U_r" in n or ".V_r" in n for n in trainable), trainable
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py::test_plain_blocktt_converts_linear_modules tests/llamafactory/model/test_compress_setup.py::test_plain_svd_converts_linear_modules -v
```

Expected: FAIL with `NotImplementedError` from Task 3 skeleton.

- [ ] **Step 3: Replace the `NotImplementedError` with the plain conversion path**

In `LlamaFactory/src/llamafactory/model/compress_setup.py`, replace the body of `init_compress_model` (after the `if not is_trainable: return model` short-circuit) with:

```python
    _ensure_compress_on_path()
    from compress.integration import (
        convert_linear_to_btt_compress,
        convert_linear_to_svd_compress,
        configure_compress_btt_trainability,
        configure_compress_svd_trainability,
        get_blocktt_target_module_names,
        get_svd_target_module_names,
        resolve_blocktt_decomp_modes,
    )

    fa = finetuning_args
    method = fa.finetuning_type

    if fa.calib_mode != "none":
        # Calibrated path lands in Task 5.
        raise NotImplementedError(
            "compress_setup: calibrated init not implemented yet (Task 5)."
        )

    rank = _resolve_rank(fa.blocktt_rank)

    if method == "blocktt":
        targets = get_blocktt_target_module_names(fa.trainable_type)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            fa.decomp_mode, include_names=targets,
        )
        convert_linear_to_btt_compress(
            model,
            target_module_names=targets,
            module_decomp_modes=module_decomp_modes,
            rank=rank,
            convert_mode=fa.convert_mode,
            s_merged_to=fa.s_merged_to,
            factorize_by_head=fa.blocktt_factorize_by_head,
        )
        configure_compress_btt_trainability(
            model,
            train_position=fa.train_position,
            train_bias=fa.train_bias,
        )
    elif method == "svd":
        targets = get_svd_target_module_names(fa.trainable_type)
        convert_linear_to_svd_compress(
            model,
            target_module_names=targets,
            s_merged_to=fa.s_merged_to,
        )
        configure_compress_svd_trainability(
            model,
            train_position=fa.train_position,
        )
    else:
        raise ValueError(
            f"compress_setup: unsupported finetuning_type={method!r}; "
            "expected 'blocktt' or 'svd'."
        )

    return model
```

Also add the `_resolve_rank` helper at module scope:

```python
def _resolve_rank(rank_arg: str):
    """Parse ``blocktt_rank`` into the value ``convert_linear_to_btt_compress``
    accepts: the literal string ``"full"`` or a positive ``int``. Mirrors
    ``run_rl.py::resolve_blocktt_rank``."""
    if rank_arg == "full":
        return "full"
    try:
        rank_int = int(rank_arg)
    except ValueError as exc:
        raise ValueError(
            f"blocktt_rank must be 'full' or a positive integer string "
            f"(got {rank_arg!r})."
        ) from exc
    if rank_int <= 0:
        raise ValueError(f"blocktt_rank must be > 0 (got {rank_int}).")
    return rank_int
```

Important: pass any kwargs that `convert_linear_to_btt_compress` / `convert_linear_to_svd_compress` accept. **Before committing**, run `python -c "from compress.integration import convert_linear_to_btt_compress, convert_linear_to_svd_compress; import inspect; print(inspect.signature(convert_linear_to_btt_compress)); print(inspect.signature(convert_linear_to_svd_compress))"` in the `verl` env (since `sft` env doesn't have the compress deps yet) and adjust the kwarg names if they differ from the design (`target_module_names`, `module_decomp_modes`, `rank`, `convert_mode`, `s_merged_to`, `factorize_by_head`). If a kwarg is unrecognized, drop it for now (defaults will be used) and note it in the README created in Task 9.

If the `sft` env cannot import `compress` because of missing deps (transformers/torch are present but other compress deps may not be), document the smoke run in `examples/train_blocktt/README.md` (Task 9) as "single-GPU smoke must use the `sft` env after `pip install -r requirements.txt` from the repo root if any compress deps are missing".

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/model/compress_setup.py \
        LlamaFactory/tests/llamafactory/model/test_compress_setup.py
git commit -m "feat(llamafactory): plain BlockTT/SVD conversion in init_compress_model"
```

---

## Task 5: Calibrated BTT / SVD path

**Files:**
- Modify: `LlamaFactory/src/llamafactory/model/compress_setup.py`
- Modify: `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`

- [ ] **Step 1: Write the failing test (calibrated BTT with tiny c4-like loader)**

Append to `LlamaFactory/tests/llamafactory/model/test_compress_setup.py`:

```python
def test_calibrated_btt_v2_runs(monkeypatch):
    """Calibrated v2 path: validates that compress_setup wires
    validate_calibrated_btt_args + build_calib_loader + apply_calibrated_btt
    together. We monkeypatch each of the three to assert dispatch."""
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    seen = {}
    fake_loader = object()

    def fake_validate(args, *, argv, hyphen_style):
        seen["validate"] = (args, argv, hyphen_style)

    def fake_build(args, *, tokenizer, training_dataset=None, training_collate_fn=None,
                   rl_rollout_fn=None, hyphen_style=True):
        seen["build"] = {"args": args, "tokenizer": tokenizer,
                         "hyphen_style": hyphen_style}
        return fake_loader

    def fake_apply_btt(model, args, *, calib_loader, device=None, hyphen_style=True):
        seen["apply_btt"] = {"loader": calib_loader, "hyphen_style": hyphen_style}
        return model, {"num_btt_layers": 1}

    monkeypatch.setattr(ci, "validate_calibrated_btt_args", fake_validate)
    monkeypatch.setattr(ci, "build_calib_loader", fake_build)
    monkeypatch.setattr(ci, "apply_calibrated_btt", fake_apply_btt)
    # The tokenizer load is also monkeypatched — calibrated path doesn't
    # need a real one for this dispatch test.
    monkeypatch.setattr(compress_setup, "_load_tokenizer", lambda model_args: object())

    model = _tiny_qwen_like_model()
    fa = _FakeFA(
        finetuning_type="blocktt", calib_mode="v2", calib_source="c4",
        calib_num_seqs=4, calib_max_length=16, calib_seed=0, calib_batch_size=1,
        calib_traces_path=None, compression_ratio=1.0,
        trainable_type="all", train_position="small", s_merged_to="frozen",
        decomp_mode="input_one_block", blocktt_rank="full", convert_mode="svd",
        train_bias=True, blocktt_normalize_after_update=False,
        blocktt_factorize_by_head=True,
    )
    out = compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )

    assert seen["validate"][2] is False        # hyphen_style=False
    assert seen["build"]["hyphen_style"] is False
    assert seen["apply_btt"]["loader"] is fake_loader
    assert seen["apply_btt"]["hyphen_style"] is False
    # apply_calibrated_btt returns (model, stats); the model in the namespace
    # is what init_compress_model returns.
    assert out is model
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py::test_calibrated_btt_v2_runs -v
```

Expected: FAIL with `NotImplementedError("compress_setup: calibrated init not implemented yet (Task 5).")`.

- [ ] **Step 3: Implement the calibrated branch**

In `LlamaFactory/src/llamafactory/model/compress_setup.py`, replace the `raise NotImplementedError` calibrated stub with:

```python
    if fa.calib_mode != "none":
        from compress.integration import (
            validate_calibrated_btt_args,
            build_calib_loader,
            apply_calibrated_btt,
            apply_calibrated_svd,
        )
        ns = _to_namespace(fa)
        validate_calibrated_btt_args(ns, argv=None, hyphen_style=False)
        tokenizer = _load_tokenizer(model_args)
        calib_loader = build_calib_loader(
            ns, tokenizer=tokenizer, hyphen_style=False,
        )
        if calib_loader is None:
            raise RuntimeError(
                f"compress_setup: build_calib_loader returned None for "
                f"calib_mode={fa.calib_mode!r}, calib_source={fa.calib_source!r}."
            )
        if method == "blocktt":
            model, _stats = apply_calibrated_btt(
                model, ns, calib_loader=calib_loader, hyphen_style=False,
            )
        elif method == "svd":
            model = apply_calibrated_svd(
                model, ns, calib_loader=calib_loader, hyphen_style=False,
            )
        else:
            raise ValueError(
                f"compress_setup: unsupported finetuning_type={method!r} for calibrated init."
            )
        return model
```

(Move this block above the plain-path branch — calibrated init replaces the plain branch when active.)

Add the `_load_tokenizer` helper at module scope:

```python
def _load_tokenizer(model_args: "ModelArguments") -> Any:
    """Load the tokenizer for calibration. Imports HF lazily to keep
    cold-start fast in the plain (non-calibrated) path."""
    from transformers import AutoTokenizer
    name = getattr(model_args, "model_name_or_path", None)
    if not name:
        raise ValueError(
            "compress_setup: model_args.model_name_or_path is required for "
            "calibrated BlockTT/SVD finetuning."
        )
    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/model/test_compress_setup.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/model/compress_setup.py \
        LlamaFactory/tests/llamafactory/model/test_compress_setup.py
git commit -m "feat(llamafactory): calibrated BlockTT/SVD path in compress_setup"
```

---

## Task 6: Wire `init_compress_model` into `init_adapter`

**Files:**
- Modify: `LlamaFactory/src/llamafactory/model/adapter.py:321-365` (`init_adapter`)

- [ ] **Step 1: Read the current branches**

Open `LlamaFactory/src/llamafactory/model/adapter.py` and locate the if/elif chain (around line 355):

```python
    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")
```

- [ ] **Step 2: Add the new branch**

Insert the new branch immediately above the final `else: raise NotImplementedError(...)`:

```python
    elif finetuning_args.finetuning_type in ("blocktt", "svd"):
        from .compress_setup import init_compress_model
        model = init_compress_model(config, model, model_args, finetuning_args, is_trainable)
        # Compress modules are float32-internally where they need to be; the
        # upcasting decision above doesn't apply to BTTLinear/SVDCompressedLinear
        # which manage their own dtype.
```

- [ ] **Step 3: Smoke-import to confirm no syntax errors**

```bash
cd LlamaFactory && conda run -n sft python -c "from llamafactory.model.adapter import init_adapter; print('OK')"
```

Expected: prints `OK` with no traceback.

- [ ] **Step 4: Re-run the prior test suites to confirm no regressions**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/hparams/test_compress_args.py tests/llamafactory/model/test_compress_setup.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/model/adapter.py
git commit -m "feat(llamafactory): dispatch blocktt/svd in init_adapter"
```

---

## Task 7: `CompressNormalizeCallback` (per-step BTT normalization)

**Files:**
- Modify: `LlamaFactory/src/llamafactory/train/callbacks.py` (append)
- Create: `LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py`

- [ ] **Step 1: Write the failing test**

Create `LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py`:

```python
"""Unit tests for CompressNormalizeCallback and CompressSaveCallback."""
import pathlib

import pytest
import torch
import torch.nn as nn


class _FakeFA:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_normalize_callback_disabled_when_flag_off():
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=False,
    ))
    assert cb.enabled is False


def test_normalize_callback_disabled_for_svd():
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="svd", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is False


def test_normalize_callback_calls_compress_helper(monkeypatch):
    """When enabled, on_step_end should call normalize_trainable_blocktt_cores_."""
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    calls = []
    monkeypatch.setattr(
        ci, "normalize_trainable_blocktt_cores_",
        lambda m: calls.append(m),
    )

    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    assert cb.enabled is True

    fake_model = nn.Linear(4, 4)
    cb.on_step_end(args=None, state=None, control=None, model=fake_model)
    assert calls == [fake_model]


def test_normalize_callback_no_model_kwarg_is_noop(monkeypatch):
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    import compress.integration as ci

    calls = []
    monkeypatch.setattr(
        ci, "normalize_trainable_blocktt_cores_",
        lambda m: calls.append(m),
    )
    from llamafactory.train.callbacks import CompressNormalizeCallback
    cb = CompressNormalizeCallback(_FakeFA(
        finetuning_type="blocktt", blocktt_normalize_after_update=True,
    ))
    cb.on_step_end(args=None, state=None, control=None, model=None)
    assert calls == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/train/test_compress_callbacks.py -v
```

Expected: FAIL — `CompressNormalizeCallback` does not exist.

- [ ] **Step 3: Implement the callback**

Append to `LlamaFactory/src/llamafactory/train/callbacks.py`:

```python
class CompressNormalizeCallback(TrainerCallback):
    """Normalize trainable BTT cores after each optimizer step.

    Mirrors run_rl.py's blocktt_normalize_after_update behavior. No-op when
    finetuning_type != blocktt or when the flag is False.
    """

    def __init__(self, finetuning_args):
        self.enabled = (
            getattr(finetuning_args, "finetuning_type", None) == "blocktt"
            and getattr(finetuning_args, "blocktt_normalize_after_update", False)
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.enabled or model is None:
            return
        from ..model.compress_setup import _ensure_compress_on_path
        _ensure_compress_on_path()
        from compress.integration import normalize_trainable_blocktt_cores_
        normalize_trainable_blocktt_cores_(model)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/train/test_compress_callbacks.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/train/callbacks.py \
        LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py
git commit -m "feat(llamafactory): CompressNormalizeCallback for BTT core normalization"
```

---

## Task 8: `CompressSaveCallback` + `_materialize_and_save`

**Files:**
- Modify: `LlamaFactory/src/llamafactory/train/callbacks.py` (append)
- Modify: `LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py`:

```python
def _build_btt_converted_tiny_model():
    """Build a tiny Qwen-shaped model and run plain BlockTT conversion on
    it so the model contains real BTTLinear modules whose forward pass
    is exercised by _materialize_and_save."""
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()

    # Reuse the tiny model factory from the compress_setup tests
    from tests.llamafactory.model.test_compress_setup import (
        _tiny_qwen_like_model,
    )

    class _FA:
        finetuning_type = "blocktt"
        calib_mode = "none"
        trainable_type = "all"
        train_position = "small"
        s_merged_to = "frozen"
        decomp_mode = "input_one_block"
        blocktt_rank = "full"
        convert_mode = "svd"
        train_bias = True
        blocktt_normalize_after_update = False
        blocktt_factorize_by_head = True

    model = _tiny_qwen_like_model()
    return compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=_FA(), is_trainable=True,
    )


def test_save_callback_plain_writes_merged_dir(tmp_path, monkeypatch):
    """Plain BTT model: _materialize_and_save should materialize BTT cores
    into dense nn.Linear weights and write a state_dict to disk."""
    from llamafactory.train.callbacks import (
        CompressSaveCallback, _materialize_and_save,
    )

    model = _build_btt_converted_tiny_model()
    out_dir = tmp_path / "merged"

    # _materialize_and_save calls model.config.to_diff_dict() and is meant
    # for real HF PreTrainedModel; for the unit test we exercise the
    # materialization helper directly and confirm the resulting state_dict
    # has dense Linear-style keys (no btt_l / btt_r).
    from llamafactory.train.callbacks import _build_materialized_state_dict
    sd = _build_materialized_state_dict(model)
    btt_keys = [k for k in sd if k.endswith(".btt_l") or k.endswith(".btt_r")]
    assert btt_keys == [], f"unexpected BTT keys: {btt_keys}"
    weight_keys = [k for k in sd if k.endswith(".weight")]
    assert len(weight_keys) > 0


def test_save_callback_rank_zero_guard(tmp_path):
    """Non-rank-0 callers must be no-ops."""
    from llamafactory.train.callbacks import CompressSaveCallback

    class _State:
        is_world_process_zero = False
        global_step = 5

    class _Args:
        output_dir = str(tmp_path)

    cb = CompressSaveCallback(_FakeFA(finetuning_type="blocktt", calib_mode="none"))
    cb.on_save(args=_Args(), state=_State(), control=None, model=object())
    # No checkpoint-<step>-merged/ should have been created
    assert not any((tmp_path / p).exists() for p in tmp_path.iterdir() if "merged" in p.name) or \
           not (tmp_path / "checkpoint-5-merged").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/train/test_compress_callbacks.py -v
```

Expected: FAIL — `CompressSaveCallback`, `_materialize_and_save`, and `_build_materialized_state_dict` do not exist.

- [ ] **Step 3: Implement the callback + helpers**

Append to `LlamaFactory/src/llamafactory/train/callbacks.py`:

```python
import pathlib as _pathlib


def _materialize_btt_weight(layer) -> torch.Tensor:
    """Reconstruct the dense ``[out, in]`` weight of a ``BTTLinear`` layer.

    Mirrors ``run_rl.py::materialize_btt_weight`` so we don't reach back
    into the RL entry point.
    """
    # Pull the per-block cores and form W = (btt_l ⊗ btt_r reduction).
    # The BTTLinear forward implements:
    #   out = (x_blocks @ btt_r).permute(1,0,2) @ btt_l_eff
    # which is equivalent to multiplying by W = btt_l_eff^T @ btt_r^T,
    # rearranged into the (out_features, in_features) layout expected
    # by ``nn.Linear.weight``. We materialize it by running the forward
    # on an identity input.
    in_features = layer.btt_r.shape[0] * layer.btt_r.shape[1] // 1  # n * b
    # Easier and exact: pass an identity through forward.
    device = layer.btt_l.device
    dtype = layer.btt_l.dtype
    eye = torch.eye(in_features, device=device, dtype=dtype)
    with torch.no_grad():
        out = layer(eye)  # (in_features, out_features) since forward is linear
    return out.t().contiguous()  # (out_features, in_features)


def _materialize_svd_weight(layer) -> torch.Tensor:
    """Use the layer's own materialize_dense_weight() method."""
    with torch.no_grad():
        return layer.materialize_dense_weight()


def _build_materialized_state_dict(model) -> "dict[str, torch.Tensor]":
    """Walk the model, replacing BTT/SVD module parameters with their dense
    equivalents under the parent-module key convention used by HF.

    For a module at path ``a.b.c.q_proj`` (a BTTLinear), this produces:
      a.b.c.q_proj.weight    -> materialized dense
      a.b.c.q_proj.bias      -> passed through if present

    All other parameters pass through unchanged (detached clone).
    """
    from llamafactory.model.compress_setup import _ensure_compress_on_path
    _ensure_compress_on_path()
    from compress.integration import BTTLinear, SVDCompressedLinear

    sd: "dict[str, torch.Tensor]" = {}
    # Track which fully-qualified parameter names are owned by a compressed module
    compressed_param_prefixes = []

    for name, module in model.named_modules():
        if isinstance(module, BTTLinear):
            w = _materialize_btt_weight(module)
            sd[f"{name}.weight" if name else "weight"] = w.detach().clone()
            if getattr(module, "bias", None) is not None:
                sd[f"{name}.bias"] = module.bias.detach().clone()
            compressed_param_prefixes.append(name + "." if name else "")
        elif isinstance(module, SVDCompressedLinear):
            w = _materialize_svd_weight(module)
            sd[f"{name}.weight" if name else "weight"] = w.detach().clone()
            if getattr(module, "bias", None) is not None:
                sd[f"{name}.bias"] = module.bias.detach().clone()
            compressed_param_prefixes.append(name + "." if name else "")

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in compressed_param_prefixes):
            continue
        sd[name] = param.detach().clone()
    for name, buf in model.named_buffers():
        if any(name.startswith(p) for p in compressed_param_prefixes):
            continue
        sd[name] = buf.detach().clone()
    return sd


def _materialize_and_save(model, ckpt_dir: str) -> None:
    """Build a peer model with dense nn.Linear weights and save it as
    plain HF format. The live training model is not mutated."""
    from transformers import AutoModelForCausalLM

    ckpt = _pathlib.Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    merged_sd = _build_materialized_state_dict(model)

    # Build a fresh peer from the same config and load the merged state_dict.
    peer = AutoModelForCausalLM.from_config(model.config)
    peer.load_state_dict(merged_sd, strict=False)
    peer.save_pretrained(str(ckpt))
    # Tokenizer save is handled by HF Trainer's regular save path; the
    # caller can copy the tokenizer files into the merged dir if needed.


class CompressSaveCallback(TrainerCallback):
    """Write a merged (dense HF) sibling checkpoint on every save and at
    end-of-train. The regular Trainer ``checkpoint-<step>/`` directory
    (factored state_dict) is what ``resume_from_checkpoint`` consumes;
    ``checkpoint-<step>-merged/`` and ``final-merged/`` are for vLLM / eval.
    """

    def __init__(self, finetuning_args):
        self.method = getattr(finetuning_args, "finetuning_type", None)
        self.calibrated = getattr(finetuning_args, "calib_mode", "none") != "none"

    def on_save(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return
        out = _pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return
        out = _pathlib.Path(args.output_dir) / "final-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out))

    def _dump(self, model, ckpt_dir: str) -> None:
        if self.calibrated:
            from llamafactory.model.compress_setup import _ensure_compress_on_path
            _ensure_compress_on_path()
            from compress.integration import save_calibrated_btt_hf_pretrained
            save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        else:
            _materialize_and_save(model, ckpt_dir)
```

Add the `torch` import at the top of `callbacks.py` if not already present (check the existing file — it likely already imports torch for `FixValueHeadModelCallback`).

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/train/test_compress_callbacks.py -v
```

Expected: all 6 tests PASS. (4 from Task 7 + 2 new.)

If `_materialize_btt_weight`'s eye-input strategy fails because forward expects a specific input shape (e.g. attention layers expect 3D inputs), fall back to building the dense weight from `btt_l` / `btt_r` cores directly using `torch.einsum`. The exact einsum is documented in `src/compress/btt/btt_linear.py:forward`; reproduce it inline rather than relying on `forward` so the materializer doesn't drag in attention masks. Concretely, the forward applies:

```python
# x: (..., d_in)  →  x_blocks = x.reshape(..., n, b)
# right = einsum("xnb,nbk->xnk", x_blocks, btt_r)   # (..., n, m*rank)
# right = right.reshape(..., n*rank, m)
# btt_l_eff = btt_l if not factorize_by_head else btt_l.reshape(m,n,rank,a).permute(0,2,1,3).reshape(m,n*rank,a)
# out = bmm(right.permute(...).reshape(m, ..., n*rank), btt_l_eff)  -> (..., m*a) = (..., d_out)
```

Build the dense weight as `W[d_out, d_in] = forward(I[d_in])^T`, where `I` is permuted into the BTT block layout matching the inverse of the reshape in the forward. The simplest correct implementation is still feeding an identity matrix shaped `(d_in, d_in)` through `forward`, treating the leading dim as a batch — that's what `run_rl.py::export_weights_for_vllm` does in practice. Verify the shape matches the original `nn.Linear.weight` (out, in).

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/train/callbacks.py \
        LlamaFactory/tests/llamafactory/train/test_compress_callbacks.py
git commit -m "feat(llamafactory): CompressSaveCallback with merged-HF export"
```

---

## Task 9: Register callbacks in `train/tuner.py`

**Files:**
- Modify: `LlamaFactory/src/llamafactory/train/tuner.py:30-75`

- [ ] **Step 1: Read the current tuner code**

Open `LlamaFactory/src/llamafactory/train/tuner.py` and locate the section near line 62:

```python
    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last
```

- [ ] **Step 2: Add the callback registration**

Modify the imports at the top of `tuner.py` — find:

```python
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
```

Replace with:

```python
from .callbacks import (
    CompressNormalizeCallback,
    CompressSaveCallback,
    LogCallback,
    PissaConvertCallback,
    ReporterCallback,
)
```

Then insert the registration immediately before the `ReporterCallback` line (since `ReporterCallback` is documented as "add to last"):

```python
    if finetuning_args.finetuning_type in ("blocktt", "svd"):
        callbacks.append(CompressNormalizeCallback(finetuning_args))
        callbacks.append(CompressSaveCallback(finetuning_args))
```

- [ ] **Step 3: Smoke-import**

```bash
cd LlamaFactory && conda run -n sft python -c "from llamafactory.train.tuner import run_exp; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Re-run all unit tests**

```bash
cd LlamaFactory && conda run -n sft pytest tests/llamafactory/ -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add LlamaFactory/src/llamafactory/train/tuner.py
git commit -m "feat(llamafactory): register compress callbacks in train/tuner"
```

---

## Task 10: Example YAML configs + README

**Files:**
- Create: `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_sft.yaml`
- Create: `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_calibrated_sft.yaml`
- Create: `LlamaFactory/examples/train_blocktt/qwen3_base_svd_sft.yaml`
- Create: `LlamaFactory/examples/train_blocktt/README.md`

- [ ] **Step 1: Create the plain BTT YAML**

Create `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_sft.yaml`:

```yaml
### model
model_name_or_path: Qwen/Qwen3-1.7B-Base
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: blocktt
deepspeed: examples/deepspeed/ds_z2_config.json
flash_attn: fa2
enable_liger_kernel: true

# Compress / BlockTT knobs
decomp_mode: input_one_block
blocktt_rank: full
convert_mode: svd
trainable_type: all
train_position: small
s_merged_to: frozen
train_bias: true
blocktt_normalize_after_update: false
blocktt_factorize_by_head: true
calib_mode: none

### dataset
dataset: openthought3_qwen3_4b
template: qwen3
enable_thinking: false
cutoff_len: 20480
preprocessing_num_workers: 64
dataloader_num_workers: 64

### output
output_dir: ../model/Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
gradient_checkpointing: true
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 0.05
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100

### swanlab / wandb
use_swanlab: false
swanlab_project: llamafactory
swanlab_run_name: Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
run_name: Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
```

- [ ] **Step 2: Create the calibrated BTT YAML**

Create `LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_calibrated_sft.yaml` — identical to step 1 except the compress block becomes:

```yaml
# Compress / BlockTT knobs (calibrated)
decomp_mode: input_one_block
blocktt_rank: full              # for calibrated, use 'full' or float in (0, 1]
convert_mode: svd
trainable_type: all
train_position: small
s_merged_to: frozen
train_bias: true
blocktt_normalize_after_update: false
blocktt_factorize_by_head: true
calib_mode: v2                  # v2 | v2_bp | v2_combined | twosteps
calib_source: c4                # c4 | traces | training_data
calib_num_seqs: 128
calib_max_length: 2048
calib_seed: 3
calib_batch_size: 8
calib_traces_path: null
compression_ratio: 1.0
```

And change `output_dir` / `run_name` / `swanlab_run_name` to substitute `BlockTT-Calibrated` for `BlockTT`.

- [ ] **Step 3: Create the plain SVD YAML**

Create `LlamaFactory/examples/train_blocktt/qwen3_base_svd_sft.yaml` — identical structure, but the compress block becomes:

```yaml
# Compress / SVD knobs
finetuning_type: svd
trainable_type: all
train_position: output          # svd: output | input | both
s_merged_to: frozen
calib_mode: none                # set to svd_v2 / svd_v2_combined for calibrated SVD
compression_ratio: 1.0          # ignored when calib_mode=none
```

Drop the BlockTT-only keys (`decomp_mode`, `blocktt_rank`, `convert_mode`, `train_bias`, `blocktt_normalize_after_update`, `blocktt_factorize_by_head`). Change `output_dir` / `run_name` / `swanlab_run_name` to substitute `SVD` for `BlockTT`.

- [ ] **Step 4: Create the README**

Create `LlamaFactory/examples/train_blocktt/README.md`:

```markdown
# BlockTT / SVD SFT recipes

These configs drive `src/compress`-backed BlockTT and SVD finetuning from
LlamaFactory. They mirror the `--train-mode blocktt` and `--train-mode svd`
paths in `run_rl.py`, but for SFT instead of RL.

## Recipes

- `qwen3_base_blocktt_sft.yaml` — plain BTT (lossless decomposition + finetune).
- `qwen3_base_blocktt_calibrated_sft.yaml` — calibrated BTT (`calib_mode: v2`).
- `qwen3_base_svd_sft.yaml` — plain SVD.

Run with:

```bash
conda activate sft
llamafactory-cli train LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_sft.yaml
```

## YAML knobs

| Key | Used by | Meaning |
|---|---|---|
| `finetuning_type` | both | `blocktt` or `svd`. |
| `trainable_type` | both | `all` / `mlp` / `attn` — which modules get compressed. |
| `train_position` | both | blocktt: `small` / `large` / `both`. svd: `output` / `input` / `both`. |
| `s_merged_to` | both | `frozen` / `trainable` / `output` / `input` / `split` / `keep_frozen` / `keep_trainable`. |
| `decomp_mode` | blocktt | `input_one_block` / `output_one_block` or dict literal. |
| `blocktt_rank` | blocktt | `"full"` or positive integer string. For calibrated mode use `"full"` or a float in `(0, 1]`. |
| `convert_mode` | blocktt | `svd` (default) or `qr`. `qr` ignores `s_merged_to`. |
| `train_bias` | blocktt | Train BTT biases. |
| `blocktt_normalize_after_update` | blocktt | Normalize trainable cores after each step. |
| `blocktt_factorize_by_head` | blocktt | Align attention BTT blocks with head structure. |
| `calib_mode` | both | `none` / `v2` / `v2_bp` / `v2_combined` / `twosteps` for BTT; `svd_v2` / `svd_v2_combined` for SVD. |
| `calib_source` | both | `c4` / `traces` / `training_data`. |
| `calib_num_seqs`, `calib_max_length`, `calib_seed`, `calib_batch_size` | both | Calibration sampling. |
| `calib_traces_path` | both | Required when `calib_source=traces`. |
| `compression_ratio` | svd calibrated | Fraction of compressible params to retain, `(0, 1]`. |

## DeepSpeed

ZeRO-2 is supported (and used by default in these recipes). **ZeRO-3 is
rejected at config-parse time** — custom BTT/SVD layers don't survive
parameter sharding under ZeRO-3.

## Checkpoint layout

Per save:

```
output_dir/
  checkpoint-200/             # factored state_dict (BTT/SVD modules);
                              # use this for resume_from_checkpoint
  checkpoint-200-merged/      # dense HF weights; drop-in for vLLM / eval
  ...
  final-merged/               # written at end-of-train, regardless of save_steps
```

## Caveats

- `learning_rate: 1.0e-4` is seeded from `run_rl.py`'s `MODE_DEFAULTS`. Tune for SFT.
- BlockTT/SVD cannot be combined with GaLore, APOLLO, or BAdam.
- `enable_liger_kernel: true` is fine — Liger patches HF attention/MLP modules but does not replace the inner `nn.Linear`, so BlockTT/SVD conversion (which runs in `init_adapter` after Liger patching) sees the post-Liger graph.
```

- [ ] **Step 5: Verify the YAMLs parse with LlamaFactory's parser**

```bash
cd LlamaFactory && conda run -n sft python -c "
from llamafactory.hparams import get_train_args
import sys
sys.argv = ['llamafactory-cli', 'train', 'examples/train_blocktt/qwen3_base_blocktt_sft.yaml']
ma, da, ta, fa, ga = get_train_args()
assert fa.finetuning_type == 'blocktt'
assert fa.train_position == 'small'
print('OK')
"
```

Expected: prints `OK`. Repeat for the calibrated and SVD YAMLs.

- [ ] **Step 6: Commit**

```bash
git add LlamaFactory/examples/train_blocktt/
git commit -m "docs(llamafactory): add BlockTT/SVD example YAMLs and README"
```

---

## Task 11: End-to-end smoke test (single GPU, tiny dataset)

**Files:** none (operational task)

This task is the integration validator — a 5-step training run on Qwen3-0.6B-Base with the plain BTT YAML, no DeepSpeed, single GPU.

- [ ] **Step 1: Prepare a tiny YAML override**

Create `LlamaFactory/examples/train_blocktt/_smoke_blocktt.yaml` (gitignored — see step 6):

```yaml
### model
model_name_or_path: Qwen/Qwen3-0.6B-Base
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: blocktt
deepspeed: null
flash_attn: auto
enable_liger_kernel: false

decomp_mode: input_one_block
blocktt_rank: full
convert_mode: svd
trainable_type: all
train_position: small
s_merged_to: frozen
calib_mode: none

### dataset
dataset: alpaca_en_demo
template: qwen3
cutoff_len: 256
preprocessing_num_workers: 2
dataloader_num_workers: 2

### output
output_dir: /tmp/smoke_blocktt
logging_steps: 1
save_steps: 5
overwrite_output_dir: true
save_only_model: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
gradient_checkpointing: false
learning_rate: 1.0e-4
max_steps: 5
lr_scheduler_type: constant
bf16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: no
```

- [ ] **Step 2: Run the smoke training**

```bash
cd LlamaFactory && conda run -n sft \
  CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_blocktt/_smoke_blocktt.yaml
```

Expected: 5 steps complete without traceback. Final log line mentions training loss. Directory `/tmp/smoke_blocktt/checkpoint-5-merged/` and `/tmp/smoke_blocktt/final-merged/` exist and contain `model.safetensors` + `config.json`.

- [ ] **Step 3: Reload the merged checkpoint**

```bash
cd LlamaFactory && conda run -n sft python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('/tmp/smoke_blocktt/final-merged', trust_remote_code=True)
print('reloaded:', type(m).__name__, sum(p.numel() for p in m.parameters()))
"
```

Expected: prints a count matching the base Qwen3-0.6B parameter count (within a small delta if compression was lossless `rank=full`).

- [ ] **Step 4: Repeat for plain SVD**

Copy `_smoke_blocktt.yaml` to `_smoke_svd.yaml`, change:

```yaml
finetuning_type: svd
train_position: output
output_dir: /tmp/smoke_svd
```

Drop the BTT-only keys (`decomp_mode`, `blocktt_rank`, `convert_mode`).

Run and reload as in steps 2-3.

- [ ] **Step 5: Repeat for calibrated BTT (`calib_mode: v2`, `calib_num_seqs: 4`)**

Copy `_smoke_blocktt.yaml` to `_smoke_blocktt_calib.yaml`, change:

```yaml
calib_mode: v2
calib_source: c4
calib_num_seqs: 4
calib_max_length: 64
calib_batch_size: 1
output_dir: /tmp/smoke_blocktt_calib
```

Run and reload.

- [ ] **Step 6: Add the smoke files to .gitignore**

If `LlamaFactory/.gitignore` does not already match the pattern, append:

```
examples/train_blocktt/_smoke_*.yaml
```

- [ ] **Step 7: Commit (only the .gitignore if changed)**

```bash
git add LlamaFactory/.gitignore
git commit -m "chore(llamafactory): ignore smoke-test YAMLs" || echo "nothing to commit"
```

---

## Task 12: Negative-path validation — ZeRO-3 rejection

**Files:** none (operational task)

- [ ] **Step 1: Try to run with ZeRO-3**

```bash
cd LlamaFactory && conda run -n sft \
  CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  examples/train_blocktt/qwen3_base_blocktt_sft.yaml \
  --deepspeed examples/deepspeed/ds_z3_config.json 2>&1 | head -40
```

Expected: process exits with a non-zero status, traceback contains `ValueError: finetuning_type='blocktt' does not support DeepSpeed ZeRO-3`. No model load happens (the error is raised in `FinetuningArguments.__post_init__`).

- [ ] **Step 2: Document the smoke results**

Record the three smoke runs (plain BTT, plain SVD, calibrated BTT) + ZeRO-3 rejection in a comment block at the bottom of `LlamaFactory/examples/train_blocktt/README.md` under a `## Verified configurations` heading:

```markdown
## Verified configurations

Last validated on YYYY-MM-DD (commit <git-hash>):
- Plain BTT on Qwen3-0.6B-Base, single GPU, 5 steps — PASS
- Plain SVD on Qwen3-0.6B-Base, single GPU, 5 steps — PASS
- Calibrated BTT (`calib_mode=v2`, `calib_source=c4`, `calib_num_seqs=4`) — PASS
- ZeRO-3 rejection — PASS (errors at config parse)
```

- [ ] **Step 3: Commit**

```bash
git add LlamaFactory/examples/train_blocktt/README.md
git commit -m "docs(llamafactory): record verified BlockTT/SVD smoke configurations"
```

---

## Self-review (done by the plan author, not for the implementer)

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 file structure | Task 1, 3, 4, 5, 6, 7, 8, 9, 10 |
| §3 CompressArguments mixin | Task 1 |
| §3.1 hard-error validators | Task 2 |
| §3.2 soft defaults | Task 2 |
| §3.3 cross-validation via compress.integration | Task 5 |
| §4.1 adapter dispatch | Task 6 |
| §4.2 compress_setup.py | Task 3 (skeleton), Task 4 (plain), Task 5 (calibrated) |
| §5.1 CompressNormalizeCallback | Task 7 |
| §5.2 CompressSaveCallback + _materialize_and_save | Task 8 |
| §5.3 callback registration in workflow | Task 9 (uses `train/tuner.py`, the canonical assembly point — *deviates from spec which named `train/sft/workflow.py`*; tuner.py is upstream of workflow.py and is where every other conditional callback is registered) |
| §6 example YAMLs | Task 10 |
| §7 edge cases | Task 10 README + Task 11/12 smokes |
| §8 testing plan | Task 11 (smokes 1-3) + Task 12 (negative test) — multi-GPU ZeRO-2 (smoke #4 in spec) deferred; flagged as follow-up |

**Placeholder scan:** no TBDs. Two callouts in Task 4 and Task 8 ("verify signatures before committing" and "verify shape") are explicit operational instructions, not placeholders.

**Type consistency:** `_materialize_btt_weight` / `_materialize_svd_weight` / `_build_materialized_state_dict` / `_materialize_and_save` are defined in Task 8 and referenced from `CompressSaveCallback._dump`. `_ensure_compress_on_path` / `_to_namespace` / `_load_tokenizer` / `_resolve_rank` / `init_compress_model` are defined in Task 3 (skeleton) and elaborated in Task 4/5, then referenced from Task 6, 7, 8.

**Deviations from spec called out inline:**
- Spec §5.3 names `train/sft/workflow.py` as the callback-registration point. In the actual LlamaFactory code, every conditional callback is registered in `train/tuner.py:62-72`; `workflow.py` only forwards what `tuner.py` assembled. Task 9 uses `tuner.py` for consistency with `PissaConvertCallback`, `swanlab`, `EarlyStoppingCallback`, etc.
- Multi-GPU ZeRO-2 smoke from §8 step 4 is deferred (requires hardware allocation that doesn't fit a unit-test cadence). All other smokes are in Task 11.
