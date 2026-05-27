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
