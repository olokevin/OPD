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

    fa = finetuning_args
    method = fa.finetuning_type

    if method not in ("blocktt", "svd"):
        raise ValueError(
            f"compress_setup: unsupported finetuning_type={method!r}; "
            "expected 'blocktt' or 'svd'."
        )

    if fa.calib_mode != "none":
        return _init_compress_calibrated(model, model_args, fa, method)

    return _init_compress_plain(model, fa, method)


def _init_compress_calibrated(model, model_args, fa, method: str):
    """Calibrated BlockTT/SVD init: validates args, builds a calib loader,
    runs the calibrated decomposition. Returns the (possibly replaced) model.
    """
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
        model, stats = apply_calibrated_btt(
            model, ns, calib_loader=calib_loader, hyphen_style=False,
        )
        logger.info_rank0(
            f"compress_setup: calibrated BTT installed "
            f"{stats.get('num_btt_layers', '?')} layers."
        )
    else:  # svd — guaranteed by the method check in init_compress_model
        model = apply_calibrated_svd(
            model, ns, calib_loader=calib_loader, hyphen_style=False,
        )
        logger.info_rank0("compress_setup: calibrated SVD applied.")
    return model


def _init_compress_plain(model, fa, method: str):
    """Plain (non-calibrated) BlockTT/SVD init: lossless decomposition or
    int-rank truncation, followed by trainability configuration."""
    from compress.integration import (
        convert_linear_to_btt_compress,
        convert_linear_to_svd_compress,
        configure_compress_btt_trainability,
        configure_compress_svd_trainability,
        get_blocktt_target_module_names,
        get_svd_target_module_names,
        resolve_blocktt_decomp_modes,
    )

    rank = _resolve_rank(fa.blocktt_rank)

    if method == "blocktt":
        targets = get_blocktt_target_module_names(fa.trainable_type)
        # resolve_blocktt_decomp_modes returns a populated dict when
        # include_names is non-empty; we discard the scalar form.
        _, module_decomp_modes = resolve_blocktt_decomp_modes(
            fa.decomp_mode, include_names=targets,
        )
        convert_linear_to_btt_compress(
            model,
            btt_rank=rank,
            decomp_mode=module_decomp_modes,
            include_names=targets,
            s_merged_to=fa.s_merged_to,
            train_position=fa.train_position,
            factorize_by_head=fa.blocktt_factorize_by_head,
            model_config=getattr(model, "config", None),
            convert_mode=fa.convert_mode,
        )
        configure_compress_btt_trainability(
            model,
            train_bias=fa.train_bias,
            train_position=fa.train_position,
            train_singular_values=(fa.s_merged_to == "keep_trainable"),
        )
    else:  # svd — guaranteed by the method check in init_compress_model
        targets = get_svd_target_module_names(fa.trainable_type)
        convert_linear_to_svd_compress(
            model,
            include_names=targets,
            s_merged_to=fa.s_merged_to,
            train_position=fa.train_position,
        )
        configure_compress_svd_trainability(
            model,
            train_position=fa.train_position,
            train_bias=fa.train_bias,
            train_embed_lm_head=(fa.train_position == "both"),
            train_singular_values=(fa.s_merged_to == "keep_trainable"),
        )
    return model


def _load_tokenizer(model_args: "ModelArguments") -> Any:
    """Load the tokenizer for calibration. Imports HF lazily to keep
    cold-start fast in the plain (non-calibrated) path."""
    from transformers import AutoTokenizer
    name = getattr(model_args, "model_name_or_path", None)
    if not name:
        raise ValueError(
            "compress_setup: model_args.model_name_or_path is required for "
            "calibrated BlockTT/SVD finetuning (got model_args=None or unset)."
        )
    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
    )


def _resolve_rank(rank_arg: str) -> "str | int":
    """Parse ``blocktt_rank``. Delegates to
    ``llamafactory.hparams.finetuning_args.resolve_blocktt_rank`` so the
    YAML-time validator in ``__post_init__`` and the model-init call here
    stay in lockstep.
    """
    from ..hparams.finetuning_args import resolve_blocktt_rank
    return resolve_blocktt_rank(rank_arg)


def _to_namespace(finetuning_args: "FinetuningArguments") -> Namespace:
    """Build a ``run_rl.py``-style argparse.Namespace from ``finetuning_args``.

    ``compress.integration`` helpers (``validate_calibrated_btt_args``,
    ``build_calib_loader``, ``apply_calibrated_btt``/``apply_calibrated_svd``)
    expect attribute access in the same shape as ``run_rl.py``'s argparse
    namespace, with ``hyphen_style=False`` for underscore field names.

    Auto-derives the field list from ``CompressArguments`` so a new field
    on the dataclass automatically propagates here — ``compress.integration``
    reads via ``getattr(args, name, default)``, so a missing field would
    silently fall back to the integration's default instead of the YAML value.
    """
    import dataclasses
    from ..hparams.finetuning_args import CompressArguments

    fa = finetuning_args
    ns = Namespace()
    ns.train_mode = fa.finetuning_type            # "blocktt" or "svd"
    for f in dataclasses.fields(CompressArguments):
        setattr(ns, f.name, getattr(fa, f.name))
    return ns
