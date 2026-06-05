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

    if method not in ("blocktt", "svd", "svd_nystrom"):
        raise ValueError(
            f"compress_setup: unsupported finetuning_type={method!r}; "
            "expected 'blocktt', 'svd' or 'svd_nystrom'."
        )

    # svd_nystrom = reasoning-aware structured compression: calibrated SVD on
    # self_attn + Nystrom on the MLP / each MoE expert. Always calibrated.
    if method == "svd_nystrom":
        return _init_compress_svd_nystrom(model, model_args, fa)

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


def _init_compress_svd_nystrom(model, model_args, fa):
    """Reasoning-aware structured compression, in-process, factors kept TRAINABLE.

    SVD-LLM-V2 on every ``self_attn`` linear (leaves trainable ``SVDCompressedLinear``
    factors) + Nystrom/MoDeGPT on every gated MLP triplet — which, via
    ``find_mlp_triplets``, covers a *dense* MLP (Qwen3: ``…mlp.{gate,up,down}_proj``)
    AND *every MoE expert* (OLMoE: ``…mlp.experts.{j}.{gate,up,down}_proj``). The
    Nystrom outputs are smaller dense ``nn.Linear``s; we re-enable their grad so the
    shrunk MLP/experts train during SFT. Calibration follows the productionized
    best practice (``docs/wiki/reasoning_aware_compress_calib.md``): sequence-reweighted
    FULL-sequence OpenThought3 covariances. The last ``skip_last_layers`` decoder
    layers are left dense (they feed the LM head). Two objectives:

      calib_mode=svd_v2            forward-only  (input-whitened SVD + forward Nystrom)
      calib_mode=svd_v2_combined   forward+backward (doubly-whitened SVD + joint Nystrom)
    """
    import torch
    from compress.loaders import build_fullseq_calib_loader
    from compress.structured.nystrom import (
        find_mlp_triplets,
        nystrom_compress_model,
        nystrom_combined_compress_model,
    )
    from compress.svd.svd_llm_v2 import svd_llm_v2_compress_model
    from compress.svd.svd_linear import SVDCompressedLinear

    ratio = float(fa.compression_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            f"svd_nystrom compression_ratio must be in (0, 1]; got {ratio}."
        )
    objective = "combined" if fa.calib_mode == "svd_v2_combined" else "forward"
    skip = ("lm_head",)

    # Calibration runs the model's forward/backward, so the model and the calib
    # batches must be co-located. Collect on CUDA when available (CPU forward on a
    # 4B/7B is impractical); the model is freshly from_pretrained at init_adapter
    # time, so move it to the compress device here. Trainer/DeepSpeed places it
    # afterward. Mirrors the existing calibrated-svd path's cuda assumption, but
    # makes the move explicit so it does not depend on prior device placement.
    if torch.cuda.is_available():
        device = "cuda"
        if next(model.parameters()).device.type != "cuda":
            model.to("cuda")
    else:
        device = "cpu"

    if fa.calib_source != "traces" or not fa.calib_traces_path:
        raise ValueError(
            "svd_nystrom requires calib_source=traces + calib_traces_path "
            "(the OpenThought3 reasoning-trace JSONL)."
        )

    tokenizer = _load_tokenizer(model_args)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- attention protection (last decoder layers) ---
    # The wiki "last layer dense" operating point is applied to ATTENTION only.
    # Rationale: SVD-compressed attn materializes back to FULL-size dense weights
    # (shape-preserving), so skipping the last layer's attn is a pure quality knob.
    # The MLP/experts, by contrast, SHRINK intermediate_size; to keep the saved HF
    # checkpoint's config valid (one global intermediate_size), every MLP/expert is
    # compressed UNIFORMLY — we do NOT leave the last layer's MLP dense.
    attn_protect = _protected_layer_set(model, int(fa.skip_last_layers))

    # --- which down_proj names are MLP/expert triplets (architecture-agnostic) ---
    # Qwen3 -> {…mlp.down_proj} (one per layer). Dense gated MLPs and *unfused*
    # per-expert MoE (experts as nn.Linear gate/up/down) are both discovered here.
    # FUSED-expert MoE (e.g. OLMoE in transformers>=5.2: experts as 3D nn.Parameter
    # gate_up_proj/down_proj, NOT nn.Linear) is NOT yet supported — fail fast with
    # an actionable message instead of a deep NotImplementedError from the finder.
    _assert_no_fused_experts(model)
    triplets = find_mlp_triplets(model, skip)
    mlp_down_keys = {down_name for (_p, _g, _u, down_name) in triplets}
    n_layers = _num_decoder_layers(model)
    logger.info_rank0(
        f"compress_setup[svd_nystrom]: ratio={ratio} objective={objective} "
        f"skip_last_layers={fa.skip_last_layers} (attn only) | {len(triplets)} "
        f"MLP/expert triplets, {n_layers} decoder layers, "
        f"attn_protect={sorted(attn_protect)}"
    )

    # --- sequence-reweighted FULL-seq calibration loader (wiki §5 default) ---
    texts = _render_calib_texts(
        tokenizer, fa.calib_traces_path, n=int(fa.calib_num_seqs) * 20,
    )
    # combined/backward path: bs=1 + truncate to 4096 to keep the backward in mem.
    calib_batch_size = 1 if objective == "combined" else 2
    loader = build_fullseq_calib_loader(
        tokenizer, texts,
        num_seqs=int(fa.calib_num_seqs),
        length_filter="full",
        batch_size=calib_batch_size,
    )

    model.eval()
    if objective == "forward":
        from compress.calibration import collect_covariances_reweighted
        full_cov = collect_covariances_reweighted(
            model, loader, device=device, skip_layers=skip, reweight="sequence",
        )
        attn_cov = _drop_protected_stats(
            {k: v for k, v in full_cov.items() if ".self_attn." in k}, attn_protect)
        # MLP: compress ALL triplets uniformly (no last-layer protection) so the
        # saved config's intermediate_size stays valid.
        down_cov = {k: v for k, v in full_cov.items() if k in mlp_down_keys}
        del full_cov
        _empty_cache()
        n_missing = len(mlp_down_keys) - len(down_cov)
        logger.info_rank0(
            f"compress_setup[svd_nystrom]: forward cov — {len(attn_cov)} attn, "
            f"{len(down_cov)} mlp-down ({n_missing} triplets had NO cov — "
            "left dense, e.g. unrouted MoE experts)."
        )
        nystrom_compress_model(
            model, {k: v.clone() for k, v in down_cov.items()},
            sparsity=1.0 - ratio, skip_layers=skip, device=device,
        )
        svd_llm_v2_compress_model(
            model, {k: v.clone() for k, v in attn_cov.items()},
            compression_ratio=ratio, skip_layers=skip, device=device,
            objective="forward",
        )
    else:  # combined (forward + backward)
        from compress.calibration import (
            collect_both_covariances_from_loader,
            collect_nystrom_combined_statistics,
        )
        fwd_cov, bwd_cov = collect_both_covariances_from_loader(
            model, loader, device=device, skip_layers=skip, reweight="sequence",
        )
        attn_fwd = _drop_protected_stats(
            {k: v for k, v in fwd_cov.items() if ".self_attn." in k}, attn_protect)
        attn_bwd = _drop_protected_stats(
            {k: v for k, v in bwd_cov.items() if ".self_attn." in k}, attn_protect)
        del fwd_cov, bwd_cov
        _empty_cache()
        # MLP/expert (C_f, C_b) pairs keyed by down_proj name (covers experts).
        # Compress ALL triplets uniformly (no last-layer protection).
        mlp_stats = collect_nystrom_combined_statistics(
            model, loader, device=device, skip_layers=skip, reweight="sequence",
        )
        n_missing = len(mlp_down_keys) - len(mlp_stats)
        logger.info_rank0(
            f"compress_setup[svd_nystrom]: combined cov — {len(attn_fwd)} attn, "
            f"{len(mlp_stats)} mlp-down ({n_missing} triplets had NO cov — "
            "left dense, e.g. unrouted MoE experts)."
        )
        nystrom_combined_compress_model(
            model, mlp_stats, sparsity=1.0 - ratio, skip_layers=skip, device=device,
        )
        svd_llm_v2_compress_model(
            model, {k: v.clone() for k, v in attn_fwd.items()},
            compression_ratio=ratio, skip_layers=skip, device=device,
            objective="combined",
            backward_covariances={k: v.clone() for k, v in attn_bwd.items()},
        )
    del loader
    _empty_cache()

    # --- guard: every MLP/expert triplet must have been compressed, else the saved
    # checkpoint would be heterogeneous (a stray full-size dense expert) and reload
    # under a single global intermediate_size would fail. With sequence-reweighted
    # full-seq calib + calib_num_seqs sized for the routing (512 for OLMoE), this
    # should be zero; if not, raise with an actionable message.
    if n_missing:
        raise RuntimeError(
            f"svd_nystrom: {n_missing} MLP/expert triplet(s) received no calibration "
            "covariance (e.g. MoE experts never routed by the calib set) and were left "
            "dense. The saved HF checkpoint requires a uniform intermediate_size — "
            "increase calib_num_seqs (and/or use length_filter=full) so every expert "
            "is routed at least once."
        )

    # --- update config.intermediate_size to the compressed width so the saved HF
    # checkpoint reloads correctly (CompressSaveCallback rebuilds from this config). ---
    new_k = _compressed_intermediate_size(model, triplets)
    old_k = getattr(model.config, "intermediate_size", None)
    if new_k is not None and new_k != old_k:
        model.config.intermediate_size = int(new_k)
        logger.info_rank0(
            f"compress_setup[svd_nystrom]: config.intermediate_size {old_k} -> {new_k} "
            "(Nystrom-shrunk MLP width; recorded for checkpoint reload)."
        )

    # --- make compressed factors TRAINABLE for SFT ---
    # SVD attn factors: SVDCompressedLinear U_r/V_r (+bias). Nystrom MLP outputs are
    # fresh nn.Linear created with requires_grad=False — flip them back on.
    n_svd = 0
    for module in model.modules():
        if isinstance(module, SVDCompressedLinear):
            n_svd += 1
            module.U_r.requires_grad_(True)
            module.V_r.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    n_nys = 0
    mods = dict(model.named_modules())
    for parent_path, _g, _u, _d in triplets:
        parent = mods[parent_path]
        for leaf in (parent.gate_proj, parent.up_proj, parent.down_proj):
            leaf.weight.requires_grad_(True)
            if leaf.bias is not None:
                leaf.bias.requires_grad_(True)
            n_nys += 1
    logger.info_rank0(
        f"compress_setup[svd_nystrom]: enabled grad on {n_svd} SVD attn linears + "
        f"{n_nys} Nystrom MLP linears; non-compressed params (embeddings, norms, "
        "lm_head, router gate, protected layers) keep their full-FT trainability."
    )
    return model


def _assert_no_fused_experts(model) -> None:
    """svd_nystrom Nystrom operates on per-expert gate/up/down nn.Linear triplets.
    Newer HF MoE modules (OLMoE OlmoeExperts, Qwen3-MoE fused experts, …) store
    experts as batched 3D nn.Parameters (gate_up_proj/down_proj of shape
    (num_experts, …)) with a functional forward — no nn.Linear to hook. Detect that
    and raise an actionable error rather than silently leaving the MoE uncompressed.
    """
    import torch.nn as _nn
    for name, module in model.named_modules():
        # an experts container that exposes a 3D gate_up_proj / down_proj Parameter
        gup = getattr(module, "gate_up_proj", None)
        dwn = getattr(module, "down_proj", None)
        if isinstance(gup, _nn.Parameter) and gup.dim() == 3:
            raise NotImplementedError(
                f"svd_nystrom: model has FUSED MoE experts at {name!r} "
                f"(gate_up_proj is a 3D Parameter {tuple(gup.shape)}). Per-expert "
                "Nystrom currently requires UNFUSED experts (gate_proj/up_proj/"
                "down_proj as nn.Linear). Fused-expert compression (e.g. OLMoE on "
                "transformers>=5.2) is a known follow-up — see "
                "examples/compress_train/olmoe_*.yaml. Use a dense or unfused-expert "
                "model, or downgrade transformers to an OLMoE version that exposes "
                "nn.ModuleList experts."
            )
        if isinstance(dwn, _nn.Parameter) and dwn.dim() == 3:
            raise NotImplementedError(
                f"svd_nystrom: model has FUSED MoE experts at {name!r} "
                f"(down_proj is a 3D Parameter {tuple(dwn.shape)}); see above."
            )


def _compressed_intermediate_size(model, triplets):
    """The post-Nystrom MLP width ``k`` (gate_proj out_features). Nystrom keeps
    ``k`` uniform across triplets (all share the same dint), so we read it from one
    compressed triplet and assert uniformity. Returns None if no triplets."""
    mods = dict(model.named_modules())
    widths = set()
    for parent_path, _g, _u, _d in triplets:
        parent = mods.get(parent_path)
        if parent is None:
            continue
        widths.add(int(parent.gate_proj.weight.shape[0]))
    if not widths:
        return None
    if len(widths) > 1:
        raise RuntimeError(
            "svd_nystrom: non-uniform compressed MLP widths "
            f"{sorted(widths)} — cannot record a single intermediate_size for the "
            "saved checkpoint. Expected uniform Nystrom width across all triplets."
        )
    return widths.pop()


def _num_decoder_layers(model) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise ValueError("svd_nystrom: model.config.num_hidden_layers is required.")
    return int(n)


def _layer_index_of(name: str):
    """Decoder-layer index embedded in a parameter/module name, or None."""
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _protected_layer_set(model, skip_last_layers: int):
    """Top-N decoder layer indices to leave DENSE (mirrors compress_common)."""
    if skip_last_layers <= 0:
        return set()
    n = _num_decoder_layers(model)
    return set(range(n - skip_last_layers, n))


def _drop_protected_stats(stats: dict, protect: set) -> dict:
    """Remove cov entries whose decoder-layer index is protected, so the compress
    core leaves those layers dense. Returns a new dict (does not mutate)."""
    if not protect:
        return dict(stats)
    return {n: s for n, s in stats.items() if _layer_index_of(n) not in protect}


def _empty_cache() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _render_calib_texts(tokenizer, jsonl_path: str, *, n: int) -> "list[str]":
    """Render OpenThought3 ``{"messages": [...]}`` rows to calibration strings via
    the model's own chat template, re-tokenized for whatever model we compress.

    Falls back to a plain user/assistant concat when the tokenizer has no chat
    template (e.g. OLMoE base → ``apply_chat_template`` raises ValueError) or does
    not accept ``enable_thinking`` (raises TypeError). Over-samples raw rows since
    the full-seq loader keeps only the first ``num_seqs`` that pass its filter.
    """
    import json
    import pathlib

    path = pathlib.Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"svd_nystrom calib_traces_path not found: {path}")

    def _render(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except (TypeError, ValueError):
                pass
        except ValueError:
            pass
        # No usable chat template (base model): plain role-tagged concat.
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

    texts: "list[str]" = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not msgs:
                continue
            texts.append(_render(msgs))
            if len(texts) >= n:
                break
    if not texts:
        raise ValueError(f"No calibration texts read from {path}")
    return texts


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
