"""BlockTT adapter: plain (SVD/QR init), calibrated (v2 / twosteps / ...),
and qfura (NF4-quantized frozen core via QBTTLinear)."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
from typing import Optional

import torch

from verl.workers.peft.base import PEFTAdapter


def _calib_cache_root() -> str:
    """Pick a stable location for the BlockTT calibration cache.

    Priority: ``CALIB_CACHE_DIR`` env var, then ``$HF_HOME/opd_calib_cache``,
    then ``/tmp/opd_calib_cache``. The launcher scripts already export both
    ``HF_HOME`` and ``CALIB_CACHE_DIR`` (when overridden).
    """
    override = os.environ.get("CALIB_CACHE_DIR")
    if override:
        return override
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "opd_calib_cache")
    return "/tmp/opd_calib_cache"


class BlockTTAdapter(PEFTAdapter):
    mode = "blocktt"

    def __init__(self, peft_cfg, model_config=None, teacher_model_path: Optional[str] = None):
        super().__init__(peft_cfg, model_config=model_config)
        self._topology_payload: Optional[dict] = None
        self._is_calibrated: bool = peft_cfg.calib.mode != "none"
        self._is_qfura: bool = peft_cfg.blocktt.qfura.enabled
        # Path to teacher LM, only used when peft.calib.loss == "opd".
        # Loaded lazily inside apply() to keep imports cheap.
        self._teacher_model_path: Optional[str] = teacher_model_path

    def needs_calibration(self) -> bool:
        return self._is_calibrated

    def _trainable_type(self) -> str:
        tm = self.peft_cfg.target_modules
        if isinstance(tm, str) and tm in {"all", "mlp", "attn"}:
            return tm
        return "all"

    def _build_compress_args(self):
        bt = self.peft_cfg.blocktt
        args = SimpleNamespace(
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
        # Forward calib_loss + OPD knobs to apply_calibrated_btt; teacher_model
        # itself is attached lazily in apply() once we know the target device.
        args.calib_loss = self.peft_cfg.calib.loss
        if self.peft_cfg.calib.loss == "opd":
            args.calib_top_k = self.peft_cfg.calib.top_k
            args.calib_top_k_strategy = self.peft_cfg.calib.top_k_strategy
            args.calib_reward_weight_mode = self.peft_cfg.calib.reward_weight_mode
            args.calib_temperature = self.peft_cfg.calib.temperature
            args.calib_teacher_temperature = self.peft_cfg.calib.teacher_temperature
        return args

    # ------------------------------------------------------------------ cache

    def _calib_cache_signature(self, actor_model_path: str) -> dict:
        """Everything that materially changes the calibrated weights.

        If two runs agree on this dict, the calibrated BTT factors are
        interchangeable and we can skip the expensive covariance pass.
        ``train_position`` is intentionally **excluded** — it only sets
        ``requires_grad`` flags, never the tensor values.
        """
        bt = self.peft_cfg.blocktt
        c = self.peft_cfg.calib
        sig = {
            "actor_model_path": str(actor_model_path),
            "target_modules": self.peft_cfg.target_modules,
            "blocktt": {
                "decomp_mode": bt.decomp_mode,
                "rank": bt.rank,
                "convert_mode": bt.convert_mode,
                "s_merged_to": bt.s_merged_to,
                "factorize_by_head": bt.factorize_by_head,
            },
            "calib": {
                "mode": c.mode,
                "source": c.source,
                "num_seqs": c.num_seqs,
                "max_length": c.max_length,
                "batch_size": c.batch_size,
                "seed": c.seed,
                "loss": c.loss,
            },
        }
        # Calibration data path (only matters when source=training_data).
        if c.source == "training_data":
            sig["calib"]["training_data_files"] = (
                os.environ.get("TRAIN_DATASET") or os.environ.get("TRAIN_FILES") or ""
            )
        if c.source == "traces":
            sig["calib"]["traces_path"] = c.traces_path
        if c.loss == "opd":
            sig["calib"]["top_k"] = c.top_k
            sig["calib"]["top_k_strategy"] = c.top_k_strategy
            sig["calib"]["reward_weight_mode"] = c.reward_weight_mode
            sig["calib"]["temperature"] = c.temperature
            sig["calib"]["teacher_temperature"] = c.teacher_temperature
            sig["calib"]["teacher_model_path"] = str(self._teacher_model_path or "")
            sig["calib"]["enable_thinking"] = os.environ.get("ENABLE_THINKING", "")
        return sig

    def _calib_cache_dir(self, actor_model_path: str) -> str:
        sig = self._calib_cache_signature(actor_model_path)
        canon = json.dumps(sig, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(canon).hexdigest()[:16]
        root = _calib_cache_root()
        return os.path.join(root, digest)

    def _try_load_calibrated_cache(self, model, cache_dir: str) -> bool:
        """If ``cache_dir`` holds a previously-saved calibrated checkpoint,
        rebuild BTT topology on ``model`` and load the cached BTT factors.

        Returns True on success (skip calibration), False otherwise.
        """
        topology_path = os.path.join(cache_dir, "btt_topology.json")
        weights_path = os.path.join(cache_dir, "model.safetensors")
        signature_path = os.path.join(cache_dir, "signature.json")
        if not (os.path.isfile(topology_path) and os.path.isfile(weights_path)):
            return False
        try:
            from compress.topology import rebuild_btt_from_topology
            from safetensors.torch import load_file
        except Exception:
            return False
        try:
            with open(topology_path) as f:
                topology = json.load(f)
            rebuild_btt_from_topology(model, topology)
            state = load_file(weights_path)
            missing, unexpected = model.load_state_dict(state, strict=False)
            # Unexpected keys mean the checkpoint and model disagree about
            # what BTT factors should exist — treat as a cache miss instead
            # of poisoning the model.
            if unexpected:
                return False
            # Record what we loaded for downstream save() / meta export.
            self._topology_payload = {"btt_topology": topology, "loaded_from_cache": cache_dir}
            if os.path.isfile(signature_path):
                with open(signature_path) as f:
                    self._topology_payload["cache_signature"] = json.load(f)
            return True
        except Exception:
            # Any deserialization issue → fall back to a fresh calibration.
            return False

    @torch.no_grad()
    def _save_calibrated_cache(self, model, cache_dir: str, signature: dict) -> None:
        """Persist the calibrated BTT factors + topology so that a future run
        with the same _calib_cache_signature can skip the covariance pass.

        Saves *post-calibration* state, before FSDP wrap, so we serialise the
        raw nn.Module on its current device. Weights are first moved to CPU
        to keep the on-disk shards device-agnostic.
        """
        try:
            from compress.topology import export_btt_topology
            from safetensors.torch import save_file
        except Exception:
            # Best-effort: cache is an optimisation, never block training.
            return
        try:
            os.makedirs(cache_dir, exist_ok=True)
            topology = export_btt_topology(model)
            with open(os.path.join(cache_dir, "btt_topology.json"), "w") as f:
                json.dump(topology, f)
            with open(os.path.join(cache_dir, "signature.json"), "w") as f:
                json.dump(signature, f, indent=2, sort_keys=True)
            # Save full state_dict (embeddings + lm_head + BTT cores + biases
            # + LayerNorm). Move shards to CPU to detach from CUDA.
            cpu_state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
            save_file(cpu_state, os.path.join(cache_dir, "model.safetensors"))
        except Exception:
            # Cache write failures are non-fatal — caller is the live trainer
            # and we don't want to abort a working run because of a flaky
            # filesystem. Silent so we don't clutter the training log; the
            # next run will simply recalibrate.
            pass

    @staticmethod
    def _untie_embeddings(model):
        """Break the input/output-embedding weight tie so FSDP1 with
        use_orig_params=True doesn't fail the update_actor writeback check
        ("Cannot writeback when the parameter shape changes"). Copies the
        embed weight into a fresh lm_head Parameter and flips the config flag.
        No-op when the model isn't tied. Returns the (same) model."""
        in_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        in_w = getattr(in_emb, "weight", None)
        out_w = getattr(out_emb, "weight", None)
        cfg_ties = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        tied = (
            in_w is not None and out_w is not None
            and in_w.data_ptr() == out_w.data_ptr()
        )
        print(
            f"[blocktt untie] cfg.tie_word_embeddings={cfg_ties} "
            f"in_emb={type(in_emb).__name__ if in_emb is not None else None} "
            f"out_emb={type(out_emb).__name__ if out_emb is not None else None} "
            f"storage_tied={tied}",
            flush=True,
        )
        # Untie whenever the storage is shared OR the config flags a tie, even
        # if get_output_embeddings() didn't expose a separate lm_head.
        if tied:
            out_emb.weight = torch.nn.Parameter(
                in_w.detach().clone(), requires_grad=out_w.requires_grad,
            )
            if getattr(model, "config", None) is not None:
                model.config.tie_word_embeddings = False
            print("[blocktt untie] untied lm_head from embed_tokens (cloned weight)", flush=True)
        elif cfg_ties and getattr(model, "config", None) is not None:
            # Storage not shared but config still claims a tie — clear the flag so
            # nothing re-ties them downstream (e.g. tie_weights() during vLLM load).
            model.config.tie_word_embeddings = False
            print("[blocktt untie] cleared stale tie_word_embeddings flag (storage already separate)", flush=True)
        return model

    def apply(self, model, *, tokenizer, calib_loader_builder):
        from compress.integration import (
            apply_calibrated_btt,
            configure_compress_btt_trainability,
            convert_and_quantize_linear_to_qbtt_streaming,
            convert_linear_to_btt_compress,
            get_blocktt_target_module_names,
            resolve_blocktt_decomp_modes,
        )
        from compress.topology import export_btt_topology

        # With the input embeddings frozen (BlockTT trains only the small BTT
        # cores) and gradient checkpointing on (use_reentrant=False), the
        # activations entering the first checkpointed block don't require grad,
        # so the backward pass raises "element 0 of tensors does not require
        # grad and does not have a grad_fn". Registering the input-require-grad
        # hook (same as the LoRA adapter) gives autograd a path into the
        # trainable cores. Must run before FSDP wraps the model.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Untie input/output embeddings for ALL blocktt branches (plain FurA
        # included, not just calibrated). FSDP1 with use_orig_params=True chokes
        # on tied embed/lm_head during the update_actor pre-unshard writeback:
        #   "Cannot writeback when the parameter shape changes
        #    Expects torch.Size([311164928]) but got torch.Size([151936, 2048])"
        # See the calibrated branch below, which does the same after calibration.
        model = self._untie_embeddings(model)

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
            device = "cuda" if torch.cuda.is_available() else None
            # Resolve actor model identity for cache keying. Use the HF config
            # name if present; fall back to env. The first run that calibrates
            # writes the cache, subsequent runs with matching settings load it
            # and skip the (expensive) covariance + decomposition pass.
            actor_model_path = (
                getattr(getattr(model, "config", None), "_name_or_path", None)
                or os.environ.get("ACTOR_MODEL_PATH")
                or ""
            )
            cache_dir = self._calib_cache_dir(actor_model_path)
            cache_signature = self._calib_cache_signature(actor_model_path)

            # Try cache hit first — move to device only after, so we don't pay
            # the CPU→CUDA copy when we can avoid the calibration entirely.
            # CALIB_CACHE_FORCE_REBUILD=1 bypasses the load step and forces a
            # fresh calibration (cache is still written afterwards).
            force_rebuild = os.environ.get("CALIB_CACHE_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes"}
            cache_hit = False if force_rebuild else self._try_load_calibrated_cache(model, cache_dir)
            if force_rebuild:
                print(f"[BlockTT calib] CALIB_CACHE_FORCE_REBUILD=1 set; ignoring any cache at {cache_dir!r}", flush=True)
            print(
                f"[BlockTT calib] cache {'HIT' if cache_hit else 'MISS'} at {cache_dir!r} "
                f"(calib.mode={self.peft_cfg.calib.mode!r}, loss={self.peft_cfg.calib.loss!r})",
                flush=True,
            )
            if cache_hit:
                if device is not None:
                    model = model.to(device)
                # Skip the rest of the calibration branch: install trainability
                # exactly as decompose_with_loader would have, mirroring the
                # post-calibration codepath.
                from compress.decomposition import configure_btt_trainability
                stats = configure_btt_trainability(
                    model,
                    train_position=self.peft_cfg.blocktt.train_position,
                    train_bias=self.peft_cfg.blocktt.train_bias,
                )
                self._topology_payload["calib_stats"] = stats
                # FSDP1-tied-embedding workaround still needs to run here so
                # the rest of the codepath agrees with a fresh calibration.
                try:
                    cfg_ties = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
                except Exception:
                    cfg_ties = False
                if cfg_ties and hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
                    in_emb = model.get_input_embeddings()
                    out_emb = model.get_output_embeddings()
                    if (
                        out_emb is not None
                        and getattr(out_emb, "weight", None) is not None
                        and getattr(in_emb, "weight", None) is not None
                        and out_emb.weight.data_ptr() == in_emb.weight.data_ptr()
                    ):
                        out_emb.weight = torch.nn.Parameter(
                            in_emb.weight.detach().clone(),
                            requires_grad=out_emb.weight.requires_grad,
                        )
                        model.config.tie_word_embeddings = False
                # Skip to the topology-record block below.
                self._topology_payload["btt_topology"] = self._topology_payload.get(
                    "btt_topology", export_btt_topology(model)
                )
                return model

            calib_loader = calib_loader_builder()
            if calib_loader is None:
                raise RuntimeError(
                    "BlockTTAdapter calibration mode requires a non-None calib_loader; "
                    "check peft.calib.* and that calib_loader_builder was passed."
                )
            # compress.collect_covariances_from_loader assumes the model is
            # already on the target device (it only moves the inputs). The
            # actor module was just built via from_pretrained() and sits on
            # CPU, so move it to cuda for the calibration pass. FSDP wrap
            # later in fsdp_workers handles its own sharding/dtype.
            if device is not None:
                model = model.to(device)
            # FSDP1 with use_orig_params=True chokes on tied input/output
            # embeddings during update_actor pre-unshard writeback:
            #   "Cannot writeback when the parameter shape changes
            #    Expects torch.Size([388956160]) but got torch.Size([151936, 2560])"
            # The tied embed/lm_head weight's storage is reported as
            # divergent from the FlatParameter slice during the per-step
            # writeback check, even though use_orig_params should expose a
            # plain view. Untie them up front: copy the embed weight to a
            # fresh lm_head weight tensor and flip the config flag so the
            # vLLM rollout path doesn't re-tie them.
            try:
                cfg_ties = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
            except Exception:
                cfg_ties = False
            if cfg_ties and hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
                in_emb = model.get_input_embeddings()
                out_emb = model.get_output_embeddings()
                if (
                    out_emb is not None
                    and getattr(out_emb, "weight", None) is not None
                    and getattr(in_emb, "weight", None) is not None
                    and out_emb.weight.data_ptr() == in_emb.weight.data_ptr()
                ):
                    out_emb.weight = torch.nn.Parameter(
                        in_emb.weight.detach().clone(),
                        requires_grad=out_emb.weight.requires_grad,
                    )
                    model.config.tie_word_embeddings = False
            # Lazy-load teacher model when calib_loss == "opd". Apply requires_grad_(False)
            # and .eval() before handing off to compress.apply_calibrated_btt.
            if self.peft_cfg.calib.loss == "opd":
                if not self._teacher_model_path:
                    raise RuntimeError(
                        "BlockTTAdapter with calib.loss='opd' requires teacher_model_path "
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
            try:
                model, stats = apply_calibrated_btt(model, args, calib_loader=calib_loader,
                                                    device=device, hyphen_style=True)
            finally:
                # Free the teacher: it's a full 4B + calibration peaks at >40GB,
                # and we don't want it competing with FSDP sharding next.
                if self.peft_cfg.calib.loss == "opd":
                    args.teacher_model = None
                    del teacher
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            self._topology_payload = {"calib_stats": stats}
            # Persist the calibrated BTT factors to the cache so subsequent
            # runs with the same _calib_cache_signature can skip the
            # covariance pass entirely. Non-fatal on failure.
            self._save_calibrated_cache(model, cache_dir, cache_signature)
        elif self._is_qfura:
            # qfura streaming path does its own Linear -> BTT conversion +
            # NF4 quantization in a single layer-streaming pass, so the plain
            # convert + trainability calls would be redundant and would also
            # leave full bf16 BTTLinear layers on-device before quantization.
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "BlockTTAdapter qfura requires a CUDA device for the "
                    "streaming NF4 quantization pass."
                )
            convert_and_quantize_linear_to_qbtt_streaming(
                model,
                btt_rank=bt.rank,
                decomp_mode=module_decomp_modes,
                train_position=bt.train_position,
                s_merged_to=bt.s_merged_to,
                quant_layout="flat",
                target_modules=include_names,
                cuda_device=torch.cuda.current_device(),
                factorize_by_head=bt.factorize_by_head,
                convert_mode=bt.convert_mode,
            )
        else:
            # The plain (SVD/QR) Linear->BTT conversion requires the target
            # Linear weights on CUDA. The model is typically still on CPU here
            # (HF from_pretrained loads to CPU; FSDP moves it to GPU later), so
            # move it to the current CUDA device first. Matches the calibrated /
            # qfura branches, which already run on CUDA.
            if torch.cuda.is_available():
                model = model.to(torch.cuda.current_device())
            convert_linear_to_btt_compress(
                model,
                btt_rank=bt.rank,
                decomp_mode=module_decomp_modes,
                include_names=include_names,
                s_merged_to=bt.s_merged_to,
                train_position=bt.train_position,
                factorize_by_head=bt.factorize_by_head,
                model_config=getattr(model, "config", None),
                convert_mode=bt.convert_mode,
            )
            configure_compress_btt_trainability(
                model,
                train_bias=bt.train_bias,
                train_position=bt.train_position,
                train_singular_values=(bt.s_merged_to == "keep_trainable"),
            )

        # Record minimal topology used by save / resume.
        self._topology_payload = self._topology_payload or {}
        self._topology_payload["btt_topology"] = export_btt_topology(model)

        return model

    @torch.no_grad()
    def export_for_vllm(self, fsdp_module):
        """Build the COMPLETE dense weight set to load into vLLM for rollout.

        Two parts, mirroring lora-without-regret/run_rl.py:export_weights_for_vllm:
          1. every BTTLinear -> its materialized dense `{name}.weight` (+bias),
          2. every OTHER parameter verbatim (embed_tokens, layernorms, lm_head,
             and any non-factorized linears) — these are NOT BTTLinear so
             `materialize_calibrated_btt_weights` skips them, and without them
             vLLM would keep stale base-model embeddings/norms/head.

        vLLM's Qwen3ForCausalLM.load_weights fuses split q/k/v/gate/up into its
        packed qkv_proj/gate_up_proj, so emitting split `*_proj.weight` is
        correct. We must, however, strip the structural `_fsdp_wrapped_module.`
        segment FSDP inserts into qualnames — vLLM's module tree has no such
        segment (otherwise: KeyError ...layers.0._fsdp_wrapped_module...).
        """
        import torch as _torch
        from compress.btt.btt_linear import BTTLinear

        def _clean(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        out = {}
        btt_prefixes = []
        for name, module in fsdp_module.named_modules():
            if not isinstance(module, BTTLinear):
                continue
            btt_prefixes.append(name + ".")
            out[_clean(f"{name}.weight")] = module.materialize_dense_weight()
            if module.bias is not None:
                out[_clean(f"{name}.bias")] = module.bias.detach()

        btt_prefixes = tuple(btt_prefixes)
        for name, param in fsdp_module.named_parameters():
            # Skip the BTT cores (their dense weight was emitted above); keep
            # everything else (embeddings / norms / lm_head / untouched linears).
            if btt_prefixes and name.startswith(btt_prefixes):
                continue
            out[_clean(name)] = param.detach() if isinstance(param, _torch.Tensor) else param
        return out

    def save_pretrained(self, fsdp_module, out_dir: str) -> None:
        from compress.integration import materialize_calibrated_btt_to_linear
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
                "qfura": {"enabled": bt.qfura.enabled},
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
