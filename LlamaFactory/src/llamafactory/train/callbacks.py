# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"),
                output_dir=output_dir,
                safe_serialization=getattr(args, "save_safetensors", True),
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=getattr(args, "save_safetensors", True),
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and getattr(args, "overwrite_output_dir", False)
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if "trackio" in args.report_to:
            import trackio

            trackio.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


class CompressNormalizeCallback(TrainerCallback):
    """Normalize trainable BTT cores after each optimizer step.

    Mirrors run_rl.py's blocktt_normalize_after_update behavior. No-op when
    finetuning_type != blocktt or when the flag is False.
    """

    def __init__(self, finetuning_args):
        # getattr defaults keep this robust to partial test fakes; in
        # production both attributes are always set by CompressArguments.
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


def _materialize_btt_weight(layer) -> torch.Tensor:
    """Reconstruct the dense ``[out_features, in_features]`` weight of a
    ``BTTLinear``. Delegates to the layer's own ``materialize_dense_weight``
    method, mirroring ``run_rl.py::materialize_btt_weight``.
    """
    with torch.no_grad():
        return layer.materialize_dense_weight()


def _materialize_svd_weight(layer) -> torch.Tensor:
    """Use the layer's own materialize_dense_weight() method."""
    with torch.no_grad():
        return layer.materialize_dense_weight()


def _build_materialized_state_dict(model) -> "dict[str, torch.Tensor]":
    """Walk ``model``, replacing each BTT/SVD module's factored parameters
    with their dense equivalents under the HF parent-module key convention.

    For a module at path ``a.b.c.q_proj`` (a BTTLinear), this produces:
      a.b.c.q_proj.weight    -> materialized dense (out_features, in_features)
      a.b.c.q_proj.bias      -> passed through if present

    All non-compressed parameters and buffers pass through unchanged as
    detached clones. The live training model is never mutated.
    """
    from ..model.compress_setup import _ensure_compress_on_path
    _ensure_compress_on_path()
    from compress.integration import BTTLinear, SVDCompressedLinear

    sd: "dict[str, torch.Tensor]" = {}
    compressed_prefixes: "list[str]" = []

    for name, module in model.named_modules():
        if isinstance(module, BTTLinear):
            sd[f"{name}.weight" if name else "weight"] = (
                _materialize_btt_weight(module).detach().clone()
            )
            if getattr(module, "bias", None) is not None:
                sd[f"{name}.bias"] = module.bias.detach().clone()
            compressed_prefixes.append(name + "." if name else "")
        elif isinstance(module, SVDCompressedLinear):
            sd[f"{name}.weight" if name else "weight"] = (
                _materialize_svd_weight(module).detach().clone()
            )
            if getattr(module, "bias", None) is not None:
                sd[f"{name}.bias"] = module.bias.detach().clone()
            compressed_prefixes.append(name + "." if name else "")

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in compressed_prefixes):
            continue
        sd[name] = param.detach().clone()
    for name, buf in model.named_buffers():
        if any(name.startswith(p) for p in compressed_prefixes):
            continue
        sd[name] = buf.detach().clone()
    return sd


def _materialize_and_save(model, ckpt_dir: str) -> None:
    """Save a dense HF checkpoint of the compressed model at ``ckpt_dir`` without
    mutating the live training model.

    The peer is built by DEEP-COPYING the live model and materializing its factored
    (BTT/SVD) modules to dense ``nn.Linear`` IN PLACE on the copy. This preserves the
    live model's exact PER-LAYER shapes — critical for svd_nystrom, whose MLP is
    HETEROGENEOUS (protected last layer at full width, others Nystrom-shrunk). A
    ``from_config`` peer would build every MLP at the single ``config.intermediate_size``
    and silently drop the mismatched last-layer MLP under ``strict=False``.

    Under DeepSpeed ZeRO-2 params are fully replicated per rank, so the copy reads
    complete data without any gather. ZeRO-3 is disallowed by the parser validator.
    """
    import copy
    import torch

    from ..model.compress_setup import _ensure_compress_on_path
    _ensure_compress_on_path()
    from compress.integration import (
        BTTLinear, SVDCompressedLinear,
        materialize_calibrated_btt_to_linear, materialize_svd_to_linear,
    )

    ckpt = pathlib.Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    # Deep-copy on CPU to avoid doubling GPU memory; detach grads first. Materialize
    # in FP32: the BTT/SVD materialize does `V_r @ U_r` matmuls, and bf16 GEMM on CPU
    # is pathologically slow on torch>=2.12 (effectively hangs for ~140 SVD linears).
    # FP32 CPU GEMM (MKL) is fast; we cast the dense result back to the model's dtype
    # before saving so the on-disk checkpoint stays bf16. Keeping it on CPU avoids
    # doubling GPU memory at save time (the optimizer states are still resident).
    orig_dtype = next(model.parameters()).dtype
    peer = copy.deepcopy(model).to("cpu", dtype=torch.float32)
    has_btt = any(isinstance(m, BTTLinear) for m in peer.modules())
    has_svd = any(isinstance(m, SVDCompressedLinear) for m in peer.modules())
    if has_btt:
        materialize_calibrated_btt_to_linear(peer)
    if has_svd:
        materialize_svd_to_linear(peer)
    peer.to(dtype=orig_dtype)
    peer.save_pretrained(str(ckpt))
    del peer


class CompressSaveCallback(TrainerCallback):
    """Write a merged (dense HF) sibling checkpoint on every save and at
    end-of-train. The regular Trainer ``checkpoint-<step>/`` directory
    (factored state_dict) is what ``resume_from_checkpoint`` consumes;
    ``checkpoint-<step>-merged/`` and ``final-merged/`` are for vLLM / eval.
    """

    def __init__(self, finetuning_args):
        del finetuning_args   # currently unused; accepted for symmetry
                              # with CompressNormalizeCallback's signature.

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Save a `checkpoint-0-merged` of the freshly-compressed (but untrained)
        # model so it can be benchmark-evaluated as the post-compression baseline.
        # ONLY on a fresh start (global_step == 0); on resume_from_checkpoint the
        # model already holds trained weights, so re-saving here would be wrong.
        if not getattr(state, "is_world_process_zero", False):
            return
        if state.global_step != 0:
            return
        out = pathlib.Path(args.output_dir) / "checkpoint-0-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out), self._extract_tokenizer(kwargs))

    def on_save(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return
        out = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out), self._extract_tokenizer(kwargs))
        # Honor save_total_limit for the dense -merged dirs too (the HF Trainer only
        # rotates the factored checkpoint-N dirs). Keep the newest N -merged dirs.
        self._rotate_merged(args, keep=getattr(args, "save_total_limit", None))

    @staticmethod
    def _rotate_merged(args, keep):
        if not keep or keep <= 0:
            return
        import shutil
        root = pathlib.Path(args.output_dir)
        # Keep `checkpoint-0-merged` (the post-compression baseline) permanently;
        # only rotate the trained (step > 0) merged dirs.
        merged = sorted(
            (p for p in root.glob("checkpoint-*-merged")
             if p.is_dir() and int(p.name.split("-")[1]) > 0),
            key=lambda p: int(p.name.split("-")[1]),
        )
        for old in merged[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return
        out = pathlib.Path(args.output_dir) / "final-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out), self._extract_tokenizer(kwargs))

    @staticmethod
    def _extract_tokenizer(kwargs):
        """HF Trainer passes the processing class via ``processing_class``
        in Transformers >= 4.46; older versions used ``tokenizer``. Accept
        either so the merged checkpoint stays self-contained across the
        rename window."""
        return kwargs.get("processing_class") or kwargs.get("tokenizer")

    def _dump(self, model, ckpt_dir: str, tokenizer=None) -> None:
        # Use the same non-mutating peer-model path for both plain and
        # calibrated runs. We deliberately avoid src/compress's
        # save_calibrated_btt_hf_pretrained because it calls
        # materialize_calibrated_btt_to_linear which mutates the live
        # model in place — that destroys subsequent training steps when
        # on_save fires multiple times.
        _materialize_and_save(model, ckpt_dir)
        # Also save the tokenizer / processor so the merged checkpoint
        # is drop-in for vLLM / eval. HF Trainer passes the processing
        # class (tokenizer or processor) via the ``processing_class``
        # kwarg to on_save/on_train_end (Transformers >= 4.46); older
        # versions used ``tokenizer``.
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(ckpt_dir)
