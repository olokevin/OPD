"""
On-Policy Distillation (OPD) training entrypoint.

Companion to run_rl.py, but instead of GRPO with rule-based rewards we:
  1. Roll out the student via vLLM on math prompts (on-policy samples).
  2. Score student-visited tokens by reverse KL against a frozen teacher.
  3. Train the student to minimise KL(student || teacher) on response tokens.

Loss (per response token):
    L = sum_v p_student(v|x) * (log p_student(v|x) - log p_teacher(v|x))

Examples:
  uv run run_opd.py --train-mode full --lr 1e-5 --teacher-id Qwen/Qwen3-4B --no-wandb
  uv run run_opd.py --train-mode lora --lr 1e-4 --lora-rank 8 \
      --teacher-id Qwen/Qwen3-4B --trainable-type all --no-wandb
  uv run run_opd.py --train-mode blocktt --lr 1e-4 --trainable-type all \
      --decomp-mode input_one_block --train-position small \
      --teacher-id Qwen/Qwen3-4B --no-wandb
  uv run run_opd.py --train-mode svd --lr 1e-4 --trainable-type all \
      --train-position output --teacher-id Qwen/Qwen3-4B --no-wandb
"""

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import is_flash_attn_2_available

from src.math_utils import last_boxed_only_string, remove_boxed
from src.optim.muon import Muon
from compress.integration import (
    BLOCKTT_DECOMP_GROUP_TO_MODULES,
    BTTLinear,
    SVDCompressedLinear,
    configure_compress_btt_trainability,
    configure_compress_svd_trainability,
    convert_linear_to_btt_compress,
    convert_linear_to_svd_compress,
    get_blocktt_target_module_names,
    get_svd_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)


MODE_DEFAULTS = {
    "full":    {"lr": 1e-5, "wandb_project": "math-opd-full",
                "micro_batch_size": 4, "gradient_accumulation_steps": 64},
    "lora":    {"lr": 9e-5, "wandb_project": "math-opd-lora",
                "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "blocktt": {"lr": 9e-5, "wandb_project": "math-opd-blocktt",
                "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "svd":     {"lr": 9e-5, "wandb_project": "math-opd-svd",
                "micro_batch_size": 2, "gradient_accumulation_steps": 128},
}


def resolve_attn_implementation():
    if torch.cuda.is_available() and is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="On-policy distillation training")
    parser.add_argument(
        "--train-mode", type=str, required=True,
        choices=["full", "lora", "blocktt", "svd"],
        help="Training mode for the student: full, lora, blocktt, or svd",
    )

    # Models
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-1.7B",
                        help="Student model HF id / local path")
    parser.add_argument("--teacher-id", type=str, required=True,
                        help="Teacher model HF id / local path (frozen)")

    # Optimization
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adamw8bit", "muon"])
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr-scheduler", type=str, default="none",
                        choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--cycle-length", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-adam", type=float, default=None,
                        help="Muon-only: AdamW fallback LR")
    parser.add_argument("--lr-embedding", type=float, default=None,
                        help="Muon-only: embedding AdamW LR")
    parser.add_argument("--norm-method", type=str, default=None,
                        choices=["row", "col", "row-col", "col-row", "shape"])

    # OPD loop knobs (mirroring run_rl.py shape)
    parser.add_argument("--n-opd-steps", type=int, default=50,
                        help="Number of outer OPD steps")
    parser.add_argument("--n-prompts-per-step", type=int, default=32,
                        help="Prompts sampled per outer step")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of student rollouts per prompt")
    parser.add_argument("--epochs-per-step", type=int, default=1,
                        help="Epochs over each rollout batch before resampling")
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)

    # OPD-specific
    parser.add_argument("--teacher-temperature", type=float, default=1.0,
                        help="Temperature applied to teacher logits before softmax")
    parser.add_argument("--student-temperature", type=float, default=1.0,
                        help="Temperature applied to student logits before softmax")
    parser.add_argument("--top-k", type=int, default=0,
                        help="If > 0, restrict the KL sum to teacher's top-k vocab.")

    # Generation
    parser.add_argument("--rollout-temperature", type=float, default=1.0,
                        help="vLLM sampling temperature for student rollouts")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="vLLM max_model_len")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4,
                        help="vLLM GPU memory fraction")

    # Bookkeeping
    parser.add_argument("--base-dir", type=str, default="/data/yequan/fura/opd_runs")
    parser.add_argument("--prompt-template", type=str, default="boxed.prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-save-ckpt", action="store_true")

    # LoRA-only
    parser.add_argument("--lora-rank", type=int, default=8)

    # BlockTT-only
    parser.add_argument("--decomp-mode", type=str, default="input_one_block")
    parser.add_argument("--blocktt-rank", type=str, default="full")
    parser.add_argument("--convert-mode", type=str, default="svd",
                        choices=["svd", "qr"])
    parser.add_argument("--no-train-bias", action="store_true")
    parser.add_argument("--blocktt-normalize-after-update", action="store_true")
    parser.add_argument("--blocktt-factorize-by-head",
                        action=argparse.BooleanOptionalAction, default=True)

    # Shared trainable selector (LoRA/BlockTT/SVD)
    parser.add_argument("--trainable-type", type=str, default="all",
                        choices=["all", "mlp", "attn"])
    parser.add_argument("--train-position", type=str, default=None,
                        choices=["output", "input", "small", "large", "both"])
    parser.add_argument(
        "--s-merged-to", type=str, default=None,
        choices=["frozen", "trainable", "output", "input", "split",
                 "keep_frozen", "keep_trainable"],
    )

    # W&B
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args(argv)
    return args


def apply_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.train_mode]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.wandb_project is None:
        args.wandb_project = defaults["wandb_project"]
    if args.micro_batch_size is None:
        args.micro_batch_size = defaults["micro_batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
    if args.train_mode == "blocktt" and args.train_position is None:
        args.train_position = "small"
    if args.train_mode == "svd" and args.train_position is None:
        args.train_position = "output"
    if args.train_mode in {"blocktt", "svd"} and args.s_merged_to is None:
        args.s_merged_to = "split" if args.train_position == "both" else "frozen"


def _flag_was_passed(argv, flag_name):
    return any(t == flag_name or t.startswith(f"{flag_name}=") for t in argv)


def validate_mode_specific_flags(args, argv):
    blocktt_flags = [
        "--decomp-mode", "--blocktt-rank", "--no-train-bias",
        "--blocktt-normalize-after-update", "--blocktt-factorize-by-head",
        "--no-blocktt-factorize-by-head", "--convert-mode",
    ]
    if args.train_mode != "blocktt":
        bad = [f for f in blocktt_flags if _flag_was_passed(argv, f)]
        if bad:
            raise ValueError(f"{', '.join(bad)} only valid with --train-mode blocktt")
    else:
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode, include_names=blocktt_targets,
            default_mode="input_one_block",
        )
        args.decomp_mode = decomp_mode
        args.blocktt_module_decomp_modes = module_decomp_modes
        args.decomp_mode_display = format_blocktt_decomp_mode(decomp_mode)

    if args.train_mode != "lora" and _flag_was_passed(argv, "--lora-rank"):
        raise ValueError("--lora-rank only valid with --train-mode lora")

    if args.train_mode == "full" and _flag_was_passed(argv, "--trainable-type"):
        raise ValueError("--trainable-type only valid with lora/blocktt/svd")

    tp_passed = _flag_was_passed(argv, "--train-position")
    if args.train_mode in {"full", "lora"} and tp_passed:
        raise ValueError("--train-position only valid with blocktt/svd")
    if args.train_mode == "blocktt" and tp_passed:
        if args.train_position not in {"small", "large", "both"}:
            raise ValueError("--train-position for blocktt must be small/large/both")
    if args.train_mode == "svd" and tp_passed:
        if args.train_position not in {"output", "input", "both"}:
            raise ValueError("--train-position for svd must be output/input/both")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_blocktt_rank(rank_arg: str):
    if rank_arg == "full":
        return "full"
    rank = int(rank_arg)
    if rank <= 0:
        raise ValueError("--blocktt-rank must be > 0 or 'full'")
    return rank


def format_blocktt_decomp_mode(decomp_mode):
    if isinstance(decomp_mode, str):
        return decomp_mode
    if isinstance(decomp_mode, dict):
        order = tuple(BLOCKTT_DECOMP_GROUP_TO_MODULES.keys())
        return ",".join(f"{g}={decomp_mode[g]}" for g in order)
    return str(decomp_mode)


def get_lora_target_modules(trainable_type):
    if trainable_type == "all":
        return ["gate_proj", "up_proj", "down_proj",
                "q_proj", "k_proj", "v_proj", "o_proj"]
    if trainable_type == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


@torch.no_grad()
def export_weights_for_vllm(model: torch.nn.Module):
    """Materialize BTT / SVD factored layers back to dense weights so the
    student's current params can be hot-swapped into the vLLM engine."""
    factorized_module_names = []
    weight_tuples = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLinear):
            factorized_module_names.append(name)
            weight_tuples.append((f"{name}.weight", module.materialize_dense_weight()))
            if module.bias is not None:
                weight_tuples.append((f"{name}.bias", module.bias))
        elif isinstance(module, SVDCompressedLinear):
            factorized_module_names.append(name)
            weight_tuples.append((f"{name}.weight", module.materialize_dense_weight()))
            if module.bias is not None:
                weight_tuples.append((f"{name}.bias", module.bias))
    factored_prefixes = tuple(f"{n}." for n in factorized_module_names)
    for name, param in model.named_parameters():
        if factored_prefixes and name.startswith(factored_prefixes):
            continue
        weight_tuples.append((name, param))
    return weight_tuples


def load_datasets_and_tokenizer(model_id, prompt_template_path):
    train_dataset = load_dataset("qwedsacf/competition_math", split="train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split="train[-500:]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()
    has_chat_template = tokenizer.chat_template is not None

    def process(example):
        with_template = template.replace("{question}", example["problem"])
        if has_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": with_template}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = with_template
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process)
    val_dataset = val_dataset.map(process)
    return train_dataset, val_dataset, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_t = [tokenizer.encode(p) for p in prompt_strs]
    output_t = [tokenizer.encode(o) for o in output_strs]
    max_len = max(len(prompt_t[i]) + len(output_t[i]) for i in range(len(prompt_t)))

    full = []
    for i in range(len(prompt_t)):
        pad = [tokenizer.pad_token_id] * (max_len - len(prompt_t[i]) - len(output_t[i]))
        full.append(torch.tensor(prompt_t[i] + output_t[i] + pad,
                                 dtype=torch.long).unsqueeze(0))
    f2 = torch.cat(full)
    input_ids = f2[:, :-1]
    labels = f2[:, 1:]

    response_mask = torch.zeros(len(prompt_strs), max_len - 1)
    for i in range(len(prompt_t)):
        # Predict tokens at positions [len(prompt)-1, len(prompt)+len(output)-1)
        response_mask[i,
                      len(prompt_t[i]) - 1:
                      len(prompt_t[i]) + len(output_t[i]) - 1] = 1

    return {"input_ids": input_ids, "labels": labels,
            "response_mask": response_mask.bool()}


def compute_kl_loss(
    student_logits: torch.Tensor,  # [B, T, V]
    teacher_logits: torch.Tensor,  # [B, T, V]
    response_mask: torch.Tensor,   # [B, T]
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    top_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse KL: KL(student || teacher) summed over vocab,
    averaged over response tokens. Returns (loss, per_token_kl_for_logging)."""
    s_logits = student_logits / max(student_temperature, 1e-6)
    t_logits = teacher_logits / max(teacher_temperature, 1e-6)

    if top_k and top_k > 0:
        # Restrict to teacher's top-k vocab per (b, t) position.
        topk = torch.topk(t_logits, k=top_k, dim=-1)
        idx = topk.indices                                   # [B, T, K]
        t_logits_k = topk.values                             # [B, T, K]
        s_logits_k = torch.gather(s_logits, dim=-1, index=idx)
        s_logp = torch.log_softmax(s_logits_k, dim=-1)
        t_logp = torch.log_softmax(t_logits_k, dim=-1)
    else:
        s_logp = torch.log_softmax(s_logits, dim=-1)
        t_logp = torch.log_softmax(t_logits, dim=-1)

    s_p = s_logp.exp()
    per_pos_kl = (s_p * (s_logp - t_logp)).sum(dim=-1)        # [B, T]
    masked_kl = per_pos_kl * response_mask
    denom = response_mask.sum().clamp_min(1)
    loss = masked_kl.sum() / denom
    return loss, per_pos_kl.detach()


def get_logits(model, input_ids):
    return model(input_ids).logits


def compute_run_name(args, mode_info: dict) -> str:
    if args.wandb_run_name is not None:
        return args.wandb_run_name
    stu = args.model_id.split("/")[-1]
    tch = args.teacher_id.split("/")[-1]
    head = f"opd_{stu}_from_{tch}_{args.lr:.1e}"
    if args.train_mode == "full":
        return f"{head}_full"
    if args.train_mode == "lora":
        return f"{head}_lora_r{args.lora_rank}_{args.trainable_type}"
    if args.train_mode == "blocktt":
        dm = mode_info.get("decomp_mode_display", args.decomp_mode)
        return f"{head}_blocktt_{dm}_{args.train_position}_{args.trainable_type}"
    return f"{head}_svd_{args.train_position}_{args.trainable_type}"


def create_run_dir(base_dir: str, train_mode: str, run_name: str, model_id: str) -> str:
    timestamp = time.strftime("%m%d-%H%M%S")
    model_name = model_id.split("/")[-1]
    run_dir = os.path.join(base_dir, model_name, train_mode, f"{run_name}-{timestamp}")
    os.makedirs(run_dir)
    return run_dir


def _build_factored_dense_state_dict(model):
    factored_prefixes = []
    extra = {}
    for module_name, module in model.named_modules():
        if isinstance(module, (BTTLinear, SVDCompressedLinear)):
            factored_prefixes.append(module_name + ".")
            dense = module.materialize_dense_weight().detach().clone()
            extra[f"{module_name}.weight"] = dense
            if module.bias is not None:
                extra[f"{module_name}.bias"] = module.bias.detach().clone()
    new_sd = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(p) for p in factored_prefixes):
            continue
        new_sd[name] = tensor
    new_sd.update(extra)
    return new_sd


def save_merged_checkpoint(model, tokenizer, ckpt_dir: str, train_mode: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    if train_mode == "full":
        model.save_pretrained(ckpt_dir)
    elif train_mode == "lora":
        model.merge_adapter()
        try:
            model.get_base_model().save_pretrained(ckpt_dir)
        finally:
            model.unmerge_adapter()
    elif train_mode in {"blocktt", "svd"}:
        sd = _build_factored_dense_state_dict(model)
        model.save_pretrained(ckpt_dir, state_dict=sd)
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")
    tokenizer.save_pretrained(ckpt_dir)


def save_checkpoint(model, tokenizer, run_dir: str, step_num: int, train_mode: str):
    ckpt_dir = os.path.join(run_dir, f"step={step_num}")
    print(f"Saving checkpoint to {ckpt_dir}")
    save_merged_checkpoint(model, tokenizer, ckpt_dir, train_mode)


def build_optimizer(args, trainable_params, trainable_named_params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(trainable_params, lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.optimizer == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "muon":
        return Muon(trainable_named_params, lr=args.lr,
                    lr_adam=args.lr_adam, lr_embedding=args.lr_embedding,
                    weight_decay=args.weight_decay, norm_method=args.norm_method)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _cosine_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float((current_step - num_warmup_steps) % cycle_length) / float(
        max(1, cycle_length - num_warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(args, optimizer, num_training_steps):
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)
    if args.lr_scheduler == "none":
        return None
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")
    if args.lr_scheduler == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps, last_epoch=-1,
        )
    if args.lr_scheduler == "cosine":
        cycle = args.cycle_length if args.cycle_length is not None else num_training_steps
        if num_training_steps % cycle != 0:
            raise ValueError(
                f"num_training_steps ({num_training_steps}) must be divisible by "
                f"cycle_length ({cycle})"
            )
        return LambdaLR(optimizer, partial(
            _cosine_lambda, num_warmup_steps=warmup_steps,
            cycle_length=cycle, min_lr_ratio=args.min_lr_ratio,
        ), last_epoch=-1)
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def compute_num_training_steps(args):
    if args.micro_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("micro/accum sizes must be > 0")
    samples_per_step = args.n_prompts_per_step * args.group_size
    micro_per_epoch = samples_per_step // args.micro_batch_size
    opt_per_epoch = micro_per_epoch // args.gradient_accumulation_steps
    return args.n_opd_steps * args.epochs_per_step * opt_per_epoch


def resolve_warmup_steps(warmup_ratio, num_training_steps):
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"--warmup-ratio in [0,1], got {warmup_ratio}")
    return int(math.ceil(num_training_steps * warmup_ratio))


def maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps):
    if args.no_wandb:
        return
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "train_mode": args.train_mode,
            "model_id": args.model_id,
            "teacher_id": args.teacher_id,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "num_training_steps": num_training_steps,
            "weight_decay": args.weight_decay,
            "n_opd_steps": args.n_opd_steps,
            "n_prompts_per_step": args.n_prompts_per_step,
            "group_size": args.group_size,
            "epochs_per_step": args.epochs_per_step,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "rollout_temperature": args.rollout_temperature,
            "student_temperature": args.student_temperature,
            "teacher_temperature": args.teacher_temperature,
            "top_k": args.top_k,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "run_dir": run_dir,
            **mode_info,
        },
        dir=run_dir,
    )


def build_local_vllm_generators(args, model, lora_mode: bool):
    """Build a vLLM engine for student rollouts, with hot-weight-swap each step."""
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(4096, args.max_model_len),
    )

    def _sync_weights():
        if args.train_mode in {"blocktt", "svd"}:
            weight_tuples = export_weights_for_vllm(model)
        elif lora_mode:
            model.merge_adapter()
            try:
                base = model.get_base_model()
                weight_tuples = []
                for name, p in base.named_parameters():
                    if any(m in name for m in (".lora_A.", ".lora_B.",
                                               ".lora_embedding_A.",
                                               ".lora_embedding_B.")):
                        continue
                    weight_tuples.append((name.replace(".base_layer.", "."),
                                          p.detach().clone()))
            finally:
                model.unmerge_adapter()
        else:
            weight_tuples = [(n, p) for n, p in model.named_parameters()]

        vllm_internal = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal.load_weights(weight_tuples)

    def generate(prompts: list[str], temperature: float, n: int):
        _sync_weights()
        sp = SamplingParams(
            max_tokens=args.max_response_length,
            temperature=temperature,
            n=n,
        )
        outputs = vllm_model.generate(prompts, sp)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts, _step):
        return generate(prompts, temperature=args.rollout_temperature, n=args.group_size)

    def generate_for_eval(prompts, _step):
        return generate(prompts, temperature=0.0, n=1)

    return generate_for_train, generate_for_eval, vllm_model


def load_teacher(teacher_id: str, device_id: int):
    print(f"Loading teacher: {teacher_id}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        attn_implementation=resolve_attn_implementation(),
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map={"": device_id},
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)

    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be > 0.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    if args.train_mode in {"blocktt", "svd"} and not torch.cuda.is_available():
        raise RuntimeError("BTT/SVD conversion requires CUDA.")

    set_seed(args.seed)

    mode_info = {}
    if args.train_mode == "lora":
        lora_target = get_lora_target_modules(args.trainable_type)
        mode_info.update({
            "lora_rank": args.lora_rank,
            "trainable_type": args.trainable_type,
            "target_modules": lora_target,
        })
    elif args.train_mode == "blocktt":
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        decomp_mode = args.decomp_mode
        decomp_mode_display = getattr(
            args, "decomp_mode_display", format_blocktt_decomp_mode(decomp_mode))
        module_decomp_modes = getattr(args, "blocktt_module_decomp_modes", None)
        if not isinstance(decomp_mode, dict):
            module_decomp_modes = None
        mode_info.update({
            "blocktt_rank": blocktt_rank,
            "trainable_type": args.trainable_type,
            "decomp_mode": decomp_mode,
            "decomp_mode_by_module": module_decomp_modes,
            "decomp_mode_display": decomp_mode_display,
            "train_position": args.train_position,
            "s_merged_to": args.s_merged_to,
            "target_modules": blocktt_targets,
            "train_bias": not args.no_train_bias,
            "blocktt_normalize_after_update": args.blocktt_normalize_after_update,
            "blocktt_factorize_by_head": args.blocktt_factorize_by_head,
            "convert_mode": args.convert_mode,
        })
    elif args.train_mode == "svd":
        svd_targets = get_svd_target_module_names(args.trainable_type)
        mode_info.update({
            "trainable_type": args.trainable_type,
            "train_position": args.train_position,
            "s_merged_to": args.s_merged_to,
            "target_modules": svd_targets,
        })

    print("OPD training configuration:")
    for k in ("train_mode", "model_id", "teacher_id", "lr", "optimizer",
              "rollout_temperature", "student_temperature", "teacher_temperature",
              "top_k", "n_opd_steps", "n_prompts_per_step", "group_size",
              "micro_batch_size", "gradient_accumulation_steps"):
        print(f"  {k}: {getattr(args, k)}")

    num_training_steps = compute_num_training_steps(args)
    print(f"  Optimizer update steps: {num_training_steps}")

    run_name = compute_run_name(args, mode_info)
    run_dir = create_run_dir(args.base_dir, args.train_mode, run_name, args.model_id)
    print(f"Created: {run_dir}")
    maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps)

    train_dataset, val_dataset, tokenizer = load_datasets_and_tokenizer(
        args.model_id, args.prompt_template,
    )

    device_id = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{device_id}"
    attn_impl = resolve_attn_implementation()
    print(f"  Attention backend: {attn_impl}")

    student = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map={"": device_id},
    )

    # Build trainable surface on the student.
    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=32,
                              target_modules=mode_info["target_modules"])
        student = get_peft_model(student, peft_cfg)
        student.print_trainable_parameters()
        trainable_params = [p for p in student.parameters() if p.requires_grad]
    elif args.train_mode == "blocktt":
        decomp_for_convert = (mode_info["decomp_mode_by_module"]
                              if mode_info["decomp_mode_by_module"] is not None
                              else mode_info["decomp_mode"])
        convert_linear_to_btt_compress(
            student,
            btt_rank=mode_info["blocktt_rank"],
            decomp_mode=decomp_for_convert,
            init_mode="default",
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=mode_info["s_merged_to"],
            train_position=mode_info["train_position"],
            factorize_by_head=mode_info["blocktt_factorize_by_head"],
            model_config=student.config,
            convert_mode=mode_info["convert_mode"],
        )
        stats = configure_compress_btt_trainability(
            student,
            train_bias=mode_info["train_bias"],
            train_position=mode_info["train_position"],
            train_singular_values=(mode_info["s_merged_to"] == "keep_trainable"),
        )
        if stats["num_btt_layers"] == 0:
            raise ValueError("No layers converted to BTT — check --trainable-type.")
        trainable_params = stats["trainable_params"]
        print(f"  BTT layers: {stats['num_btt_layers']}, "
              f"trainable params: {stats['trainable_param_count']:,}")
    elif args.train_mode == "svd":
        convert_linear_to_svd_compress(
            student,
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            s_merged_to=mode_info["s_merged_to"],
            train_position=args.train_position,
        )
        stats = configure_compress_svd_trainability(
            student,
            train_position=args.train_position,
            train_bias=True,
            train_embed_lm_head=(args.train_position == "both"),
            train_singular_values=(mode_info["s_merged_to"] == "keep_trainable"),
        )
        if stats["num_svd_layers"] == 0:
            raise ValueError("No layers converted to SVD — check --trainable-type.")
        trainable_params = stats["trainable_params"]
        print(f"  SVD layers: {stats['num_svd_layers']}, "
              f"trainable params: {stats['trainable_param_count']:,}")
    else:
        trainable_params = list(student.parameters())

    trainable_named_params = [
        (n, p) for n, p in student.named_parameters() if p.requires_grad
    ]
    trainable_params = [p for _, p in trainable_named_params]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")

    if args.gradient_checkpointing:
        base_for_gc = getattr(student, "base_model", None)
        gc_target = (base_for_gc if base_for_gc is not None
                     and hasattr(base_for_gc, "gradient_checkpointing_enable")
                     else student)
        gc_target.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    # Teacher: frozen, eval mode.
    teacher = load_teacher(args.teacher_id, device_id)
    if teacher.config.vocab_size != student.config.vocab_size:
        raise ValueError(
            f"Teacher / student vocab mismatch: "
            f"{teacher.config.vocab_size} vs {student.config.vocab_size}. "
            "Reverse-KL OPD assumes a shared vocabulary."
        )

    # vLLM for student rollouts, hot-swap-updated each step.
    generate_for_train, generate_for_eval, _vllm = build_local_vllm_generators(
        args, student, lora_mode=(args.train_mode == "lora"),
    )

    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    scheduler = build_lr_scheduler(args, optimizer, num_training_steps)
    normalize_blocktt_after_update = (
        args.train_mode == "blocktt" and args.blocktt_normalize_after_update
    )

    student = student.to(device)
    student.train()

    optimizer_steps = 0
    for i in range(args.n_opd_steps):
        step_start = time.time()

        sample_indices = random.sample(range(len(train_dataset)),
                                       args.n_prompts_per_step)
        batch = train_dataset[sample_indices]

        gen_start = time.time()
        outputs = generate_for_train(batch["prompt"], i)
        gen_time = time.time() - gen_start

        prompts_expanded = [p for p in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)

        # Pre-compute teacher logits once per rollout batch (frozen).
        teacher_logits_all = []
        with torch.inference_mode():
            for b in range(len(input_ids) // args.micro_batch_size):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                teacher_logits_all.append(get_logits(teacher, input_ids[idx:end]))
        # Concatenate; teacher logits are read-only from here on.

        epoch_losses = []
        epoch_kls = []
        epoch_grad_norms = []

        train_start = time.time()
        for epoch in range(args.epochs_per_step):
            for b in tqdm(
                range(len(input_ids) // args.micro_batch_size),
                desc=f"OPD step {i + 1}/{args.n_opd_steps} "
                     f"epoch {epoch + 1}/{args.epochs_per_step}",
            ):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                x = input_ids[idx:end]
                mask = response_mask[idx:end]
                t_logits = teacher_logits_all[b]

                s_logits = get_logits(student, x)

                loss, per_pos_kl = compute_kl_loss(
                    s_logits, t_logits, mask,
                    student_temperature=args.student_temperature,
                    teacher_temperature=args.teacher_temperature,
                    top_k=args.top_k,
                )
                epoch_losses.append(loss.item())
                with torch.no_grad():
                    masked_kl = per_pos_kl * mask
                    kl_mean = (masked_kl.sum() / mask.sum().clamp_min(1)).item()
                    epoch_kls.append(kl_mean)

                (loss / args.gradient_accumulation_steps).backward()

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    if normalize_blocktt_after_update:
                        normalize_trainable_blocktt_cores_(student)
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1

        train_time = time.time() - train_start
        step_time = time.time() - step_start

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        mean_kl = sum(epoch_kls) / max(1, len(epoch_kls))
        mean_gn = (sum(epoch_grad_norms) / len(epoch_grad_norms)
                   if epoch_grad_norms else 0.0)
        total_resp_tokens = response_mask.sum().item()
        avg_gen_len = int(response_mask.sum(dim=-1).float().mean().item())

        if not args.no_wandb:
            wandb.log({
                "train/loss": mean_loss,
                "train/kl_student_teacher": mean_kl,
                "train/grad_norm": mean_gn,
                "train/avg_gen_length": avg_gen_len,
                "train/response_tokens": total_resp_tokens,
                "time/step_time": step_time,
                "time/generation_time": gen_time,
                "time/training_time": train_time,
            }, step=i + 1)

        print(f"Step {i + 1}/{args.n_opd_steps} | "
              f"loss: {mean_loss:.4f} | KL: {mean_kl:.4f} | "
              f"gen_len: {avg_gen_len} | time: {step_time:.1f}s")

        if args.enable_save_ckpt and (
            (i + 1) == 10 or (i + 1) == 30 or (i + 1) == args.n_opd_steps
        ):
            save_checkpoint(student, tokenizer, run_dir, i + 1, args.train_mode)

    final_step = args.n_opd_steps
    final_dir = os.path.join(run_dir, f"step={final_step}")
    if not os.path.isdir(final_dir):
        os.makedirs(final_dir, exist_ok=True)
        try:
            save_merged_checkpoint(student, tokenizer, final_dir, args.train_mode)
            print(f"Final checkpoint: {final_dir}")
        except Exception as exc:
            print(f"WARNING: final save_merged_checkpoint failed: {exc!r}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
