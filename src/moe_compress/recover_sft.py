"""Recovery SFT for compressed OLMoE experts — the inversion-test training leg.

Loads a compressed (per-Linear) OLMoE checkpoint, FREEZES self-attention, trains
the MoE blocks (experts + router `mlp.*`), and fine-tunes on OpenThoughts3 for a
fixed number of samples. Runs in the VERL env (tfm 4.56) because the compressed
checkpoints store per-expert nn.Linear, incompatible with the sft-env tfm-5.2
fused-3D OlmoeExperts. Minimal HF Trainer (no DeepSpeed; OLMoE active is 1.3B so
single-GPU fits). Logs to wandb.

Protocol (user-confirmed):
  - attention FROZEN, experts + router TRAINABLE (router weights untouched at
    compression; they re-adapt during recovery — fixes the frozen-router confound).
  - recovery data = OpenThoughts3 (the original SFT distribution).
  - completion-only loss (mask the prompt tokens).

Run (verl env):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:verl HF_HOME=/data/yequan/huggingface \\
  WANDB_PROJECT=olmoe_compress_sft /home/yequan/miniconda3/envs/verl/bin/python \\
    -m moe_compress.recover_sft \\
      --ckpt /data/yequan/moe_compress/ckpts/nystrom_r0.50_s0 \\
      --tag nystrom_r0.50 --num-samples 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, TrainerCallback)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "verl"))

DATA = REPO / "datasets" / "OpenThought3-Qwen3-4B" / "data" / "train.jsonl"
BASE_TOK = "allenai/OLMoE-1B-7B-0924-Instruct"
EVAL_STEPS_DEFAULT = [0, 100, 500, 2000]


def freeze_attention(model) -> tuple[int, int]:
    """Freeze self_attn; train everything in the MoE block (experts + router gate).
    Returns (n_trainable_params, n_frozen_params)."""
    tr = fr = 0
    for name, p in model.named_parameters():
        # train mlp.* (experts + router .gate); freeze self_attn, embeddings, lm_head, norms
        train = (".mlp." in name)
        p.requires_grad_(train)
        if train:
            tr += p.numel()
        else:
            fr += p.numel()
    return tr, fr


class SFTDataset(torch.utils.data.Dataset):
    """OpenThoughts3 chat → input_ids + completion-only labels (prompt masked)."""

    def __init__(self, tokenizer, n: int, max_len: int = 4096):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = []
        with open(DATA) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msgs = json.loads(line).get("messages")
                if msgs and len(msgs) >= 2:
                    self.rows.append(msgs)
                if len(self.rows) >= n:
                    break
        logger.info(f"SFT dataset: {len(self.rows)} samples (max_len={max_len})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]
        full = self.tok.apply_chat_template(msgs, tokenize=False)
        prompt = self.tok.apply_chat_template(msgs[:-1], tokenize=False,
                                              add_generation_prompt=True)
        full_ids = self.tok(full, truncation=True, max_length=self.max_len,
                            add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt, truncation=True, max_length=self.max_len,
                              add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        n_mask = min(len(prompt_ids), len(labels))
        for j in range(n_mask):
            labels[j] = -100  # completion-only loss
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids, lab = b["input_ids"], b["labels"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


class EvalAtStepsCallback(TrainerCallback):
    """Run the 4-task eval at the planned optim-steps; log to wandb."""

    def __init__(self, model, tokenizer, steps, tag, eval_limit, out_dir):
        self.model, self.tok = model, tokenizer
        self.steps = set(steps)
        self.tag, self.eval_limit, self.out_dir = tag, eval_limit, out_dir
        self.done = set()

    def _run(self, step):
        from moe_compress.eval_tasks import eval_all
        self.model.eval()
        with torch.no_grad():
            res = eval_all(self.model, self.tok, self.model.device,
                           limit=self.eval_limit, batch_size=8)
        row = {f"eval/{t}": (v.get("value") or 0.0) for t, v in res.items()}
        row["eval/step"] = step
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(row, step=step)
        except Exception:  # noqa: BLE001
            pass
        (Path(self.out_dir) / f"eval_step{step}.json").write_text(json.dumps(res, indent=2, default=str))
        logger.info(f"[{self.tag}] step {step} eval: "
                    + " ".join(f"{t}={ (v.get('value') or 0):.3f}" for t, v in res.items()))
        self.model.train()

    def on_train_begin(self, args, state, control, **kw):
        if 0 in self.steps:
            self._run(0); self.done.add(0)

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if s in self.steps and s not in self.done:
            self._run(s); self.done.add(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--out-root", default="/data/yequan/moe_compress/sft")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_TOK, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    tr, fr = freeze_attention(model)
    # gradient checkpointing + frozen embeddings breaks the backward graph
    # ("element 0 ... does not require grad") unless the embedding output is made
    # to require grad via this hook.
    model.enable_input_require_grads()
    logger.info(f"trainable (mlp.*) {tr/1e9:.3f}B / frozen {fr/1e9:.3f}B "
                f"({100*tr/(tr+fr):.1f}% trainable)")

    ds = SFTDataset(tok, args.num_samples, max_len=args.max_len)
    # eff batch = bs*grad_accum; total optim steps = ceil(num_samples / eff_batch)
    eff = args.bs * args.grad_accum
    max_steps = (args.num_samples + eff - 1) // eff
    logger.info(f"eff_batch={eff} -> ~{max_steps} optim steps for {args.num_samples} samples")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb",
        run_name=f"olmoe_compress_sft_{args.tag}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    eval_cb = EvalAtStepsCallback(model, tok, EVAL_STEPS_DEFAULT + [max_steps],
                                  args.tag, args.eval_limit, out_dir)
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        callbacks=[eval_cb],
    )
    trainer.train()
    logger.info(f"[{args.tag}] recovery SFT done; evals in {out_dir}/eval_step*.json")


if __name__ == "__main__":
    main()
