"""In-trainer TRUE task-accuracy eval for svd_nystrom SFT.

Every ``task_eval_steps`` optimization steps, generate with the live (compressed)
model on MATH-500 / MMLU-Pro (greedy) and AIME24 / AIME25 / AMC23 (avg@k sampling),
grade with the verl ``ttrl_math`` grader, and log per-benchmark accuracy to the
active wandb run. Generation uses HF ``model.generate`` (no vLLM/Ray); grading is
done in a SUBPROCESS using the ``verl`` conda env (which has ttrl_math +
latex2sympy2_extended + math_verify), so the ``sft`` training env stays clean.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TrainerCallback

from ..extras.logging import get_logger

if TYPE_CHECKING:
    from ..hparams.finetuning_args import FinetuningArguments
    from ..hparams.model_args import ModelArguments

logger = get_logger(__name__)

# Repo root: eval_callbacks.py is LlamaFactory/src/llamafactory/train/eval_callbacks.py
# parents: [0]=train [1]=llamafactory [2]=src [3]=LlamaFactory [4]=<repo>
_REPO = Path(__file__).resolve().parents[4]
_TEST_DATA = _REPO / "datasets" / "test_data"
_VERL_PY = "/home/yequan/miniconda3/envs/verl/bin/python"
_GRADER = _REPO / "scripts" / "compress_sft" / "grade_responses.py"

# Greedy benchmarks: (name, parquet subdir, n_problems limit, max_new_tokens)
# AIME/AMC handled separately (avg@k sampling).
_AIME_AMC = ("AIME24", "AIME25", "AMC23")


def _render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            pass
    except ValueError:
        pass
    body = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
    return body + "\nassistant:"


def _stratified(df, limit):
    if limit <= 0 or limit >= len(df):
        return list(range(len(df)))
    if "category" not in df.columns:
        return list(range(min(limit, len(df))))
    by_cat: dict = {}
    for i, c in enumerate(df["category"].tolist()):
        by_cat.setdefault(c, []).append(i)
    order, cats, pos = [], list(by_cat.values()), 0
    while len(order) < limit and any(pos < len(c) for c in cats):
        for c in cats:
            if pos < len(c):
                order.append(c[pos])
                if len(order) >= limit:
                    break
        pos += 1
    return sorted(order[:limit])


@torch.no_grad()
def _generate(model, tokenizer, prompts, *, device, max_new_tokens, do_sample,
              batch_size, temperature=1.0):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i:i + batch_size]
        enc = tokenizer(bp, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    tokenizer.padding_side = prev_side
    return outs


def _grade(items):
    """Grade [{response, ground_truth}] via the verl-env ttrl_math grader subprocess.
    Returns accuracy in [0,1]; -1.0 on grader failure (logged, non-fatal)."""
    if not items:
        return 0.0
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d) / "resp.json", Path(d) / "res.json"
        inp.write_text(json.dumps(items))
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{_REPO/'src'}:{_REPO/'verl'}"
        try:
            subprocess.run(
                [_VERL_PY, str(_GRADER), "--in", str(inp), "--out", str(outp)],
                check=True, capture_output=True, env=env, timeout=1800)
            return json.loads(outp.read_text())["accuracy"]
        except Exception as e:
            logger.warning_rank0(f"TaskAccuracyEval: grader subprocess failed: {e}")
            return -1.0


class TaskAccuracyEvalCallback(TrainerCallback):
    def __init__(self, finetuning_args: "FinetuningArguments", model_args: "ModelArguments"):
        self.steps = int(finetuning_args.task_eval_steps)
        self.math_limit = int(finetuning_args.task_eval_math_limit)
        self.mmlu_limit = int(finetuning_args.task_eval_mmlu_limit)
        self.do_aime_amc = bool(finetuning_args.task_eval_aime_amc)
        self.k = int(finetuning_args.task_eval_aime_amc_k)
        self.max_new_tokens = int(finetuning_args.task_eval_max_new_tokens)
        self.batch_size = 8
        self._last = -1

    def _eval_bench(self, model, tokenizer, device, name, limit, *, max_new_tokens,
                    do_sample, k, stratify):
        import pandas as pd
        df = pd.read_parquet(_TEST_DATA / name / "test.parquet")
        idxs = _stratified(df, limit) if stratify else list(range(min(limit, len(df)) if limit > 0 else len(df)))
        sub = df.iloc[idxs]
        prompts = [_render(tokenizer, list(r["prompt"])) for _, r in sub.iterrows()]
        gts = [r["reward_model"]["ground_truth"] for _, r in sub.iterrows()]
        # avg@k: replicate prompts k times, grade each, average
        rep_prompts = [p for p in prompts for _ in range(k)]
        rep_gts = [g for g in gts for _ in range(k)]
        responses = _generate(model, tokenizer, rep_prompts, device=device,
                              max_new_tokens=max_new_tokens, do_sample=do_sample,
                              batch_size=self.batch_size, temperature=1.0)
        acc = _grade([{"response": r, "ground_truth": str(g)}
                      for r, g in zip(responses, rep_gts)])
        return acc

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.steps <= 0 or model is None:
            return
        if state.global_step == self._last or state.global_step % self.steps != 0:
            return
        if not getattr(state, "is_world_process_zero", True):
            return
        self._last = state.global_step
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tokenizer is None:
            logger.warning_rank0("TaskAccuracyEval: no tokenizer in kwargs; skipping.")
            return
        device = next(model.parameters()).device

        was_training = model.training
        model.eval()
        metrics: dict = {}
        try:
            metrics["eval/math500_acc"] = self._eval_bench(
                model, tokenizer, device, "MATH-500", self.math_limit,
                max_new_tokens=self.max_new_tokens, do_sample=False, k=1, stratify=False)
            metrics["eval/mmlu_pro_acc"] = self._eval_bench(
                model, tokenizer, device, "MMLU-Pro", self.mmlu_limit,
                max_new_tokens=512, do_sample=False, k=1, stratify=True)
            if self.do_aime_amc:
                for bench in _AIME_AMC:
                    metrics[f"eval/{bench.lower()}_avg{self.k}"] = self._eval_bench(
                        model, tokenizer, device, bench, 0,
                        max_new_tokens=self.max_new_tokens, do_sample=True,
                        k=self.k, stratify=False)
        finally:
            if was_training:
                model.train()

        msg = " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in metrics.items())
        logger.info_rank0(f"[TaskAccuracyEval step {state.global_step}] {msg}")
        if "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except Exception as e:
                logger.warning_rank0(f"TaskAccuracyEval: wandb.log failed: {e}")
