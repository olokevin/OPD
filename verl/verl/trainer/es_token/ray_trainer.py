"""es_token trainer: per-token weight-perturbation ES for OPD.

Subclasses RayNPTrainer for the engine-launch / NCCL / eval scaffolding and
replaces the fit loop: graphed packed es_token decode (1 clean + N rail rows
per token, rank-1 weight perturbation), ONE teacher prefill per rollout for the
sampled-token loss, chunked-GEMM assembly on the worker. Phase wall-clocks
(decode / teacher / assemble) are logged per step -- they are the benchmark
deliverable. See docs/plans/es_token_trainer.md.
"""
import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from vllm import SamplingParams

from verl.trainer.es_token.grad_estimator import (rail_scales,
                                                  sampled_token_losses)
from verl.trainer.np.ray_trainer import RayNPTrainer
from verl.utils.tracking import Tracking
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    _assign_rollout_ids, _pad_waves_to_pack_width)


class SampledTokenTeacher:
    """ONE teacher prefill per rollout; reads log q(y_t) of each response token
    via vLLM prompt_logprobs (the actual token's logprob is always included).
    Batched over prompts like NP's TeacherScorer.score_wave."""

    def __init__(self, teacher_engine, teacher_temperature, teacher_batch_size):
        self.engine = teacher_engine
        self.temp = float(teacher_temperature)
        self.batch = max(1, int(teacher_batch_size))

    def _sp(self):
        return SamplingParams(temperature=self.temp, max_tokens=1,
                              prompt_logprobs=1)

    def logq_wave(self, fulls: List[List[int]], resp_lens: List[int]):
        """fulls[i] = prompt+response token ids; resp_lens[i] = response length.
        Returns [tensor [T_i] of teacher logprobs of the response tokens]."""
        assert len(fulls) == len(resp_lens)
        out: List[torch.Tensor] = [None] * len(fulls)
        sp = self._sp()
        for s0 in range(0, len(fulls), self.batch):
            idxs = list(range(s0, min(s0 + self.batch, len(fulls))))
            prompts = [{"prompt_token_ids": list(fulls[i])} for i in idxs]
            outs = ray.get(self.engine.generate.remote(prompts, sp,
                                                       use_tqdm=False))
            for o, i in zip(outs, idxs):
                T = int(resp_lens[i])
                if T == 0:
                    out[i] = torch.zeros(0)
                    continue
                plp = o.prompt_logprobs[-T:]
                ids = fulls[i][-T:]
                out[i] = torch.tensor(
                    [plp[t][ids[t]].logprob for t in range(T)],
                    dtype=torch.float32)
        return out


class RayESTokenTrainer(RayNPTrainer):
    def __init__(self, config: DictConfig, tokenizer, reward_fn,
                 val_reward_fn=None, train_data=None, eval_data=None,
                 prompt_processor=None):
        super().__init__(config, tokenizer, reward_fn, val_reward_fn,
                         train_data, eval_data, prompt_processor)
        # Rebind the parent's config slot to the es_token group so every
        # inherited method (_launch_engines, _launch_teacher_engine, eval, ...)
        # reads es_token.* keys.
        self.np_config = config.es_token
        self.es = config.es_token
        if self.es.get("global_seed") is not None:
            self._set_global_seed(self.es.global_seed)
        self.teacher = None
        self.matched: List[str] = []

    # ---------------------------------------------------------------- init ---
    def init_workers(self, model_path: str):
        print(f"Launching {self.es.num_engines} student vLLM engines...")
        self._launch_engines(model_path)
        print("Initializing inter-engine NCCL group...")
        self._init_inter_engine_group()
        print("Installing es_token layers on all engines...")
        matched_per_engine = ray.get([
            e.collective_rpc.remote(
                "install_es_layers",
                args=(list(self.es.perturb_rules), int(self.es.n_sample),
                      int(self.es.global_seed)))
            for e in self.engines
        ])
        self.matched = list(matched_per_engine[0][0])
        print(f"Matched {len(self.matched)} layers "
              f"(first: {self.matched[:2]} ... last: {self.matched[-1:]})")
        teacher_path = self.es.teacher_model_path
        if not teacher_path:
            raise ValueError("es_token requires es_token.teacher_model_path")
        print(f"Launching teacher engine ({teacher_path})...")
        self._launch_teacher_engine(teacher_path)
        self.teacher = SampledTokenTeacher(
            self.teacher_engine, self.es.teacher_temperature,
            self.es.get("teacher_batch_size", 16))
        print("Workers initialized successfully.")

    # --------------------------------------------------------------- probe ---
    def _heldout_clean_loss(self, heldout_pids, sp, es_cfg):
        """Mean clean sampled-token loss (log p0(y_t) - log q(y_t)) on FIXED
        held-out prompts -- the honest progress signal (single-sample reverse-KL
        estimate; lower = closer to teacher). The clean rail is unperturbed, so
        this reuses the training decode at the configured sigma."""
        if not heldout_pids or self.teacher is None:
            return None
        pack_width = int(self.es.get("pack_width", 4))
        vals = []
        for w0 in range(0, len(heldout_pids), pack_width):
            wave = heldout_pids[w0:w0 + pack_width]
            out = ray.get(self.engines[0].collective_rpc.remote(
                "run_es_decode_packed",
                args=(wave, sp, es_cfg, list(range(len(wave))), True)))[0]
            fulls, lens, p0 = [], [], []
            for i, pid in enumerate(wave):
                toks = out["clean_tokens"][i]
                if not toks:
                    continue
                fulls.append(list(pid) + list(toks))
                lens.append(len(toks))
                p0.append(out["payload"][i][:, 0])
            if not fulls:
                continue
            logqs = self.teacher.logq_wave(fulls, lens)
            for lp0, lq in zip(p0, logqs):
                vals.append(float((lp0 - lq).mean().item()))
        return float(np.mean(vals)) if vals else None

    # ----------------------------------------------------------------- fit ---
    def fit(self):
        cfg = self.es
        base_dir = self.config.trainer.get("default_local_dir",
                                           "/tmp/verl/es_token_checkpoints")
        logging_dir = os.path.join(
            base_dir, f"es_token_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(logging_dir, exist_ok=True)
        logger = Tracking(
            project_name=self.config.trainer.get("project_name", "OPD-ES-TOKEN"),
            experiment_name=self.config.trainer.get("experiment_name",
                                                    "es-token-run"),
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=OmegaConf.to_container(self.config, resolve=True)
            if isinstance(self.config, DictConfig) else vars(self.config),
        )
        with open(os.path.join(logging_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.config, resolve=True), f,
                      indent=4)

        if self.prompt_processor:
            prompts = [self.prompt_processor(d, self.tokenizer)
                       for d in self.train_data]
        else:
            prompts = [d.get("prompt", d.get("context"))
                       for d in self.train_data]

        n_heldout = int(cfg.get("heldout_probe_size", 16))
        heldout = prompts[-n_heldout:] if len(prompts) > 2 * n_heldout else []
        if heldout:
            prompts = prompts[: len(prompts) - n_heldout]
        heldout_pids = [(p["prompt_token_ids"] if isinstance(p, dict) else p)
                        for p in heldout]

        num_iterations = (self.config.trainer.get("total_epochs", None)
                          or cfg.num_iterations)
        eval_interval = (self.config.trainer.get("test_freq", None)
                         or cfg.get("eval_interval", 25))

        sp = SamplingParams(temperature=cfg.get("temperature", 0.0),
                            max_tokens=int(cfg.max_tokens))
        es_cfg = dict(
            n_sample=int(cfg.n_sample),
            max_tokens=int(cfg.max_tokens),
            global_seed=int(cfg.global_seed),
            sigma=float(cfg.sigma),
            sigma_mode=cfg.get("sigma_mode", "absolute"),
            sample_method=cfg.sample_method,
            b_pack_buckets=list(cfg.get("b_pack_buckets", [2, 4])),
            token_agg=cfg.get("token_agg", "sum"),
        )
        batch_size = int(cfg.get("batch_size", 1))
        pack_width = int(cfg.get("pack_width", 4))
        n_rails = int(cfg.n_sample)
        weight_mode = cfg.get("reward_weight_mode", "student_iw")
        iw_clamp = cfg.get("iw_clamp", 10.0)
        scale_mode = cfg.get("grad_estimate_sample", "mean_baseline")
        verify_update = bool(cfg.get("verify_update", True))
        use_graph = bool(cfg.get("use_cuda_graph", True))
        ES_DEBUG = os.environ.get("ES_DEBUG_DECODE", "0") == "1"

        progress = tqdm(range(num_iterations), desc="ES-token Training")
        for step in progress:
            t0 = time.time()
            pids = [prompts[(step * batch_size + b) % len(prompts)]
                    for b in range(batch_size)]
            pids = [(p["prompt_token_ids"] if isinstance(p, dict) else p)
                    for p in pids]
            rollout_ids = _assign_rollout_ids(step, batch_size, 1)
            waves = _pad_waves_to_pack_width(pids, rollout_ids, pack_width)

            # ---- Phase 1: graphed packed rail decode --------------------- #
            t_dec0 = time.time()
            roll_pids, roll_rids, roll_toks, roll_payload = [], [], [], []
            for wi, (wave_pids, wave_rids, real_count) in enumerate(waves):
                if ES_DEBUG:
                    print(f"[esdbg s{step} wave {wi} real={real_count}] decode",
                          flush=True)
                    _tw = time.time()
                out = ray.get(self.engines[0].collective_rpc.remote(
                    "run_es_decode_packed",
                    args=(wave_pids, sp, es_cfg, wave_rids, use_graph)))[0]
                if ES_DEBUG:
                    print(f"[esdbg s{step} wave {wi}] decode done "
                          f"dt={time.time()-_tw:.2f}s", flush=True)
                for i in range(real_count):
                    if not out["clean_tokens"][i]:
                        continue
                    roll_pids.append(wave_pids[i])
                    roll_rids.append(int(wave_rids[i]))
                    roll_toks.append(list(out["clean_tokens"][i]))
                    roll_payload.append(out["payload"][i])
            decode_s = time.time() - t_dec0

            if not roll_toks:
                logger.log(data={"train/step_time": time.time() - t0,
                                 "training/global_step": step}, step=step)
                continue

            # ---- Phase 2: ONE teacher prefill per rollout ---------------- #
            t_tch0 = time.time()
            fulls = [list(p) + t for p, t in zip(roll_pids, roll_toks)]
            lens = [len(t) for t in roll_toks]
            logqs = self.teacher.logq_wave(fulls, lens)
            teacher_s = time.time() - t_tch0

            # ---- Phase 3: losses -> scales -> assemble+apply ------------- #
            t_asm0 = time.time()
            rec_rids: List[int] = []
            rec_t: List[int] = []
            rec_scales: List[torch.Tensor] = []
            clean_means: List[float] = []
            for rid, payload, logq in zip(roll_rids, roll_payload, logqs):
                losses, clean = sampled_token_losses(
                    payload, logq, weight_mode, iw_clamp)
                # RAW rail differences; the 1/sigma_l is applied per layer in
                # the worker assemble (sigma_mode=relative stays unbiased).
                sc = rail_scales(losses, clean, 1.0, scale_mode)   # [T, N]
                T = sc.shape[0]
                rec_rids += [rid] * T
                rec_t += list(range(T))
                rec_scales.append(sc)
                clean_means.append(float(clean.mean().item()))
            scales = torch.cat(rec_scales, dim=0)                  # [M, N]
            assert scales.shape[1] == n_rails

            w_before = {}
            if verify_update:
                for ln in self.matched:
                    w_before[ln] = ray.get(
                        self.engines[0].collective_rpc.remote(
                            "layer_weight_norm", args=(ln,)))[0]
            dws = ray.get(self.engines[0].collective_rpc.remote(
                "es_assemble_and_apply",
                args=(rec_rids, rec_t, scales, es_cfg, float(cfg.lr),
                      cfg.get("update_clip"),
                      int(cfg.get("assemble_chunk", 1024)))))[0]
            for ln in self.matched:
                ray.get([
                    e.collective_rpc.remote("broadcast_layer_weights",
                                            args=(ln, 0))
                    for e in self.engines
                ])
            assemble_s = time.time() - t_asm0

            w_deltas, w_sync_ok = {}, {}
            if verify_update:
                for ln in self.matched:
                    norms = ray.get([
                        e.collective_rpc.remote("layer_weight_norm",
                                                args=(ln,))
                        for e in self.engines
                    ])
                    norms = [n[0] for n in norms]
                    w_deltas[ln] = abs(norms[0] - w_before[ln])
                    w_sync_ok[ln] = all(abs(n - norms[0]) < 1e-3
                                        for n in norms)

            step_time = time.time() - t0
            metrics: Dict[str, Any] = {
                "train/step_time": step_time,
                "train/decode_s": decode_s,
                "train/teacher_s": teacher_s,
                "train/assemble_s": assemble_s,
                "train/n_token_records": int(scales.shape[0]),
                "train/L_clean_mean": float(np.mean(clean_means)),
                "train/dW_norm_max": float(max(dws.values())),
                "train/dW_norm_mean": float(np.mean(list(dws.values()))),
                "training/global_step": step,
            }
            if w_deltas:
                metrics["train/weight_delta_mean"] = float(
                    np.mean(list(w_deltas.values())))
                metrics["train/weight_sync_ok"] = (
                    1.0 if all(w_sync_ok.values()) else 0.0)
            logger.log(data=metrics, step=step)
            progress.set_postfix({
                "dW": f"{metrics['train/dW_norm_max']:.3e}",
                "L_clean": f"{metrics['train/L_clean_mean']:.3f}",
                "dec": f"{decode_s:.1f}s",
                "tch": f"{teacher_s:.1f}s",
                "asm": f"{assemble_s:.1f}s",
            }, refresh=False)

            if eval_interval and (step % eval_interval == 0
                                  or step == num_iterations - 1):
                eval_metrics = self._evaluate_model(
                    self.engines[0], self.eval_data, step, logger)
                if eval_metrics:
                    logger.log(data=eval_metrics, step=step)
                hk = self._heldout_clean_loss(heldout_pids, sp, es_cfg)
                if hk is not None:
                    logger.log(data={"eval/heldout_clean_loss": hk}, step=step)
                    print(f"[Probe @ step {step}] heldout_clean_loss={hk:.4f} "
                          f"(fixed {len(heldout_pids)} prompts; lower=better)")

            gc.collect()
            torch.cuda.empty_cache()

        progress.close()
        if hasattr(logger, "finish"):
            logger.finish()
        self._cleanup()
        print(f"es_token training completed. Results saved to {logging_dir}")
