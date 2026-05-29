"""Node-Perturbation trainer with Ray single-controller (mirrors RayESTrainer).

Reuses the ES engine-launch / NCCL / eval scaffolding verbatim; the material
differences are: NPNcclLLM forces enforce_eager + prefix caching; a teacher
engine is launched for per-step scoring; fit() drives the custom n_sample-wide
decode + per-token NP update instead of weight-perturbation ES.
See docs/superpowers/specs/2026-05-28-np-trainer-design.md.
"""
import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from verl.trainer.np.layer_resolve import active_layers_for_step
from verl.trainer.np.teacher_scorer import TeacherScorer
from verl.utils.tracking import Tracking


class NPNcclLLM(LLM):
    """vLLM wrapper for NP. enforce_eager mandatory (eager-Python hooks must run);
    prefix caching ON (shared-prefix decode across the 1+n_sample row batch)."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        kwargs["enforce_eager"] = True
        kwargs["enable_prefix_caching"] = True
        super().__init__(*args, **kwargs)


class RayNPTrainer:
    """Node-Perturbation trainer using Ray for distributed execution.

    Per step, for each active layer:
      1. n_rollout x run_np_decode on engine 0 -> (clean_tokens, candidate_logits, captured_u).
      2. For each rollout: score per-step candidate logits against the teacher
         (TeacherScorer) -> (L_q_per_step, L_clean_per_step).
      3. _capture_x: re-run the clean sequence in mode=capture to extract per-step
         x_t into the active layer.
      4. assemble_and_apply: build δW from (L_q, L_clean, u, x) and apply locally
         on engine 0; broadcast the updated layer to all other engines.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        reward_fn: Callable,
        val_reward_fn: Optional[Callable] = None,
        train_data: Optional[List[Dict[str, Any]]] = None,
        eval_data: Optional[List[Dict[str, Any]]] = None,
        prompt_processor: Optional[Callable] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn or reward_fn
        self.train_data = train_data or []
        self.eval_data = eval_data or []
        self.prompt_processor = prompt_processor

        # Extract NP config (mirrors ES's es_config pattern)
        self.np_config = config.np if hasattr(config, "np") else config

        self.engines = []
        self.placement_groups = []
        self.teacher_engine = None
        self.teacher_placement_group = None
        self.scorer: Optional[TeacherScorer] = None

        if self.np_config.get("global_seed") is not None:
            self._set_global_seed(self.np_config.global_seed)

    def _set_global_seed(self, seed: int):
        """Set random seeds for reproducibility (mirrors ES)."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # ------------------------------------------------------------------ launch

    def _launch_engines(self, model_path: str):
        """Launch student vLLM engines with placement groups (mirrors ES)."""
        num_engines = self.np_config.num_engines
        precision = self.np_config.get("precision", "bfloat16")
        worker_ext = self.np_config.get(
            "worker_extension_cls",
            "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension",
        )

        pgs = [
            placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
            for _ in range(num_engines)
        ]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for pg in pgs
        ]

        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(NPNcclLLM).remote(
                model=model_path,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls=worker_ext,
                dtype=precision,
                gpu_memory_utilization=self.np_config.get("gpu_memory_utilization", 0.9),
            )
            for strategy in strategies
        ]

        self.engines = engines
        self.placement_groups = pgs
        return engines, pgs

    def _launch_teacher_engine(self, model_path: str):
        """Launch ONE teacher vLLM engine on its own placement group.

        One-engine variant of _launch_engines. Stored on self.teacher_engine; its
        placement group is tracked separately so _cleanup tears it down too.
        """
        precision = self.np_config.get("precision", "bfloat16")
        worker_ext = self.np_config.get(
            "worker_extension_cls",
            "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension",
        )

        pg = placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        ray.get(pg.ready())
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(NPNcclLLM).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls=worker_ext,
            dtype=precision,
            gpu_memory_utilization=self.np_config.get("teacher_gpu_memory_utilization",
                                                     self.np_config.get("gpu_memory_utilization", 0.9)),
        )
        self.teacher_engine = engine
        self.teacher_placement_group = pg
        return engine

    def _init_inter_engine_group(self):
        """Initialize NCCL group across student engines for per-layer weight broadcast."""
        master_address = get_ip()
        master_port = get_open_port()
        num_engines = len(self.engines)

        ray.get([
            self.engines[i].collective_rpc.remote(
                "init_inter_engine_group",
                args=(master_address, master_port, i, num_engines),
            )
            for i in range(num_engines)
        ])

    def _cleanup(self):
        """Tear down student + teacher engines and their placement groups."""
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        if self.teacher_engine is not None:
            try:
                ray.kill(self.teacher_engine)
            except Exception:
                pass
        for pg in self.placement_groups:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        if self.teacher_placement_group is not None:
            try:
                remove_placement_group(self.teacher_placement_group)
            except Exception:
                pass

    # ----------------------------------------------------------------- evaluate

    def _evaluate_with_engine(self, engine, prompts, seed: int):
        """Submit a generate.remote on the given engine; return the ObjectRef."""
        sampling_params = SamplingParams(
            temperature=self.np_config.get("temperature", 0.0),
            seed=seed,
            max_tokens=self.np_config.get("max_tokens", 1024),
        )
        return engine.generate.remote(prompts, sampling_params, use_tqdm=False)

    def _compute_metrics(self, outputs, task_datas) -> Dict[str, Any]:
        """Compute metrics from model outputs (mirrors ES)."""
        rewards = []
        avg_rewards = []
        format_rewards = []
        answer_rewards = []

        for output, data in zip(outputs, task_datas):
            response = output.outputs[0].text
            r = self.reward_fn(response, data)
            rewards.append(r)

            if isinstance(r, dict):
                avg_rewards.append(r.get("reward", 0.0))
                if "reward_info" in r:
                    format_rewards.append(r["reward_info"].get("format_reward", 0.0))
                    answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
            else:
                avg_rewards.append(float(r))

        avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
        avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
        accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0

        return {
            "rewards": rewards,
            "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            "avg_format": avg_format,
            "avg_answer": avg_answer,
            "accuracy": accuracy,
        }

    def _evaluate_model(self, engine, eval_data: List[Dict], step: int, logger) -> Dict[str, float]:
        """Run evaluation on held-out data (mirrors ES, uses self.np_config)."""
        if not eval_data:
            return {}

        batch_size = self.np_config.get("eval_batch_size", 512)
        eval_seed = self.np_config.get("global_seed", 999)
        sampling_params = SamplingParams(
            temperature=0.0,
            seed=eval_seed,
            max_tokens=self.np_config.get("max_tokens", 1024),
        )

        all_rewards = []
        format_rewards = []
        answer_rewards = []
        start = time.time()

        for b in range(0, len(eval_data), batch_size):
            batch = eval_data[b:b + batch_size]

            if self.prompt_processor:
                prompts = [self.prompt_processor(d, self.tokenizer) for d in batch]
            else:
                prompts = [d.get("prompt", d.get("context")) for d in batch]

            outputs = ray.get(
                engine.generate.remote(prompts, sampling_params, use_tqdm=False)
            )

            for idx, (out, data) in enumerate(zip(outputs, batch)):
                response = out.outputs[0].text
                r = self.val_reward_fn(response, data)

                if isinstance(r, dict):
                    all_rewards.append(r.get("reward", 0.0))
                    if "reward_info" in r:
                        format_rewards.append(r["reward_info"].get("format_reward", 0.0))
                        answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
                else:
                    all_rewards.append(float(r))

                if idx == 0 and (step == 0 or self.np_config.get("verbose", False)):
                    print(f"\n[Debug] Eval Sample (step {step}):")
                    print(f"Ground truth: {data.get('reward_model', {}).get('ground_truth', data.get('answer', 'N/A'))}")
                    print(f"Response (first 500 chars): {response[:500]}...")
                    print(f"Reward result: {r}\n")

            del outputs
            gc.collect()

        elapsed = time.time() - start

        metrics = {
            "eval/avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "eval/std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
            "eval/min_reward": float(np.min(all_rewards)) if all_rewards else 0.0,
            "eval/max_reward": float(np.max(all_rewards)) if all_rewards else 0.0,
            "eval/format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "eval/answer_reward": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
            "eval/accuracy": (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0)
                if answer_rewards else 0.0,
            "eval/time": elapsed,
        }

        print(f"[Eval @ step {step}] avg_reward={metrics['eval/avg_reward']:.4f} ± {metrics['eval/std_reward']:.4f} "
              f"acc={metrics['eval/accuracy']:.1f}% time={elapsed:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

        return metrics

    # ---------------------------------------------------------------- workers

    def init_workers(self, model_path: str):
        """Launch student engines + NCCL + install perturb layers + (opt) teacher.

        Order: _launch_engines -> _init_inter_engine_group -> install_perturb_layers
        on every student engine. If loss_type=="opd" (default), spin up the teacher
        engine and build the TeacherScorer that the fit loop will call per rollout.
        """
        print(f"Launching {self.np_config.num_engines} student vLLM engines...")
        self._launch_engines(model_path)
        print("Initializing inter-engine NCCL group...")
        self._init_inter_engine_group()
        print("Installing perturb layers on all engines...")
        ray.get([
            e.collective_rpc.remote(
                "install_perturb_layers",
                args=(list(self.np_config.perturb_rules),),
            )
            for e in self.engines
        ])
        if self.np_config.get("loss_type", "opd") == "opd":
            teacher_path = self.np_config.teacher_model_path
            print(f"Launching teacher engine ({teacher_path})...")
            self._launch_teacher_engine(teacher_path)
            self.scorer = TeacherScorer(
                self.teacher_engine,
                top_k=self.np_config.log_prob_top_k,
                top_k_strategy=self.np_config.top_k_strategy,
                teacher_temperature=self.np_config.teacher_temperature,
                weight_mode=self.np_config.reward_weight_mode,
            )
        print("Workers initialized successfully.")

    # --------------------------------------------------------- capture helper

    def _capture_x(self, engine, prompt_ids, clean_tokens, layer_name, np_cfg):
        """Re-run the fixed [prompt + clean response] once with mode=capture.

        Thin wrapper around the worker-extension method `run_capture_pass`. The
        engine replays the committed sequence one step at a time using
        `_np_step_forward(..., n_sample=0)` (single clean row, no perturbed
        companions) -- on each step the PerturbedLinear hook records the clean
        row's input x_t for `layer_name`. Returns list[Tensor[d_in]] (CPU),
        aligned with `clean_tokens`.
        """
        return ray.get(engine.collective_rpc.remote(
            "run_capture_pass",
            args=(prompt_ids, clean_tokens, layer_name, np_cfg),
        ))[0]

    # ---------------------------------------------------------------- fit

    def fit(self):
        """Main NP training loop. Mirrors RayESTrainer.fit's scaffolding, replaces
        the perturbation/restore/update body with per-step NP update per layer."""
        base_dir = self.config.trainer.get("default_local_dir", "/tmp/verl/np_checkpoints")
        logging_dir = os.path.join(
            base_dir,
            f"np_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(logging_dir, exist_ok=True)

        logger = Tracking(
            project_name=self.config.trainer.get("project_name", "OPD-NP"),
            experiment_name=self.config.trainer.get("experiment_name", "np-run"),
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=OmegaConf.to_container(self.config, resolve=True)
                if isinstance(self.config, DictConfig) else vars(self.config),
        )

        config_path = os.path.join(logging_dir, "config.json")
        with open(config_path, "w") as f:
            if isinstance(self.config, DictConfig):
                json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=4)
            else:
                json.dump(vars(self.config), f, indent=4)

        # Prepare prompts
        if self.prompt_processor:
            prompts = [self.prompt_processor(d, self.tokenizer) for d in self.train_data]
        else:
            prompts = [d.get("prompt", d.get("context")) for d in self.train_data]

        cfg = self.np_config
        trainer_total_epochs = self.config.trainer.get("total_epochs", None)
        trainer_test_freq = self.config.trainer.get("test_freq", None)
        num_iterations = trainer_total_epochs if trainer_total_epochs else cfg.num_iterations
        eval_interval = trainer_test_freq if trainer_test_freq else cfg.get("eval_interval", 25)

        sp = SamplingParams(temperature=cfg.get("temperature", 0.0),
                            max_tokens=int(cfg.max_tokens))

        # Query engine 0 once for the resolved (matched) module name list.
        # install_perturb_layers is idempotent and returns the resolved list.
        matched = ray.get(self.engines[0].collective_rpc.remote(
            "install_perturb_layers", args=(list(cfg.perturb_rules),)))[0]

        np_cfg = dict(
            n_sample=int(cfg.n_sample),
            max_tokens=int(cfg.max_tokens),
            global_seed=int(cfg.global_seed),
            sigma=float(cfg.sigma),
            sample_method=cfg.sample_method,
        )

        progress_bar = tqdm(range(num_iterations), desc="NP Training")
        for step in progress_bar:
            t0 = time.time()
            active = active_layers_for_step(
                matched, step, cfg.en_layerwise_perturbation)
            dw_norms: Dict[str, float] = {}
            L_clean_means: List[float] = []
            for layer_name in active:
                prompt = prompts[step % len(prompts)]
                pid = (prompt["prompt_token_ids"]
                       if isinstance(prompt, dict) else prompt)

                L_q_steps: List[Any] = []
                L_clean_steps: List[float] = []
                u_steps: List[Any] = []
                x_steps: List[Any] = []

                for r in range(int(cfg.n_rollout)):
                    out = ray.get(self.engines[0].collective_rpc.remote(
                        "run_np_decode",
                        args=(pid, sp, layer_name, np_cfg, r),
                    ))[0]
                    full = list(pid) + list(out["clean_tokens"])
                    L_q, L_clean = self.scorer.score_rollout(full, out["candidate_logits"])
                    L_q_steps += L_q
                    L_clean_steps += L_clean
                    u_steps += [out["captured_u"][t]
                                for t in range(len(out["candidate_logits"]))]
                    x_steps += self._capture_x(
                        self.engines[0], pid, out["clean_tokens"], layer_name, np_cfg)

                dw = ray.get(self.engines[0].collective_rpc.remote(
                    "assemble_and_apply",
                    args=(layer_name, L_q_steps, L_clean_steps, u_steps, x_steps,
                          float(cfg.sigma), cfg.grad_estimate_sample, True,
                          cfg.token_agg, float(cfg.lr), cfg.get("update_clip")),
                ))[0]
                dw_norms[layer_name] = dw
                # Broadcast updated layer from engine 0 to all engines.
                ray.get([
                    e.collective_rpc.remote("broadcast_layer_weights",
                                             args=(layer_name, 0))
                    for e in self.engines
                ])
                if L_clean_steps:
                    L_clean_means.append(float(np.mean(L_clean_steps)))

            iter_time = time.time() - t0
            train_metrics: Dict[str, Any] = {
                "train/step_time": iter_time,
                "training/global_step": step,
            }
            for k, v in dw_norms.items():
                train_metrics[f"train/dW_norm/{k}"] = v
            if L_clean_means:
                train_metrics["train/L_clean_mean"] = float(np.mean(L_clean_means))
            logger.log(data=train_metrics, step=step)

            progress_bar.set_postfix({
                "dW": f"{max(dw_norms.values()) if dw_norms else 0:.3e}",
                "L_clean": f"{train_metrics.get('train/L_clean_mean', 0.0):.3f}",
                "time": f"{iter_time:.2f}s",
            }, refresh=False)

            if eval_interval and (step % eval_interval == 0 or step == num_iterations - 1):
                eval_metrics = self._evaluate_model(self.engines[0], self.eval_data, step, logger)
                logger.log(data=eval_metrics, step=step)

            gc.collect()
            torch.cuda.empty_cache()

        progress_bar.close()
        logger.finish()
        self._cleanup()
        print(f"NP training completed. Results saved to {logging_dir}")
