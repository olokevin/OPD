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
        # With the Ray distributed executor, Ray re-derives the worker's device
        # from the placement group, so we drop CUDA_VISIBLE_DEVICES to avoid a
        # conflicting pin. With the uni (in-process) executor there is no child
        # worker -- popping it would send vLLM to physical GPU0 regardless of the
        # CUDA_VISIBLE_DEVICES the launcher set, so KEEP the pin in that case.
        if os.environ.get("NP_KEEP_CUDA_VISIBLE", "0") != "1":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        kwargs["enforce_eager"] = True
        kwargs["enable_prefix_caching"] = True
        super().__init__(*args, **kwargs)


class RayNPTrainer:
    """Node-Perturbation trainer using Ray for distributed execution.

    Per step, for each active layer:
      1. n_rollout x run_np_decode on engine 0 -> (clean_tokens, candidate_logits,
         captured_u, captured_x). u_t AND x_t are captured in the SAME perturbed
         forward (no separate capture re-decode).
      2. For each rollout: score per-step candidate logits against the teacher
         (TeacherScorer) -> (L_q_per_step, L_clean_per_step).
      3. assemble_and_apply: build δW from (L_q, L_clean, u, x) and apply locally
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
        # When co-locating the teacher on the same GPU (single-GPU-per-run sweeps),
        # request a fractional GPU per engine so student + teacher fit one device.
        gpu_frac = float(self.np_config.get("gpu_fraction", 1.0))

        pgs = [
            placement_group([{"GPU": gpu_frac, "CPU": 0}], lifetime="detached")
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

        # With backend="ray", vLLM spawns a child RayWorkerWrapper that demands a
        # FULL GPU, which cannot fit a fractional PG bundle. For tensor_parallel=1
        # use "uni" (in-process worker): the engine actor itself owns the GPU slice
        # granted by its PG, so co-locating student+teacher on one fractional GPU works.
        exec_backend = self.np_config.get("distributed_executor_backend", "ray")
        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(NPNcclLLM).remote(
                model=model_path,
                tensor_parallel_size=1,
                distributed_executor_backend=exec_backend,
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

        gpu_frac = float(self.np_config.get("gpu_fraction", 1.0))
        pg = placement_group([{"GPU": gpu_frac, "CPU": 0}], lifetime="detached")
        ray.get(pg.ready())
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        # vLLM caps prompt_logprobs at 20 by default; the teacher needs to return
        # the OPD top-k set (np.log_prob_top_k, default 256), so raise the cap.
        top_k = int(self.np_config.get("log_prob_top_k", 256))
        exec_backend = self.np_config.get("distributed_executor_backend", "ray")
        engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(NPNcclLLM).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend=exec_backend,
            worker_extension_cls=worker_ext,
            dtype=precision,
            gpu_memory_utilization=(self.np_config.get("teacher_gpu_memory_utilization")
                                    or self.np_config.get("gpu_memory_utilization", 0.9)),
            max_logprobs=max(20, top_k),
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

    def _heldout_kl(self, fixed_prompts, layer_name, sp, np_cfg):
        """Mean clean teacher-KL over a FIXED set of prompts (no perturbation).

        This is the honest training-progress signal: per-step train/L_clean_mean is
        on the step's shifting prompt batch, so it cannot show learning. Here we
        re-decode the SAME held-out prompts every probe and average the clean-row
        reverse-KL to the teacher -> a curve that should DECREASE if the LR trains
        and INCREASE if it diverges. Greedy clean decode, so it is deterministic
        given the (frozen) weights.
        """
        if not fixed_prompts or self.scorer is None:
            return None
        kls = []
        for prompt in fixed_prompts:
            pid = (prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt)
            out = ray.get(self.engines[0].collective_rpc.remote(
                "run_np_decode", args=(pid, sp, layer_name, np_cfg, 0)))[0]
            if not out["clean_tokens"]:
                continue
            full = list(pid) + list(out["clean_tokens"])
            _, L_clean = self.scorer.score_rollout(full, out["candidate_logits"])
            if L_clean:
                kls.append(float(np.mean(L_clean)))
        return float(np.mean(kls)) if kls else None

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
        loss_type = self.np_config.get("loss_type", "opd")
        if loss_type == "opd":
            teacher_path = self.np_config.teacher_model_path
            if not teacher_path:
                raise ValueError(
                    "loss_type=opd requires np.teacher_model_path to be set."
                )
            print(f"Launching teacher engine ({teacher_path})...")
            self._launch_teacher_engine(teacher_path)
            self.scorer = TeacherScorer(
                self.teacher_engine,
                top_k=self.np_config.log_prob_top_k,
                top_k_strategy=self.np_config.top_k_strategy,
                teacher_temperature=self.np_config.teacher_temperature,
                weight_mode=self.np_config.reward_weight_mode,
                teacher_batch_size=int(self.np_config.get("teacher_batch_size", 16)),
            )
        elif loss_type == "grpo":
            raise NotImplementedError(
                "np.loss_type='grpo' (rule-reward only) is not implemented in v1. "
                "Use loss_type='opd' (teacher reverse-KL). Tracking issue: see plan §5."
            )
        else:
            raise ValueError(
                f"np.loss_type={loss_type!r} is not supported. Use 'opd' (v1)."
            )
        print("Workers initialized successfully.")

    # --------------------------------------------------------- capture helper

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
        # Fixed held-out prompts for the honest progress probe (last N, never used
        # in update batches). Distinct slice so the probe measures generalization-ish
        # loss, not memorized batches. Disabled if too few prompts.
        n_heldout = int(cfg.get("heldout_probe_size", 16))
        heldout_prompts = prompts[-n_heldout:] if len(prompts) > 2 * n_heldout else []
        if heldout_prompts:
            prompts = prompts[: len(prompts) - n_heldout]
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
            topk_store_k=max(int(cfg.get("log_prob_top_k", 256)),
                             int(cfg.get("topk_store_k", 512))),
            b_pack_buckets=list(cfg.get("b_pack_buckets", [2, 4, 8, 16])),
        )

        batch_size = int(cfg.get("batch_size", 1))
        normalize_anp = bool(cfg.get("normalize_anp", False))
        verify_update = bool(cfg.get("verify_update", True))
        # V2 decode-driver selection (spec §6). Default 'eager' = V1 (parity oracle).
        decode_mode = cfg.get("decode_mode", "eager")
        use_cuda_graph = bool(cfg.get("use_cuda_graph", False))
        pack_width = int(cfg.get("pack_width", 8))
        if decode_mode not in ("eager", "graphed", "packed", "packed_graphed"):
            raise ValueError(
                f"np.decode_mode={decode_mode!r} must be 'eager', 'graphed', "
                f"'packed', or 'packed_graphed'.")
        # Packed modes need the rollout-id helper; import once (pure-Python, no
        # GPU deps) rather than per-step inside the layer loop. packed_graphed
        # (all-layer graphed) also needs the wave-padding helper that keeps every
        # wave at pack_width so only ONE bucket is ever captured.
        if decode_mode == "packed":
            from verl.workers.rollout.vllm_rollout.np_worker_extension import (
                _assign_rollout_ids,
            )
        if decode_mode == "packed_graphed":
            from verl.workers.rollout.vllm_rollout.np_worker_extension import (
                _assign_rollout_ids,
                _pad_waves_to_pack_width,
            )
        # Env-gated per-prompt boundary instrumentation (decode / score / assemble).
        # Removable: set NP_DEBUG_DECODE=1 only when diagnosing a stall.
        NP_DEBUG_DECODE = os.environ.get("NP_DEBUG_DECODE", "0") == "1"
        progress_bar = tqdm(range(num_iterations), desc="NP Training")
        for step in progress_bar:
            t0 = time.time()
            active = active_layers_for_step(
                matched, step, cfg.en_layerwise_perturbation)
            dw_norms: Dict[str, float] = {}
            w_deltas: Dict[str, float] = {}
            w_changed_fracs: Dict[str, float] = {}
            w_sync_ok: Dict[str, bool] = {}
            L_clean_means: List[float] = []
            if len(active) > 1:
                # ===== ALL-LAYER branch (eager): ONE decode over all active
                # layers -> ONE teacher score per prompt (shared across layers)
                # -> ONE all-layer assemble+apply per step. The teacher loss is
                # scored on the CLEAN rollout, which is IDENTICAL for every layer
                # (perturbation only affects captured u/x, not the sampled clean
                # tokens), so L_q/L_clean is scored ONCE per prompt and reused for
                # every layer (the C-5 landmine: re-scoring per layer is wrong AND
                # T*L times more work). Per-layer u/x are accumulated into
                # layer_u_steps[ln]/layer_x_steps[ln], built in lockstep with the
                # shared L_q_steps so each layer's list aligns positionally.
                #
                # All-layer EAGER routes through run_np_decode with the LIST arg
                # (run_np_decode_packed raises on a list -- all-layer packed is
                # Stage E). The clean rollout's candidate_logits is layer-agnostic.
                L_q_steps: List[Any] = []
                L_clean_steps: List[float] = []
                layer_u_steps: Dict[str, List[Any]] = {ln: [] for ln in active}
                layer_x_steps: Dict[str, List[Any]] = {ln: [] for ln in active}

                # Shared per-prompt score+accumulate tail. `full_pid` is the prompt
                # token ids; `out_*` are this prompt's slice of a decode result
                # (clean_tokens/candidate_logits are this prompt's; captured_u/x are
                # {layer: {t: tensor}} for THIS prompt). ONE teacher score per
                # prompt, reused for every layer; per-layer u/x appended in lockstep
                # so each layer's list aligns positionally with L_q_steps. Both the
                # eager loop and the packed_graphed wave loop feed THIS, so the
                # assemble+broadcast+verify tail below runs once, identically.
                def _accum_prompt(full_pid, out_clean_tokens,
                                  out_candidate_logits, out_captured_u,
                                  out_captured_x):
                    if not out_clean_tokens:
                        return  # empty rollout -> no signal
                    full = list(full_pid) + list(out_clean_tokens)
                    L_q, L_clean = self.scorer.score_rollout(
                        full, out_candidate_logits)
                    nT = len(out_candidate_logits)
                    L_q_steps.extend(L_q)
                    L_clean_steps.extend(L_clean)
                    for ln in active:
                        layer_u_steps[ln] += [
                            out_captured_u[ln][t] for t in range(nT)]
                        layer_x_steps[ln] += [
                            out_captured_x[ln][t] for t in range(nT)]

                if decode_mode == "packed_graphed":
                    # ALL-LAYER graphed packed decode: ONE wide forward perturbs
                    # EVERY matched layer (pass the FULL `active` list, not
                    # active[0]). Build (prompt,rollout) slots exactly like the
                    # eager packed branch, then pad every wave to pack_width so
                    # _select_bucket picks the SAME bucket each wave -> ONE captured
                    # graph (no multi-bucket allocator assert).
                    all_pids = [
                        (prompts[(step * batch_size + b) % len(prompts)])
                        for b in range(batch_size)
                    ]
                    all_pids = [
                        (p["prompt_token_ids"] if isinstance(p, dict) else p)
                        for p in all_pids
                    ]
                    rollout_ids_full = _assign_rollout_ids(
                        step, batch_size, int(cfg.n_rollout))
                    if int(cfg.n_rollout) > 1:
                        slot_pids, slot_rids = [], []
                        for b in range(batch_size):
                            for r in range(int(cfg.n_rollout)):
                                slot_pids.append(all_pids[b])
                                slot_rids.append(
                                    rollout_ids_full[b * int(cfg.n_rollout) + r])
                    else:
                        slot_pids, slot_rids = all_pids, rollout_ids_full

                    waves = _pad_waves_to_pack_width(
                        slot_pids, slot_rids, pack_width)
                    for wi, (wave_pids, wave_rids, real_count) in enumerate(waves):
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step} ALL-LAYER GRAPHED L={len(active)} "
                                  f"wave {wi} real={real_count}/{pack_width}] "
                                  f"packed_graphed decode start", flush=True)
                            _tw = time.time()
                        out = ray.get(self.engines[0].collective_rpc.remote(
                            "run_np_decode_packed_graphed",
                            args=(wave_pids, sp, active, np_cfg, wave_rids),
                        ))[0]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step}] wave decode done "
                                  f"dt={time.time()-_tw:.2f}s", flush=True)
                        # Score+accumulate ONLY the real prompts; discard padded tail.
                        for pidx in range(real_count):
                            _accum_prompt(
                                wave_pids[pidx],
                                out["clean_tokens"][pidx],
                                out["candidate_logits"][pidx],
                                out["captured_u"][pidx],
                                out["captured_x"][pidx])
                else:
                    # EAGER all-layer: per-prompt eager decode through run_np_decode
                    # with the LIST `active`. The clean rollout's candidate_logits is
                    # layer-agnostic; per-prompt u/x are {layer: {t: tensor}}.
                    for b in range(batch_size):
                        prompt = prompts[(step * batch_size + b) % len(prompts)]
                        pid = (prompt["prompt_token_ids"]
                               if isinstance(prompt, dict) else prompt)
                        for r in range(int(cfg.n_rollout)):
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} ALL-LAYER L={len(active)} "
                                      f"b{b}/{batch_size} r{r}] decode start "
                                      f"plen={len(pid)}", flush=True)
                                _td = time.time()
                            out = ray.get(self.engines[0].collective_rpc.remote(
                                "run_np_decode",
                                args=(pid, sp, active, np_cfg, r),
                            ))[0]
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] all-layer decode done "
                                      f"ntok={len(out['clean_tokens'])} "
                                      f"dt={time.time()-_td:.2f}s", flush=True)
                                _ts = time.time()
                            _accum_prompt(pid, out["clean_tokens"],
                                          out["candidate_logits"],
                                          out["captured_u"], out["captured_x"])
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] all-layer score done "
                                      f"dt={time.time()-_ts:.2f}s", flush=True)

                if not L_q_steps:
                    pass  # nothing decoded this step -> fall through to logging
                else:
                    if NP_DEBUG_DECODE:
                        print(f"[npdbg s{step} ALL-LAYER] batch decode+score DONE; "
                              f"assemble start nsig={len(L_q_steps)} "
                              f"nlayer={len(active)}", flush=True)
                        _ta = time.time()
                    # Weight norms BEFORE the update (engine 0), per layer.
                    w_before = {}
                    if verify_update:
                        for ln in active:
                            w_before[ln] = ray.get(
                                self.engines[0].collective_rpc.remote(
                                    "layer_weight_norm", args=(ln,)))[0]
                    # ONE all-layer assemble+apply: shared L_q/L_clean, per-layer
                    # u/x. Returns {ln: ||dW||}.
                    layer_signals = {
                        ln: {"u": layer_u_steps[ln], "x": layer_x_steps[ln]}
                        for ln in active
                    }
                    dws = ray.get(self.engines[0].collective_rpc.remote(
                        "assemble_all_layers_and_apply",
                        args=(layer_signals, L_q_steps, L_clean_steps,
                              float(cfg.sigma), cfg.grad_estimate_sample,
                              normalize_anp, cfg.token_agg, float(cfg.lr),
                              cfg.get("update_clip")),
                    ))[0]
                    if NP_DEBUG_DECODE:
                        print(f"[npdbg s{step} ALL-LAYER] assemble+apply DONE "
                              f"dt={time.time()-_ta:.2f}s nlayer={len(dws)}",
                              flush=True)
                    for ln, dw in dws.items():
                        dw_norms[ln] = dw
                    # Broadcast EVERY updated layer from engine 0 to all engines.
                    # (last_changed_frac is per-layer state set by the most-recent
                    # apply_node_update, so read it right after each broadcast.)
                    for ln in active:
                        ray.get([
                            e.collective_rpc.remote("broadcast_layer_weights",
                                                     args=(ln, 0))
                            for e in self.engines
                        ])
                    # Per-layer verify (same machinery as the single-layer path,
                    # looped over active). NOTE: last_changed_frac is overwritten
                    # by each apply inside assemble_all_layers_and_apply, so it is
                    # NOT reliable per-layer here -- read weight_changed_frac via a
                    # per-layer norm-delta is the honest signal we keep.
                    if verify_update:
                        for ln in active:
                            norms = ray.get([
                                e.collective_rpc.remote("layer_weight_norm",
                                                         args=(ln,))
                                for e in self.engines
                            ])
                            norms = [n[0] for n in norms]
                            w_deltas[ln] = abs(norms[0] - w_before[ln])
                            w_sync_ok[ln] = all(
                                abs(n - norms[0]) < 1e-3 for n in norms)
                    if L_clean_steps:
                        L_clean_means.append(float(np.mean(L_clean_steps)))
                # End all-layer branch; logging tail iterates dw_norms etc.
                active_iter = []  # skip the single-layer loop below
            else:
                active_iter = active
            for layer_name in active_iter:
                # Accumulate batch_size distinct prompts (one update per step).
                # Each prompt contributes n_rollout rollouts (default 1). The
                # per-token signals from ALL prompts/rollouts are concatenated and
                # fed once to assemble_and_apply -> a single delta_W per step, i.e.
                # a true mini-batch of effective size batch_size * n_rollout prompts.
                L_q_steps: List[Any] = []
                L_clean_steps: List[float] = []
                u_steps: List[Any] = []
                x_steps: List[Any] = []

                if decode_mode == "packed":
                    all_pids = [
                        (prompts[(step * batch_size + b) % len(prompts)])
                        for b in range(batch_size)
                    ]
                    all_pids = [
                        (p["prompt_token_ids"] if isinstance(p, dict) else p)
                        for p in all_pids
                    ]
                    rollout_ids_full = _assign_rollout_ids(
                        step, batch_size, int(cfg.n_rollout))
                    # n_rollout>1: expand prompts to (prompt,rollout) slots.
                    if int(cfg.n_rollout) > 1:
                        slot_pids, slot_rids = [], []
                        for b in range(batch_size):
                            for r in range(int(cfg.n_rollout)):
                                slot_pids.append(all_pids[b])
                                slot_rids.append(
                                    rollout_ids_full[b * int(cfg.n_rollout) + r])
                    else:
                        slot_pids, slot_rids = all_pids, rollout_ids_full

                    for w0 in range(0, len(slot_pids), pack_width):
                        wave_pids = slot_pids[w0:w0 + pack_width]
                        wave_rids = slot_rids[w0:w0 + pack_width]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step} L={layer_name} "
                                  f"wave {w0}-{w0+len(wave_pids)}/{len(slot_pids)}] "
                                  f"packed decode start", flush=True)
                            _tw = time.time()
                        out = ray.get(self.engines[0].collective_rpc.remote(
                            "run_np_decode_packed",
                            args=(wave_pids, sp, layer_name, np_cfg, wave_rids),
                        ))[0]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step}] wave decode done "
                                  f"dt={time.time()-_tw:.2f}s", flush=True)
                        if NP_DEBUG_DECODE:
                            _tsw = time.time()
                        for pidx in range(len(wave_pids)):
                            if not out["clean_tokens"][pidx]:
                                continue
                            full = (list(wave_pids[pidx])
                                    + list(out["clean_tokens"][pidx]))
                            L_q, L_clean = self.scorer.score_rollout(
                                full, out["candidate_logits"][pidx])
                            L_q_steps += L_q
                            L_clean_steps += L_clean
                            nT = len(out["candidate_logits"][pidx])
                            u_steps += [out["captured_u"][pidx][t]
                                        for t in range(nT)]
                            x_steps += [out["captured_x"][pidx][t]
                                        for t in range(nT)]
                        if NP_DEBUG_DECODE:
                            print(f"[npdbg s{step}] wave score done "
                                  f"dt={time.time()-_tsw:.2f}s", flush=True)
                else:
                    for b in range(batch_size):
                        prompt = prompts[(step * batch_size + b) % len(prompts)]
                        pid = (prompt["prompt_token_ids"]
                               if isinstance(prompt, dict) else prompt)
                        for r in range(int(cfg.n_rollout)):
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} L={layer_name} b{b}/{batch_size} "
                                      f"r{r}] decode start plen={len(pid)}", flush=True)
                                _td = time.time()
                            if decode_mode == "graphed":
                                out = ray.get(self.engines[0].collective_rpc.remote(
                                    "run_np_decode_graphed",
                                    args=(pid, sp, layer_name, np_cfg, r, use_cuda_graph),
                                ))[0]
                            else:
                                out = ray.get(self.engines[0].collective_rpc.remote(
                                    "run_np_decode",
                                    args=(pid, sp, layer_name, np_cfg, r),
                                ))[0]
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] decode done "
                                      f"ntok={len(out['clean_tokens'])} "
                                      f"dt={time.time()-_td:.2f}s", flush=True)
                                _ts = time.time()
                            if not out["clean_tokens"]:
                                continue  # empty rollout -> no signal
                            full = list(pid) + list(out["clean_tokens"])
                            L_q, L_clean = self.scorer.score_rollout(full, out["candidate_logits"])
                            if NP_DEBUG_DECODE:
                                print(f"[npdbg s{step} b{b}] score done "
                                      f"dt={time.time()-_ts:.2f}s", flush=True)
                            L_q_steps += L_q
                            L_clean_steps += L_clean
                            # u_t and x_t are both captured inside run_np_decode (same
                            # forward) -- no separate capture re-decode (see deleted
                            # _capture_x / run_capture_pass). Aligned per step.
                            u_steps += [out["captured_u"][t]
                                        for t in range(len(out["candidate_logits"]))]
                            x_steps += [out["captured_x"][t]
                                        for t in range(len(out["candidate_logits"]))]

                if not L_q_steps:
                    continue  # nothing decoded this step

                if NP_DEBUG_DECODE:
                    print(f"[npdbg s{step} L={layer_name}] batch decode+score DONE; "
                          f"assemble start nsig={len(L_q_steps)} "
                          f"u_steps={len(u_steps)}", flush=True)
                    _ta = time.time()

                # Weight norm BEFORE the update (engine 0) for the propagation check.
                w_before = (ray.get(self.engines[0].collective_rpc.remote(
                    "layer_weight_norm", args=(layer_name,)))[0]
                    if verify_update else None)

                dw = ray.get(self.engines[0].collective_rpc.remote(
                    "assemble_and_apply",
                    args=(layer_name, L_q_steps, L_clean_steps, u_steps, x_steps,
                          float(cfg.sigma), cfg.grad_estimate_sample, normalize_anp,
                          cfg.token_agg, float(cfg.lr), cfg.get("update_clip")),
                ))[0]
                if NP_DEBUG_DECODE:
                    print(f"[npdbg s{step} L={layer_name}] assemble+apply DONE "
                          f"dt={time.time()-_ta:.2f}s dW={dw:.3e}", flush=True)
                dw_norms[layer_name] = dw
                # Broadcast updated layer from engine 0 to all engines.
                ray.get([
                    e.collective_rpc.remote("broadcast_layer_weights",
                                             args=(layer_name, 0))
                    for e in self.engines
                ])
                # Verify: (1) the update LANDED -- measured by the fraction of weight
                # ELEMENTS that changed (the norm difference badly under-reports this;
                # a 20%-of-elements bf16 update can show ~0 norm delta), (2) every
                # other engine now holds the SAME weight (broadcast worked, so the
                # next run_np_decode on any engine reads the updated param).
                if verify_update:
                    w_changed_fracs[layer_name] = ray.get(
                        self.engines[0].collective_rpc.remote("last_changed_frac"))[0]
                    norms = ray.get([
                        e.collective_rpc.remote("layer_weight_norm", args=(layer_name,))
                        for e in self.engines
                    ])
                    norms = [n[0] for n in norms]
                    w_deltas[layer_name] = abs(norms[0] - w_before)
                    w_sync_ok[layer_name] = all(
                        abs(n - norms[0]) < 1e-3 for n in norms)
                if L_clean_steps:
                    L_clean_means.append(float(np.mean(L_clean_steps)))

            iter_time = time.time() - t0
            train_metrics: Dict[str, Any] = {
                "train/step_time": iter_time,
                "training/global_step": step,
            }
            for k, v in dw_norms.items():
                train_metrics[f"train/dW_norm/{k}"] = v
            for k, v in w_deltas.items():
                train_metrics[f"train/weight_delta/{k}"] = v
            for k, v in w_changed_fracs.items():
                train_metrics[f"train/weight_changed_frac/{k}"] = v
            if L_clean_means:
                train_metrics["train/L_clean_mean"] = float(np.mean(L_clean_means))
            if w_deltas:
                train_metrics["train/weight_delta_mean"] = float(np.mean(list(w_deltas.values())))
            if w_changed_fracs:
                train_metrics["train/weight_changed_frac_mean"] = float(np.mean(list(w_changed_fracs.values())))
            if w_sync_ok:
                all_synced = all(w_sync_ok.values())
                train_metrics["train/weight_sync_ok"] = 1.0 if all_synced else 0.0
                if not all_synced:
                    print(f"[WARN step {step}] weight broadcast OUT OF SYNC across "
                          f"engines for {[k for k,v in w_sync_ok.items() if not v]} "
                          f"-- next rollout would use stale weights!")
            # A nonzero dW that changed ZERO weight elements = a true bf16 no-op.
            # (weight_delta/norm is NOT used here -- it under-reports element changes.)
            # Gated on `k in w_changed_fracs`: the all-layer branch cannot measure a
            # reliable per-layer changed_frac (last_changed_frac is overwritten by
            # each apply inside assemble_all_layers_and_apply), so it omits the key
            # rather than emit a false 0%-changed no-op warning.
            for k, v in dw_norms.items():
                if (verify_update and v > 0 and k in w_changed_fracs
                        and w_changed_fracs[k] == 0.0):
                    print(f"[WARN step {step}] dW_norm={v:.3e} for {k} but 0% of "
                          f"weight elements changed -- true bf16 no-op (lr too small).")
            logger.log(data=train_metrics, step=step)

            progress_bar.set_postfix({
                "dW": f"{max(dw_norms.values()) if dw_norms else 0:.3e}",
                "chg%": f"{100*train_metrics.get('train/weight_changed_frac_mean', 0.0):.1f}",
                "L_clean": f"{train_metrics.get('train/L_clean_mean', 0.0):.3f}",
                "time": f"{iter_time:.2f}s",
            }, refresh=False)

            if eval_interval and (step % eval_interval == 0 or step == num_iterations - 1):
                eval_metrics = self._evaluate_model(self.engines[0], self.eval_data, step, logger)
                logger.log(data=eval_metrics, step=step)
                # Honest progress signal: clean teacher-KL on FIXED held-out prompts.
                if heldout_prompts:
                    probe_layer = active[0] if active else matched[0]
                    hk = self._heldout_kl(heldout_prompts, probe_layer, sp, np_cfg)
                    if hk is not None:
                        logger.log(data={"eval/heldout_kl": hk}, step=step)
                        print(f"[Probe @ step {step}] heldout_kl={hk:.4f} "
                              f"(fixed {len(heldout_prompts)} prompts; lower=better)")

            gc.collect()
            torch.cuda.empty_cache()

        progress_bar.close()
        # Tracking has no .finish(); per-backend cleanup runs in __del__. Be defensive
        # so individual backends (wandb/swanlab) can still publish if exposed.
        if hasattr(logger, "finish"):
            logger.finish()
        self._cleanup()
        print(f"NP training completed. Results saved to {logging_dir}")
