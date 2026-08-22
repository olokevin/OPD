"""
Evolution Strategy (ES) Trainer with Ray-based single controller.
This trainer implements zeroth-order optimization for LLM fine-tuning.

The ES algorithm:
1. Perturbs model weights with random noise (scaled by sigma)
2. Evaluates perturbed models in parallel using vLLM engines
3. Updates weights using normalized fitness-weighted noise
"""

import gc
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.utils import get_ip, get_open_port

from verl.utils.tracking import Tracking


@dataclass
class ESConfig:
    """Configuration for Evolution Strategy training."""
    # ES hyperparameters
    sigma: float = 0.001  # Noise scale for perturbation
    alpha: float = 0.0005  # Learning rate
    population_size: int = 30  # Number of perturbations per iteration
    num_engines: int = 4  # Number of parallel vLLM engines
    num_iterations: int = 800  # Total training iterations
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    precision: str = "bfloat16"  # float16, bfloat16, or float32
    
    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.0
    
    # Evaluation settings
    eval_interval: int = 25
    eval_batch_size: int = 512
    
    # Experiment settings
    experiment_dir: str = "es-ft-experiment"
    global_seed: Optional[int] = None
    verbose: bool = False
    
    # Worker extension path (relative to project root)
    worker_extension_cls: str = "utils.worker_extn.WorkerExtension"


class ESNcclLLM(LLM):
    """vLLM wrapper for ES training with NCCL support."""
    
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


class RayESTrainer:
    """
    Evolution Strategy trainer using Ray for distributed execution.
    
    This trainer implements the OpenAI ES algorithm:
    θ_{t+1} = θ_t + α * (1/nσ) * Σ F_i * ε_i
    
    where:
    - θ: model parameters
    - α: learning rate
    - σ: noise scale
    - n: population size
    - F_i: normalized fitness (reward)
    - ε_i: perturbation noise
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
        """
        Initialize the ES trainer.
        
        Args:
            config: Training configuration
            tokenizer: HuggingFace tokenizer
            reward_fn: Function to compute rewards from model outputs
            val_reward_fn: Optional separate reward function for validation
            train_data: Training data (list of task dictionaries)
            eval_data: Evaluation data (list of task dictionaries)
            prompt_processor: Function to process task data into prompts
        """
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn or reward_fn
        self.train_data = train_data or []
        self.eval_data = eval_data or []
        self.prompt_processor = prompt_processor
        
        # Extract ES config
        self.es_config = config.es if hasattr(config, 'es') else config
        
        # Initialize engines list
        self.engines = []
        self.placement_groups = []
        
        # Set random seeds if specified
        if self.es_config.get('global_seed') is not None:
            self._set_global_seed(self.es_config.global_seed)
    
    def _set_global_seed(self, seed: int):
        """Set random seeds for reproducibility."""
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
    
    def _launch_engines(self, model_path: str):
        """Launch vLLM engines with placement groups."""
        num_engines = self.es_config.num_engines
        precision = self.es_config.get('precision', 'bfloat16')
        worker_ext = self.es_config.get('worker_extension_cls', 'utils.worker_extn.WorkerExtension')
        
        # Create placement groups
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
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
                model=model_path,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls=worker_ext,
                dtype=precision,
                enable_prefix_caching=False,
                enforce_eager=False,
                max_model_len=self.es_config.get('max_model_len', None),
                seed=int(self.es_config.get('global_seed', 0) or 0),
                gpu_memory_utilization=self.es_config.get('gpu_memory_utilization', 0.9)
            )
            for strategy in strategies
        ]
        
        self.engines = engines
        self.placement_groups = pgs
        return engines, pgs
    
    def _init_inter_engine_group(self):
        """Initialize NCCL group for weight synchronization between engines."""
        master_address = get_ip()
        master_port = get_open_port()
        num_engines = len(self.engines)
        
        ray.get([
            self.engines[i].collective_rpc.remote(
                "init_inter_engine_group", 
                args=(master_address, master_port, i, num_engines)
            )
            for i in range(num_engines)
        ])
    
    def _cleanup(self):
        """Clean up Ray resources."""
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in self.placement_groups:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
    
    def _evaluate_with_engine(self, engine, prompts, seed: int):
        """Evaluate prompts using a specific engine."""
        sampling_params = SamplingParams(
            temperature=self.es_config.get('temperature', 0.0),
            seed=seed,
            max_tokens=self.es_config.get('max_tokens', 1024)
        )
        return engine.generate.remote(prompts, sampling_params, use_tqdm=False)
    
    def _compute_metrics(self, outputs, task_datas) -> Dict[str, Any]:
        """Compute metrics from model outputs."""
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
        """Run evaluation on held-out data."""
        if not eval_data:
            return {}
        
        batch_size = self.es_config.get('eval_batch_size', 512)
        eval_seed = self.es_config.get('global_seed', 999)
        # Held-out eval gets its own (larger) token budget: it is run once every
        # eval_interval iterations, whereas the training rollout budget is paid
        # population_size times per iteration.
        sampling_params = SamplingParams(
            temperature=0.0,
            seed=eval_seed,
            max_tokens=self.es_config.get('eval_max_tokens', None)
                       or self.es_config.get('max_tokens', 1024)
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
                
                # Print sample for inspection (always print first sample on step 0 for debugging)
                if idx == 0 and (step == 0 or self.es_config.get('verbose', False)):
                    print(f"\n[Debug] Eval Sample (step {step}):")
                    print(f"Ground truth: {data.get('reward_model', {}).get('ground_truth', data.get('answer', 'N/A'))}")
                    print(f"Response (first 500 chars): {response[:500]}...")
                    print(f"Reward result: {r}\n")
            
            # Clean up after each eval batch
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
            "eval/accuracy": (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0,
            "eval/time": elapsed,
        }
        
        print(f"[Eval @ step {step}] avg_reward={metrics['eval/avg_reward']:.4f} ± {metrics['eval/std_reward']:.4f} "
              f"acc={metrics['eval/accuracy']:.1f}% time={elapsed:.2f}s")
        
        # Clean up GPU memory after evaluation
        gc.collect()
        torch.cuda.empty_cache()
        
        return metrics
    
    def init_workers(self, model_path: str):
        """Initialize vLLM workers and NCCL communication."""
        print(f"Launching {self.es_config.num_engines} vLLM engines...")
        self._launch_engines(model_path)
        print("Initializing inter-engine NCCL group...")
        self._init_inter_engine_group()
        self._init_es_state()
        print("Workers initialized successfully.")

    def _init_es_state(self):
        """Install the subspace-restricted perturbation state on every engine.

        `perturb_mode=off` keeps the original in-place bf16 add/subtract path
        (bit-compatible with previous runs).  Any other mode switches the worker to
        the `W = W_base + P(coef)` formulation with an fp32 coefficient master --
        see `es_worker_extension.StructuredESMixin`.
        """
        mode = self.es_config.get('perturb_mode', 'off')
        if mode == 'off':
            return
        cfg = {
            'calib_path': self.es_config.get('calib_path', None),
            'rank': self.es_config.get('subspace_rank', 1),
            'density': self.es_config.get('insparse_density', 0.01),
            'swap_blocks': self.es_config.get('fura_swap_blocks', False),
            'iso_block_size': self.es_config.get('iso_block_size', 128),
            'iso_perm': self.es_config.get('iso_perm', True),
        }
        print(f"Installing ES perturbation mode '{mode}' with cfg={cfg} ...")
        infos = ray.get([
            e.collective_rpc.remote("init_es_state", args=(mode, cfg))
            for e in self.engines
        ])
        self.es_state_info = infos[0][0] if isinstance(infos[0], list) else infos[0]
        print(f"ES perturbation state: {self.es_state_info}")
    
    def fit(self):
        """
        Main ES training loop.
        
        Training process:
        1. For each iteration:
           a. Generate random seeds for perturbations
           b. For each batch of seeds:
              - Perturb weights on each engine
              - Generate completions
              - Restore weights
              - Compute rewards
           c. Normalize rewards and update weights
           d. Broadcast updated weights to all engines
           e. Optionally evaluate on held-out data
        """
        # Setup logging directory
        base_dir = self.config.trainer.get('default_local_dir', '/tmp/verl/es_checkpoints')
        logging_dir = os.path.join(
            base_dir,
            f"es_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(logging_dir, exist_ok=True)
        
        # Initialize logger
        logger = Tracking(
            project_name=self.config.trainer.get('project_name', 'es-training'),
            experiment_name=self.config.trainer.get('experiment_name', 'es-run'),
            default_backend=self.config.trainer.get('logger', ['tensorboard']),
            config=OmegaConf.to_container(self.config, resolve=True) if isinstance(self.config, DictConfig) else vars(self.config),
        )
        
        # Save config
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
        
        # ES hyperparameters
        sigma = self.es_config.sigma
        alpha = self.es_config.alpha
        population_size = self.es_config.population_size
        num_engines = self.es_config.num_engines
        
        # Allow trainer config to override ES config for iterations/eval
        trainer_total_epochs = self.config.trainer.get('total_epochs', None)
        trainer_test_freq = self.config.trainer.get('test_freq', None)
        
        num_iterations = trainer_total_epochs if trainer_total_epochs else self.es_config.num_iterations
        eval_interval = trainer_test_freq if trainer_test_freq else self.es_config.get('eval_interval', 25)
        global_seed = self.es_config.get('global_seed', 42)
        
        # Step 0 = the untouched base model, so the training curve starts from the
        # published base-model score instead of from "after one ES update".
        best_metric = float('-inf')
        ckpt_path = os.path.join(logging_dir, "es_coef_best.pt")
        save_best = bool(self.es_config.get('save_best_coef', False))
        if self.es_config.get('eval_before_train', True) and self.eval_data:
            base_metrics = self._evaluate_model(self.engines[0], self.eval_data, 0, logger)
            logger.log(data=base_metrics, step=0)
            best_metric = base_metrics.get('eval/accuracy', float('-inf'))

        # Training loop
        progress_bar = tqdm(range(num_iterations), desc="ES Training")
        
        for iteration in progress_bar:
            total_iter_start = time.time()
            
            # Generate deterministic seeds for this iteration
            loop_rng = np.random.default_rng(seed=global_seed + iteration)
            seeds = loop_rng.integers(0, 2**30, size=population_size, dtype=np.int64).tolist()
            
            seeds_perf: Dict[int, Dict[str, Any]] = {}
            
            # Static batching: process seeds in batches of num_engines
            for b in range(0, len(seeds), num_engines):
                batch_seeds = seeds[b:b + num_engines]
                
                # 1) Perturb weights on each engine
                ray.get([
                    self.engines[eng_idx].collective_rpc.remote(
                        "perturb_self_weights", 
                        args=(int(seed), sigma, False)
                    )
                    for eng_idx, seed in enumerate(batch_seeds)
                ])
                
                # 2) Generate completions
                gen_seed = global_seed + iteration
                handles = [
                    self._evaluate_with_engine(self.engines[eng_idx], prompts, seed=gen_seed)
                    for eng_idx, _ in enumerate(batch_seeds)
                ]
                outputs_per_engine = ray.get(handles)
                
                # 3) Restore weights
                ray.get([
                    self.engines[eng_idx].collective_rpc.remote(
                        "restore_self_weights", 
                        args=(int(seed), sigma)
                    )
                    for eng_idx, seed in enumerate(batch_seeds)
                ])
                
                # 4) Compute rewards
                for eng_idx, seed in enumerate(batch_seeds):
                    metrics = self._compute_metrics(outputs_per_engine[eng_idx], self.train_data)
                    seeds_perf[int(seed)] = metrics
                
                # Clean up GPU memory after each batch
                del outputs_per_engine
                gc.collect()
                torch.cuda.empty_cache()
            
            # Aggregate metrics
            all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
            mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
            std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
            min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
            max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0
            
            # Aggregate format and answer rewards
            all_avg_formats = [v.get("avg_format", 0.0) for v in seeds_perf.values()]
            all_avg_answers = [v.get("avg_answer", 0.0) for v in seeds_perf.values()]
            all_accuracies = [v.get("accuracy", 0.0) for v in seeds_perf.values()]
            mean_format = float(np.mean(all_avg_formats)) if all_avg_formats else 0.0
            mean_answer = float(np.mean(all_avg_answers)) if all_avg_answers else 0.0
            mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0
            
            # Normalize rewards
            for k in seeds_perf:
                seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            
            # Update weights on engine 0
            coeffs = [float(seeds_perf[seed]["norm_reward"]) for seed in seeds]
            ray.get(self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(seeds, coeffs, alpha, population_size)
            ))
            
            # Broadcast from engine 0 to all engines
            ray.get([
                e.collective_rpc.remote("broadcast_all_weights", args=(0,)) 
                for e in self.engines
            ])
            torch.cuda.synchronize()
            
            iter_time = time.time() - total_iter_start
            step = iteration + 1
            
            # Constraint health for the fixed-spectrum (ISO) modes: ||W||_F is an exact
            # invariant of the bi-orthogonal orbit, so any drift is pure fp32 round-off.
            extra_metrics = {}
            if self.es_config.get('perturb_mode', 'off') in ('iso', 'isobtt'):
                try:
                    got = ray.get(self.engines[0].collective_rpc.remote("es_get_metrics"))
                    extra_metrics = (got[0] if isinstance(got, list) else got) or {}
                except Exception as e:
                    print(f"[ES] es_get_metrics failed: {e}")
            
            # Log metrics
            train_metrics = {
                "train/reward_mean": mean_reward,
                "train/reward_std": std_reward,
                "train/reward_min": min_reward,
                "train/reward_max": max_reward,
                "train/format_reward": mean_format,
                "train/answer_reward": mean_answer,
                "train/accuracy": mean_accuracy,
                "train/iteration_time": iter_time,
                "training/global_step": step,
            }
            train_metrics.update(extra_metrics)
            
            logger.log(data=train_metrics, step=step)
            
            progress_bar.set_postfix({
                "reward": f"{mean_reward:.4f}",
                "acc": f"{mean_accuracy:.1f}%",
                "time": f"{iter_time:.2f}s"
            }, refresh=False)
            
            if self.es_config.get('verbose', False):
                print(f"Iteration {iteration}: mean_reward={mean_reward:.4f}, std={std_reward:.4f}, "
                      f"format={mean_format:.4f}, answer={mean_answer:.4f}, acc={mean_accuracy:.1f}%")
            
            # Evaluation
            if eval_interval > 0 and (step % eval_interval == 0 or iteration == num_iterations - 1):
                eval_metrics = self._evaluate_model(self.engines[0], self.eval_data, step, logger)
                logger.log(data=eval_metrics, step=step)
                acc = eval_metrics.get('eval/accuracy', float('-inf'))
                logger.log(data={"eval/best_accuracy": max(best_metric, acc)}, step=step)
                if acc > best_metric:
                    best_metric = acc
                    if save_best:
                        # Only the ES coefficients are saved (fp32).  For the
                        # structured modes that is 1-2% of the model; combined with the
                        # frozen base/basis it reconstructs the trained weights exactly.
                        ray.get(self.engines[0].collective_rpc.remote(
                            "es_save_coef", args=(ckpt_path,)))
                        print(f"[Ckpt] new best eval/accuracy={acc:.2f} -> {ckpt_path}")
            
            # Periodic memory cleanup at end of each iteration
            seeds_perf.clear()
            gc.collect()
            torch.cuda.empty_cache()
        
        progress_bar.close()
        if hasattr(logger, "finish"):
            logger.finish()
        else:  # verl's Tracking closes its backends in __del__
            for _b in getattr(logger, "logger", {}).values():
                if hasattr(_b, "finish"):
                    try:
                        _b.finish()
                    except Exception:
                        pass
        
        # Cleanup
        self._cleanup()
        
        print(f"Training completed. Results saved to {logging_dir}")
