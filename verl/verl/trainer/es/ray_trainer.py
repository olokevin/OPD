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


class TeacherKLScorer:
    """OPD fitness for sequence-level ES: the reverse KL between a PERTURBED
    student's own rollout and the teacher.

    Each ES rail generates its own trajectory y ~ pi_n under W + sigma*eps_n, so
    the perturbation propagates through the whole rollout (unlike es_token, whose
    rails read the clean row's KV and can only see the detached-history gradient
    -- docs/results/zo_opd.md 12.4/12.5). The fitness of rail n is

        KL_n = mean_t [ log pi_n(y_t) - log q(y_t) ],   y ~ pi_n

    a single-sample estimate of KL(pi_n || q) that is unbiased exactly because y
    is SAMPLED from pi_n (so es.temperature must be > 0). ES maximises reward, so
    the reward returned is -KL.

    log pi_n(y_t) comes free from generation (SamplingParams.logprobs=0); log q
    needs ONE teacher prefill per rollout, read via prompt_logprobs.
    """

    def __init__(self, teacher_engine, teacher_temperature=1.0, batch_size=16):
        self.engine = teacher_engine
        self.temp = float(teacher_temperature)
        self.batch = max(1, int(batch_size))

    def logq(self, fulls: List[List[int]], resp_lens: List[int]) -> List[List[float]]:
        """Teacher logprob of each response token. fulls[i] = prompt+response ids."""
        sp = SamplingParams(temperature=self.temp, max_tokens=1, prompt_logprobs=1)
        out: List[List[float]] = [None] * len(fulls)
        for s0 in range(0, len(fulls), self.batch):
            idxs = list(range(s0, min(s0 + self.batch, len(fulls))))
            reqs = [{"prompt_token_ids": list(fulls[i])} for i in idxs]
            outs = ray.get(self.engine.generate.remote(reqs, sp, use_tqdm=False))
            for o, i in zip(outs, idxs):
                T = int(resp_lens[i])
                if T <= 0:
                    out[i] = []
                    continue
                plp = o.prompt_logprobs[-T:]
                ids = fulls[i][-T:]
                out[i] = [plp[t][ids[t]].logprob for t in range(T)]
        return out

    @staticmethod
    def score_fixed(engine, fulls: List[List[int]], resp_lens: List[int],
                    batch: int = 16) -> List[List[float]]:
        """log pi(y_t) for a FIXED token sequence under whatever weights `engine`
        currently holds. Teacher-forced via prompt_logprobs, so no generation and
        no dependence on what this policy would have sampled."""
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        out: List[List[float]] = [None] * len(fulls)
        for s0 in range(0, len(fulls), batch):
            idxs = list(range(s0, min(s0 + batch, len(fulls))))
            reqs = [{"prompt_token_ids": list(fulls[i])} for i in idxs]
            outs = ray.get(engine.generate.remote(reqs, sp, use_tqdm=False))
            for o, i in zip(outs, idxs):
                T = int(resp_lens[i])
                if T <= 0:
                    out[i] = []
                    continue
                plp = o.prompt_logprobs[-T:]
                ids = fulls[i][-T:]
                out[i] = [plp[t][ids[t]].logprob for t in range(T)]
        return out

    @staticmethod
    def student_logp(output) -> List[float]:
        """Per-token logprob of the tokens the perturbed student actually sampled."""
        comp = output.outputs[0]
        lps = getattr(comp, "logprobs", None)
        if not lps:
            return []
        vals = []
        for tok, d in zip(comp.token_ids, lps):
            lp = d.get(tok)
            vals.append(float(lp.logprob) if lp is not None else 0.0)
        return vals


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
        # With the Ray distributed executor, Ray re-derives the worker's device
        # from the placement group, so we drop CUDA_VISIBLE_DEVICES to avoid a
        # conflicting pin. With the uni (in-process) executor there is no child
        # worker -- popping it sends vLLM to physical GPU0 regardless of the
        # CUDA_VISIBLE_DEVICES the launcher set, so KEEP the pin in that case.
        # (Same fix and same failure mode as NPNcclLLM.)
        if os.environ.get("ES_KEEP_CUDA_VISIBLE", "0") != "1":
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
        
        # Create placement groups.
        # engine_gpu_fraction < 1.0 leaves room in the bundle for a co-located
        # teacher engine (es.fitness=opd_kl): a student PG that reserves a whole
        # GPU makes the teacher's own PG unschedulable and pg.ready() hangs.
        eng_frac = float(self.es_config.get('engine_gpu_fraction', 1.0))
        pgs = [
            placement_group([{"GPU": eng_frac, "CPU": 0}], lifetime="detached")
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
        
        # vLLM's "ray" executor spawns a RayWorkerWrapper that requests a FULL GPU,
        # which cannot fit a fractional PG bundle. For tensor_parallel=1 use "uni"
        # (in-process worker): the engine actor itself owns the GPU slice granted
        # by its PG, so co-locating student + teacher on one card works.
        exec_backend = self.es_config.get('distributed_executor_backend', 'ray')
        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
                model=model_path,
                tensor_parallel_size=1,
                distributed_executor_backend=exec_backend,
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
    
    def _launch_teacher_engine(self, model_path: str):
        """ONE teacher vLLM engine, CO-LOCATED with the student engines.

        gpu_fraction < 1.0 lets its placement-group bundle share a card with a
        student engine, which is what a single-GPU OPD run needs.
        """
        precision = self.es_config.get('teacher_precision', None) or \
            self.es_config.get('precision', 'bfloat16')
        worker_ext = self.es_config.get(
            'worker_extension_cls',
            'verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension')
        gpu_frac = float(self.es_config.get('teacher_gpu_fraction', 0.01))
        pg = placement_group([{"GPU": gpu_frac, "CPU": 0}], lifetime="detached")
        ray.get(pg.ready())
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        # Uncapped, the teacher sizes its KV pool (and its profiling activation
        # peak) for the model's full 32k context -- unaffordable co-located.
        tmax = self.es_config.get('teacher_max_model_len', None)
        engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_path,
            tensor_parallel_size=1,
            distributed_executor_backend=self.es_config.get(
                'distributed_executor_backend', 'ray'),
            worker_extension_cls=worker_ext,
            dtype=precision,
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=(self.es_config.get('teacher_gpu_memory_utilization')
                                    or self.es_config.get('gpu_memory_utilization', 0.9)),
            **({"max_model_len": int(tmax)} if tmax else {}),
        )
        self.teacher_engine = engine
        self.teacher_placement_group = pg
        return engine

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
        for llm in [getattr(self, "teacher_engine", None)]:
            if llm is not None:
                try:
                    ray.kill(llm)
                except Exception:
                    pass
        tpg = getattr(self, "teacher_placement_group", None)
        if tpg is not None:
            try:
                remove_placement_group(tpg)
            except Exception:
                pass
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
    
    def _evaluate_with_engine(self, engine, prompts, seed: int, want_logprobs: bool = False):
        """Evaluate prompts using a specific engine.

        want_logprobs=True adds `logprobs=0`, which makes vLLM return the logprob
        of the token it actually sampled -- the `log pi_n(y_t)` half of the OPD
        reverse-KL fitness, for free.
        """
        sampling_params = SamplingParams(
            temperature=self.es_config.get('temperature', 0.0),
            top_p=float(self.es_config.get('top_p', 1.0)),
            seed=seed,
            max_tokens=self.es_config.get('max_tokens', 1024),
            **({"logprobs": 0} if want_logprobs else {})
        )
        return engine.generate.remote(prompts, sampling_params, use_tqdm=False)

    def _opd_metrics(self, outputs, task_datas, with_reward: bool = False) -> Dict[str, Any]:
        """OPD fitness: reward = -KL(pi_n || q), averaged over the batch.

        The task reward is still computed alongside, purely as a diagnostic --
        it is NOT what drives the ES update in this mode.
        """
        fulls, lens, stu = [], [], []
        for o in outputs:
            resp = list(o.outputs[0].token_ids)
            lp = TeacherKLScorer.student_logp(o)
            n = min(len(resp), len(lp))
            fulls.append(list(o.prompt_token_ids) + resp[:n])
            lens.append(n)
            stu.append(lp[:n])
        logqs = self.kl_scorer.logq(fulls, lens)

        # Aggregation over the response is NOT a free choice: ES compares rails, so any
        # length-dependent term in the fitness becomes selection pressure.
        #   "mean" divides by length, so a rail can lower its score by appending easy,
        #   low-KL filler that dilutes the high-KL reasoning tokens. Measured: response
        #   length +21 tok/iter (t=+14.9), corr(length, KL) = -0.88, MATH-500 74.8 -> 68.2.
        #   "sum" is the true sequence-level reverse KL (the log-ratio of the whole
        #   trajectory, EOS included), so padding costs what it is worth and stopping
        #   early is rewarded only when the teacher agrees. Averaged over prompts it
        #   estimates E_x[KL(pi(.|x) || q(.|x))].
        agg = self.es_config.get('opd_kl_agg', 'sum')
        kls = []
        for sp_, lq in zip(stu, logqs):
            if not sp_ or not lq:
                continue
            m = min(len(sp_), len(lq))
            d = [sp_[t] - lq[t] for t in range(m)]
            kls.append(float(np.sum(d) if agg == 'sum' else np.mean(d)))
        kl = float(np.mean(kls)) if kls else 0.0

        # The task reward is a DIAGNOSTIC here, not the fitness, and its sympy
        # grader costs ~50 ms/prompt -- at N=30 rails x 32 prompts that is ~48 s
        # an iteration. Grade only when asked (default: only the first rail, via
        # `with_reward`), and report zeros otherwise.
        if with_reward:
            base = self._compute_metrics(outputs, task_datas)
        else:
            base = {"rewards": [], "avg_reward": 0.0, "avg_format": 0.0,
                    "avg_answer": 0.0, "accuracy": float("nan")}
        base["kl"] = kl
        base["avg_reward"] = -kl          # ES maximises reward: fitness = -KL
        base["resp_len"] = float(np.mean(lens)) if lens else 0.0
        return base
    
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
        if self.es_config.get('fitness', 'reward') == 'opd_kl':
            tpath = self.es_config.get('teacher_model_path', None)
            if not tpath:
                raise ValueError("es.fitness=opd_kl requires es.teacher_model_path")
            print(f"Launching teacher engine ({tpath}) for OPD reverse-KL fitness...")
            self._launch_teacher_engine(tpath)
            self.kl_scorer = TeacherKLScorer(
                self.teacher_engine,
                self.es_config.get('teacher_temperature', 1.0),
                int(self.es_config.get('teacher_batch_size', 16)))
            if float(self.es_config.get('temperature', 0.0)) <= 0.0:
                raise ValueError(
                    "es.fitness=opd_kl needs SAMPLED rollouts (es.temperature > 0): "
                    "mean_t[log pi_n(y_t) - log q(y_t)] estimates KL(pi_n||q) only "
                    "when y ~ pi_n. Set es.temperature=1.0.")
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
        
        global_seed = self.es_config.get('global_seed', 42)

        # Training batch policy.
        #
        # The reference implementation (VsonicV/es-at-scale, train.py) wraps the training
        # set in a `DataLoader(..., batch_size=args.batch_size, shuffle=True)` and draws a
        # FRESH batch every ES iteration -- for math, batch_size=1024 out of an 8.5k pool.
        # Every member of the population still sees the *same* batch within an iteration
        # (that is what makes the z-scores comparable); the batch changes between
        # iterations. Holding one fixed batch instead makes ES overfit it, which is exactly
        # what we measured (train acc 77.7 vs held-out 71.5 at step 150).
        #
        # es.train_batch_size = 0 keeps the old fixed-batch behaviour.
        train_batch_size = int(self.es_config.get('train_batch_size', 0) or 0)
        resample = train_batch_size > 0 and train_batch_size < len(self.train_data)
        batch_rng = np.random.default_rng(seed=global_seed)
        epoch_order = []

        def _draw_batch():
            """Sampling-without-replacement over shuffled epochs, like DataLoader(shuffle=True)."""
            nonlocal epoch_order
            if not resample:
                return list(self.train_data)
            if len(epoch_order) < train_batch_size:
                epoch_order = list(batch_rng.permutation(len(self.train_data)))
            idx = [epoch_order.pop() for _ in range(train_batch_size)]
            return [self.train_data[i] for i in idx]

        def _to_prompts(batch):
            if self.prompt_processor:
                return [self.prompt_processor(d, self.tokenizer) for d in batch]
            return [d.get("prompt", d.get("context")) for d in batch]

        train_batch = _draw_batch()
        prompts = _to_prompts(train_batch)
        if resample:
            print(f"Training batch: {train_batch_size} problems resampled per iteration "
                  f"from a pool of {len(self.train_data)}")
        else:
            print(f"Training batch: fixed set of {len(self.train_data)} problems")
        
        opd = self.es_config.get('fitness', 'reward') == 'opd_kl'
        fixed_traj = opd and bool(self.es_config.get('opd_fixed_traj', True))

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

            if resample and iteration > 0:
                train_batch = _draw_batch()
                prompts = _to_prompts(train_batch)

            # Generate deterministic seeds for this iteration
            loop_rng = np.random.default_rng(seed=global_seed + iteration)
            seeds = loop_rng.integers(0, 2**30, size=population_size, dtype=np.int64).tolist()
            
            seeds_perf: Dict[int, Dict[str, Any]] = {}

            # ---- FIXED-TRAJECTORY OPD fitness ------------------------------------
            # Both length-sensitive aggregations are gameable (zo_opd.md 13.5/13.6):
            # `mean` rewards padding, `sum` rewards truncation, because the per-token
            # log-ratio is positive so total KL tracks length. Remove length from the
            # comparison entirely: generate ONE rollout per prompt from the CLEAN
            # policy, then teacher-force that same sequence through every rail. All
            # rails then score the identical tokens, so no length strategy can win,
            # and the perturbation still acts through the whole sequence.
            # Bonus: 1 generation + N prefills instead of N generations, and log q is
            # computed ONCE per iteration instead of once per rail.
            if opd and fixed_traj:
                gen_seed = global_seed + iteration
                cl = ray.get(self._evaluate_with_engine(
                    self.engines[0], prompts, seed=gen_seed))
                fulls, lens = [], []
                for o in cl:
                    resp = list(o.outputs[0].token_ids)
                    fulls.append(list(o.prompt_token_ids) + resp)
                    lens.append(len(resp))
                logqs = self.kl_scorer.logq(fulls, lens)
                clean_len = float(np.mean(lens)) if lens else 0.0
                del cl

                for b in range(0, len(seeds), num_engines):
                    batch_seeds = seeds[b:b + num_engines]
                    ray.get([self.engines[i].collective_rpc.remote(
                        "perturb_self_weights", args=(int(sd), sigma, False))
                        for i, sd in enumerate(batch_seeds)])
                    scored = [TeacherKLScorer.score_fixed(
                        self.engines[i], fulls, lens,
                        int(self.es_config.get('teacher_batch_size', 16)))
                        for i, _ in enumerate(batch_seeds)]
                    ray.get([self.engines[i].collective_rpc.remote(
                        "restore_self_weights", args=(int(sd), sigma))
                        for i, sd in enumerate(batch_seeds)])
                    for i, sd in enumerate(batch_seeds):
                        # Teacher-probability-weighted student log-likelihood.
                        #
                        # NOT sum_t[log pi_n - log q]: on a trajectory sampled from
                        # pi_0 rather than pi_n that scalar has expectation
                        #   KL(pi_0||q) - KL(pi_0||pi_n),
                        # whose first term is constant across rails, so minimising it
                        # MAXIMISES KL(pi_0||pi_n) -- it selects the rails furthest
                        # from the current policy. Measured: the score fell monotonically
                        # with sigma (306 -> 224 -> -133 -> -5551 -> -17643), i.e. a
                        # bigger perturbation always won (zo_opd.md 13.7).
                        #
                        # Reverse KL needs samples from pi_n, which a fixed trajectory
                        # cannot supply. The forward/distillation form IS well defined on
                        # another policy's samples: weight each token by the teacher's
                        # probability of it and push up the student's log-prob there.
                        # This is the repo's reward_weight_mode=teacher_p, it is bounded
                        # (weights in [0,1]), and perturbation can only lower it.
                        tot, ntok = [], 0
                        for lp, lq in zip(scored[i], logqs):
                            if not lp or not lq:
                                continue
                            m = min(len(lp), len(lq))
                            tot.append(float(np.sum(
                                [np.exp(lq[t]) * lp[t] for t in range(m)])))
                            ntok += m
                        fit = float(np.mean(tot)) if tot else 0.0
                        seeds_perf[int(sd)] = {
                            "avg_reward": fit, "kl": -fit,
                            "kl_per_tok": -fit / max(1.0, ntok / max(1, len(tot))),
                            "resp_len": clean_len, "accuracy": float("nan"),
                            "avg_format": 0.0, "avg_answer": 0.0, "rewards": [],
                        }
                    gc.collect(); torch.cuda.empty_cache()

            # Static batching: process seeds in batches of num_engines
            for b in (range(0, len(seeds), num_engines)
                      if not (opd and fixed_traj) else []):
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
                    self._evaluate_with_engine(self.engines[eng_idx], prompts,
                                               seed=gen_seed, want_logprobs=opd)
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
                
                # 4) Compute rewards (OPD reverse-KL, or the task reward)
                for eng_idx, seed in enumerate(batch_seeds):
                    metrics = (self._opd_metrics(outputs_per_engine[eng_idx], train_batch,
                                                 with_reward=(b == 0 and eng_idx == 0))
                               if opd else
                               self._compute_metrics(outputs_per_engine[eng_idx], train_batch))
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
            all_accuracies = [v.get("accuracy", 0.0) for v in seeds_perf.values()
                              if not np.isnan(v.get("accuracy", 0.0))]
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
            if opd:
                _kls = [v["kl"] for v in seeds_perf.values() if "kl" in v]
                if _kls:
                    train_metrics["train/kl_mean"] = float(np.mean(_kls))
                    train_metrics["train/kl_min"] = float(np.min(_kls))
                    train_metrics["train/kl_spread"] = float(np.std(_kls))
                _rl = [v["resp_len"] for v in seeds_perf.values() if "resp_len" in v]
                if _rl:
                    train_metrics["train/resp_len"] = float(np.mean(_rl))
                _pt = [v["kl_per_tok"] for v in seeds_perf.values() if "kl_per_tok" in v]
                if _pt:
                    train_metrics["train/kl_per_tok"] = float(np.mean(_pt))
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
