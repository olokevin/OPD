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
                                                  sampled_token_losses,
                                                  topk_rail_losses)
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


class TopKTeacherHF:
    """Eager HF teacher for loss_impl=topk: log q at ARBITRARY per-token id
    sets (vLLM prompt_logprobs can only return the teacher's own top-k, not
    the student's clean top-K). One bf16 forward per rollout on the training
    GPU; lm_head + log_softmax run in position chunks so the [chunk, V] fp32
    logits stay bounded (~300 MB at chunk=512)."""

    def __init__(self, model_path, temperature, device="cuda", chunk=512):
        from transformers import AutoModelForCausalLM
        self.temp = float(temperature)
        self.chunk = int(chunk)
        self.device = device
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2")
        except Exception as e:
            print(f"[topk-teacher] flash_attention_2 unavailable ({e}); sdpa")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def logq_topk(self, fulls, resp_lens, topk_ids):
        """fulls[i] = prompt+response ids; topk_ids[i] = [T_i, K] int ids.
        Returns [tensor [T_i, K] fp32] of teacher logprobs at those ids."""
        out = []
        for full, T, kids in zip(fulls, resp_lens, topk_ids):
            T = int(T)
            if T == 0:
                out.append(torch.zeros(0, 0))
                continue
            ids = torch.tensor([list(full)], dtype=torch.long,
                               device=self.device)
            h = self.model.model(input_ids=ids).last_hidden_state[0]
            hs = h[-T - 1:-1]                        # predicts fulls[-T:]
            kid = kids.to(self.device).long()        # [T, K]
            lq = torch.empty(T, kid.shape[1], dtype=torch.float32)
            for s in range(0, T, self.chunk):
                e = min(T, s + self.chunk)
                lg = self.model.lm_head(hs[s:e]).float()
                if self.temp != 1.0:
                    lg = lg / self.temp
                lsm = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
                lq[s:e] = lsm.gather(1, kid[s:e]).cpu()
            out.append(lq)
        return out

    def logq_wave(self, fulls, resp_lens):
        """Sampled-token logq (K=1 gather) -- keeps the held-out probe's
        clean reverse-KL metric identical to the SampledTokenTeacher arms."""
        kids = [torch.tensor(f[-int(T):], dtype=torch.long).view(-1, 1)
                if int(T) > 0 else torch.zeros(0, 1, dtype=torch.long)
                for f, T in zip(fulls, resp_lens)]
        return [lq[:, 0] for lq in self.logq_topk(fulls, resp_lens, kids)]


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
        self.rail_mode = str(self.es.get("rail_mode", "token"))
        noise_rank = self.es.get("noise_rank", 1)
        noise_rank = "full" if str(noise_rank) == "full" else int(noise_rank)
        matched_per_engine = ray.get([
            e.collective_rpc.remote(
                "install_es_layers",
                args=(list(self.es.perturb_rules), int(self.es.n_sample),
                      int(self.es.global_seed), self.rail_mode, noise_rank))
            for e in self.engines
        ])
        self.matched = list(matched_per_engine[0][0])
        print(f"Matched {len(self.matched)} layers "
              f"(first: {self.matched[:2]} ... last: {self.matched[-1:]})")
        teacher_path = self.es.teacher_model_path
        if not teacher_path:
            raise ValueError("es_token requires es_token.teacher_model_path")
        if str(self.es.get("loss_impl", "sampled")) == "topk":
            # No vLLM teacher engine: top-K scoring needs logprobs at
            # arbitrary ids, which prompt_logprobs cannot return. The HF
            # teacher (~3.5 GB bf16) rides on the freed engine fraction.
            print(f"Loading HF top-K teacher ({teacher_path})...")
            self.teacher = TopKTeacherHF(
                teacher_path, self.es.teacher_temperature,
                chunk=int(self.es.get("teacher_topk_chunk", 512)))
        else:
            print(f"Launching teacher engine ({teacher_path})...")
            self._launch_teacher_engine(teacher_path)
            self.teacher = SampledTokenTeacher(
                self.teacher_engine, self.es.teacher_temperature,
                self.es.get("teacher_batch_size", 16))
        print("Workers initialized successfully.")

    # ---------------------------------------------------------- checkpoint ---
    def _save_hf_checkpoint(self, step: int, base_dir: str, keep_last: int = 2):
        """Write a plain HF checkpoint of the CURRENT perturbed weights.

        The es trainer keeps the model inside the vLLM engine, whose decoder
        linears are FUSED (`qkv_proj`, `gate_up_proj`) while HF stores them split
        (`q_proj`/`k_proj`/`v_proj`, `gate_proj`/`up_proj`). So: pull the perturbed
        tensors off the engine, split them back, drop them into a CPU copy of the
        base model, and `save_pretrained`. Everything the trainer never perturbs
        (embeddings, norms, lm_head) comes from that base copy unchanged.
        """
        import shutil
        from transformers import AutoModelForCausalLM

        if getattr(self, "_ckpt_model", None) is None:
            self._ckpt_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.path, torch_dtype=torch.bfloat16, device_map="cpu")
            self._ckpt_model.eval()
        model = self._ckpt_model
        cfg = model.config
        n_q = cfg.num_attention_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        n_kv = cfg.num_key_value_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        inter = cfg.intermediate_size

        weights = ray.get(self.engines[0].collective_rpc.remote("es_export_weights"))[0]
        sd = dict(model.state_dict())
        missing = []
        for ln, t in weights.items():
            if ln.endswith("self_attn.qkv_proj"):
                base = ln[: -len("qkv_proj")]
                q, k, v = torch.split(t, [n_q, n_kv, n_kv], dim=0)
                parts = {base + "q_proj.weight": q, base + "k_proj.weight": k,
                         base + "v_proj.weight": v}
            elif ln.endswith("mlp.gate_up_proj"):
                base = ln[: -len("gate_up_proj")]
                g, u = torch.split(t, [inter, inter], dim=0)
                parts = {base + "gate_proj.weight": g, base + "up_proj.weight": u}
            else:
                parts = {ln + ".weight": t}
            for k2, v2 in parts.items():
                if k2 in sd:
                    sd[k2].copy_(v2.to(sd[k2].dtype))
                else:
                    missing.append(k2)
        if missing:
            print(f"[es ckpt] WARNING {len(missing)} unmatched keys, first: {missing[:3]}")

        out = os.path.join(base_dir, f"step_{step}")
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)
        print(f"[es ckpt] saved step {step} -> {out}")

        # keep only the newest `keep_last` step_* dirs (disk is tight)
        steps = sorted(
            (int(d.split("_")[1]) for d in os.listdir(base_dir)
             if d.startswith("step_") and d.split("_")[1].isdigit()))
        for old in steps[:-keep_last]:
            shutil.rmtree(os.path.join(base_dir, f"step_{old}"), ignore_errors=True)
        return out

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

        # Drop overlong prompts, exactly as BP does via data.filter_overlong_prompts.
        # Two things break without this, both only on the rare long prompt:
        #   * the teacher engine is capped at teacher_max_model_len and refuses a
        #     prompt+response longer than it ("decoder prompt ... is longer than
        #     the maximum model length"), and
        #   * the packed decode reserves (longest prompt + max_tokens) of scratch
        #     KV per slot, so one long prompt in a wave can exceed the pool.
        # On DAPO-Math-17k this drops 9 of 17,917 rows (0.05%).
        max_pl = cfg.get("max_prompt_length", None)
        if max_pl:
            max_pl = int(max_pl)
            _len = lambda p: len(p["prompt_token_ids"] if isinstance(p, dict) else p)
            kept = [p for p in prompts if _len(p) <= max_pl]
            if len(kept) != len(prompts):
                print(f"[es data] dropped {len(prompts) - len(kept)}/{len(prompts)} "
                      f"prompts longer than {max_pl} tokens")
            prompts = kept

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
        save_freq = int(self.config.trainer.get("save_freq", 0) or 0)

        sp = SamplingParams(temperature=cfg.get("temperature", 0.0),
                            max_tokens=int(cfg.max_tokens))
        # The probe must rank learning rates, so it has to be quieter than the
        # effect it is measuring. Scoring SAMPLED rollouts at T=1.0 gives it a
        # +-8% floor (results/zo_opd.md 9.1) that swamps everything short of
        # divergence; a GREEDY clean trajectory on the same fixed prompts is
        # deterministic, so run-to-run spread reflects the weights alone.
        probe_sp = SamplingParams(temperature=0.0, max_tokens=int(cfg.max_tokens))
        es_cfg = dict(
            n_sample=int(cfg.n_sample),
            max_tokens=int(cfg.max_tokens),
            global_seed=int(cfg.global_seed),
            sigma=float(cfg.sigma),
            sigma_mode=cfg.get("sigma_mode", "absolute"),
            sample_method=cfg.sample_method,
            b_pack_buckets=list(cfg.get("b_pack_buckets", [2, 4])),
            token_agg=cfg.get("token_agg", "sum"),
            fp32_master=bool(cfg.get("fp32_master", True)),
            # Default 1.0 reproduces the pre-2026-08-28 decode exactly; set to
            # 0.95 to match BP's rollout and every eval (results/zo_opd.md 12.6).
            top_p=float(cfg.get("top_p", 1.0)),
            # rail-aware kernels (es_profile_results.md): rows = shipping path
            attn_impl=str(cfg.get("attn_impl", "rows")),
            lm_head_impl=str(cfg.get("lm_head_impl", "full")),
            rail_impl=str(cfg.get("rail_impl", "kernel")),
            step_impl=str(cfg.get("step_impl", "eager")),
            # loss_impl=topk: the worker also returns the clean top-K ids and
            # per-rail logprobs at them (run_es_decode_packed).
            topk_k=(int(cfg.get("topk_k", 16))
                    if str(cfg.get("loss_impl", "sampled")) == "topk" else 0),
            # es-decode (rail_mode=seq): one held perturbation per rail
            rail_mode=str(cfg.get("rail_mode", "token")),
            noise_rank=str(cfg.get("noise_rank", 1)),
        )
        rail_mode = str(cfg.get("rail_mode", "token"))
        es_alpha = float(cfg.get("es_alpha", 1.25e-3))
        es_antithetic = bool(cfg.get("es_antithetic", True))
        es_normalize = str(cfg.get("es_normalize", "zscore"))
        if rail_mode == "seq":
            assert str(cfg.get("loss_impl", "sampled")) == "sampled", "es-decode uses the k1 fitness"
            assert cfg.sample_method == "bernoulli", "es-decode noise is Rademacher (packed bits)"
            if es_antithetic:
                assert int(cfg.n_sample) % 2 == 0, "antithetic es-decode needs an even n_sample"
        ckpt_keep_last = int(cfg.get("ckpt_keep_last", 2))
        # es-decode: after the update, re-score the FIRST wave's rollouts (first
        # es_post_gain_tokens tokens, teacher-forced, clean rail only) to log
        # F(W_new) - F(W_0) on the batch -- es_update.py's post_update_gain.
        es_post_gain_tokens = int(cfg.get("es_post_gain_tokens", 0) or 0)
        # A bare SamplingParams leaves _all_stop_token_ids empty, so _np_is_eos
        # falls back to config.json's single eos_token_id and misses 151643
        # (<|endoftext|>, declared only in generation_config.json). Opt-in so the
        # LR sweep keeps the old rollout-length semantics.
        if cfg.get("use_generation_config_eos", False):
            eos = set()
            for src in (getattr(self.tokenizer, "eos_token_id", None),):
                if isinstance(src, int):
                    eos.add(src)
            try:
                from transformers import GenerationConfig
                gc_ = GenerationConfig.from_pretrained(self.config.model.path)
                e = gc_.eos_token_id
                eos |= set(e) if isinstance(e, (list, tuple)) else {e}
            except Exception as _e:
                print(f"[es] generation_config eos lookup failed: {_e}")
            eos = {int(x) for x in eos if x is not None}
            if eos:
                sp.stop_token_ids = sorted(eos)
                sp._all_stop_token_ids = set(eos)
                probe_sp.stop_token_ids = sorted(eos)
                probe_sp._all_stop_token_ids = set(eos)
                print(f"[es] stop_token_ids = {sorted(eos)}")
        batch_size = int(cfg.get("batch_size", 1))
        pack_width = int(cfg.get("pack_width", 4))
        n_rails = int(cfg.n_sample)
        loss_impl = str(cfg.get("loss_impl", "sampled"))
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

            # ---- es-decode: draw this step's held rail noise ------------- #
            if rail_mode == "seq":
                rng = np.random.default_rng(int(cfg.global_seed) + step)
                n_eff = max(1, n_rails // 2) if es_antithetic else n_rails
                seq_seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=n_eff)]
                ray.get([e.collective_rpc.remote("es_seq_draw", args=(seq_seeds, es_antithetic))
                         for e in self.engines])

            # ---- Phase 1: graphed packed rail decode --------------------- #
            t_dec0 = time.time()
            roll_pids, roll_rids, roll_toks, roll_payload = [], [], [], []
            roll_tp, roll_tids = [], []
            first_wave_n = None
            for wi, (wave_pids, wave_rids, real_count) in enumerate(waves):
                if wi == 1:
                    first_wave_n = len(roll_toks)
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
                    if loss_impl == "topk":
                        roll_tp.append(out["topk_payload"][i])
                        roll_tids.append(out["topk_ids"][i])
            decode_s = time.time() - t_dec0
            if first_wave_n is None:
                first_wave_n = len(roll_toks)

            if not roll_toks:
                logger.log(data={"train/step_time": time.time() - t0,
                                 "training/global_step": step}, step=step)
                continue

            # ---- Phase 2: ONE teacher prefill per rollout ---------------- #
            t_tch0 = time.time()
            fulls = [list(p) + t for p, t in zip(roll_pids, roll_toks)]
            lens = [len(t) for t in roll_toks]
            if loss_impl == "topk":
                logqs = self.teacher.logq_topk(fulls, lens, roll_tids)
            else:
                logqs = self.teacher.logq_wave(fulls, lens)
            teacher_s = time.time() - t_tch0

            # ---- es-decode: k1 fitness per rail -> OpenAI-ES step --------- #
            if rail_mode == "seq":
                t_asm0 = time.time()
                num = torch.zeros(n_rails, dtype=torch.float64)
                den = 0
                clean_means = []
                for payload, logq in zip(roll_payload, logqs):
                    lp = payload.double()                      # [T, 1+N]
                    lq = logq.double()
                    A = lq - lp[:, 0]                          # k1 advantage (frozen)
                    dlp = lp[:, 1:] - lp[:, :1]                # delta log pi_n(y_t)
                    num += (A[:, None] * dlp).sum(0)
                    den += lp.shape[0]
                    clean_means.append(float((lp[:, 0] - lq).mean()))
                F = (num / max(den, 1)).numpy()                # [N] F(W+sigma eps_n) - F(W)
                if es_antithetic:
                    fp_, fm_ = F[0::2], F[1::2]
                    d = 0.5 * (fp_ - fm_)
                    n_eff = len(d)
                    scale = float(np.sqrt(np.mean(d ** 2))) if n_eff > 1 else max(abs(float(d[0])), 1e-12)
                else:
                    fp_, fm_ = F, np.full_like(F, float(F.mean()))
                    d = F - F.mean()
                    n_eff = len(d)
                    scale = float(d.std()) + 1e-12
                if es_normalize == "zscore":
                    coef_eff = (es_alpha / n_eff) * d / (scale + 1e-12)
                elif es_normalize == "raw":
                    coef_eff = es_alpha * d / (n_eff * float(cfg.sigma))
                else:
                    raise ValueError(f"unknown es_normalize={es_normalize!r}")
                if es_antithetic:   # rail 2i = +eps_i, rail 2i+1 = -eps_i
                    coeffs = np.zeros(n_rails)
                    coeffs[0::2] = coef_eff
                else:
                    coeffs = coef_eff
                _res = ray.get(self.engines[0].collective_rpc.remote(
                    "es_seq_apply", args=([float(c) for c in coeffs], es_cfg)))[0]
                for ln in self.matched:
                    ray.get([e.collective_rpc.remote("broadcast_layer_weights", args=(ln, 0))
                             for e in self.engines])
                assemble_s = time.time() - t_asm0
                post_gain = float("nan")
                if es_post_gain_tokens > 0 and first_wave_n > 0:
                    t_pg0 = time.time()
                    npg = first_wave_n
                    force = [list(roll_toks[j][:es_post_gain_tokens]) for j in range(npg)]
                    cfg_pg = dict(es_cfg, n_sample=0, max_tokens=es_post_gain_tokens,
                                  force_tokens=force, b_pack_buckets=[npg])
                    out_pg = ray.get(self.engines[0].collective_rpc.remote(
                        "run_es_decode_packed",
                        args=(roll_pids[:npg], sp, cfg_pg, roll_rids[:npg], use_graph)))[0]
                    num_pg, den_pg = 0.0, 0
                    for j in range(npg):
                        Tj = min(len(out_pg["clean_tokens"][j]), len(force[j]))
                        if Tj == 0:
                            continue
                        lp_new = out_pg["payload"][j][:Tj, 0].double()
                        lp0 = roll_payload[j][:Tj, 0].double()
                        A = logqs[j][:Tj].double() - lp0
                        num_pg += float((A * (lp_new - lp0)).sum())
                        den_pg += Tj
                    post_gain = num_pg / max(den_pg, 1)
                    assemble_s += time.time() - t_pg0
                upd_rms = float(np.sqrt(np.sum(coef_eff ** 2)))
                rms_w = float(_res["rms_w"])
                self._es_cum_sq = getattr(self, "_es_cum_sq", 0.0) + upd_rms ** 2
                step_time = time.time() - t0
                metrics = {
                    "train/step_time": step_time,
                    "train/decode_s": decode_s,
                    "train/teacher_s": teacher_s,
                    "train/assemble_s": assemble_s,
                    "train/n_token_records": int(den),
                    "train/L_clean_mean": float(np.mean(clean_means)),
                    "train/dW_norm_max": float(max(_res["norms"].values())),
                    "train/dW_norm_mean": float(np.mean(list(_res["norms"].values()))),
                    "train/update_footprint": upd_rms / rms_w,
                    "es/post_update_gain": post_gain,      # F(W_new)-F(W_0), first wave, >0 = ascent
                    "es/cum_footprint": float(np.sqrt(self._es_cum_sq)) / rms_w,
                    "es/fitness_plus_mean": float(fp_.mean()),
                    "es/fitness_minus_mean": float(fm_.mean()),
                    "es/d_mean": float(d.mean()),
                    "es/d_std": scale,
                    "es/d_snr": float(abs(d.mean()) / (scale + 1e-12)),
                    "es/update_rms": upd_rms,
                    "es/update_rms_measured": float(_res["update_rms_measured"]),
                    "es/update_footprint": upd_rms / rms_w,
                    "es/probe_footprint": float(cfg.sigma) / rms_w,
                    "es/n_rails": n_rails,
                    "es/sigma": float(cfg.sigma),
                    "es/alpha": es_alpha,
                    "training/global_step": step,
                }
                logger.log(data=metrics, step=step)
                progress.set_postfix({
                    "fp": f"{metrics['train/update_footprint']:.2e}",
                    "cum": f"{metrics['es/cum_footprint']:.3f}",
                    "snr": f"{metrics['es/d_snr']:.2f}",
                    "L_clean": f"{metrics['train/L_clean_mean']:.3f}",
                    "dec": f"{decode_s:.1f}s", "tch": f"{teacher_s:.1f}s", "asm": f"{assemble_s:.1f}s",
                }, refresh=False)
                if save_freq and (step > 0 and step % save_freq == 0
                                  or step == num_iterations - 1):
                    try:
                        self._save_hf_checkpoint(step, logging_dir, keep_last=ckpt_keep_last)
                    except Exception as e:
                        print(f"[es ckpt] save failed at step {step}: {e}")
                if eval_interval and (step % eval_interval == 0
                                      or step == num_iterations - 1):
                    eval_metrics = self._evaluate_model(
                        self.engines[0], self.eval_data, step, logger)
                    if eval_metrics:
                        logger.log(data=eval_metrics, step=step)
                    hk = self._heldout_clean_loss(heldout_pids, probe_sp, es_cfg)
                    if hk is not None:
                        logger.log(data={"eval/heldout_clean_loss": hk}, step=step)
                        print(f"[Probe @ step {step}] heldout_clean_loss={hk:.4f} "
                              f"(fixed {len(heldout_pids)} prompts; lower=better)")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # ---- Phase 3: losses -> scales -> assemble+apply ------------- #
            t_asm0 = time.time()
            rec_rids: List[int] = []
            rec_t: List[int] = []
            rec_scales: List[torch.Tensor] = []
            clean_means: List[float] = []
            for ri, (rid, payload, logq) in enumerate(
                    zip(roll_rids, roll_payload, logqs)):
                if loss_impl == "topk":
                    if ES_DEBUG and ri == 0:
                        # gate: where the sampled token IS in the K set, the
                        # clean rail's K-gather must equal payload col 0.
                        _hit = (roll_tids[ri].long()
                                == torch.tensor(roll_toks[ri])[:, None])
                        if _hit.any():
                            _err = (roll_tp[ri][:, 0, :][_hit]
                                    - payload[:, 0][_hit.any(1)]).abs().max()
                            print(f"[esdbg topk] clean-token K-gather "
                                  f"max|d|={float(_err):.3e}", flush=True)
                    losses, clean = topk_rail_losses(roll_tp[ri], logq)
                else:
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
            _res = ray.get(self.engines[0].collective_rpc.remote(
                "es_assemble_and_apply",
                args=(rec_rids, rec_t, scales, es_cfg, float(cfg.lr),
                      cfg.get("update_clip"),
                      int(cfg.get("assemble_chunk", 1024)))))[0]
            dws = _res["norms"]
            foots, dwcos = _res["footprint"], _res["dw_cos_prev"]
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
            # The two numbers that decide whether the run can learn at all.
            # footprint: per-step RMS(lr*dW)/RMS(W). The ES arms in this repo
            #   that learn sit at 1.6e-2..5e-2; the 200-step flat run sat at
            #   1.4e-4 (docs/results/zo_opd.md 12).
            # dw_cos_prev: coherent fraction of the estimate. ~0 = random walk.
            if foots:
                metrics["train/update_footprint"] = float(np.mean(list(foots.values())))
            if dwcos:
                metrics["train/dW_cos_prev_mean"] = float(np.mean(list(dwcos.values())))
            if w_deltas:
                metrics["train/weight_delta_mean"] = float(
                    np.mean(list(w_deltas.values())))
                metrics["train/weight_sync_ok"] = (
                    1.0 if all(w_sync_ok.values()) else 0.0)
            logger.log(data=metrics, step=step)
            progress.set_postfix({
                "fp": f"{metrics.get('train/update_footprint', float('nan')):.2e}",
                "cos": f"{metrics.get('train/dW_cos_prev_mean', float('nan')):+.4f}",
                "L_clean": f"{metrics['train/L_clean_mean']:.3f}",
                "dec": f"{decode_s:.1f}s",
                "tch": f"{teacher_s:.1f}s",
                "asm": f"{assemble_s:.1f}s",
            }, refresh=False)

            if save_freq and (step > 0 and step % save_freq == 0
                              or step == num_iterations - 1):
                try:
                    self._save_hf_checkpoint(step, logging_dir, keep_last=ckpt_keep_last)
                except Exception as e:
                    print(f"[es ckpt] save failed at step {step}: {e}")

            if eval_interval and (step % eval_interval == 0
                                  or step == num_iterations - 1):
                eval_metrics = self._evaluate_model(
                    self.engines[0], self.eval_data, step, logger)
                if eval_metrics:
                    logger.log(data=eval_metrics, step=step)
                hk = self._heldout_clean_loss(heldout_pids, probe_sp, es_cfg)
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
