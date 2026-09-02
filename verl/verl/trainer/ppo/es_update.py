# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Forward-only (ES) actor update for OPD -- a drop-in replacement for `update_actor`.

Same rollout, same teacher scores, same per-token advantages as BP-OPD; only the
gradient estimator differs.  Each "rail" is one teacher-forced forward of the actor
over the fixed rollout under a seeded Gaussian weight perturbation (antithetic
pairs), scored with the objective BP maximises,

    F(W) = sum_t m_t * A_t * (log pi_W(y_t) - log pi_0(y_t)) / sum_t m_t          (k1)
    F(W) = sum_{t,k} m_t * A_{t,k} * (log pi_W(k|s_t) - log pi_0(k|s_t)) / sum m   (top-K)

whose gradient at W_0 is exactly BP's policy-gradient.  The update is OpenAI-ES:

    W += (alpha / n_pairs) * sum_i z_i * eps_i,   z_i = d_i / std(d),  d_i = (F(+eps_i) - F(-eps_i)) / 2

(`es_normalize=raw` instead applies alpha times the unbiased gradient estimate
 sum_i d_i eps_i / (n_pairs * sigma)).  Noise is regenerated from seeds inside the
actor worker (`es_perturb_weights` / `es_apply_update`); nothing but seeds and
scalars crosses Ray.
"""
import time

import numpy as np
import torch

from verl import DataProto


def _fitness_fn(trainer, batch):
    """Returns a closure evaluating F(W_current) - F(W_0) on the fixed rollout."""
    wg = trainer.actor_rollout_wg
    mask = batch.batch["response_mask"].float()
    adv = batch.batch["advantages"]
    denom = mask.sum().clamp(min=1.0)
    if adv.dim() == 3:
        base = batch.batch["student_top_k_log_probs"].float()
        ids = batch.batch["student_top_k_ids"]
        m = mask.unsqueeze(-1)
        sub = batch.select(batch_keys=["input_ids", "attention_mask", "position_ids", "responses"])
        sub.batch["target_ids"] = ids
        sub.meta_info = dict(batch.meta_info)

        def fit():
            out = wg.compute_log_probs_for_ids(sub)
            lp = out.batch["student_log_probs_on_teacher_ids"].float().to(base.device)
            return float(((lp - base) * adv * m).sum() / denom)
    else:
        base = batch.batch["old_log_probs"].float()
        m = mask
        sub = batch.select(batch_keys=["input_ids", "attention_mask", "position_ids", "responses", "response_mask"])
        sub.meta_info = dict(batch.meta_info)

        def fit():
            out = wg.compute_log_prob(sub)
            lp = out.batch["old_log_probs"].float().to(base.device)
            return float(((lp - base) * adv * m).sum() / denom)
    return fit


def es_update_actor(trainer, batch: DataProto) -> dict:
    cfg = trainer.config.algorithm
    wg = trainer.actor_rollout_wg
    sigma = float(cfg.es_sigma)
    alpha = float(cfg.es_alpha)
    antithetic = bool(cfg.get("es_antithetic", True))
    rng = np.random.default_rng(int(cfg.es_seed) + int(trainer.global_steps))
    fit = _fitness_fn(trainer, batch)
    t0 = time.time()
    if antithetic:
        n_eff = max(1, int(cfg.es_n_rails) // 2)   # pairs; 2 evals each
        seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_eff)]
        fp, fm = [], []
        for s in seeds:
            wg.es_perturb_weights(s, +sigma)
            fp.append(fit())
            wg.es_perturb_weights(s, -2.0 * sigma)
            fm.append(fit())
            wg.es_perturb_weights(s, +sigma)      # back to W_0 (fp32 add/sub, exact to ~1e-7)
        fp, fm = np.array(fp), np.array(fm)
        d = 0.5 * (fp - fm)                        # zero-mean by construction
        # Normalise by the RMS so sum(coef^2) = alpha^2 / n_eff EXACTLY (std would inflate
        # the step by sqrt(1 + snr^2): up to 4x at 4 pairs -- seen in ES-D, 2026-09-01).
        scale = float(np.sqrt(np.mean(d**2))) if n_eff > 1 else max(abs(float(d[0])), 1e-12)
    else:
        # plain OpenAI-ES sampling: one eval per rail, sample-mean baseline
        n_eff = max(2, int(cfg.es_n_rails))
        seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_eff)]
        f = []
        for s in seeds:
            wg.es_perturb_weights(s, +sigma)
            f.append(fit())
            wg.es_perturb_weights(s, -sigma)
        f = np.array(f)
        fp, fm = f, np.full_like(f, float(f.mean()))
        d = f - f.mean()                           # centering plays the antithetic role
        scale = float(d.std()) + 1e-12             # std of centered == RMS(d)
    t_rails = time.time() - t0
    mode = str(cfg.get("es_normalize", "zscore"))
    if mode == "zscore":
        coeffs = (alpha / n_eff) * d / (scale + 1e-12)
    elif mode == "raw":
        coeffs = alpha * d / (n_eff * sigma)       # alpha * gradient estimate
    else:
        raise ValueError(f"unknown es_normalize={mode!r}")
    t0 = time.time()
    stats = wg.es_apply_update(seeds, [float(c) for c in coeffs])
    stats = stats[0] if isinstance(stats, list) else stats
    t_apply = time.time() - t0
    # Did the step actually ascend the batch objective?  One more rail at W_new.
    post_gain = fit()
    upd_rms = float(np.sqrt(np.sum(coeffs**2)))   # independent unit-variance noises
    rms_w = float(stats.get("rms_w", float("nan")))
    # random-walk bookkeeping: cumulative RMS displacement of all ES steps so far
    trainer._es_cum_sq = getattr(trainer, "_es_cum_sq", 0.0) + upd_rms**2
    cum_footprint = float(np.sqrt(trainer._es_cum_sq)) / rms_w if rms_w == rms_w else float("nan")
    return {
        "es/post_update_gain": post_gain,          # F(W_new) - F(W_0) on this batch (>0 = descent)
        "es/cum_footprint": cum_footprint,         # sqrt(sum_s update_rms_s^2) / RMS(W)
        "es/fitness_plus_mean": float(fp.mean()),
        "es/fitness_minus_mean": float(fm.mean()),
        "es/d_mean": float(d.mean()),
        "es/d_std": scale,                           # ES signal strength (spread across rails)
        "es/d_snr": float(abs(d.mean()) / (scale + 1e-12)),
        "es/update_rms": upd_rms,
        "es/update_footprint": upd_rms / rms_w if rms_w == rms_w else float("nan"),
        "es/probe_footprint": sigma / rms_w if rms_w == rms_w else float("nan"),
        "es/n_rails": (2 * n_eff) if antithetic else n_eff,
        "es/sigma": sigma,
        "es/alpha": alpha,
        "timing_s/es_rails": t_rails,
        "timing_s/es_apply": t_apply,
        "timing_s/es_per_rail": t_rails / ((2 * n_eff) if antithetic else n_eff),
    }
