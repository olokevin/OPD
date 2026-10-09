"""Parameter-space-noise exploration for BP-OPD (algorithm.pn_explore).

Plappert et al., "Parameter Space Noise for Exploration" (ICLR 2018), applied to token-level
OPD with greedy decoding.  Of the ``rollout.n`` rollouts per prompt, slot 0 (``pn_n_clean``
slots) is the clean policy's greedy decode and every other slot k is the greedy decode of the
perturbed policy ``W + pn_sigma * eps_k`` -- one fresh seeded perturbation per slot per step,
shared by all prompts of that slot.  Greedy makes each slot deterministic, so all exploration
comes from parameter space.

Everything after generation is BP-OPD unchanged: the teacher scores every token, the advantage
is ``log q(y_t) - log pi_clean(y_t)`` (clean ``old_log_probs``), and the clean weights are updated
by backward.  The perturbed slots are off-policy for the clean policy; their behaviour
log-probs ``log pi_{W+sigma eps_k}(y_t)`` (teacher-forced FSDP forward under the perturbation, T=1)
go into ``rollout_log_probs`` so verl's rollout correction applies the truncated token-level IS
weight ``min(pi_clean / pi_perturbed, threshold)`` on top of the PPO-clipped loss (decoupled
PPO: the trust region stays around the clean policy).  Clean slots get weight exactly 1.

The perturbation reuses the ES machinery (``es_perturb_weights``: seeded noise on the fp32
master, regenerated, never stored); the hybrid engine re-syncs FSDP weights into vLLM on every
``generate_sequences``, so the perturbed weights are what vLLM decodes with.
"""
import time

import numpy as np
import torch

from verl import DataProto


def pn_generate(trainer, gen_batch: DataProto):
    """Generate ``rollout.n`` slots per prompt, interleaved like ``gen_batch.repeat(n, interleave=True)``.

    Returns (gen_output, metrics, timing).  gen_output carries ``rollout_log_probs`` = behaviour
    log-probs (zeros on clean rows -- filled by ``pn_fill_clean`` once ``old_log_probs`` exist)
    and ``pn_is_clean`` (bool per row).
    """
    cfg = trainer.config.algorithm
    wg = trainer.actor_rollout_wg
    n = int(trainer.config.actor_rollout_ref.rollout.n)
    n_clean = int(cfg.get("pn_n_clean", 1))
    sigma = float(cfg.pn_sigma)
    pset = str(cfg.get("pn_perturb_set", "all"))
    rng = np.random.default_rng(int(cfg.get("pn_seed", 4321)) + int(trainer.global_steps))
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]
    b = len(gen_batch)

    outs, timing = [], {"gen_pn_generate": 0.0, "gen_pn_behavior_logprob": 0.0}
    for k in range(n):
        perturbed = k >= n_clean
        if perturbed:
            wg.es_perturb_weights(seeds[k], +sigma, pset)
        try:
            t0 = time.time()
            out = wg.generate_sequences(gen_batch)
            timing["gen_pn_generate"] += time.time() - t0
            out.meta_info.pop("timing", None)
            if perturbed:
                t0 = time.time()
                sub = out.select(batch_keys=["input_ids", "attention_mask", "position_ids", "responses"])
                beh = wg.compute_log_prob(sub).batch["old_log_probs"].float().cpu()
                timing["gen_pn_behavior_logprob"] += time.time() - t0
            else:
                beh = torch.zeros_like(out.batch["responses"], dtype=torch.float32)
        finally:
            if perturbed:
                wg.es_perturb_weights(seeds[k], -sigma, pset)   # back to W (fp32 add/sub, ~1e-7 rel)
        out.batch["rollout_log_probs"] = beh
        out.batch["pn_is_clean"] = torch.full((b,), not perturbed, dtype=torch.bool)
        outs.append(out)

    # pad all slots to the same response width before concatenating
    width = max(o.batch["responses"].shape[1] for o in outs)
    assert all(o.batch["responses"].shape[1] == width for o in outs), "rollout pads to max_response_length"
    cat = DataProto.concat(outs)
    order = torch.tensor([k * b + i for i in range(b) for k in range(n)])
    cat.reorder(order)

    # exploration diagnostics: how different are the perturbed decodes from the clean one?
    resp = cat.batch["responses"].view(b, n, -1)
    rmask = cat.batch["attention_mask"][:, -resp.shape[-1]:].view(b, n, -1).bool()
    clean = resp[:, :1]
    same_tok = (resp == clean) & rmask & rmask[:, :1]
    any_diff = ((resp != clean) & (rmask | rmask[:, :1])).any(-1)          # (b, n)
    # first position where slot k differs from the clean slot (= length if never)
    diff = ((resp != clean) | (rmask != rmask[:, :1]))
    first = torch.where(diff.any(-1), diff.float().argmax(-1), rmask.sum(-1)).float()
    lens = rmask.sum(-1).float()
    pk = slice(n_clean, n)
    metrics = {
        "pn/frac_identical_to_clean": float((~any_diff[:, pk]).float().mean()),
        "pn/first_divergence_tok_mean": float(first[:, pk].mean()),
        "pn/first_divergence_frac_mean": float((first[:, pk] / lens[:, pk].clamp(min=1)).mean()),
        "pn/shared_prefix_tok_frac": float(same_tok[:, pk].sum() / rmask[:, pk].sum().clamp(min=1)),
        "pn/resp_len_clean": float(lens[:, :n_clean].mean()),
        "pn/resp_len_perturbed": float(lens[:, pk].mean()),
        "pn/sigma": sigma,
        "pn/n_perturbed_slots": n - n_clean,
    }
    return cat, metrics, timing


def pn_fill_clean(batch: DataProto) -> dict:
    """Clean rows: behaviour == clean policy, so rollout_log_probs := old_log_probs (IS weight 1).

    Also logs the clean-vs-perturbed policy gap on the perturbed rows.
    """
    old = batch.batch["old_log_probs"].float()
    clean = batch.batch["pn_is_clean"].to(old.device)
    beh = batch.batch["rollout_log_probs"].float().to(old.device)
    beh = torch.where(clean[:, None], old, beh)
    batch.batch["rollout_log_probs"] = beh
    m = batch.batch["response_mask"].float().to(old.device) * (~clean)[:, None].float()
    denom = m.sum().clamp(min=1.0)
    lr = (old - beh) * m                      # log pi_clean - log pi_perturbed on perturbed tokens
    w = torch.exp(lr.clamp(max=20.0))
    return {
        "pn/log_ratio_mean": float(lr.sum() / denom),        # -KL(pert || clean) estimate (<= 0)
        "pn/log_ratio_abs_mean": float(lr.abs().sum() / denom),
        "pn/is_weight_mean": float((w * m).sum() / denom),
        "pn/is_weight_frac_gt2": float(((w > 2.0).float() * m).sum() / denom),
        "pn/is_weight_frac_lt0.5": float(((w < 0.5).float() * m).sum() / denom),
    }
