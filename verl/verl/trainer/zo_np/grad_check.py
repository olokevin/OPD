"""Offline NP-vs-BP gradient check for the zeroth-order node-perturbation estimator.

Goal
----
Measure, for ONE perturb layer on ONE frozen (prompt + greedy response):

  1. cos( delta_W_NP , dL/dW_BP )      direction agreement
  2. ||delta_W_NP||, ||dL/dW_BP||      magnitudes of each
  3. the implied scale ratio           -> diagnoses improper scaling
  4. a learning-rate suggestion        -> lr that makes the NP step match a
                                          reference first-order BP step size

It does this entirely in eager HuggingFace + autograd on a single GPU, so the
true gradient dL/dW is available from loss.backward(). The NP estimate reuses
the SHIPPING estimator math (seeding / sample_scale / accumulate_delta_w /
assemble_layer_delta) so the number we report is the real thing, not a model of it.

Loss
----
The trainer's objective (loss_type=opd) is per-token reverse-KL of the student
to the teacher over a top-K token set:

    L_t = sum_{v in topK} w_v * (log p_student(v) - log p_teacher(v))

with w_v = softmax(student)_v under reward_weight_mode=student_p (the default).
L = sum_t L_t over response tokens. This module reproduces exactly
verl.trainer.np.teacher_scorer.reverse_kl_topk, but in an autograd-friendly way
for the BP gradient.

The teacher log-probs over the chosen token set are computed ONCE (frozen
targets, detached). Both the NP estimate and the BP gradient are then taken of
the SAME loss with the SAME frozen teacher targets and the SAME top-K id set per
token -- the only difference is how the gradient w.r.t. W is obtained
(node-perturbation forward differences vs. autograd backward).

Node convention
---------------
The perturb layer is a linear  y = x W^T  (HF Linear stores weight as
[d_out, d_in]).  The NP estimator perturbs the OUTPUT node y:  y^(q) = y + sigma*u_q,
u_q in R^{d_out}.  To first order  L_t^(q) - L_t ~= sigma * <dL_t/dy_t, u_q>, so
g_t = mean_q s(L_t^(q)) u_q estimates dL_t/dy_t and delta_W = sum_t g_t (x) x_t
estimates dL/dW = sum_t (dL_t/dy_t) (x) x_t.  Hence cos(delta_W_NP, dL/dW_BP) is
the quantity of interest.
"""
import argparse
import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---- shipping estimator math (single source of truth) ----------------------
from verl.trainer.np.seeding import noise_seed, draw_noise
from verl.workers.rollout.vllm_rollout.np_worker_extension import assemble_layer_delta


# ===========================================================================
# Teacher target + reverse-KL loss (autograd-friendly, matches reverse_kl_topk)
# ===========================================================================
def reverse_kl_topk_autograd(
    student_logp_set: torch.Tensor,   # [k] student log-probs over the top-k set (grad-tracking)
    teacher_logp_set: torch.Tensor,   # [k] teacher log-probs over the SAME k tokens (frozen)
    weight_mode: str = "student_p",
) -> torch.Tensor:
    """Same formula as verl.trainer.np.teacher_scorer.reverse_kl_topk.

    sum_v w_v * (log p_student(v) - log p_teacher(v)).  weight_mode picks w_v.
    Kept separate from the shipping fn only so the student side keeps its graph
    (the shipping one is called on detached CPU logits inside vLLM).
    """
    diff = student_logp_set - teacher_logp_set
    if weight_mode == "student_p":
        w = student_logp_set.exp()
    elif weight_mode == "teacher_p":
        w = teacher_logp_set.exp()
    elif weight_mode == "none":
        w = torch.ones_like(diff)
    else:
        raise ValueError(f"unknown reward_weight_mode: {weight_mode!r}")
    return (w * diff).sum()


def per_token_loss_from_logits(
    student_logits_t: torch.Tensor,   # [vocab] student logits at response step t (grad-tracking)
    topk_ids_t: torch.Tensor,         # [k] frozen top-k id set for this step
    teacher_logp_t: torch.Tensor,     # [k] frozen teacher log-probs aligned to topk_ids_t
    weight_mode: str,
) -> torch.Tensor:
    """L_t for one step, differentiable in student_logits_t."""
    s_full_logp = torch.log_softmax(student_logits_t.float(), dim=-1)  # [vocab]
    s_set = s_full_logp[topk_ids_t]                                    # [k]
    return reverse_kl_topk_autograd(s_set, teacher_logp_t, weight_mode)


# ===========================================================================
# Model / tokenizer / data helpers
# ===========================================================================
def load_model(path: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",  # deterministic, autograd-safe
    ).to(device)
    model.eval()
    return model, tok


def build_prompt_ids(tok, problem: str, enable_thinking: bool) -> List[int]:
    """Apply the chat template exactly as the dataset prompt would be formed."""
    messages = [{"role": "user", "content": problem}]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def greedy_decode(model, tok, prompt_ids: List[int], max_new: int, device: str) -> List[int]:
    """Greedy-decode a frozen response. temp=0 so both NP and BP condition on
    the identical token sequence (sampling noise removed from the comparison)."""
    ids = torch.tensor([prompt_ids], device=device)
    eos = tok.eos_token_id
    eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos])
    out = []
    past = None
    cur = ids
    for _ in range(max_new):
        res = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = res.past_key_values
        nxt = int(res.logits[0, -1].argmax().item())
        out.append(nxt)
        if nxt in eos_set:
            break
        cur = torch.tensor([[nxt]], device=device)
    return out


# ===========================================================================
# Forward passes that expose the perturb layer's input x_t and output y_t
# ===========================================================================
def find_module(model, name: str) -> torch.nn.Module:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        cand = [n for n, _ in model.named_modules() if name in n]
        raise KeyError(f"module {name!r} not found. close matches: {cand[:8]}")
    return mod


class IOCapture:
    """Forward hook that records a linear layer's input x and output y for the
    LAST forward call (full-sequence teacher-forced pass)."""
    def __init__(self, module: torch.nn.Module):
        self.x = None
        self.y = None
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.x = inp[0].detach()            # [1, seq, d_in]
        self.y = (out[0] if isinstance(out, tuple) else out).detach()  # [1, seq, d_out]

    def remove(self):
        self._h.remove()


class OutputPerturb:
    """Forward hook that ADDS a precomputed per-position delta to a linear
    layer's output. Used to inject sigma*u at the node output for one decode
    step in the NP forward-difference, replaying the full frozen sequence."""
    def __init__(self, module: torch.nn.Module):
        self.delta = None   # [1, seq, d_out] or None
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if self.delta is None:
            return out
        y, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        y = y + self.delta.to(y.dtype)
        return (y, *rest) if rest is not None else y

    def remove(self):
        self._h.remove()


# ===========================================================================
# Build frozen teacher targets + the per-step top-K id set + clean student logits
# ===========================================================================
@torch.no_grad()
def teacher_targets(teacher, tok, full_ids: List[int], resp_start: int,
                    topk_ids_per_step: List[torch.Tensor],
                    teacher_temperature: float, device: str) -> List[torch.Tensor]:
    """Teacher log-probs aligned to each step's top-K id set.

    The teacher scores position t's *next-token* distribution, which predicts
    full_ids[t+1]. For response token at full index j (j in [resp_start, L-1]),
    the predicting position is j-1. Returns list aligned to response steps.
    """
    ids = torch.tensor([full_ids], device=device)
    logits = teacher(input_ids=ids).logits[0]          # [L, vocab]
    out = []
    for k, j in enumerate(range(resp_start, len(full_ids))):
        pred_pos = j - 1
        logp = torch.log_softmax(logits[pred_pos].float() / teacher_temperature, dim=-1)
        out.append(logp[topk_ids_per_step[k]].detach())
    return out


# ===========================================================================
# NP estimate (forward differences) and BP truth (autograd) for ONE layer
# ===========================================================================
def run_grad_check(args) -> dict:
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"[load] student {args.student}")
    student, tok = load_model(args.student, device, dtype)
    print(f"[load] teacher {args.teacher}")
    teacher, _ = load_model(args.teacher, device, dtype)

    # ---- problem + frozen greedy response -------------------------------
    problem = args.problem
    prompt_ids = build_prompt_ids(tok, problem, args.enable_thinking)
    if len(prompt_ids) > args.max_prompt_len:
        prompt_ids = prompt_ids[: args.max_prompt_len]
    print(f"[decode] greedy response (max_new={args.max_resp_len}) ...")
    resp_ids = greedy_decode(student, tok, prompt_ids, args.max_resp_len, device)
    if args.max_steps > 0:
        resp_ids = resp_ids[: args.max_steps]
    full_ids = list(prompt_ids) + list(resp_ids)
    resp_start = len(prompt_ids)
    T = len(resp_ids)
    print(f"[decode] prompt_len={len(prompt_ids)} resp_len={T}")
    if T == 0:
        raise RuntimeError("empty response; nothing to score")

    layer_name = args.layer
    smod = find_module(student, layer_name)
    d_out, d_in = smod.weight.shape
    print(f"[layer] {layer_name}  weight=[{d_out},{d_in}]")

    full_t = torch.tensor([full_ids], device=device)

    # ---- clean student logits + captured x_t, y_t (one teacher-forced pass)
    cap = IOCapture(smod)
    with torch.no_grad():
        clean_logits = student(input_ids=full_t).logits[0]   # [L, vocab]
    x_full = cap.x[0]            # [L, d_in]
    cap.remove()

    # response step k predicts full_ids[resp_start+k] from position resp_start+k-1
    pred_positions = [resp_start + k - 1 for k in range(T)]
    # student top-K id set per step (only_stu strategy, K=log_prob_top_k)
    K = int(args.log_prob_top_k)
    topk_ids_per_step: List[torch.Tensor] = []
    for k in range(T):
        lg = clean_logits[pred_positions[k]]                 # [vocab]
        topk_ids_per_step.append(torch.topk(lg.float(), K).indices)

    # ---- frozen teacher targets aligned to those id sets ----------------
    print("[teacher] scoring frozen targets ...")
    teacher_logp_per_step = teacher_targets(
        teacher, tok, full_ids, resp_start, topk_ids_per_step,
        args.teacher_temperature, device)

    # x_t captured at the PREDICTING position (the input that produced the
    # next-token logits) -- this is the x that pairs with dL_t/dy_t.
    x_steps = [x_full[pred_positions[k]].detach().float().cpu() for k in range(T)]

    # ======================================================================
    # (A) TRUE gradient dL/dW via autograd
    # ======================================================================
    print("[BP] autograd backward of summed reverse-KL ...")
    student.zero_grad(set_to_none=True)
    smod.weight.requires_grad_(True)
    logits_g = student(input_ids=full_t).logits[0]           # [L, vocab], grad-tracking
    L_total = 0.0
    L_clean_steps: List[float] = []
    for k in range(T):
        lt = per_token_loss_from_logits(
            logits_g[pred_positions[k]], topk_ids_per_step[k],
            teacher_logp_per_step[k], args.reward_weight_mode)
        L_total = L_total + lt
        L_clean_steps.append(float(lt.detach().item()))
    L_total.backward()
    dW_bp = smod.weight.grad.detach().float().cpu()          # [d_out, d_in] = dL/dW
    smod.weight.requires_grad_(False)
    student.zero_grad(set_to_none=True)

    # ======================================================================
    # (B) NP estimate delta_W via node-output forward differences
    # ======================================================================
    print(f"[NP] node-perturbation forward differences  "
          f"n_sample={args.n_sample} n_rollout={args.n_rollout} sigma={args.sigma} "
          f"method={args.sample_method} ...")
    pert = OutputPerturb(smod)
    # FAITHFUL forward difference: perturb EXACTLY ONE token's node output per
    # forward (the step under test), never the others. This matches production --
    # the vLLM NP decode scores each step n_sample-wide against the shared CLEAN
    # prefix, so token t's perturbation never contaminates token t'<t. Perturbing
    # all predicting positions at once (a tempting batched shortcut) lets earlier
    # tokens' noise leak into later tokens' loss through causal attention, which
    # decorrelates the per-token forward difference and destroys the estimate.
    #
    # We still batch the n_sample perturbations of a SINGLE token into one
    # forward via the batch dimension: each row replays the same sequence but the
    # hook adds a different sigma*u_q at this token's predicting position. Rows do
    # not attend across the batch, so the n_sample losses are independent and
    # correct. Trainer seeds by (seed,step,layer,rollout,q); we mirror it exactly.
    L_q_steps: List[torch.Tensor] = []        # per step: [R*n_sample]
    u_steps: List[torch.Tensor] = []          # per step: [R*n_sample, d_out]
    n_sample = int(args.n_sample)
    n_rollout = int(args.n_rollout)
    Lp = len(full_ids)

    for t in range(T):
        pos = pred_positions[t]
        Lq_t, u_t = [], []
        for r in range(n_rollout):
            # Draw u on CPU (seed -> device-independent); assemble accumulates on
            # CPU like production. .to(device) only to inject the per-row delta.
            u_rt = torch.stack([
                draw_noise(noise_seed(args.global_seed, t, layer_name, r, q),
                           (d_out,), torch.device("cpu"), torch.float32,
                           args.sample_method)
                for q in range(n_sample)
            ], 0)                                       # [n_sample, d_out] (CPU)
            # batch the n_sample rows: [n_sample, L] ids, delta only at `pos`.
            batch_ids = full_t.expand(n_sample, Lp)
            delta = torch.zeros(n_sample, Lp, d_out, device=device, dtype=dtype)
            delta[:, pos] = (args.sigma * u_rt).to(device, dtype)
            pert.delta = delta
            with torch.no_grad():
                pl = student(input_ids=batch_ids).logits  # [n_sample, L, vocab]
            pert.delta = None
            Lq_r = torch.stack([
                per_token_loss_from_logits(
                    pl[q, pos], topk_ids_per_step[t],
                    teacher_logp_per_step[t], args.reward_weight_mode).detach().float().cpu()
                for q in range(n_sample)
            ], 0)                                        # [n_sample]
            Lq_t.append(Lq_r)
            u_t.append(u_rt)
        L_q_steps.append(torch.cat(Lq_t, 0))            # [R*n_sample]
        u_steps.append(torch.cat(u_t, 0))               # [R*n_sample, d_out]
    pert.remove()

    # assemble's "average" mode uses the per-step clean baseline; grpo ignores it.
    L_clean_for_assemble = L_clean_steps

    # ---- delta_W from the SHIPPING assemble_layer_delta -----------------
    dW_np = assemble_layer_delta(
        L_q_steps, L_clean_for_assemble, u_steps, x_steps,
        sigma=float(args.sigma), sample_mode=args.grad_estimate_sample,
        normalize=args.normalize, token_agg=args.token_agg,
    ).float()

    # ======================================================================
    # (C) metrics
    # ======================================================================
    def cos(a, b):
        return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())

    metrics = {
        "layer": layer_name,
        "d_out": int(d_out), "d_in": int(d_in),
        "T_steps": T,
        "n_sample": int(args.n_sample), "n_rollout": int(args.n_rollout),
        "sigma": float(args.sigma), "sample_method": args.sample_method,
        "grad_estimate_sample": args.grad_estimate_sample,
        "normalize": bool(args.normalize), "token_agg": args.token_agg,
        "reward_weight_mode": args.reward_weight_mode,
        "cosine_NP_vs_BP": cos(dW_np, dW_bp),
        "norm_dW_NP": float(dW_np.norm().item()),
        "norm_dW_BP": float(dW_bp.norm().item()),
        "ratio_NP_over_BP": float((dW_np.norm() / (dW_bp.norm() + 1e-30)).item()),
        "L_clean_mean": float(sum(L_clean_steps) / max(T, 1)),
    }

    # Also report the "average"-mode (unnormalized) NP estimate, which is the
    # theoretically unbiased one -- the clean diagnostic for scaling.
    dW_np_avg = assemble_layer_delta(
        L_q_steps, L_clean_for_assemble, u_steps, x_steps,
        sigma=float(args.sigma), sample_mode="average",
        normalize=False, token_agg=args.token_agg,
    ).float()
    metrics["cosine_NPavg_vs_BP"] = cos(dW_np_avg, dW_bp)
    metrics["norm_dW_NPavg"] = float(dW_np_avg.norm().item())
    metrics["ratio_NPavg_over_BP"] = float((dW_np_avg.norm() / (dW_bp.norm() + 1e-30)).item())

    # LR suggestion: pick lr_np so that ||lr_np * dW_np|| == ||lr_ref * dW_BP||,
    # i.e. the NP update lands the same step size a BP step of lr_ref would.
    lr_ref = float(args.lr_ref)
    metrics["lr_ref"] = lr_ref
    metrics["lr_suggest_match_BP_step"] = (
        lr_ref * metrics["norm_dW_BP"] / (metrics["norm_dW_NP"] + 1e-30))

    print("\n================ GRAD CHECK ================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.6g}")
        else:
            print(f"  {k:28s}: {v}")
    print("===========================================\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[out] wrote {args.out}")

    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline NP-vs-BP gradient check")
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    p.add_argument("--problem", default=(
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did she sell altogether in "
        "April and May? Put your final answer in \\boxed{}."))
    p.add_argument("--enable-thinking", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--max-resp-len", type=int, default=7168)
    p.add_argument("--max-steps", type=int, default=64,
                   help="cap response steps scored (cost control; 0 = all)")
    # NP knobs (mirror np_trainer)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--n-sample", type=int, default=16)
    p.add_argument("--n-rollout", type=int, default=4)
    p.add_argument("--sample-method", default="bernoulli")
    p.add_argument("--grad-estimate-sample", default="grpo")
    p.add_argument("--normalize", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--token-agg", default="sum")
    p.add_argument("--global-seed", type=int, default=42)
    # teacher / OPD
    p.add_argument("--log-prob-top-k", type=int, default=16)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--reward-weight-mode", default="student_p")
    # misc
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--lr-ref", type=float, default=1e-6,
                   help="reference BP learning rate for the lr suggestion")
    p.add_argument("--out", default="")
    return p


def main():
    args = build_argparser().parse_args()
    run_grad_check(args)


if __name__ == "__main__":
    main()
