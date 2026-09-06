"""es-decode flow debug: is the decode-rail fitness a faithful proxy of the
prefill fitness of the SAME held perturbation?

For N held rank-1 rails (the worker's exact noise: same layout, same seeds)
on real sampled rollouts of the ds15b student:

  F_dec[n]  = sum_t A_t (log pi_n(y_t) - log pi_0(y_t)) / T   from the decode
              rails riding the clean KV (worker payload)          -- what training uses
  F_pre[n]  = the same, with pi_n = HF model whose EVERY position carries
              dW_n = sigma a_n b_n^T (forward hooks), teacher-forced          -- es-prefill's
  A_t       = log q(y_t) - log pi_0(y_t)  (HF teacher, vLLM clean rail)

Reports, per sigma: corr(F_dec, F_pre) over rails, corr of the antithetic
differences d = (F+ - F-)/2 (this IS the cosine of the two assembled ES
updates up to the noise Gram), the spread ratio RMS(d_dec)/RMS(d_pre), and
on a few layers the cosine of each assembled rank-1 update with the TRUE k1
gradient (HF autograd) -- how much of es-prefill's information the detached-
history rail keeps.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_token_checks/check_es_decode_fitness.py --N 64
"""
import argparse
import json
import math
import os
import re
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_rail_kernels import RULES, WEXT  # noqa: E402
from verl.trainer.es_token.rail_seq_kernels import SeqNoise  # noqa: E402
from verl.trainer.es_token.seeding import build_noise_layout  # noqa: E402

STUDENT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER = "hbx/JustRL-DeepSeek-1.5B"
DATA = "datasets/DAPO-Math-17k/DAPO-Math.parquet"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


class HeldPerturb:
    """Apply the worker's held rank-1 noise to an HF model at EVERY position.
    vLLM fused linears map onto HF's split ones: qkv_proj -> q/k/v_proj (a
    sliced, b shared), gate_up_proj -> gate/up_proj, o/down direct."""

    def __init__(self, model, vllm_names, layout, sn, cfg, dev):
        self.sn, self.layout = sn, layout
        self.on, self.sigma, self.rail = False, 0.0, 0
        self.handles, self.parts = [], []
        n_q = cfg.num_attention_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        n_kv = cfg.num_key_value_heads * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        I = cfg.intermediate_size
        for vn in vllm_names:
            off_u, d_out, off_v, d_in = layout[vn]
            if vn.endswith("self_attn.qkv_proj"):
                base = vn[: -len("qkv_proj")]
                splits = [(base + "q_proj", 0, n_q), (base + "k_proj", n_q, n_kv), (base + "v_proj", n_q + n_kv, n_kv)]
            elif vn.endswith("mlp.gate_up_proj"):
                base = vn[: -len("gate_up_proj")]
                splits = [(base + "gate_proj", 0, I), (base + "up_proj", I, I)]
            else:
                splits = [(vn, 0, d_out)]
            for hn, a0, alen in splits:
                mod = model.get_submodule(hn)
                self.parts.append((hn, vn, a0, alen))
                self.handles.append(mod.register_forward_hook(self._mk(vn, a0, alen)))

    def _vec(self, vn, rail):
        off_u, d_out, off_v, d_in = self.layout[vn]
        r = int(self.sn.rank)
        row = self.sn.noise[rail]
        a = row[r * off_u: r * off_u + r * d_out].view(r, d_out).float()
        b = row[r * off_v: r * off_v + r * d_in].view(r, d_in).float()
        return a, b

    def _mk(self, vn, a0, alen):
        def hook(mod, inp, out):
            if not self.on or self.sigma == 0.0:
                return out
            a, b = self._vec(vn, self.rail)
            r = a.shape[0]
            x = inp[0].float()                                     # [..., d_in]
            coef = (x @ b.t()) * (self.sigma / math.sqrt(r))       # [..., r]
            return out + (coef @ a[:, a0:a0 + alen]).to(out.dtype)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--sigmas", default="1e-3,3e-3")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--gmu", type=float, default=0.3)
    ap.add_argument("--measure-layers", default=(
        "model.layers.3.mlp.down_proj,model.layers.3.self_attn.o_proj,"
        "model.layers.14.mlp.down_proj,model.layers.14.self_attn.o_proj,"
        "model.layers.25.mlp.down_proj,model.layers.25.self_attn.o_proj"))
    ap.add_argument("--out", default="scripts/zo_opd/es_profile/results/es_decode_fitness.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    N = args.N
    assert N % 2 == 0
    seeds = [1000 + i for i in range(N // 2)]

    # ---- rollouts + decode-rail payloads from the worker (vLLM) --------------
    llm = LLM(model=STUDENT, enforce_eager=True, enable_prefix_caching=False,
              worker_extension_cls=WEXT, dtype="bfloat16", gpu_memory_utilization=args.gmu,
              max_model_len=4096)
    tok = llm.get_tokenizer()
    df = pd.read_parquet(DATA)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    pids = []
    for i in range(args.n_prompts):
        msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i][col]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text)["input_ids"]
        if len(ids) <= 1024:
            pids.append(ids)
    B = len(pids)
    llm.collective_rpc("install_es_layers", args=(RULES, N, 42, "seq", args.rank))
    llm.collective_rpc("es_seq_draw", args=(seeds, True))
    w_names = list(llm.collective_rpc("es_export_weights")[0].keys())   # matched order
    shapes = llm.collective_rpc("es_export_weights")[0]
    layout, d_total = build_noise_layout([(n, shapes[n].shape[0], shapes[n].shape[1]) for n in w_names])
    del shapes
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.max_tokens)
    outs = {}
    for s in [float(x) for x in args.sigmas.split(",")]:
        llm.collective_rpc("es_reset_graphs")
        cfg = dict(n_sample=N, max_tokens=args.max_tokens, global_seed=42, sigma=s, sigma_mode="absolute",
                   sample_method="bernoulli", b_pack_buckets=[B], token_agg="mean",
                   attn_impl="seq", lm_head_impl="stream", rail_impl="fused", step_impl="eager",
                   rail_mode="seq", noise_rank=str(args.rank), top_p=0.95)
        if not outs:   # sample the rollouts ONCE at the first sigma, then force them
            o = llm.collective_rpc("run_es_decode_packed", args=(pids, sp, cfg, list(range(B)), True))[0]
            force = o["clean_tokens"]
        cfg["force_tokens"] = force
        outs[s] = llm.collective_rpc("run_es_decode_packed", args=(pids, sp, cfg, list(range(B)), True))[0]
        assert outs[s]["clean_tokens"] == force
    toks = force
    lens = [len(t) for t in toks]
    print(f"[fit] {B} rollouts, lengths {lens}", flush=True)
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- teacher log q (HF) --------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to(dev).eval()
    logqs = []
    with torch.no_grad():
        for p, t in zip(pids, toks):
            seq = list(p) + list(t)
            lp = torch.log_softmax(teacher(torch.tensor([seq], device=dev)).logits[0].float(), -1)
            P = len(p)
            logqs.append(torch.stack([lp[i, seq[i + 1]] for i in range(P - 1, len(seq) - 1)]).cpu())
    del teacher; gc.collect(); torch.cuda.empty_cache()

    # ---- HF student: prefill fitness of the SAME held noise + true gradient ---
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    sn = SeqNoise(layout, N, args.rank, dev, torch.bfloat16)
    sn.draw(seeds, True)
    pert = HeldPerturb(model, w_names, layout, sn, model.config, dev)
    meas = [m for m in args.measure_layers.split(",") if m]
    # clean HF log pi_0 and the true k1 gradient on measured layers
    A_all, lp0_hf, g_true = [], [], {m: torch.zeros_like(model.get_submodule(m).weight, dtype=torch.float32) for m in meas}
    seqs = []
    for p, t, lq, o in zip(pids, toks, logqs, [outs[float(args.sigmas.split(",")[0])]["payload"][i] for i in range(B)]):
        seq = list(p) + list(t); P = len(p)
        ids = torch.tensor([seq], device=dev)
        idx = torch.arange(P - 1, len(seq) - 1, device=dev)
        tgt = torch.tensor(seq[P:], device=dev)
        with torch.no_grad():
            lp0 = torch.log_softmax(model(ids).logits[0].float(), -1)[idx, tgt]
        A = (lq.to(dev) - o[:, 0].to(dev).float())        # k1 advantage from the vLLM clean rail
        for m in meas:
            model.get_submodule(m).weight.requires_grad_(True)
        lp = torch.log_softmax(model(ids).logits[0].float(), -1)[idx, tgt]
        (A.detach() * lp).sum().backward()
        for m in meas:
            w = model.get_submodule(m).weight
            g_true[m] += w.grad.float(); w.grad = None; w.requires_grad_(False)
        seqs.append((ids, idx, tgt)); A_all.append(A); lp0_hf.append(lp0)
    T_tot = sum(lens)
    res = {"N": N, "rank": args.rank, "n_prompts": B, "lens": lens, "sigmas": {}}
    for s in [float(x) for x in args.sigmas.split(",")]:
        pay = outs[s]["payload"]
        F_dec = torch.zeros(N, dtype=torch.float64)
        for i in range(B):
            lp = pay[i].double()                                  # [T, 1+N]
            F_dec += (A_all[i].cpu().double()[:, None] * (lp[:, 1:] - lp[:, :1])).sum(0)
        F_dec /= T_tot
        F_pre = torch.zeros(N, dtype=torch.float64)
        pert.sigma = s
        with torch.no_grad():
            for n in range(N):
                pert.on, pert.rail = True, n
                acc = 0.0
                for i, (ids, idx, tgt) in enumerate(seqs):
                    lpn = torch.log_softmax(model(ids).logits[0].float(), -1)[idx, tgt]
                    acc += float((A_all[i] * (lpn - lp0_hf[i])).sum())
                F_pre[n] = acc / T_tot
        pert.on = False
        d_dec = 0.5 * (F_dec[0::2] - F_dec[1::2]).numpy()
        d_pre = 0.5 * (F_pre[0::2] - F_pre[1::2]).numpy()
        e_dec = 0.5 * (F_dec[0::2] + F_dec[1::2]).numpy()       # even (curvature) part
        e_pre = 0.5 * (F_pre[0::2] + F_pre[1::2]).numpy()
        cF = float(np.corrcoef(F_dec.numpy(), F_pre.numpy())[0, 1])
        cd = float(np.corrcoef(d_dec, d_pre)[0, 1])
        sign_agree = float(np.mean(np.sign(d_dec) == np.sign(d_pre)))
        ratio = float(np.sqrt(np.mean(d_dec ** 2)) / np.sqrt(np.mean(d_pre ** 2)))
        # assembled rank-1 updates on the measured layers vs the true gradient
        z_dec = d_dec / (np.sqrt(np.mean(d_dec ** 2)) + 1e-12)
        z_pre = d_pre / (np.sqrt(np.mean(d_pre ** 2)) + 1e-12)
        cos_dec, cos_pre, cos_dp, bound = {}, {}, {}, {}
        for m in meas:
            off_u, d_out, off_v, d_in = layout[m]
            gd = torch.zeros(d_out, d_in, device=dev); gp = torch.zeros_like(gd)
            for i in range(N // 2):
                a, b = pert._vec(m, 2 * i)                          # rail 2i = +eps_i
                outer = (a.t() @ b) / math.sqrt(args.rank)
                gd += float(z_dec[i]) * outer; gp += float(z_pre[i]) * outer
            cos_dec[m] = cos(gd, g_true[m]); cos_pre[m] = cos(gp, g_true[m]); cos_dp[m] = cos(gd, gp)
            bound[m] = math.sqrt((N // 2) / (N // 2 + d_out * d_in))
        res["sigmas"][str(s)] = dict(corr_F=cF, corr_d=cd, sign_agree=sign_agree, spread_ratio=ratio,
                                     rms_d_dec=float(np.sqrt(np.mean(d_dec ** 2))), rms_d_pre=float(np.sqrt(np.mean(d_pre ** 2))),
                                     even_over_odd_dec=float(np.sqrt(np.mean(e_dec ** 2)) / np.sqrt(np.mean(d_dec ** 2))),
                                     even_over_odd_pre=float(np.sqrt(np.mean(e_pre ** 2)) / np.sqrt(np.mean(d_pre ** 2))),
                                     cos_true_dec=cos_dec, cos_true_pre=cos_pre, cos_dec_pre=cos_dp, bound=bound)
        print(f"\n[sigma={s:g}] N={N} rails ({N//2} pairs), {T_tot} tokens", flush=True)
        print(f"  corr(F_dec, F_pre) = {cF:+.3f}   corr(d_dec, d_pre) = {cd:+.3f}   sign agreement {sign_agree:.2f}", flush=True)
        print(f"  RMS(d): decode {np.sqrt(np.mean(d_dec**2)):.2e}  prefill {np.sqrt(np.mean(d_pre**2)):.2e}  ratio {ratio:.2f}", flush=True)
        print(f"  even/odd (2nd order / 1st order): decode {res['sigmas'][str(s)]['even_over_odd_dec']:.2f}  prefill {res['sigmas'][str(s)]['even_over_odd_pre']:.2f}", flush=True)
        for m in meas:
            print(f"  {m:40s} cos(update, g_true): decode {cos_dec[m]:+.4f}  prefill {cos_pre[m]:+.4f}  "
                  f"cos(dec, pre) {cos_dp[m]:+.3f}   bound sqrt(K/(K+D)) {bound[m]:.4f}", flush=True)
    pert.close()
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
