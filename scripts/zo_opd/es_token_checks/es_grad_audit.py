"""es_token gradient audit AT THE TRAINING CONFIGURATION.

The shipped gate (check_es_grad_cosine.py) validated ONE layer at sigma=1e-3 on a
plain cross-entropy loss. Training perturbs ALL 112 decoder linears at sigma=1e-2
with the importance-weighted sampled-token OPD loss. This measures the estimator
in that regime, against two exact autograd references:

  g_direct : sum_t c_t * grad log pi(y_t | DETACHED clean prefix)
             -- EXACTLY what the es probe can see (rails read the clean KV cache
                and are perturbed only at the current decode step)
  g_full   : grad of sum_t c_t * log pi(y_t)   (normal teacher-forced backward)
             -- what BP-OPD actually descends
  c_t      = (log p0(y_t) - log q(y_t)) + 1
             the first-order coefficient of the student_iw rail loss
             l = exp(a - a0) * (a - b)  at a = a0.

Reported:
  cos(dW_es, g_direct)  estimator quality        (is sigma in the linear regime?)
  cos(g_direct, g_full) myopia cost              (structural ceiling of the design)
  cos(dW_es, g_full)    end-to-end
  plus rail Delta-logp spread and IW clamp/underflow rates.
"""
import argparse, json, math, os, re, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get("VERL_PATH", "verl"))
from verl.trainer.es_token.signs import build_layer_signs

RULE = re.compile(r'^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$')


def matched_linears(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and RULE.match(n)]


class Perturb:
    """Rank-1 rail perturbation on rows 1..N of the batch, mirroring ESTokenLinear:
        y[n] += sigma * ((R[n] . v)^T x[n]) * (S[n] . u)
    u, v are redrawn per probe token (shared across rails); S, R are the fixed
    Hadamard sign rails."""

    def __init__(self, model, names, n_rails, seed, device, dtype):
        self.on = False
        self.sigma = 0.0
        self.n = n_rails
        self.mods, self.signs, self.noise = {}, {}, {}
        self.handles = []
        for nm in names:
            mod = model.get_submodule(nm)
            d_out, d_in = mod.weight.shape
            S, R = build_layer_signs(nm, n_rails, d_out, d_in, seed, dtype, device)
            self.signs[nm] = (S, R)
            self.mods[nm] = mod
            self.handles.append(mod.register_forward_hook(self._mk(nm)))

    def _mk(self, nm):
        def hook(mod, inp, out):
            if not self.on or self.sigma == 0.0:
                return out
            x = inp[0]                                   # [B, ..., d_in]
            S, R = self.signs[nm]
            u, v = self.noise[nm]                        # [d_out], [d_in]
            xr = x[1:1 + self.n]                         # rail rows
            v_eff = R * v                                # [N, d_in]
            u_eff = S * u                                # [N, d_out]
            # xr: [N, L, d_in] -> alpha [N, L, 1]
            alpha = (xr * v_eff.unsqueeze(1)).sum(-1, keepdim=True)
            out = out.clone()
            out[1:1 + self.n] = out[1:1 + self.n] + self.sigma * alpha * u_eff.unsqueeze(1)
            return out
        return hook

    def draw(self, gen, device, dtype):
        for nm, mod in self.mods.items():
            d_out, d_in = mod.weight.shape
            u = (torch.randint(0, 2, (d_out,), generator=gen, device=device,
                               dtype=torch.int8).to(dtype) * 2 - 1)
            v = (torch.randint(0, 2, (d_in,), generator=gen, device=device,
                               dtype=torch.int8).to(dtype) * 2 - 1)
            self.noise[nm] = (u, v)

    def close(self):
        for h in self.handles:
            h.remove()


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500")
    ap.add_argument("--data", default="datasets/dapo-math-17k.parquet")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--n-probe", type=int, default=96, help="probe positions per rollout")
    ap.add_argument("--n-rails", type=int, default=8)
    ap.add_argument("--sigmas", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--iw-clamp", type=float, default=10.0)
    ap.add_argument("--weight-mode", default="student_iw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--measure-layers", default=(
        "model.layers.0.mlp.down_proj,model.layers.7.self_attn.o_proj,"
        "model.layers.14.mlp.down_proj,model.layers.21.self_attn.o_proj,"
        "model.layers.27.mlp.down_proj"))
    ap.add_argument("--perturb-layers", default="all",
                    help="'all' = every matched linear (training regime), or a "
                         "comma list = perturb only these (the shipped offline "
                         "gate's regime). Controls the all-layer cross-talk.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    dt = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt,
                                                 attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    names = matched_linears(model)
    if args.perturb_layers != "all":
        names = [n for n in args.perturb_layers.split(",") if n]
    meas = [m for m in args.measure_layers.split(",") if m]
    print(f"[audit] PERTURBING {len(names)} linears, measuring {len(meas)}, dtype={args.dtype}")

    # ---- prompts ----
    import pandas as pd
    df = pd.read_parquet(args.data)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    rollouts = []
    for i in range(args.n_prompts):
        msgs = df.iloc[i][col]
        msgs = list(msgs) if not isinstance(msgs, str) else [{"role": "user", "content": msgs}]
        msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        pids = tok(text, return_tensors="pt").input_ids[0].tolist()
        with torch.no_grad():
            gen = model.generate(torch.tensor([pids], device=dev),
                                 do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=args.max_new,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        seq = gen[0].tolist()
        rollouts.append((len(pids), seq))
        print(f"[audit] rollout {i}: prompt={len(pids)} resp={len(seq)-len(pids)}")

    # ---- teacher logq ----
    print("[audit] loading teacher for log q ...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                   attn_implementation="eager").to(dev).eval()
    logqs = []
    with torch.no_grad():
        for P, seq in rollouts:
            ids = torch.tensor([seq], device=dev)
            lg = teacher(ids).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            logqs.append(torch.stack([lp[i, seq[i + 1]] for i in range(P - 1, len(seq) - 1)]))
    del teacher
    torch.cuda.empty_cache()

    pert = Perturb(model, names, args.n_rails, seed=0, device=dev, dtype=dt)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    results = {"dtype": args.dtype, "n_rails": args.n_rails, "sigmas": {}, "meta": {}}

    # ---------- per-rollout: references and probes ----------
    g_direct = {m: torch.zeros_like(model.get_submodule(m).weight, dtype=torch.float32) for m in meas}
    g_full = {m: torch.zeros_like(model.get_submodule(m).weight, dtype=torch.float32) for m in meas}
    dW = {s: {m: torch.zeros_like(g_direct[m]) for m in meas} for s in sigmas}
    dlp_stats = {s: [] for s in sigmas}
    n_probe_tot = 0

    for ri, (P, seq) in enumerate(rollouts):
        ids = torch.tensor([seq], device=dev)
        Tresp = len(seq) - P
        logq = logqs[ri]

        # clean pass: full KV cache + logp0
        pert.on = False
        with torch.no_grad():
            out = model(ids, use_cache=True)
            cache = out.past_key_values
            lp_all = torch.log_softmax(out.logits[0].float(), -1)
        logp0 = torch.stack([lp_all[i, seq[i + 1]] for i in range(P - 1, len(seq) - 1)])
        c = (logp0 - logq) + 1.0                       # first-order coefficient

        step = max(1, Tresp // args.n_probe)
        probe_j = list(range(0, Tresp, step))[:args.n_probe]
        n_probe_tot += len(probe_j)
        print(f"[audit] rollout {ri}: {len(probe_j)} probe positions, "
              f"c mean={c.mean():.3f} std={c.std():.3f} "
              f"KL mean={(logp0-logq).mean():.3f}")

        # ---- g_full: teacher-forced backward of sum_t c_t log pi(y_t) ----
        # restricted to the SAME probe positions g_direct/dW_es use, so the
        # only difference between the two references is the history path.
        mask = torch.zeros(Tresp, device=dev)
        mask[torch.tensor(probe_j, device=dev)] = 1.0
        for m in meas:
            model.get_submodule(m).weight.requires_grad_(True)
        lg = model(ids).logits[0]
        lp = torch.log_softmax(lg.float(), -1)
        tgt = torch.tensor([seq[i + 1] for i in range(P - 1, len(seq) - 1)], device=dev)
        sel = lp[torch.arange(P - 1, len(seq) - 1, device=dev), tgt]
        loss = ((c.detach() * mask) * sel).sum()
        loss.backward()
        for m in meas:
            w = model.get_submodule(m).weight
            g_full[m] += w.grad.float()
            w.grad = None
            w.requires_grad_(False)

        # ---- per-probe-position: g_direct and the ES estimate ----
        kcache = [ (k.clone(), v.clone()) for k, v in
                   zip([l[0] for l in cache.to_legacy_cache()],
                       [l[1] for l in cache.to_legacy_cache()]) ]

        def sliced_cache(L, B):
            from transformers.cache_utils import DynamicCache
            legacy = tuple((k[:, :, :L].expand(B, -1, -1, -1),
                            v[:, :, :L].expand(B, -1, -1, -1)) for k, v in kcache)
            return DynamicCache.from_legacy_cache(legacy)

        gen_noise = torch.Generator(device=dev); gen_noise.manual_seed(args.seed + 1000 * ri)

        for j in probe_j:
            i = P - 1 + j                       # position that predicts y_j
            y = seq[i + 1]
            tok_in = torch.tensor([[seq[i]]], device=dev)
            pos = torch.tensor([[i]], device=dev)

            # -- g_direct: grad of c_j * log pi(y | detached prefix) --
            for m in meas:
                model.get_submodule(m).weight.requires_grad_(True)
            pert.on = False
            o = model(tok_in, past_key_values=sliced_cache(i, 1), position_ids=pos,
                      use_cache=False)
            lpj = torch.log_softmax(o.logits[0, -1].float(), -1)[y]
            (c[j].detach() * lpj).backward()
            for m in meas:
                w = model.get_submodule(m).weight
                g_direct[m] += w.grad.float()
                w.grad = None
                w.requires_grad_(False)

            # -- ES probe: (1+N) rows, same noise across sigmas --
            pert.draw(gen_noise, dev, dt)
            B = 1 + args.n_rails
            toks = tok_in.expand(B, -1)
            poss = pos.expand(B, -1)
            for s in sigmas:
                pert.on = True; pert.sigma = s
                with torch.no_grad():
                    o = model(toks, past_key_values=sliced_cache(i, B),
                              position_ids=poss, use_cache=False)
                    lpn = torch.log_softmax(o.logits[:, -1].float(), -1)[:, y]   # [1+N]
                pert.on = False
                a0, an = lpn[0], lpn[1:]
                b = logq[j]
                if args.weight_mode == "student_iw":
                    w_iw = (an - a0).exp().clamp(max=args.iw_clamp)
                    l_n = w_iw * (an - b)
                elif args.weight_mode == "student_p":
                    l_n = an.exp() * (an - b)
                else:
                    l_n = an - b
                sc = (l_n - l_n.mean())                       # RAW, /sigma applied below
                dlp_stats[s].append((float((an - a0).abs().mean()),
                                     float((an - a0).max()),
                                     float(((an - a0).exp() > args.iw_clamp).float().mean()),
                                     float(((an - a0) < -10).float().mean()),
                                     float(l_n.std())))
                for m in meas:
                    S, R = pert.signs[m]
                    u, v = pert.noise[m]
                    left = (S * u).float() * sc[:, None].float()    # [N, d_out]
                    right = (R * v).float()                         # [N, d_in]
                    dW[s][m] += left.t() @ right

    # ---------- report ----------
    print(f"\n[audit] probe positions total = {n_probe_tot}, "
          f"K = n_probe * n_rails = {n_probe_tot * args.n_rails}")
    for m in meas:
        d_out, d_in = g_direct[m].shape
        print(f"  {m:<44} d={d_out*d_in/1e6:.1f}M  "
              f"bound(K)={math.sqrt(n_probe_tot*args.n_rails/(n_probe_tot*args.n_rails + d_out*d_in)):.4f}")
    cdf = {m: cos(g_direct[m], g_full[m]) for m in meas}
    print(f"\n=== MYOPIA: cos(g_direct, g_full) — structural ceiling ===")
    for m in meas:
        print(f"  {m:<44} {cdf[m]:+.4f}")
    results["myopia_cos"] = cdf
    print("  (cos of the objective ES can see vs the objective BP descends,")
    print("   on identical token positions -- 1.0 would mean no structural loss)")

    K = n_probe_tot * args.n_rails
    bound = {m: math.sqrt(K / (K + g_direct[m].numel())) for m in meas}
    print(f"\n=== ESTIMATOR: cos(dW_es, g_direct) / cos(dW_es, g_full) vs sigma ===")
    hdr = f"{'sigma':>8} | " + " | ".join(f"{m.split('.')[2]+'.'+m.split('.')[-1]:>16}" for m in meas)
    print(hdr); print("-" * len(hdr))
    for s in sigmas:
        row_d = [cos(dW[s][m], g_direct[m]) for m in meas]
        row_f = [cos(dW[s][m], g_full[m]) for m in meas]
        print(f"{s:>8.0e} | " + " | ".join(f"{a:+.4f}/{b:+.4f}" for a, b in zip(row_d, row_f)))
        print(f"{'  ratio':>8} | " + " | ".join(
            f"{a/bound[m]:>15.2f}x" for a, m in zip(row_d, meas)))
        st = dlp_stats[s]
        n = len(st)
        results["sigmas"][str(s)] = {
            "cos_direct": dict(zip(meas, row_d)),
            "cos_full": dict(zip(meas, row_f)),
            "dlogp_absmean": sum(x[0] for x in st) / n,
            "dlogp_max": max(x[1] for x in st),
            "iw_clamped_frac": sum(x[2] for x in st) / n,
            "iw_underflow_frac": sum(x[3] for x in st) / n,
            "rail_loss_std": sum(x[4] for x in st) / n,
        }

    print(f"\n=== RAIL PERTURBATION MAGNITUDE (|log pi_n(y) - log pi_0(y)|) ===")
    print(f"{'sigma':>8} | {'mean|dlogp|':>12} | {'max dlogp':>10} | {'iw>clamp':>9} | {'iw<e-10':>9} | {'std(l_n)':>9}")
    for s in sigmas:
        r = results["sigmas"][str(s)]
        print(f"{s:>8.0e} | {r['dlogp_absmean']:12.4f} | {r['dlogp_max']:10.3f} | "
              f"{r['iw_clamped_frac']:9.3f} | {r['iw_underflow_frac']:9.3f} | {r['rail_loss_std']:9.4f}")

    results["meta"] = {"n_probe": n_probe_tot, "K": n_probe_tot * args.n_rails,
                       "n_matched": len(names)}
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[audit] wrote {args.out}")
    pert.close()


if __name__ == "__main__":
    main()
