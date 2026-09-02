"""opd_curvature.py -- can ANY forward-only (ES / rail) estimator match BP on this objective?

Measures, on one on-policy batch of the student, for the fixed-trajectory reverse-KL
objective f(W) = mean_t KL(pi_W(.|s_t) || q(.|s_t))  (exact, full vocab; this is the
zero-variance version of the k1 policy-gradient loss BP-OPD minimises):

  g        = grad f (autograd)                        -> ||g||
  kappa_g  = g^T H g / ||g||^2   (FD along g/||g||)   -> curvature BP "sees"
  tr(H)    = E_eps[eps^T H eps]  (FD, isotropic eps)  -> curvature isotropic ES noise sees
  r_eff    = tr(H) / kappa_g

An unbiased ES estimator with N rails and its OPTIMAL step makes per-step progress
     (1/2)||g||^2 / (kappa_g + tr(H)/N)   vs   GD's (1/2)||g||^2 / kappa_g,
i.e. a fraction 1 / (1 + r_eff/N) of BP's.  So r_eff decides the question; N is the
only lever, and sqrt(N/D) cosines are a red herring.

Also reports the KL-rise curve vs sigma (the ES sigma calibration at production
length) and cos(g, W_teacher - W_student) (the teacher is the RL'd student here).

  python scripts/zo_opd/ds15b/opd_curvature.py --gpu 5 --out logs/ds15b/curv.json
"""
import argparse, gc, json, math, os, time
import numpy as np
import torch


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--teacher", default="hbx/JustRL-DeepSeek-1.5B")
    ap.add_argument("--data", default="datasets/DAPO-Math-17k/DAPO-Math.parquet")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigmas", default="2.5e-4,5e-4,1e-3,2e-3")   # isotropic, per-element
    ap.add_argument("--n-probes", type=int, default=2)             # antithetic pairs per sigma
    ap.add_argument("--dir-steps", default="3e-3,1e-2,3e-2")       # along g/||g|| (unit direction)
    ap.add_argument("--exclude-embed", action="store_true", help="do not perturb embed/lm_head")
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.35)
    ap.add_argument("--out", default="logs/ds15b/opd_curvature.json")
    ap.add_argument("--rollout-cache", default="", help="reuse token ids from a prior run")
    ap.add_argument("--groups", action="store_true",
                    help="per parameter-group ||g_G||^2 and tr(H_G) (embed/lm_head, attn, mlp, norms) "
                         "-- the displacement-limited figure of merit ||g_G||^2 / tr(H_G) per group")
    ap.add_argument("--group-sigma", type=float, default=1e-3)
    ap.add_argument("--subspace", default="", choices=["", "zoact"],
                    help="zoact: per decoder linear, dW_l = A_l V_l^T with V_l = top-r eigenvectors of the "
                         "layer-input second moment (calibrated on this batch); reports ||g_S||^2 / tr(H_S)")
    ap.add_argument("--subspace-rank", default="1,4,16")
    ap.add_argument("--subspace-sigma", type=float, default=1e-3)
    ap.add_argument("--n-seqs", type=int, default=0, help="use only the first n cached sequences (0 = all)")
    return ap.parse_args()


# ----------------------------------------------------------------------------- rollout
def rollout(args, tok):
    if args.rollout_cache and os.path.exists(args.rollout_cache):
        d = torch.load(args.rollout_cache)
        print(f"[rollout] loaded {len(d['full'])} sequences from {args.rollout_cache}")
        return d["full"], d["resp_len"]
    import pandas as pd
    from vllm import LLM, SamplingParams
    df = pd.read_parquet(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=args.n_prompts, replace=False)
    prompts = []
    for i in idx:
        msgs = [dict(m) for m in df.iloc[int(i)]["prompt"]]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    llm = LLM(model=args.student, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=1024 + args.max_tokens, seed=args.seed, enable_prefix_caching=False)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=args.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    full, resp_len = [], []
    for o in outs:
        r = list(o.outputs[0].token_ids)
        full.append(list(o.prompt_token_ids) + r)
        resp_len.append(len(r))
    print(f"[rollout] {len(full)} seqs, resp_len mean {np.mean(resp_len):.0f} "
          f"max {max(resp_len)} (cap {args.max_tokens}), {time.time()-t0:.0f}s")
    del llm; gc.collect(); torch.cuda.empty_cache()
    if args.rollout_cache:
        torch.save({"full": full, "resp_len": resp_len}, args.rollout_cache)
    return full, resp_len


# ----------------------------------------------------------------------------- models
def load_hf(path, dtype, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def teacher_logq(teacher, full, resp_len, device):
    """log q over the full vocab at every response position; cached on CPU in bf16."""
    cache = []
    for ids, T in zip(full, resp_len):
        x = torch.tensor(ids, device=device)[None]
        logits = teacher(x).logits[0, -T - 1:-1].float()      # predicts response tokens
        cache.append(torch.log_softmax(logits, -1).to(torch.bfloat16).cpu())
        del logits
    return cache


def seq_kl(student, ids, T, logq_cpu, device, need_grad=False):
    """sum_t KL(pi_W(.|s_t) || q(.|s_t)) over the T response positions of one sequence."""
    x = torch.tensor(ids, device=device)[None]
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        logits = student(x).logits[0, -T - 1:-1].float()
        logp = torch.log_softmax(logits, -1)
        logq = logq_cpu.to(device, non_blocking=True).float()
        kl = (logp.exp() * (logp - logq)).sum(-1)              # [T]
        tot = kl.sum()
        if need_grad:
            return tot
        return tot.item()


def f_eval(student, full, resp_len, logq, device):
    s, n = 0.0, 0
    for ids, T, lq in zip(full, resp_len, logq):
        s += seq_kl(student, ids, T, lq, device); n += T
    return s / n


def grad_f(student, full, resp_len, logq, device):
    n = sum(resp_len)
    params = [p for p in student.parameters()]
    for p in params:
        p.requires_grad_(True); p.grad = None
    student.gradient_checkpointing_enable()
    student.train()  # needed for checkpointing; no dropout in Qwen2
    for ids, T, lq in zip(full, resp_len, logq):
        (seq_kl(student, ids, T, lq, device, need_grad=True) / n).backward()
    student.eval(); student.gradient_checkpointing_disable()
    g = {name: p.grad.detach().clone() for name, p in student.named_parameters()}
    for p in params:
        p.requires_grad_(False); p.grad = None
    return g


# ----------------------------------------------------------------------------- perturbations
def pert_names(student, exclude_embed):
    names = []
    for name, p in student.named_parameters():
        if exclude_embed and ("embed_tokens" in name or "lm_head" in name):
            continue
        names.append(name)
    return names


def add_noise(student, names, seed, scale):
    """W += scale * eps, eps ~ N(0, I) regenerated from `seed` (never stored)."""
    pd = dict(student.named_parameters())
    for name in names:
        p = pd[name]
        gen = torch.Generator(device=p.device); gen.manual_seed(int(seed))
        p.data.add_(torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen), alpha=scale)


def add_dir(student, direction, scale):
    pd = dict(student.named_parameters())
    for name, d in direction.items():
        pd[name].data.add_(d, alpha=scale)


def dot(a, b):
    return sum((a[k].double() * b[k].double()).sum().item() for k in a)


def main():
    args = parse()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student)
    full, resp_len = rollout(args, tok)
    if args.n_seqs > 0:
        full, resp_len = full[:args.n_seqs], resp_len[:args.n_seqs]
    n_tok = sum(resp_len)

    teacher = load_hf(args.teacher, torch.bfloat16, device)
    t0 = time.time(); logq = teacher_logq(teacher, full, resp_len, device)
    print(f"[teacher] log q cached for {n_tok} tokens in {time.time()-t0:.0f}s")
    W_T = {n: p.detach().float().cpu() for n, p in teacher.named_parameters()}
    del teacher; gc.collect(); torch.cuda.empty_cache()

    student = load_hf(args.student, torch.float32, device)
    names = pert_names(student, args.exclude_embed)
    pd = dict(student.named_parameters())
    D = sum(pd[n].numel() for n in names)
    rmsW = math.sqrt(sum((pd[n].double() ** 2).sum().item() for n in names) / D)
    print(f"[student] D = {D/1e9:.3f} B perturbed params, RMS(W) = {rmsW:.4f}")

    res = dict(setting=vars(args), n_tok=n_tok, resp_len_mean=float(np.mean(resp_len)),
               D=D, rmsW=rmsW)

    f0 = f_eval(student, full, resp_len, logq, device)
    res["f0_kl_per_tok"] = f0
    print(f"[f0] base reverse-KL per token = {f0:.5f}")

    # ---- gradient, its norm, and the teacher-delta cosine
    t0 = time.time(); g = grad_f(student, full, resp_len, logq, device)
    gn = math.sqrt(dot(g, g)); res["grad_norm"] = gn
    delta = {n: (W_T[n] - pd[n].detach().cpu().float()) for n in names}
    dn = math.sqrt(dot(delta, delta))
    gsub = {n: g[n] for n in names}
    cos_td = dot(gsub, {n: delta[n].to(device) for n in names}) / (gn * dn + 1e-30)
    res["teacher_delta_norm"] = dn; res["cos_grad_teacher_delta"] = -cos_td  # descent = -g
    print(f"[grad] ||g|| = {gn:.4e}  ({time.time()-t0:.0f}s);  ||W_T - W_S|| = {dn:.3f} "
          f"(rel {dn/(rmsW*math.sqrt(D)):.3e});  cos(-g, W_T - W_S) = {-cos_td:+.4f}")
    del W_T, delta; gc.collect()

    # ---- curvature along the gradient direction (unit vector), symmetric FD
    ghat = {n: (g[n] / gn) for n in names}
    res["dir"] = {}
    for s_str in args.dir_steps.split(","):
        s = float(s_str)
        add_dir(student, ghat, +s); fp = f_eval(student, full, resp_len, logq, device)
        add_dir(student, ghat, -2 * s); fm = f_eval(student, full, resp_len, logq, device)
        add_dir(student, ghat, +s)
        kappa = (fp + fm - 2 * f0) / s ** 2
        dd = (fp - fm) / (2 * s)                        # should equal ||g||
        res["dir"][s_str] = dict(f_plus=fp, f_minus=fm, kappa_g=kappa, fd_dirderiv=dd)
        print(f"[dir] step {s:.0e}: f+ {fp:.5f} f- {fm:.5f}  kappa_g = {kappa:.4e}  "
              f"FD dir-deriv {dd:.4e} vs autograd {gn:.4e}")

    # ---- isotropic probes: KL rise (sigma calibration) and tr(H) by Hutchinson
    res["iso"] = {}
    for s_str in args.sigmas.split(","):
        s = float(s_str)
        rows = []
        for k in range(args.n_probes):
            seed = 1000 * k + 7
            add_noise(student, names, seed, +s); fp = f_eval(student, full, resp_len, logq, device)
            add_noise(student, names, seed, -2 * s); fm = f_eval(student, full, resp_len, logq, device)
            add_noise(student, names, seed, +s)
            # first-order term cancels in fp+fm; eps has ||eps||^2 ~ D so the quadratic
            # term is (1/2) s^2 eps^T H eps  ->  tr(H) ~ (fp + fm - 2 f0) / s^2
            trH = (fp + fm - 2 * f0) / s ** 2
            # FD directional derivative along eps vs autograd <g, eps>
            fd = (fp - fm) / (2 * s)
            pdict = dict(student.named_parameters())
            geps = 0.0
            for n in names:
                gen = torch.Generator(device=device); gen.manual_seed(int(seed))
                eps = torch.randn(pdict[n].shape, dtype=torch.float32, device=device, generator=gen)
                geps += (g[n].double() * eps.double()).sum().item()
            rows.append(dict(seed=seed, f_plus=fp, f_minus=fm, trH=trH, fd_dirderiv=fd, autograd_dirderiv=geps))
            print(f"[iso] sigma {s:.1e} probe {k}: f+ {fp:.5f} f- {fm:.5f} (x{(fp+fm)/(2*f0):.3f} base)"
                  f"  tr(H) ~ {trH:.4e}   FD <g,eps> {fd:+.4e} vs autograd {geps:+.4e}")
        res["iso"][s_str] = dict(rows=rows, trH_mean=float(np.mean([r["trH"] for r in rows])),
                                  kl_rise=float(np.mean([(r["f_plus"] + r["f_minus"]) / 2 for r in rows]) / f0),
                                  probe_footprint=s / rmsW)

    # ---- per-group figure of merit: which parameters carry gradient energy per unit curvature?
    if args.groups:
        def grp(name):
            if "embed_tokens" in name or "lm_head" in name:
                return "embed+lm_head"
            if "self_attn" in name:
                return "attn"
            if "mlp" in name:
                return "mlp"
            return "norms+other"
        groups = {}
        for n in names:
            groups.setdefault(grp(n), []).append(n)
        res["groups"] = {}
        for gname, gnames in groups.items():
            dG = sum(pd[n].numel() for n in gnames)
            g2 = sum((g[n].double() ** 2).sum().item() for n in gnames)
            sG = args.group_sigma
            trs = []
            for k in range(args.n_probes):
                seed = 5000 * k + 3
                add_noise(student, gnames, seed, +sG); fp = f_eval(student, full, resp_len, logq, device)
                add_noise(student, gnames, seed, -2 * sG); fm = f_eval(student, full, resp_len, logq, device)
                add_noise(student, gnames, seed, +sG)
                trs.append((fp + fm - 2 * f0) / sG ** 2)
            trG = float(np.mean(trs))
            res["groups"][gname] = dict(D=dG, g2=g2, trH=trG, ratio=g2 / trG if trG > 0 else float("nan"),
                                        g2_frac=g2 / gn ** 2)
            print(f"[group] {gname:>14s}: D {dG/1e6:8.1f} M  ||g_G||^2 {g2:.3e} ({100*g2/gn**2:5.1f} % of ||g||^2)"
                  f"  tr(H_G) {trG:.3e}  ||g_G||^2/tr(H_G) {g2/max(trG,1e-30):.3e}")
        print(f"[group] {'ALL':>14s}: ||g||^2/tr(H) {gn**2/trH if 'trH' in dir() else float('nan'):.3e}")

    # ---- calibrated activation subspace (zoact-style): does an ALIGNED low-D subspace beat 7e-4?
    if args.subspace == "zoact":
        import torch.nn as nn
        lin = {n: m for n, m in student.named_modules()
               if isinstance(m, nn.Linear) and "layers." in n}
        # second moments live on the HOST in fp32 (9 GB) so this can run next to a training job
        acc = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32, pin_memory=True) for n, m in lin.items()}
        hooks = []
        def mk(n):
            def h(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                acc[n].add_((x.t() @ x).cpu())
            return h
        for n, m in lin.items():
            hooks.append(m.register_forward_hook(mk(n)))
        with torch.no_grad():
            for ids in full:
                student(torch.tensor(ids, device=device)[None])
        for h in hooks:
            h.remove()
        res["subspace"] = {}
        rmax = max(int(r) for r in args.subspace_rank.split(","))
        V = {}
        for n, C in acc.items():
            evals, evecs = torch.linalg.eigh(C.to(device))      # ascending
            V[n] = evecs[:, -rmax:].flip(-1).float().contiguous()  # [d_in, rmax], top first
            del evecs
        acc.clear(); torch.cuda.empty_cache()
        pdict = dict(student.named_parameters())
        for r in [int(x) for x in args.subspace_rank.split(",")]:
            names_w = [n + ".weight" for n in lin]
            dS = sum(pdict[w].shape[0] * r for w in names_w)
            g2 = sum(((g[w].float() @ V[n][:, :r]) ** 2).sum().item() for n, w in zip(lin, names_w))
            sS = args.subspace_sigma
            trs = []
            for k in range(args.n_probes):
                seed = 9000 * k + 1
                def pert(scale):
                    for pidx, (n, w) in enumerate(zip(lin, names_w)):
                        gen = torch.Generator(device=device); gen.manual_seed(seed * 100003 + pidx)
                        A = torch.randn(pdict[w].shape[0], r, device=device, generator=gen)
                        pdict[w].data.add_(A @ V[n][:, :r].t(), alpha=scale)
                pert(+sS); fp = f_eval(student, full, resp_len, logq, device)
                pert(-2 * sS); fm = f_eval(student, full, resp_len, logq, device)
                pert(+sS)
                trs.append((fp + fm - 2 * f0) / sS ** 2)
            trS = float(np.mean(trs))
            res["subspace"][r] = dict(D=dS, g2=g2, trH=trS, ratio=g2 / trS if trS > 0 else float("nan"),
                                      g2_frac=g2 / gn ** 2, kl_rise=(fp + fm) / (2 * f0))
            print(f"[zoact r={r:2d}] D_S {dS/1e6:7.2f} M  ||g_S||^2 {g2:.3e} ({100*g2/gn**2:5.1f} % of ||g||^2)"
                  f"  tr(H_S) {trS:.3e}  ||g_S||^2/tr(H_S) {g2/max(trS,1e-30):.3e}  (full-space {gn**2/trH:.3e})")

    # ---- the verdict
    # take kappa_g from the smallest dir step whose quadratic fit is sane, tr(H) from the
    # smallest sigma (most linear); print all so the reader can pick.
    kap = res["dir"][args.dir_steps.split(",")[0]]["kappa_g"]
    trH = res["iso"][args.sigmas.split(",")[0]]["trH_mean"]
    r_eff = trH / kap if kap > 0 else float("nan")
    res.update(kappa_g=kap, trH=trH, r_eff=r_eff)
    print("\n================ verdict ================")
    print(f"base KL/tok {f0:.4f}   ||g|| {gn:.3e}   kappa_g {kap:.3e}   tr(H) {trH:.3e}   "
          f"r_eff = tr(H)/kappa_g = {r_eff:.3e}")
    gd = 0.5 * gn ** 2 / kap
    print(f"GD (optimal step) per-step KL decrease: {gd:.3e}  ({100*gd/f0:.2f}% of base KL)")
    res["pred"] = {}
    for N in [8, 32, 128, 512, 4096]:
        es = 0.5 * gn ** 2 / (kap + trH / N)
        res["pred"][N] = dict(es_per_step=es, ratio=es / gd, steps_for_10pct=0.1 * f0 / es)
        print(f"  ES N={N:5d}: per-step {es:.3e} = {es/gd:.2e} x GD;  steps to cut KL by 10%: {0.1*f0/es:.3e}")
    # Displacement-limited regime (the one that binds in practice: the isotropic noise of
    # every ES step accumulates as a random walk and must stay inside the quadratic basin):
    # with a total noise-damage budget delta_max (KL units) and C total rail forwards,
    # first-order coherent KL gain ~ ||g|| * sqrt(2 * delta_max * C / tr(H)).
    lam_avg = trH / D
    res["lambda_avg"] = lam_avg
    res["efficiency_g2_over_trH"] = gn ** 2 / trH
    print(f"lambda_avg = tr(H)/D = {lam_avg:.3e};  ||g||^2/tr(H) = {gn**2/trH:.3e}")
    # Report the coherent MOTION (weight-norm units) and its equivalent in fully aligned BP steps
    # of norm lr*sqrt(D) (Adam at lr=1e-6); a first-order KL number would overshoot the objective.
    bp_step = 1e-6 * math.sqrt(D)
    for dmax in [0.1 * f0, 0.36 * f0]:
        for C in [32 * 300, 128 * 300, 512 * 300, 1_000_000]:
            motion = math.sqrt(2 * dmax * C / trH)
            print(f"  damage budget {dmax:.3f} KL ({100*dmax/f0:.0f}% of base), C={C:>8d} rails: "
                  f"coherent motion {motion:.3f} = {motion/bp_step:.1f} aligned BP steps (lr 1e-6)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
