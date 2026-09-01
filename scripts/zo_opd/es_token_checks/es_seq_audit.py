"""A/B: sequence-level ES perturbation vs the shipped token-level (clean-KV) one.

Token-level (shipped): rails are perturbed only at the current decode step and
read the CLEAN row's KV -> they can only ever see the detached-history gradient.
Sequence-level (this): ONE fixed rank-1 direction per rail for the whole rollout,
scored with a teacher-forced forward, so the perturbation propagates through the
history exactly as a real weight change would -> it sees the TRUE gradient.

Both are scored against the same autograd reference g_full.
"""
import argparse, json, math, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get("VERL_PATH", "verl"))
from verl.trainer.es_token.signs import build_layer_signs

RULE = re.compile(r'^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$')


class SeqPerturb:
    """Fixed rank-1 perturbation of every matched linear, applied to the WHOLE
    forward (all positions), for one rail at a time: y += sigma*((R.v)^T x)*(S.u)."""

    def __init__(self, model, names, n_rails, seed, device, dtype):
        self.on = False; self.sigma = 0.0; self.rail = 0
        self.signs, self.noise, self.mods, self.handles = {}, {}, {}, []
        for nm in names:
            mod = model.get_submodule(nm)
            d_out, d_in = mod.weight.shape
            self.signs[nm] = build_layer_signs(nm, n_rails, d_out, d_in, seed, dtype, device)
            self.mods[nm] = mod
            self.handles.append(mod.register_forward_hook(self._mk(nm)))

    def _mk(self, nm):
        def hook(mod, inp, out):
            if not self.on or self.sigma == 0.0:
                return out
            S, R = self.signs[nm]
            u, v = self.noise[nm]
            ve = R[self.rail] * v          # [d_in]
            ue = S[self.rail] * u          # [d_out]
            alpha = (inp[0] * ve).sum(-1, keepdim=True)
            return out + self.sigma * alpha * ue
        return hook

    def draw(self, gen, device, dtype):
        for nm, mod in self.mods.items():
            d_out, d_in = mod.weight.shape
            self.noise[nm] = (
                (torch.randint(0, 2, (d_out,), generator=gen, device=device, dtype=torch.int8).to(dtype) * 2 - 1),
                (torch.randint(0, 2, (d_in,), generator=gen, device=device, dtype=torch.int8).to(dtype) * 2 - 1))

    def close(self):
        for h in self.handles: h.remove()


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500")
    ap.add_argument("--data", default="datasets/dapo-math-17k.parquet")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--n-rails", type=int, default=8)
    ap.add_argument("--sigmas", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--iw-clamp", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--measure-layers", default=(
        "model.layers.0.mlp.down_proj,model.layers.7.self_attn.o_proj,"
        "model.layers.14.self_attn.k_proj,model.layers.21.self_attn.o_proj,"
        "model.layers.27.mlp.down_proj"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda"); dt = torch.float32
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt,
                                                 attn_implementation="eager").to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    names = matched = [n for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Linear) and RULE.match(n)]
    meas = [m for m in args.measure_layers.split(",") if m]
    print(f"[seq] matched {len(names)} linears")

    import pandas as pd
    df = pd.read_parquet(args.data)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    rollouts = []
    for i in range(args.n_prompts):
        msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i][col]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        pids = tok(text, return_tensors="pt").input_ids[0].tolist()
        with torch.no_grad():
            g = model.generate(torch.tensor([pids], device=dev), do_sample=True,
                               temperature=1.0, top_p=1.0, max_new_tokens=args.max_new,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        rollouts.append((len(pids), g[0].tolist()))
    print(f"[seq] {len(rollouts)} rollouts")

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                   attn_implementation="eager").to(dev).eval()
    logqs = []
    with torch.no_grad():
        for P, seq in rollouts:
            lp = torch.log_softmax(teacher(torch.tensor([seq], device=dev)).logits[0].float(), -1)
            logqs.append(torch.stack([lp[i, seq[i+1]] for i in range(P-1, len(seq)-1)]))
    del teacher; torch.cuda.empty_cache()

    pert = SeqPerturb(model, names, args.n_rails, 0, dev, dt)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    g_full = {m: torch.zeros_like(model.get_submodule(m).weight, dtype=torch.float32) for m in meas}
    dW = {s: {m: torch.zeros_like(g_full[m]) for m in meas} for s in sigmas}
    dlp = {s: [] for s in sigmas}

    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    pert.draw(gen, dev, dt)          # ONE shared direction set for the batch
    railL = {s: [0.0] * args.n_rails for s in sigmas}
    for ri, (P, seq) in enumerate(rollouts):
        ids = torch.tensor([seq], device=dev)
        idx = torch.arange(P-1, len(seq)-1, device=dev)
        tgt = torch.tensor([seq[i+1] for i in range(P-1, len(seq)-1)], device=dev)
        logq = logqs[ri]

        pert.on = False
        with torch.no_grad():
            lp0 = torch.log_softmax(model(ids).logits[0].float(), -1)[idx, tgt]
        c = (lp0 - logq) + 1.0

        for m in meas: model.get_submodule(m).weight.requires_grad_(True)
        lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        ((c.detach() * lp[idx, tgt]).sum()).backward()
        for m in meas:
            w = model.get_submodule(m).weight
            g_full[m] += w.grad.float(); w.grad = None; w.requires_grad_(False)

        for s in sigmas:
            for n in range(args.n_rails):
                pert.on = True; pert.sigma = s; pert.rail = n
                with torch.no_grad():
                    lpn = torch.log_softmax(model(ids).logits[0].float(), -1)[idx, tgt]
                pert.on = False
                w_iw = (lpn - lp0).exp().clamp(max=args.iw_clamp)
                railL[s][n] += float((w_iw * (lpn - logq)).mean()) / len(rollouts)
                dlp[s].append(float((lpn - lp0).abs().mean()))
        if (ri + 1) % 6 == 0:
            print(f"[seq] rollout {ri+1}/{len(rollouts)}")

    # assemble ONCE from the batch-averaged per-rail losses
    for s in sigmas:
        Lt = torch.tensor(railL[s])
        sc = (Lt - Lt.mean()).to(dev)
        for m in meas:
            S, R = pert.signs[m]; u, v = pert.noise[m]
            dW[s][m] += ((S * u).float() * sc[:, None]).t() @ (R * v).float()

    K = args.n_rails   # distinct probe DIRECTIONS (shared across the batch)
    print(f"\n=== SEQUENCE-LEVEL ES vs g_full  (K = {K} directions) ===")
    print(f"{'sigma':>8} | " + " | ".join(f"{m.split('.')[2]+'.'+m.split('.')[-1]:>14}" for m in meas) + " | mean|dlogp|")
    res = {}
    for s in sigmas:
        row = [cos(dW[s][m], g_full[m]) for m in meas]
        bnd = [math.sqrt(K/(K+g_full[m].numel())) for m in meas]
        print(f"{s:>8.0e} | " + " | ".join(f"{a:+.5f}" for a in row) +
              f" | {sum(dlp[s])/len(dlp[s]):.4f}")
        print(f"{'  x bound':>8} | " + " | ".join(f"{a/b:>13.2f}x" for a, b in zip(row, bnd)))
        res[str(s)] = {"cos_full": dict(zip(meas, row)),
                       "ratio": dict(zip(meas, [a/b for a, b in zip(row, bnd)])),
                       "dlogp": sum(dlp[s])/len(dlp[s])}
    print(f"\nbound(K={K}) per layer: " + ", ".join(
        f"{m.split('.')[2]}.{m.split('.')[-1]}={math.sqrt(K/(K+g_full[m].numel())):.5f}" for m in meas))
    if args.out:
        json.dump({"K": K, "sigmas": res}, open(args.out, "w"), indent=2)
    pert.close()


if __name__ == "__main__":
    main()
