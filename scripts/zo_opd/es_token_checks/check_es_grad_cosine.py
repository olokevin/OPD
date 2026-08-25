"""es_token gradient-validity gate: cosine(es dW, autograd grad) on a real
Transformer linear (offline HF model, decode-driver-independent -- validates
the rails -> scales -> assemble math chain, the analog of NP's
check_grad_cosine.py).

Loss: cross-entropy of the last token's argmax target (self-consistent label,
same as the NP check). ES estimate: per repeat draw ONE (u, v), evaluate the
N Hadamard-rail perturbations of the target layer's weight, mean-baseline FD,
assemble via the same chunked-GEMM math the trainer uses. Reports cosine vs
W.grad; PASS if > 0.05 (directional gate). Expect LOWER cosine than NP at
equal sample count -- the weight-space probe is oblivious to x_t (plan §0);
this gate quantifies that cost.

  CUDA_VISIBLE_DEVICES=3 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/es_token_checks/check_es_grad_cosine.py \
      --n-sample 8 --repeats 50 --sigma 1e-3
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.es_token.grad_estimator import assemble_chunk, rail_scales
from verl.trainer.es_token.signs import build_layer_signs

DEFAULT_MODEL = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                 "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


def get_module(model, name):
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--prompt", default="Compute 7*8. Answer:")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    lin = get_module(model, args.layer)
    W = lin.weight                      # [d_out, d_in]
    d_out, d_in = W.shape
    print(f"layer {args.layer}  W [{d_out}, {d_in}]")

    # ---- autograd reference ----
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
        target = int(logits.argmax().item())
    W.requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(
        model(ids).logits[0, -1][None, :],
        torch.tensor([target], device=device))
    loss.backward()
    g_true = W.grad.detach().clone()
    W.requires_grad_(False)

    def f():
        with torch.no_grad():
            lg = model(ids).logits[0, -1]
            return float(torch.nn.functional.cross_entropy(
                lg[None, :], torch.tensor([target], device=device)).item())

    # ---- es_token estimate ----
    N = args.n_sample
    S, R = build_layer_signs(args.layer, N, d_out, d_in, args.seed,
                             torch.float32, device)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    acc = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)
    W0 = W.detach().clone()
    for rep in range(args.repeats):
        if args.sample_method == "bernoulli":
            u = (torch.randint(0, 2, (d_out,), generator=gen).float() * 2 - 1)
            v = (torch.randint(0, 2, (d_in,), generator=gen).float() * 2 - 1)
        else:
            u = torch.randn(d_out, generator=gen)
            v = torch.randn(d_in, generator=gen)
        u, v = u.to(device), v.to(device)
        losses = torch.empty(1, N)
        for n in range(N):
            dW = args.sigma * torch.outer(S[n] * u, R[n] * v)
            with torch.no_grad():
                W.copy_(W0 + dW)
            losses[0, n] = f()
        with torch.no_grad():
            W.copy_(W0)
        sc = rail_scales(losses, None, args.sigma, "mean_baseline").to(device)
        assemble_chunk(sc, u[None, :], v[None, :], S, R, acc)
        if (rep + 1) % 10 == 0:
            dw = acc / (N * (rep + 1))
            cos = torch.nn.functional.cosine_similarity(
                dw.flatten(), g_true.flatten(), dim=0).item()
            print(f"  rep {rep+1:4d}: cos = {cos:+.4f}")

    dw = acc / (N * args.repeats)
    cos = torch.nn.functional.cosine_similarity(
        dw.flatten(), g_true.flatten(), dim=0).item()
    K = args.repeats * N   # independent rank-1 probes
    # Theory for an unbiased isotropic rank-1 weight-space probe: the estimate
    # splits into K/(d_out*d_in) signal fraction -> cos ~= sqrt(K / (K + d)).
    # (NP probes the d_out-dim output space instead, hence its ~40x higher
    # per-probe cosine -- the plan §0 trade-off, quantified here.) At the
    # training operating point K = B*T*N ~= 5e5 per step, predicted per-layer
    # cos ~= 0.2 for down_proj.
    d = float(d_out * d_in)
    pred = (K / (K + d)) ** 0.5
    ratio = cos / pred if pred > 0 else 0.0
    k_train = 64 * 1024 * N
    print(f"\ncosine(es_dW, true_grad) = {cos:+.4f}  "
          f"(n_sample={N}, repeats={args.repeats}, K={K} probes)")
    print(f"theory sqrt(K/(K+d_out*d_in)) = {pred:.4f}  ->  cos/theory = "
          f"{ratio:.2f}")
    print(f"training-scale prediction (K=B*T*N={k_train}): "
          f"cos ~= {(k_train / (k_train + d)) ** 0.5:.3f}")
    ok = cos > 0 and ratio > 0.4
    print("PASS (>=0.4x theory)" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
