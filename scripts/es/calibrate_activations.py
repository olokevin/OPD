"""One-shot activation calibration for the structured ES perturbation modes.

Produces, for every target linear layer of the base model:

  * ``v``       (r, d) float32 -- top-r **right singular vectors of the input
                activation matrix** X (rows = token activations).  This is the
                ZO-Act basis: ``X ~= U_r D_r V_r^T`` (arXiv:2607.01125), and the
                ZO-Act effective weight is ``W_eff = W + V_r B`` (x@W convention),
                i.e. ``dW_torch = B^T V_r^T`` in PyTorch's (out, in) layout.
  * ``act_rms`` (d,)   float32 -- per-input-channel activation RMS, used by the
                ``insparse`` mode to pick the large-magnitude input channels.

The top-r subspace is obtained with randomized subspace iteration on the
(never materialized) Gram matrix ``G = X^T X``:

    Y_0 = G Omega ;  Q_k = orth(Y_k) ;  Y_{k+1} = G Q_k ;  M = Q^T G Q

Each ``G @ M`` product is one streaming pass over the calibration set, so the
whole thing costs ``passes`` forward passes and O(d * sketch) memory per layer.

Calibration text = training prompt (Qwen-Math template) + the base model's own
greedy rollout, which is what ES actually runs the model on.

Output: a torch ``.pt`` dict keyed by **HF module name**
(``model.layers.0.self_attn.q_proj``).  The ES worker maps vLLM's fused names
onto these (``qkv_proj -> q_proj``, ``gate_up_proj -> gate_proj``).
"""

import argparse
import json
import os

import pandas as pd
import torch

TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def is_target(name: str) -> bool:
    return name.endswith(TARGET_SUFFIXES)


# --------------------------------------------------------------------------- #
# step 1: rollouts
# --------------------------------------------------------------------------- #
def build_calib_texts(args, tokenizer):
    if os.path.exists(args.rollout_cache) and not args.regenerate:
        with open(args.rollout_cache) as f:
            texts = [json.loads(line)["text"] for line in f]
        print(f"[calib] loaded {len(texts)} cached rollouts from {args.rollout_cache}")
        return texts

    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.train_file).iloc[: args.num_samples]
    prompts = [
        tokenizer.apply_chat_template(list(r), add_generation_prompt=True, tokenize=False)
        for r in df["prompt"]
    ]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,
    )
    outs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.rollout_max_tokens),
    )
    texts = [p + o.outputs[0].text for p, o in zip(prompts, outs)]
    os.makedirs(os.path.dirname(args.rollout_cache) or ".", exist_ok=True)
    with open(args.rollout_cache, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"[calib] wrote {len(texts)} rollouts to {args.rollout_cache}")

    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return texts


# --------------------------------------------------------------------------- #
# step 2: streaming Gram passes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_pass(model, modules, batches, accum_fn):
    """Run one forward pass over `batches`, calling accum_fn(name, x) per module."""
    handles = []

    def mk(name):
        def hook(_mod, inp, _out):
            x = inp[0]
            accum_fn(name, x.reshape(-1, x.shape[-1]))

        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(mk(name)))
    try:
        for ids in batches:
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout-cache", required=True)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--rank", type=int, default=1, help="r for the ZO-Act basis")
    ap.add_argument("--sketch", type=int, default=8, help="randomized sketch width (>= rank)")
    ap.add_argument("--passes", type=int, default=3, help="subspace-iteration passes over the data")
    ap.add_argument("--rollout-max-tokens", type=int, default=1024)
    ap.add_argument("--max-calib-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = build_calib_texts(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()

    modules = {n: m for n, m in model.named_modules() if is_target(n)}
    print(f"[calib] {len(modules)} target linear layers")

    dev = next(model.parameters()).device
    batches = [
        tokenizer(t, return_tensors="pt").input_ids[:, : args.max_calib_tokens].to(dev)
        for t in texts
    ]
    print(f"[calib] {len(batches)} sequences, {sum(b.shape[1] for b in batches)} tokens")

    dims = {n: m.in_features for n, m in modules.items()}
    q = max(args.sketch, args.rank)
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # Q_k: current orthonormal sketch basis per layer, initialized randomly.
    Q = {
        n: torch.linalg.qr(torch.randn(d, q, generator=g).to(dev, torch.float32))[0]
        for n, d in dims.items()
    }
    sq_sum = {n: torch.zeros(d, dtype=torch.float64, device=dev) for n, d in dims.items()}
    n_tok = {n: 0 for n in dims}

    for it in range(args.passes):
        Y = {n: torch.zeros(d, q, dtype=torch.float32, device=dev) for n, d in dims.items()}
        first = it == 0

        def accum(name, x):
            xf = x.float()
            Y[name] += xf.T @ (xf @ Q[name])  # (d,q) += G @ Q
            if first:
                sq_sum[name] += (xf * xf).sum(0).double()
                n_tok[name] += xf.shape[0]

        run_pass(model, modules, batches, accum)
        for n in dims:
            Q[n] = torch.linalg.qr(Y[n])[0]
        print(f"[calib] subspace-iteration pass {it + 1}/{args.passes} done")

    # Final Rayleigh-Ritz pass: M = Q^T G Q  (q x q), then eig -> top-r directions.
    M = {n: torch.zeros(q, q, dtype=torch.float64, device=dev) for n in dims}

    def accum_M(name, x):
        z = x.float() @ Q[name]  # (T, q)
        M[name] += (z.T @ z).double()

    run_pass(model, modules, batches, accum_M)

    out = {}
    for n, d in dims.items():
        evals, evecs = torch.linalg.eigh(M[n])  # ascending
        top = evecs[:, -args.rank:].flip(-1).float()  # (q, r), descending
        v = (Q[n] @ top).T.contiguous()  # (r, d)
        v = v / v.norm(dim=-1, keepdim=True)
        ev = evals.flip(0)
        out[n] = {
            "v": v.cpu(),
            "act_rms": (sq_sum[n] / max(n_tok[n], 1)).sqrt().float().cpu(),
            "eig_top": ev[: args.rank].float().cpu(),
            "eig_ratio": float(ev[0] / ev.sum().clamp_min(1e-12)),
            "in_features": d,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"meta": vars(args), "layers": out}, args.out)
    ratios = [out[n]["eig_ratio"] for n in out]
    print(
        f"[calib] wrote {args.out}: {len(out)} layers, "
        f"top-1 sketch energy share min/mean/max = "
        f"{min(ratios):.3f}/{sum(ratios) / len(ratios):.3f}/{max(ratios):.3f}"
    )


if __name__ == "__main__":
    main()
