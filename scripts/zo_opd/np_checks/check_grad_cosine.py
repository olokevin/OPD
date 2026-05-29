"""GPU verification (spec Verification #2): NP's estimated delta_W should align
with the true autograd gradient on one layer/batch. Reports cosine similarity;
PASS if >= 0.05 (token-granularity expected >= 0.1 with enough samples).

This loads the model TWICE: once via HF (eager, autograd) for the reference
gradient, once via the NP worker path for the estimate. Run on 1 GPU.

  conda run -n verl python scripts/zo_opd/np_checks/check_grad_cosine.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' \
      --n-sample 64 --repeats 50
"""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.np.seeding import noise_seed, draw_noise
from verl.trainer.np.grad_estimator import sample_scale, accumulate_delta_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--prompt", default="Compute 7*8. Answer:")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
    model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

    # locate the HF analog of the layer (HF uses split names; map down_proj directly).
    # For this check we target a real HF Linear by attribute walk.
    hf_name = args.layer  # HF Qwen has model.layers.N.mlp.down_proj as a real module
    mod = model
    for p in hf_name.split("."):
        mod = getattr(mod, p)
    W = mod.weight  # [d_out, d_in]

    # --- reference gradient: dL/dW where L = next-token CE on the prompt's last token ---
    captured = {}
    h = mod.register_forward_hook(lambda m, i, o: captured.__setitem__("x", i[0].detach()))
    W.requires_grad_(True)
    out = model(ids)
    logits = out.logits[:, -1, :]
    target = logits.argmax(-1)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    g_true = W.grad.detach().clone()
    h.remove()
    W.requires_grad_(False)
    x_t = captured["x"].reshape(-1, captured["x"].shape[-1])[-1]  # last-token input [d_in]

    # --- NP estimate: perturb W's OUTPUT row-wise via u, measure loss delta ---
    d_out = W.shape[0]
    dw = torch.zeros_like(g_true, dtype=torch.float32)
    base = F.cross_entropy(model(ids).logits[:, -1, :], target).item()
    for rep in range(args.repeats):
        u = torch.stack([
            draw_noise(noise_seed(0, rep, args.layer, 0, q), (d_out,), dev, torch.float32, "gaussian")
            for q in range(args.n_sample)
        ])  # [n_sample, d_out]
        L_q = []
        for q in range(args.n_sample):
            hh = mod.register_forward_hook(
                lambda m, i, o, uq=u[q]: o + args.sigma * uq)
            L_q.append(F.cross_entropy(model(ids).logits[:, -1, :], target).item())
            hh.remove()
        scales = sample_scale(torch.tensor(L_q, device=dev), L_clean=base, sigma=args.sigma, mode="average")
        accumulate_delta_w(dw, scales=scales, u=u, x_t=x_t, normalize=False)
    dw.div_(args.repeats)

    cos = F.cosine_similarity(dw.flatten(), g_true.flatten(), dim=0).item()
    print(f"cosine(NP_dW, true_grad) = {cos:.4f}  (n_sample={args.n_sample}, repeats={args.repeats})")
    assert cos > 0.05, f"FAIL: cosine {cos:.4f} <= 0.05 (sign/scale bug?)"
    print("PASS")


if __name__ == "__main__":
    main()
