"""GPU parity gate: per-prompt graphed (oracle) vs packed driver on the SAME
prompts/seeds. clean_tokens identical, u bit-identical (same noise_seed key),
logits/x within bf16 reduction-order tol. Proves packing is a tiling change only.

Usage:
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_packed_parity.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --n-sample 8 --b-pack 4 --sigma 0.01
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--b-pack", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--logit-rtol", type=float, default=1e-2)
    args = ap.parse_args()

    prompts = [
        "Compute 7*8. Answer:", "Differentiate x^3. Answer:",
        "What is 10/2? Answer:", "Square root of 81? Answer:",
        "What is 3-1? Answer:", "Name a prime. Answer:",
        "Integral of 2x? Answer:", "What is 5! ? Answer:",
    ][: args.b_pack]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=42, sigma=args.sigma,
                  sample_method=args.sample_method)
    rollout_ids = list(range(len(pids)))

    # Oracle: per-prompt graphed driver, seeded with the SAME rollout_id per prompt.
    serial = []
    for p, pid in enumerate(pids):
        o = llm.collective_rpc(
            "run_np_decode_graphed",
            args=(pid, SamplingParams(temperature=0.0), args.layer, np_cfg,
                  rollout_ids[p], False))[0]
        serial.append(o)

    # Packed.
    packed = llm.collective_rpc(
        "run_np_decode_packed",
        args=(pids, SamplingParams(temperature=0.0), args.layer, np_cfg,
              rollout_ids))[0]

    for p in range(len(pids)):
        sa, ta = serial[p]["clean_tokens"], packed["clean_tokens"][p]
        n = min(len(sa), len(ta))
        assert sa[:n] == ta[:n] and len(sa) == len(ta), (
            f"prompt {p}: clean tokens diverged\n serial={sa}\n packed={ta}")
        for t in range(n):
            us = serial[p]["captured_u"][t].detach().cpu().float()
            up = packed["captured_u"][p][t].detach().cpu().float()
            assert torch.equal(us, up), (
                f"prompt {p} step {t}: u NOT bit-identical "
                f"(max {(us-up).abs().max():.3e}) -- seed/key bug")
            ls = serial[p]["candidate_logits"][t].float()
            lp = packed["candidate_logits"][p][t].float()
            assert torch.allclose(ls, lp, rtol=args.logit_rtol, atol=1e-2), (
                f"prompt {p} step {t}: logits beyond tol "
                f"(max {(ls-lp).abs().max():.3e})")
            xs = serial[p]["captured_x"][t].float()
            xp = packed["captured_x"][p][t].float()
            assert torch.allclose(xs, xp, rtol=args.logit_rtol, atol=1e-2), (
                f"prompt {p} step {t}: x beyond tol "
                f"(max {(xs-xp).abs().max():.3e})")
    print(f"PASS [serial vs packed]: {len(pids)} prompts, u bit-identical, "
          f"logits/x within rtol={args.logit_rtol}, b_pack={args.b_pack}.")


if __name__ == "__main__":
    main()
