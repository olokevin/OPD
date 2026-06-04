"""GPU parity gate for the V2 buffer-in-graph decode driver
(spec docs/superpowers/specs/2026-06-03-np-v2-cudagraph-rails.md §7.2).

Runs the SAME prompt/seeds through two drivers and asserts per-token agreement:

  --stage m1 : V1 eager (run_np_decode, in-forward RNG)
               vs  eager-with-u_buf (run_np_decode_graphed, use_cuda_graph=False)
               Isolates the NOISE RELOCATION. u must be BIT-IDENTICAL (same
               noise_seed+draw_noise key); logits/x within bf16 reduction tol.

  --stage m2 : eager-with-u_buf (use_cuda_graph=False)
               vs  graphed (use_cuda_graph=True)
               Isolates THE GRAPH. u bit-identical; logits/x within capture tol.

Usage (1 GPU + small model):
  conda run -n verl python scripts/zo_opd/np_checks/check_graphed_parity.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' \
      --n-sample 4 --sigma 0.01 --stage m1
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def _decode(llm, which, prompt_ids, layer, np_cfg, use_cuda_graph=False):
    sp = SamplingParams(temperature=0.0)
    if which == "v1_eager":
        return llm.collective_rpc(
            "run_np_decode", args=(prompt_ids, sp, layer, np_cfg, 0))[0]
    return llm.collective_rpc(
        "run_np_decode_graphed",
        args=(prompt_ids, sp, layer, np_cfg, 0, use_cuda_graph))[0]


def _assert_parity(a, b, name_a, name_b, logit_rtol):
    # clean tokens identical
    ta, tb = a["clean_tokens"], b["clean_tokens"]
    n = min(len(ta), len(tb))
    assert ta[:n] == tb[:n], (
        f"clean tokens diverged:\n {name_a}={ta}\n {name_b}={tb}")
    assert len(ta) == len(tb), (
        f"different #tokens: {name_a}={len(ta)} {name_b}={len(tb)}")

    for t in range(n):
        ua = a["captured_u"][t].float()
        ub = b["captured_u"][t].float()
        # u must be BIT-IDENTICAL (same seed -> same draw_noise). This is the
        # parity-by-construction guarantee; any mismatch is a seed/key bug.
        assert torch.equal(ua, ub), (
            f"step {t}: u NOT bit-identical between {name_a} and {name_b} "
            f"(max abs diff {(ua-ub).abs().max().item():.3e}) -- seed/key bug")
        # logits within bf16 reduction-order / capture tolerance
        la = a["candidate_logits"][t].float()
        lb = b["candidate_logits"][t].float()
        assert torch.allclose(la, lb, rtol=logit_rtol, atol=1e-2), (
            f"step {t}: logits differ beyond tol "
            f"(max abs {(la-lb).abs().max().item():.3e})")
        xa = a["captured_x"][t].float()
        xb = b["captured_x"][t].float()
        assert torch.allclose(xa, xb, rtol=logit_rtol, atol=1e-2), (
            f"step {t}: x_t differs beyond tol "
            f"(max abs {(xa-xb).abs().max().item():.3e})")
    print(f"PASS [{name_a} vs {name_b}]: {n} tokens, u bit-identical, "
          f"logits/x within rtol={logit_rtol}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.mlp.down_proj")
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--prompt", default="Compute 7*8. Answer:")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--stage", choices=["m1", "m2"], default="m1")
    ap.add_argument("--logit-rtol", type=float, default=1e-2)
    args = ap.parse_args()

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    prompt_ids = tok(args.prompt)["input_ids"]
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))

    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=42, sigma=args.sigma,
                  sample_method=args.sample_method)

    if args.stage == "m1":
        a = _decode(llm, "v1_eager", prompt_ids, args.layer, np_cfg)
        b = _decode(llm, "graphed", prompt_ids, args.layer, np_cfg,
                    use_cuda_graph=False)
        _assert_parity(a, b, "v1_eager", "eager+u_buf", args.logit_rtol)
    else:
        a = _decode(llm, "graphed", prompt_ids, args.layer, np_cfg,
                    use_cuda_graph=False)
        b = _decode(llm, "graphed", prompt_ids, args.layer, np_cfg,
                    use_cuda_graph=True)
        _assert_parity(a, b, "eager+u_buf", "graphed", args.logit_rtol)


if __name__ == "__main__":
    main()
