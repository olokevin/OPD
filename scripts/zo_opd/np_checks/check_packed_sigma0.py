"""GPU gate: with sigma=0 the packed decode must reproduce stock greedy
generate() for EVERY prompt in the wave -- proving per-prompt prefix routing
(disjoint scratch KV, per-row seq_lens/block_table) is correct.

Usage (1 GPU + small model):
  CUDA_VISIBLE_DEVICES=6 /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_packed_sigma0.py --model Qwen/Qwen3-1.7B \
      --layer 'model.layers.0.mlp.down_proj' --n-sample 4 --b-pack 4
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
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--b-pack", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    prompts = [
        "What is 2+2? Answer:",
        "Compute 7*8. Answer:",
        "What is the capital of France? Answer:",
        "Differentiate x^3. Answer:",
        "What is 10/2? Answer:",
        "Name a prime number. Answer:",
        "What is 3-1? Answer:",
        "Square root of 81? Answer:",
    ][: args.b_pack]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]

    # Stock greedy references, one per prompt.
    refs = []
    for pid in pids:
        r = llm.generate({"prompt_token_ids": pid},
                         SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                         use_tqdm=False)
        refs.append(list(r[0].outputs[0].token_ids))

    llm.collective_rpc("install_perturb_layers", args=([args.layer],))
    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=42, sigma=0.0, sample_method="gaussian")
    rollout_ids = list(range(len(pids)))
    out = llm.collective_rpc(
        "run_np_decode_packed",
        args=(pids, SamplingParams(temperature=0.0), args.layer, np_cfg,
              rollout_ids))[0]

    for p in range(len(pids)):
        np_tok = out["clean_tokens"][p]
        ref = refs[p]
        assert np_tok[: len(ref)] == ref, (
            f"prompt {p}: packed sigma=0 diverged from greedy:\n"
            f" ref={ref}\n np ={np_tok}")
        for i, cl in enumerate(out["candidate_logits"][p]):
            assert cl.shape[0] == 1 + args.n_sample, (
                f"prompt {p} step {i} width {cl.shape[0]} != {1+args.n_sample}")
    print(f"PASS [packed sigma=0]: all {len(pids)} prompts match greedy; "
          f"width=1+{args.n_sample}; b_pack={args.b_pack}.")


if __name__ == "__main__":
    main()
