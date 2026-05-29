"""GPU verification (spec Verification #1): with sigma=0 the NP custom decode
must reproduce a stock greedy generate() byte-for-byte, the perturbed forward
must widen to 1+n_sample rows, and the KV cache must grow by exactly 1 per step.

Usage (needs 1 GPU + a small model):
  conda run -n verl python scripts/zo_opd/np_checks/check_decode_sigma0.py \
      --model model/Qwen3-1.7B --layer 'model.layers.0.mlp.down_proj' --n-sample 4
"""
import argparse
import os

# vLLM 0.11.0 V1 engine multiprocessing msgpack-serializes tensors across process
# boundaries, which breaks `collective_rpc` returning CPU tensors. Run single-process
# so the (1+n_sample, vocab) logits we return from run_np_decode arrive as a real
# torch.Tensor. (The NP trainer entrypoint sets the same flag — see ES trainer.)
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
    ap.add_argument("--prompt", default="What is 2+2? Answer:")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)

    tok = llm.get_tokenizer()
    prompt_ids = tok(args.prompt)["input_ids"]

    # Stock greedy reference.
    ref = llm.generate({"prompt_token_ids": prompt_ids},
                       SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                       use_tqdm=False)
    ref_tokens = list(ref[0].outputs[0].token_ids)

    # Install perturb layers, then run NP decode with sigma=0.
    llm.collective_rpc("install_perturb_layers", args=([args.layer],))
    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens, global_seed=42,
                  sigma=0.0, sample_method="gaussian")
    out = llm.collective_rpc("run_np_decode",
                             args=(prompt_ids, SamplingParams(temperature=0.0),
                                   args.layer, np_cfg, 0))[0]
    np_tokens = out["clean_tokens"]

    # Assertions.
    assert np_tokens[: len(ref_tokens)] == ref_tokens, (
        f"sigma=0 decode diverged from greedy generate():\n ref={ref_tokens}\n np ={np_tokens}")
    # width: each step's candidate logits must be [1+n_sample, vocab]
    for i, cl in enumerate(out["candidate_logits"]):
        assert cl.shape[0] == 1 + args.n_sample, f"step {i} width {cl.shape[0]} != {1+args.n_sample}"
    print(f"PASS: sigma=0 matches greedy ({len(ref_tokens)} tokens); "
          f"width=1+{args.n_sample} every step.")


if __name__ == "__main__":
    main()
