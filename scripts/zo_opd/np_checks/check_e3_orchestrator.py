"""GPU smoke for Stage E3: run_np_decode_packed_graphed -- the fully CUDA-graphed
all-layer PACKED decode orchestrator that ties E1 (capture) + E2 (replay) into a
full decode driver.

E3 scope: prove the orchestrator runs end-to-end (prefill -> one capture per
bucket -> padded replay decode -> per-prompt per-layer signals) without error and
returns the run_np_decode_packed shape (per-prompt lists/dicts). As a parity
sanity-check (cheap), we ALSO compare each prompt's graphed clean tokens against
the EAGER all-layer oracle (run_np_decode with a layer LIST, decoded per-prompt
serially with the SAME rollout id) -- if the padded decode corrupts an active
prompt's attention, the tokens diverge.

The bucket here is chosen by _select_bucket: B=4 prompts with default buckets
[2,4,8,16] -> bucket 4 (exact fit, no PAD slots). To exercise PAD slots too, pass
--n-prompts 3 (B=3 -> bucket 4, one PAD slot).

Usage (one free GPU among 1/2/3):
  cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
  PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=1 NP_KEEP_CUDA_VISIBLE=1 \
      /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_e3_orchestrator.py \
      --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: F401  (import side-effects / device init parity with E1/E2)
from vllm import LLM, SamplingParams


def _orchestrate_on_worker(self, prompt_ids_list, layer_names, n_sample,
                           max_tokens, sigma, global_seed, sample_method,
                           rollout_ids):
    """Runs ON the worker: drive run_np_decode_packed_graphed and return a small
    facts dict the host asserts on (no GPU tensors cross the RPC boundary -- the
    per-layer signal shapes/counts are reduced to python ints/bools here)."""
    mr = self.model_runner
    sp = SamplingParams(temperature=0.0)
    np_cfg = dict(n_sample=int(n_sample), max_tokens=int(max_tokens),
                  global_seed=int(global_seed), sigma=float(sigma),
                  sample_method=sample_method)

    out = self.run_np_decode_packed_graphed(
        prompt_ids_list, sp, list(layer_names), np_cfg, list(rollout_ids))

    B = len(prompt_ids_list)
    clean = out["clean_tokens"]
    cand = out["candidate_logits"]
    cu = out["captured_u"]
    cx = out["captured_x"]

    facts = {
        "B": B,
        "n_prompts_returned": len(clean),
        "token_counts": [len(clean[p]) for p in range(B)],
        "clean_tokens": [list(clean[p]) for p in range(B)],
        # candidate_logits[p] must be a list of (topk_logp, ids) TUPLES.
        "cand_is_tuple": all(
            isinstance(c, tuple) and len(c) == 2
            for p in range(B) for c in cand[p]),
        "cand_counts": [len(cand[p]) for p in range(B)],
        # per-layer u/x present for EVERY layer of EVERY prompt, with a key for
        # every decoded token t.
        "u_layers_per_prompt": [sorted(cu[p].keys()) for p in range(B)],
        "x_layers_per_prompt": [sorted(cx[p].keys()) for p in range(B)],
        "u_t_counts": [
            {ln: len(cu[p][ln]) for ln in cu[p]} for p in range(B)],
        # shape of one captured tensor (u: [n_sample, d_out], x: [d_in]).
        "u_shape0": (tuple(cu[0][layer_names[0]][0].shape)
                     if cu[0][layer_names[0]] else None),
        "x_shape0": (tuple(cx[0][layer_names[0]][0].shape)
                     if cx[0][layer_names[0]] else None),
        "ext_file": type(self).__module__,
    }

    # Cheap parity oracle: per-prompt EAGER all-layer decode with the SAME
    # rollout id -> graphed clean tokens should match BIT-FOR-BIT.
    oracle = []
    for p, pid in enumerate(prompt_ids_list):
        o = self.run_np_decode(
            pid, SamplingParams(temperature=0.0), list(layer_names), np_cfg,
            int(rollout_ids[p]))
        oracle.append(list(o["clean_tokens"]))
    facts["oracle_tokens"] = oracle
    facts["parity_match"] = [
        list(clean[p]) == oracle[p] for p in range(B)]
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--layers", nargs="+", default=[
        "model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"])
    args = ap.parse_args()

    import verl.workers.rollout.vllm_rollout.np_worker_extension as npx
    print("np_worker_extension.__file__ =", npx.__file__)
    assert "np-alllayer-graphed" in npx.__file__, (
        "STALE verl loaded -- prepend PYTHONPATH=$PWD/verl. "
        f"got {npx.__file__}")

    prompts = [
        "Compute 7*8. Answer:",
        "Differentiate x^3. Answer:",
        "What is 10/2? Answer:",
        "Square root of 81? Answer:",
    ][: args.n_prompts]
    B = len(prompts)

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]
    llm.collective_rpc("install_perturb_layers", args=(args.layers,))

    # slot p bound to prompt p; rollout id = p (the oracle uses the same id).
    rollout_ids = list(range(B))

    facts = llm.collective_rpc(
        _orchestrate_on_worker,
        args=(pids, list(args.layers), args.n_sample, args.max_tokens,
              args.sigma, args.global_seed, args.sample_method, rollout_ids))[0]

    print("\nworker ext module :", facts["ext_file"])
    print("B / returned      :", facts["B"], "/", facts["n_prompts_returned"])
    print("token counts/prompt:", facts["token_counts"])
    print("cand counts/prompt :", facts["cand_counts"])
    print("cand are tuples    :", facts["cand_is_tuple"])
    print("u layers (prompt0) :", facts["u_layers_per_prompt"][0])
    print("x layers (prompt0) :", facts["x_layers_per_prompt"][0])
    print("u t-counts(prompt0):", facts["u_t_counts"][0])
    print("u shape[0]         :", facts["u_shape0"], "(expect [n_sample, d_out])")
    print("x shape[0]         :", facts["x_shape0"], "(expect [d_in])")
    print("parity vs oracle   :", facts["parity_match"])

    layer_set = sorted(args.layers)
    # --- Assertions ---
    assert facts["n_prompts_returned"] == B, "returned != B prompts"
    assert all(tc > 0 for tc in facts["token_counts"]), (
        "a prompt produced zero clean tokens")
    assert facts["cand_is_tuple"], "candidate_logits[p] entries are not tuples"
    assert all(cc == tc for cc, tc in zip(
        facts["cand_counts"], facts["token_counts"])), (
        "candidate_logits count != clean-token count per prompt")
    for p in range(B):
        assert facts["u_layers_per_prompt"][p] == layer_set, (
            f"prompt {p}: captured_u missing layers "
            f"(got {facts['u_layers_per_prompt'][p]}, want {layer_set})")
        assert facts["x_layers_per_prompt"][p] == layer_set, (
            f"prompt {p}: captured_x missing layers")
        # every layer has one signal per decoded token.
        for ln, n_t in facts["u_t_counts"][p].items():
            assert n_t == facts["token_counts"][p], (
                f"prompt {p} layer {ln}: {n_t} u entries != "
                f"{facts['token_counts'][p]} tokens")
    assert facts["u_shape0"][0] == args.n_sample, (
        f"u_buf row dim {facts['u_shape0']} != n_sample {args.n_sample}")
    assert all(facts["parity_match"]), (
        "PARITY FAILED: graphed clean tokens diverged from the eager all-layer "
        "oracle -- the padded decode likely corrupted an active prompt's "
        f"attention. graphed={facts['clean_tokens']} oracle={facts['oracle_tokens']}")

    print(f"\nPASS [E3 orchestrator]: run_np_decode_packed_graphed decoded "
          f"{B} prompts (bucket >= B, one capture, padded decode), per-layer "
          f"u/x present for all {len(args.layers)} layers, candidate_logits are "
          f"tuples, and clean tokens match the eager all-layer oracle BIT-FOR-BIT "
          f"(n_sample={args.n_sample}, max_tokens={args.max_tokens}).")


if __name__ == "__main__":
    main()
