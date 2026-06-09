"""GPU staggered-EOS parity gate for Stage E2 (_np_replay_step_packed).

The DEEP staggered-EOS test (not the shallow "one prompt inactive" case): drive
a packed all-layer CUDA-graph replay over a FIXED bucket of B_pack prompts whose
EOS is STAGGERED -- prompt 0 finishes early, prompt 1 mid-decode, prompt 2 runs
longest -- and assert each prompt's CLEAN tokens match the EAGER all-layer oracle
(`run_np_decode` with a layer LIST, decoded per-prompt serially) BIT-FOR-BIT up
to that prompt's finish. The crossing-a-bucket-boundary case (a prompt finishing
mid-bucket while its bucket-mates keep decoding) is exactly what this exercises:
the finished prompts' PADDED rows (slot=-1, last-valid seq_len/position -- C-4)
must NOT corrupt the still-active prompts' attention, so the active tokens stay
identical to the per-prompt oracle.

EOS is INJECTED on a fixed schedule (eos_at[p]) rather than waiting for the model
to emit a real EOS, so the staggered pattern is deterministic and short. A prompt
marked finished at step eos_at[p] is dropped from the active set but KEPT IN ITS
SLOT (its KV block_table was baked into the graph) and pad-handled by C-4. The
oracle reproduces the SAME injected schedule (it just truncates each prompt's
clean-token list at eos_at[p]); the parity claim is that the active prompts'
clean tokens are unaffected by their bucket-mates finishing.

Usage (one free GPU among 1/2/3):
  cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
  PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=1 NP_KEEP_CUDA_VISIBLE=1 \
      /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_e2_staggered_eos.py \
      --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams


def _replay_staggered_on_worker(self, prompt_ids_list, layer_names, n_sample,
                                max_tokens, eos_at, sigma, global_seed,
                                sample_method):
    """Runs ON the worker. Captures ONE packed all-layer graph at bucket width
    B_pack = len(prompts), then drives a staggered-EOS replay decode, returning
    each prompt's clean-token list (truncated at its injected EOS step). Pure
    Python ints cross the RPC boundary (no GPU tensors)."""
    mr = self.model_runner
    model = mr.model
    device = mr.device
    sp = SamplingParams(temperature=0.0)

    b_pack = len(prompt_ids_list)
    st = self._ensure_np_state()
    st["sigma"] = float(sigma)

    # Prefill all B_pack prompts (each into its own disjoint scratch-KV slice).
    states = self._np_prefill_packed(model, device, prompt_ids_list)
    max_seq_len_cap = max(s["prompt_len"] for s in states) + int(max_tokens)

    # Capture the packed all-layer graph at the FIXED bucket width.
    gs = self._np_capture_step_packed(
        model, device, b_pack, n_sample, layer_names, states, max_seq_len_cap)

    np_cfg = dict(n_sample=n_sample, max_tokens=max_tokens,
                  global_seed=int(global_seed), sigma=float(sigma),
                  sample_method=sample_method)
    # slot p is bound to prompt p; rollout id = p (the oracle uses the same).
    slot_rollout_ids = list(range(b_pack))

    clean_tokens = [[] for _ in range(b_pack)]
    try:
        for t in range(max_tokens):
            active_idx = [p for p in range(b_pack) if states[p]["active"]]
            if not active_idx:
                break
            logits = self._np_replay_step_packed(
                model, states, active_idx, n_sample, layer_names, np_cfg, t,
                slot_rollout_ids, gs)  # [R, vocab]
            width = 1 + n_sample
            for p in range(b_pack):
                if not states[p]["active"]:
                    continue
                base = p * width
                block = logits[base:base + width]          # [1+N, vocab]
                next_tok = self._np_sample_clean(block[0], sp)
                clean_tokens[p].append(int(next_tok))
                # INJECTED staggered EOS: finish prompt p at its scheduled step.
                if len(clean_tokens[p]) >= int(eos_at[p]):
                    states[p]["active"] = False
                else:
                    self._np_commit_clean(states[p], next_tok)
    finally:
        st["mode"] = "off"
        for k in ("u_buf", "x_buf", "perturbed_row_idx", "clean_row_idx"):
            st[k] = None

    return {"clean_tokens": clean_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--layers", nargs="+", default=[
        "model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"])
    # Staggered EOS: prompt 0 finishes at 3, prompt 1 at 6, prompt 2 runs longest.
    ap.add_argument("--eos-at", nargs="+", type=int, default=[3, 6, 12])
    args = ap.parse_args()

    import verl.workers.rollout.vllm_rollout.np_worker_extension as npx
    print("np_worker_extension.__file__ =", npx.__file__)
    assert "np-alllayer-graphed" in npx.__file__, (
        "STALE verl loaded -- prepend PYTHONPATH=$PWD/verl. "
        f"got {npx.__file__}")

    prompts = [
        "What is 10/2? Answer:",
        "Differentiate x^3. Answer:",
        "Square root of 81? Answer:",
    ]
    b_pack = len(prompts)
    assert len(args.eos_at) == b_pack

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]
    llm.collective_rpc("install_perturb_layers", args=(args.layers,))

    np_cfg = dict(n_sample=args.n_sample, max_tokens=args.max_tokens,
                  global_seed=args.global_seed, sigma=args.sigma,
                  sample_method=args.sample_method)

    # ---- Oracle: per-prompt EAGER all-layer decode (run_np_decode with a LIST).
    # Same rollout id per prompt (p) so the noise matches the packed-graph seed.
    oracle = []
    for p, pid in enumerate(pids):
        o = llm.collective_rpc(
            "run_np_decode",
            args=(pid, SamplingParams(temperature=0.0), list(args.layers),
                  np_cfg, p))[0]
        oracle.append(o["clean_tokens"])

    # ---- Graphed packed replay with INJECTED staggered EOS.
    res = llm.collective_rpc(
        _replay_staggered_on_worker,
        args=(pids, list(args.layers), args.n_sample, args.max_tokens,
              args.eos_at, args.sigma, args.global_seed, args.sample_method))[0]
    graphed = res["clean_tokens"]

    print("\nEOS schedule (eos_at):", args.eos_at)
    ok = True
    for p in range(b_pack):
        n = int(args.eos_at[p])
        oracle_trunc = oracle[p][:n]
        got = graphed[p]
        match = (len(got) == len(oracle_trunc)) and (got == oracle_trunc)
        ok = ok and match
        first_div = next((i for i in range(min(len(got), len(oracle_trunc)))
                          if got[i] != oracle_trunc[i]), None)
        print(f"\nprompt {p}  (finishes at token {n})")
        print(f"  graphed: {got}")
        print(f"  oracle : {oracle_trunc}")
        print(f"  MATCH  : {match}"
              + ("" if first_div is None
                 else f"  (first divergence at idx {first_div})"))

    assert ok, ("STAGGERED-EOS PARITY FAILED: a prompt's graphed clean tokens "
                "diverged from the eager all-layer oracle -- finished prompts' "
                "padded rows likely corrupted an active prompt's attention (C-4).")
    print(f"\nPASS [E2 staggered-EOS]: {b_pack} prompts, staggered EOS "
          f"{args.eos_at}, graphed packed all-layer replay == eager all-layer "
          f"oracle BIT-FOR-BIT up to each prompt's EOS "
          f"(n_sample={args.n_sample}, layers={len(args.layers)}).")


if __name__ == "__main__":
    main()
