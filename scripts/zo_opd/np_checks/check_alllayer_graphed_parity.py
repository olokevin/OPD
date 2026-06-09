"""Task F1: consolidated GPU parity gate for the fully-CUDA-graphed packed
all-layer decode (`run_np_decode_packed_graphed`).

Proves the graphed path equals the eager all-layer oracle on THREE axes, in one
self-contained script:

  Gate (a) -- sigma=0 routing. Run the graphed packed decode with sigma=0 and
    assert each prompt's CLEAN tokens equal a plain greedy decode (the eager
    all-layer `run_np_decode` list-mode at sigma=0, itself the validated
    oracle). sigma=0 makes the perturbation a no-op (0*u=0) and makes seeding
    irrelevant, so the graphed clean tokens must be the model's plain greedy.

  Gate (b) -- graphed vs eager all-layer, per-layer u BIT-IDENTITY + logits tol.
    sigma>0, MATCHING rollout_ids on both sides (the packed family seeds noise
    via rollout_ids[p]; the eager list-mode oracle seeds with the rollout_idx
    arg -- so we feed the eager oracle prompt p the SAME rollout_ids[p]). Assert
    every (prompt, layer, token) captured `u` is torch.equal between graphed and
    eager (same seed key -> bit-identical by construction), and the stored top-k
    clean-row logprobs agree within rtol=1e-2 on their common ids (graph vs eager
    fp differences are tiny but nonzero; a tiny top-k boundary reshuffle is
    tolerated by comparing on the id intersection).

  Gate (c) -- staggered-EOS bucket-crossing parity. 3 prompts with staggered
    INJECTED EOS, bucket crossing: a prompt finishing mid-bucket must NOT corrupt
    its still-active bucket-mates' attention (C-4 padded rows). Assert each
    prompt's graphed clean tokens match the eager all-layer oracle BIT-FOR-BIT up
    to its EOS step (matching rollout_ids). Self-contained replica of
    check_e2_staggered_eos.py's deep test.

Reuses the E1/E2/E3 worker-boot boilerplate and oracle patterns. Exits 0 only if
all three gates PASS; nonzero on any fail.

Usage (one free GPU among 1/2/3):
  cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
  PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=1 NP_KEEP_CUDA_VISIBLE=1 \
      /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_alllayer_graphed_parity.py \
      --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""
import argparse
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM, SamplingParams

DEFAULT_MODEL = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                 "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


# --------------------------------------------------------------------------- #
# Gate (a) + (b): graphed packed all-layer decode vs eager all-layer oracle.
# Both run ON the worker; only python-native facts cross the RPC boundary (the
# u bit-identity + logits-tol comparisons are reduced to ints/floats/bools here,
# so no GPU tensors are shipped back to the host).
# --------------------------------------------------------------------------- #
def _gate_ab_on_worker(self, prompt_ids_list, layer_names, n_sample, max_tokens,
                       sigma, global_seed, sample_method, rollout_ids,
                       logits_rtol):
    import torch
    sp = SamplingParams(temperature=0.0)
    layer_names = list(layer_names)
    B = len(prompt_ids_list)

    def _np_cfg(sig):
        return dict(n_sample=int(n_sample), max_tokens=int(max_tokens),
                    global_seed=int(global_seed), sigma=float(sig),
                    sample_method=sample_method)

    out = {}

    # ---- Gate (a): sigma=0 routing. Graphed clean tokens must equal the plain
    # greedy oracle (eager all-layer list-mode at sigma=0). sigma=0 -> seeding
    # irrelevant, so any rollout id works; reuse rollout_ids for symmetry.
    cfg0 = _np_cfg(0.0)
    g0 = self.run_np_decode_packed_graphed(
        list(prompt_ids_list), sp, layer_names, cfg0, list(rollout_ids))
    greedy0 = []
    for p, pid in enumerate(prompt_ids_list):
        o = self.run_np_decode(pid, SamplingParams(temperature=0.0),
                               list(layer_names), cfg0, int(rollout_ids[p]))
        greedy0.append(list(o["clean_tokens"]))
    out["a_graphed_clean"] = [list(g0["clean_tokens"][p]) for p in range(B)]
    out["a_greedy_clean"] = greedy0
    out["a_match"] = [list(g0["clean_tokens"][p]) == greedy0[p]
                      for p in range(B)]

    # ---- Gate (b): sigma>0, MATCHING rollout_ids. u bit-identity + logits tol.
    cfg = _np_cfg(sigma)
    gg = self.run_np_decode_packed_graphed(
        list(prompt_ids_list), sp, layer_names, cfg, list(rollout_ids))
    # Eager oracle per prompt with the SAME rollout id (seed parity).
    eager = []
    for p, pid in enumerate(prompt_ids_list):
        o = self.run_np_decode(pid, SamplingParams(temperature=0.0),
                               list(layer_names), cfg, int(rollout_ids[p]))
        eager.append(o)

    g_clean = [list(gg["clean_tokens"][p]) for p in range(B)]
    e_clean = [list(eager[p]["clean_tokens"]) for p in range(B)]
    out["b_clean_match"] = [g_clean[p] == e_clean[p] for p in range(B)]
    out["b_graphed_clean"] = g_clean
    out["b_eager_clean"] = e_clean

    # u bit-identity over every (prompt, layer, token). gg.captured_u[p][ln][t]
    # and eager[p].captured_u[ln][t] are both CPU [n_sample, d_out] clones.
    u_total = 0
    u_equal = 0
    u_first_mismatch = None  # (p, ln, t) of the first torch.equal failure.
    g_cu = gg["captured_u"]
    for p in range(B):
        e_cu = eager[p]["captured_u"]
        n_tok = len(g_clean[p])
        for ln in layer_names:
            for t in range(n_tok):
                gut = g_cu[p][ln].get(t)
                eut = e_cu[ln].get(t)
                u_total += 1
                if (gut is not None and eut is not None
                        and tuple(gut.shape) == tuple(eut.shape)
                        and torch.equal(gut, eut)):
                    u_equal += 1
                elif u_first_mismatch is None:
                    u_first_mismatch = (p, ln, t)
    out["u_total"] = u_total
    out["u_equal"] = u_equal
    out["u_first_mismatch"] = u_first_mismatch

    # logits tol: compare the stored top-k CLEAN-row logprobs (row 0) on their
    # COMMON ids (robust to a tiny boundary reshuffle near the top-k cutoff).
    # _topk_store -> (logp[1+N, k] cpu, ids[k] cpu); row 0 is the clean/student
    # row whose ids define the gather.
    max_rtol = 0.0
    cmp_count = 0
    worst = None  # (p, t, rtol, n_common)
    g_cand = gg["candidate_logits"]
    for p in range(B):
        e_cand = eager[p]["candidate_logits"]
        n_tok = min(len(g_cand[p]), len(e_cand))
        for t in range(n_tok):
            g_lp, g_ids = g_cand[p][t]
            e_lp, e_ids = e_cand[t]
            g_ids_l = g_ids.tolist()
            e_ids_l = e_ids.tolist()
            g_pos = {int(i): j for j, i in enumerate(g_ids_l)}
            e_pos = {int(i): j for j, i in enumerate(e_ids_l)}
            common = [i for i in g_ids_l if i in e_pos]
            if not common:
                continue
            g_vals = g_lp[0][[g_pos[i] for i in common]].float()
            e_vals = e_lp[0][[e_pos[i] for i in common]].float()
            denom = e_vals.abs().clamp_min(1e-8)
            rtol = (g_vals - e_vals).abs() / denom
            r = float(rtol.max().item())
            cmp_count += 1
            if r > max_rtol:
                max_rtol = r
                worst = (p, t, r, len(common))
    out["logits_max_rtol"] = max_rtol
    out["logits_cmp_count"] = cmp_count
    out["logits_worst"] = worst
    out["logits_rtol_thresh"] = float(logits_rtol)
    out["B"] = B
    out["n_sample"] = int(n_sample)
    out["layers"] = list(layer_names)
    out["ext_file"] = type(self).__module__
    return out


# --------------------------------------------------------------------------- #
# Gate (c): staggered-EOS bucket-crossing parity (self-contained replica of
# check_e2_staggered_eos's deep test).
# --------------------------------------------------------------------------- #
def _gate_c_replay_on_worker(self, prompt_ids_list, layer_names, n_sample,
                             max_tokens, eos_at, sigma, global_seed,
                             sample_method, rollout_ids):
    mr = self.model_runner
    model = mr.model
    device = mr.device
    sp = SamplingParams(temperature=0.0)

    b_pack = len(prompt_ids_list)
    st = self._ensure_np_state()
    st["sigma"] = float(sigma)

    states = self._np_prefill_packed(model, device, prompt_ids_list)
    max_seq_len_cap = max(s["prompt_len"] for s in states) + int(max_tokens)

    gs = self._np_capture_step_packed(
        model, device, b_pack, n_sample, list(layer_names), states,
        max_seq_len_cap)

    np_cfg = dict(n_sample=int(n_sample), max_tokens=int(max_tokens),
                  global_seed=int(global_seed), sigma=float(sigma),
                  sample_method=sample_method)
    # slot p bound to prompt p; rollout id = rollout_ids[p] (oracle uses same).
    slot_rollout_ids = [int(r) for r in rollout_ids]

    clean_tokens = [[] for _ in range(b_pack)]
    try:
        for t in range(int(max_tokens)):
            active_idx = [p for p in range(b_pack) if states[p]["active"]]
            if not active_idx:
                break
            logits = self._np_replay_step_packed(
                model, states, active_idx, n_sample, list(layer_names), np_cfg,
                t, slot_rollout_ids, gs)
            width = 1 + n_sample
            for p in range(b_pack):
                if not states[p]["active"]:
                    continue
                base = p * width
                block = logits[base:base + width]
                next_tok = self._np_sample_clean(block[0], sp)
                clean_tokens[p].append(int(next_tok))
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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--sample-method", default="bernoulli")
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--logits-rtol", type=float, default=1e-2)
    ap.add_argument("--layers", nargs="+", default=[
        "model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"])
    # Gate (c) staggered EOS: prompt 0 finishes early, 1 mid, 2 runs longest.
    ap.add_argument("--c-eos-at", nargs="+", type=int, default=[3, 6, 12])
    ap.add_argument("--c-max-tokens", type=int, default=12)
    args = ap.parse_args()

    # Verify the worktree verl is the one loaded (stale-import landmine).
    import verl.workers.rollout.vllm_rollout.np_worker_extension as npx
    print("np_worker_extension.__file__ =", npx.__file__)
    assert "np-alllayer-graphed" in npx.__file__, (
        "STALE verl loaded -- prepend PYTHONPATH=$PWD/verl. "
        f"got {npx.__file__}")

    # Gate (a)/(b) prompts: B=4 -> bucket 4 (exact fit). Gate (c): 3 prompts.
    ab_prompts = [
        "Compute 7*8. Answer:",
        "Differentiate x^3. Answer:",
        "What is 10/2? Answer:",
        "Square root of 81? Answer:",
    ]
    c_prompts = [
        "What is 10/2? Answer:",
        "Differentiate x^3. Answer:",
        "Square root of 81? Answer:",
    ]
    assert len(args.c_eos_at) == len(c_prompts)

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    llm.collective_rpc("install_perturb_layers", args=(list(args.layers),))

    ab_pids = [tok(p)["input_ids"] for p in ab_prompts]
    c_pids = [tok(p)["input_ids"] for p in c_prompts]

    # rollout id = prompt index, MATCHED on graphed + eager sides (seed parity).
    ab_rollout_ids = list(range(len(ab_pids)))
    c_rollout_ids = list(range(len(c_pids)))

    # ==================== Gate (a) + (b) ====================
    ab = llm.collective_rpc(
        _gate_ab_on_worker,
        args=(ab_pids, list(args.layers), args.n_sample, args.max_tokens,
              args.sigma, args.global_seed, args.sample_method, ab_rollout_ids,
              args.logits_rtol))[0]
    B = ab["B"]

    print("\nworker ext module :", ab["ext_file"])
    print("layers            :", ab["layers"])

    # ---- Gate (a) report ----
    print("\n=== Gate (a) sigma=0 routing (graphed clean == plain greedy) ===")
    for p in range(B):
        g = ab["a_graphed_clean"][p]
        gr = ab["a_greedy_clean"][p]
        fd = next((i for i in range(min(len(g), len(gr))) if g[i] != gr[i]),
                  None)
        print(f"  prompt {p}: match={ab['a_match'][p]}"
              + ("" if fd is None else f"  (first div idx {fd})"))
        if not ab["a_match"][p]:
            print(f"    graphed: {g}")
            print(f"    greedy : {gr}")
    gate_a_ok = all(ab["a_match"])

    # ---- Gate (b) report ----
    print("\n=== Gate (b) graphed vs eager all-layer: u bit-identity + logits "
          "tol ===")
    print(f"  clean-token match per prompt : {ab['b_clean_match']}")
    print(f"  u tensors compared           : {ab['u_equal']}/{ab['u_total']} "
          f"torch.equal")
    if ab["u_first_mismatch"] is not None:
        p, ln, t = ab["u_first_mismatch"]
        print(f"  FIRST u MISMATCH             : prompt={p} layer={ln} token={t}")
    print(f"  logits compared (clean rows) : {ab['logits_cmp_count']} tokens")
    print(f"  logits max rtol              : {ab['logits_max_rtol']:.3e} "
          f"(thresh {ab['logits_rtol_thresh']:.1e})")
    if ab["logits_worst"] is not None:
        p, t, r, nc = ab["logits_worst"]
        print(f"  worst-rtol token             : prompt={p} t={t} rtol={r:.3e} "
              f"(on {nc} common ids)")
    u_ok = (ab["u_equal"] == ab["u_total"]) and ab["u_total"] > 0
    clean_b_ok = all(ab["b_clean_match"])
    logits_ok = ab["logits_max_rtol"] <= ab["logits_rtol_thresh"]
    gate_b_ok = u_ok and clean_b_ok and logits_ok

    # ==================== Gate (c) ====================
    c_cfg = dict(n_sample=args.n_sample, max_tokens=args.c_max_tokens,
                 global_seed=args.global_seed, sigma=args.sigma,
                 sample_method=args.sample_method)
    # Eager all-layer oracle per prompt, SAME rollout id as the graphed side.
    c_oracle = []
    for p, pid in enumerate(c_pids):
        o = llm.collective_rpc(
            "run_np_decode",
            args=(pid, SamplingParams(temperature=0.0), list(args.layers),
                  c_cfg, c_rollout_ids[p]))[0]
        c_oracle.append(list(o["clean_tokens"]))
    cres = llm.collective_rpc(
        _gate_c_replay_on_worker,
        args=(c_pids, list(args.layers), args.n_sample, args.c_max_tokens,
              list(args.c_eos_at), args.sigma, args.global_seed,
              args.sample_method, c_rollout_ids))[0]
    c_graphed = cres["clean_tokens"]

    print("\n=== Gate (c) staggered-EOS bucket-crossing parity ===")
    print(f"  EOS schedule (eos_at): {args.c_eos_at}")
    gate_c_ok = True
    for p in range(len(c_pids)):
        n = int(args.c_eos_at[p])
        oracle_trunc = c_oracle[p][:n]
        got = c_graphed[p]
        match = (len(got) == len(oracle_trunc)) and (got == oracle_trunc)
        gate_c_ok = gate_c_ok and match
        fd = next((i for i in range(min(len(got), len(oracle_trunc)))
                   if got[i] != oracle_trunc[i]), None)
        print(f"  prompt {p} (finishes at token {n}): match={match}"
              + ("" if fd is None else f"  (first div idx {fd})"))
        if not match:
            print(f"    graphed: {got}")
            print(f"    oracle : {oracle_trunc}")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    if gate_a_ok:
        print(f"PASS [Gate a] sigma=0 routing: graphed clean tokens == plain "
              f"greedy for all {B} prompts (perturbation is a no-op at sigma=0).")
    else:
        print("FAIL [Gate a] sigma=0 routing: a prompt's graphed clean tokens "
              "diverged from plain greedy.")
    if gate_b_ok:
        print(f"PASS [Gate b] graphed vs eager all-layer: u BIT-IDENTICAL "
              f"({ab['u_equal']}/{ab['u_total']} torch.equal), clean tokens "
              f"match, logits max rtol {ab['logits_max_rtol']:.3e} <= "
              f"{ab['logits_rtol_thresh']:.1e}.")
    else:
        why = []
        if not u_ok:
            why.append(f"u {ab['u_equal']}/{ab['u_total']} (check rollout_id "
                       f"alignment FIRST)")
        if not clean_b_ok:
            why.append("clean-token mismatch")
        if not logits_ok:
            why.append(f"logits rtol {ab['logits_max_rtol']:.3e} > "
                       f"{ab['logits_rtol_thresh']:.1e}")
        print("FAIL [Gate b]: " + "; ".join(why))
    if gate_c_ok:
        print(f"PASS [Gate c] staggered-EOS: {len(c_pids)} prompts, EOS "
              f"{args.c_eos_at}, graphed packed all-layer replay == eager "
              f"all-layer oracle BIT-FOR-BIT up to each prompt's EOS.")
    else:
        print("FAIL [Gate c] staggered-EOS: a prompt's graphed clean tokens "
              "diverged from the eager all-layer oracle (C-4 padded-row "
              "corruption of an active prompt's attention).")

    all_ok = gate_a_ok and gate_b_ok and gate_c_ok
    print("=" * 60)
    print("ALL GATES PASS" if all_ok else "SOME GATES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
