"""GPU smoke for Stage E1: capture an all-layer PACKED step forward into a CUDA
graph at a FIXED bucket width (_np_capture_step_packed).

E1 scope: prove the capture runs without error and returns a real
torch.cuda.CUDAGraph. Replay/parity (E2) and the orchestrator (E3) are NOT
exercised here. The C-2 buffer-dict pinning correctness is covered by the CPU
packed-scatter test + the E3 orchestrator parity gate.

Usage (one free GPU among 1/2/3):
  cd /home/yequan/Project/compression/OPD/.claude/worktrees/np-alllayer-graphed
  PYTHONPATH=$PWD/verl CUDA_VISIBLE_DEVICES=1 NP_KEEP_CUDA_VISIBLE=1 \
      /home/yequan/miniconda3/envs/verl/bin/python \
      scripts/zo_opd/np_checks/check_e1_capture.py \
      --model /data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from vllm import LLM


def _capture_on_worker(self, prompt_ids_list, layer_names, bucket_b_pack,
                       n_sample, max_tokens):
    """Runs ON the worker (collective_rpc callable; self == worker). Prefills
    bucket_b_pack prompts, calls _np_capture_step_packed, returns a small dict of
    facts the host asserts on (no GPU tensors crossing the RPC boundary)."""
    mr = self.model_runner
    model = mr.model
    device = mr.device

    # A captured-time sigma must be installed before capture (the forward reads
    # st["sigma"]). Use a nonzero sigma so the captured op is the real add.
    st = self._ensure_np_state()
    st["sigma"] = 0.01

    states = self._np_prefill_packed(model, device, prompt_ids_list)
    max_seq_len_cap = max(s["prompt_len"] for s in states) + int(max_tokens)

    gs = self._np_capture_step_packed(
        model, device, bucket_b_pack, n_sample, layer_names, states,
        max_seq_len_cap)

    R = bucket_b_pack * (1 + n_sample)
    facts = {
        "graph_type": type(gs["graph"]).__module__ + "." + type(gs["graph"]).__name__,
        "is_cuda_graph": isinstance(gs["graph"], torch.cuda.CUDAGraph),
        "input_ids_len": int(gs["input_ids_buf"].shape[0]),
        "R": R,
        "u_buf_layers": sorted(gs["u_buf"].keys()),
        "u_buf_shapes": {ln: tuple(gs["u_buf"][ln].shape) for ln in gs["u_buf"]},
        "x_buf_shapes": {ln: tuple(gs["x_buf"][ln].shape) for ln in gs["x_buf"]},
        "perturbed_row_idx": gs["perturbed_row_idx"].tolist(),
        "clean_row_idx": gs["clean_row_idx"].tolist(),
        # C-2: the dicts installed on st must be the SAME objects the graph holds.
        "u_buf_is_pinned": st["u_buf"] is gs["u_buf"],
        "x_buf_is_pinned": st["x_buf"] is gs["x_buf"],
        # slot_mapping perturbed rows must be PAD(-1) after the post-fix.
        "perturbed_slots": gs["meta_bufs"]["slot_mapping"][
            gs["perturbed_row_idx"]].tolist(),
        "ext_file": type(self).__module__,
    }
    # Sanity: replay once (in-place buffers already valid from capture) to prove
    # the captured graph at least runs without error (not a parity check).
    gs["graph"].replay()
    torch.cuda.synchronize()
    facts["replayed_ok"] = True
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bucket-b-pack", type=int, default=2)
    ap.add_argument("--n-sample", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--layers", nargs="+", default=[
        "model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"])
    args = ap.parse_args()

    # Verify the worktree verl is the one loaded (stale-import landmine).
    import verl.workers.rollout.vllm_rollout.np_worker_extension as npx
    print("np_worker_extension.__file__ =", npx.__file__)
    assert "np-alllayer-graphed" in npx.__file__, (
        "STALE verl loaded -- prepend PYTHONPATH=$PWD/verl. "
        f"got {npx.__file__}")

    prompts = [
        "Compute 7*8. Answer:", "Differentiate x^3. Answer:",
        "What is 10/2? Answer:", "Square root of 81? Answer:",
    ][: args.bucket_b_pack]

    wext = "verl.workers.rollout.vllm_rollout.np_worker_extension.WorkerExtension"
    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=wext, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tok = llm.get_tokenizer()
    pids = [tok(p)["input_ids"] for p in prompts]
    llm.collective_rpc("install_perturb_layers", args=(args.layers,))

    facts = llm.collective_rpc(
        _capture_on_worker,
        args=(pids, args.layers, args.bucket_b_pack, args.n_sample,
              args.max_tokens))[0]

    print("worker ext module:", facts["ext_file"])
    print("graph_type       :", facts["graph_type"])
    print("R / input_ids_len:", facts["R"], "/", facts["input_ids_len"])
    print("u_buf layers     :", facts["u_buf_layers"])
    print("u_buf shapes     :", facts["u_buf_shapes"])
    print("x_buf shapes     :", facts["x_buf_shapes"])
    print("clean_row_idx    :", facts["clean_row_idx"])
    print("perturbed_row_idx:", facts["perturbed_row_idx"])
    print("perturbed slots  :", facts["perturbed_slots"], "(expect all -1)")
    print("C-2 u_buf pinned :", facts["u_buf_is_pinned"])
    print("C-2 x_buf pinned :", facts["x_buf_is_pinned"])
    print("replayed_ok      :", facts["replayed_ok"])

    assert facts["is_cuda_graph"], (
        f"graph is not a torch.cuda.CUDAGraph (got {facts['graph_type']})")
    assert facts["input_ids_len"] == facts["R"], "input buffer width != R"
    assert facts["u_buf_is_pinned"] and facts["x_buf_is_pinned"], (
        "C-2 VIOLATION: st buffer dicts are NOT the graph-pinned objects")
    assert all(s == -1 for s in facts["perturbed_slots"]), (
        "perturbed-row slots not PAD(-1) after post-fix")
    assert facts["replayed_ok"]
    print("PASS [E1 capture]: _np_capture_step_packed returned a CUDAGraph "
          f"(bucket_b_pack={args.bucket_b_pack}, n_sample={args.n_sample}, "
          f"layers={len(args.layers)}), C-2 buffers pinned, perturbed slots PAD.")


if __name__ == "__main__":
    main()
