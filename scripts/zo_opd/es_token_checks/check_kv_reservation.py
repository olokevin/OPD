"""es_token scratch-KV gate: budget-sized reservation must be safe at any width.

The packed decode driver carves its own KV region out of the top of vLLM's block
pool, one disjoint slice per slot (`_np_prefill_packed`). That reservation used
to be sized at `max_model_len` -- ~20x what a 1024-token generation needs, which
capped pack_width at 9 slots. It is now sized at (longest prompt + max_tokens).

What this gate must check is NOT bit-equality with stock `llm.generate`. The
hand-driven packed forward batches differently from vLLM's scheduler, so bf16
rounding differs and greedy argmax flips on near-ties; that divergence is
prompt-dependent and present at the shipping pack_width=4 as well, where the
reservation change is provably output-neutral. Comparing to stock therefore
measures rounding, not KV safety.

The invariants that DO matter, and that a bad reservation would break:

  [A] output-neutrality -- for widths the OLD full reservation could also serve
      (<= 9 slots), the budget-sized carving must produce byte-identical clean
      tokens and payload. Checked ACROSS PROCESSES by check_kv_output_neutral.sh:
      flipping the reservation inside one process reuses the CUDA graph already
      captured for that bucket, so an in-process comparison is meaningless.
  [B] neighbour-independence -- a slot's output must not depend on what the
      OTHER slots contain. This is the direct test for slices aliasing each
      other, and it works at ANY width (including those the old reservation
      could not reach).
  [C] sufficiency -- every slot must decode the full max_tokens without
      tripping the slice-size assert.

  CUDA_VISIBLE_DEVICES=6 python scripts/zo_opd/es_token_checks/check_kv_reservation.py
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

DEFAULT_MODEL = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                 "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
WEXT = ("verl.workers.rollout.vllm_rollout."
        "es_token_worker_extension.WorkerExtension")
RULES = [r"^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)"
         r"|mlp\.(gate_up_proj|down_proj))$"]

PROBE = ["Q1. Compute 7*8 step by step.", "Q2. Differentiate x^3 and explain.",
         "Q3. What is 10/2? Show work.", "Q4. Square root of 81, explain."]
FILL_A = ["Explain why the sky is blue.", "Name three prime numbers.",
          "What is the capital of France?", "Define a derivative.",
          "State the Pythagorean theorem.", "What is 15% of 200?",
          "Convert 100 F to Celsius.", "List the first 5 squares.",
          "What is a matrix determinant?", "Explain modular arithmetic.",
          "Define a limit informally.", "What is Euler's number?"]
FILL_B = ["Describe photosynthesis briefly.", "What is an integer?",
          "Explain gravity in one line.", "What is a polynomial?",
          "Define the median of a list.", "What is 8 factorial?",
          "Convert 2 km to metres.", "List the first 5 cubes.",
          "What is a vector norm?", "Explain a geometric series.",
          "Define continuity informally.", "What is pi to 3 places?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pack-widths", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--gmu", type=float, default=0.55)
    args = ap.parse_args()

    llm = LLM(model=args.model, enforce_eager=True, enable_prefix_caching=True,
              worker_extension_cls=WEXT, dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=args.gmu)
    tok = llm.get_tokenizer()
    llm.collective_rpc("install_es_layers", args=(RULES, args.n_sample, 42))
    T = args.max_tokens

    def run(prompts, full_reserve=False, sigma=0.0):
        B = len(prompts)
        if full_reserve:
            os.environ["ES_KV_FULL_RESERVE"] = "1"
        else:
            os.environ.pop("ES_KV_FULL_RESERVE", None)
        pids = [tok(p)["input_ids"] for p in prompts]
        sp = SamplingParams(temperature=0.0, max_tokens=T)
        cfg = dict(n_sample=args.n_sample, max_tokens=T, global_seed=42,
                   sigma=sigma, sigma_mode="absolute", sample_method="bernoulli",
                   b_pack_buckets=[B], token_agg="mean")
        return llm.collective_rpc(
            "run_es_decode_packed",
            args=(pids, sp, cfg, list(range(B)), True))[0]

    def fill(pool, n):
        return [pool[i % len(pool)] + f" (v{i // len(pool)})" for i in range(n)]

    ok = True
    for B in args.pack_widths:
        pa = PROBE + fill(FILL_A, B - len(PROBE))
        pb = PROBE + fill(FILL_B, B - len(PROBE))
        try:
            oa = run(pa)
            ob = run(pb)
        except AssertionError as e:
            print(f"  pack_width={B:<3d} [C] REFUSED: {str(e)[:100]}")
            ok = False
            continue

        # [C] sufficiency
        lens = [len(t) for t in oa["clean_tokens"]]
        suff = min(lens) == T
        # [B] neighbour-independence on the shared probe slots
        indep = all(list(oa["clean_tokens"][p]) == list(ob["clean_tokens"][p])
                    for p in range(len(PROBE)))
        # [A] is NOT checked here: flipping the reservation inside one process
        # reuses the CUDA graph already captured for this bucket, which makes the
        # comparison meaningless. It is checked across separate processes by
        # check_kv_output_neutral.sh instead.
        ok &= suff and indep
        print(f"  pack_width={B:<3d} [C] all slots reached {T} tok: "
              f"{'PASS' if suff else f'FAIL min={min(lens)}'}   "
              f"[B] neighbour-independent: {'PASS' if indep else 'FAIL'}")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
