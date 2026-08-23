"""Dump packed clean tokens + payload for a fixed wave, one process, one setting."""
import os, sys, pickle
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
B, T, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
M = ("/data/yequan/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/"
     "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
WEXT = "verl.workers.rollout.vllm_rollout.es_token_worker_extension.WorkerExtension"
RULES = [r"^model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))$"]
PROBE = ["Q1. Compute 7*8 step by step.", "Q2. Differentiate x^3 and explain.",
         "Q3. What is 10/2? Show work.", "Q4. Square root of 81, explain."]
FILL = ["Explain why the sky is blue.", "Name three prime numbers.",
        "What is the capital of France?", "Define a derivative.",
        "State the Pythagorean theorem.", "What is 15% of 200?",
        "Convert 100 F to Celsius.", "List the first 5 squares.",
        "What is a matrix determinant?", "Explain modular arithmetic.",
        "Define a limit informally.", "What is Euler's number?"]
prompts = (PROBE + [FILL[i % len(FILL)] + f" (v{i//len(FILL)})"
                    for i in range(B - len(PROBE))])[:B]
llm = LLM(model=M, enforce_eager=True, enable_prefix_caching=True,
          worker_extension_cls=WEXT, dtype="bfloat16", tensor_parallel_size=1,
          gpu_memory_utilization=0.55)
tok = llm.get_tokenizer()
llm.collective_rpc("install_es_layers", args=(RULES, 8, 42))
pids = [tok(p)["input_ids"] for p in prompts]
sp = SamplingParams(temperature=0.0, max_tokens=T)
cfg = dict(n_sample=8, max_tokens=T, global_seed=42, sigma=0.01,
           sigma_mode="absolute", sample_method="bernoulli",
           b_pack_buckets=[B], token_agg="mean")
o = llm.collective_rpc("run_es_decode_packed",
                       args=(pids, sp, cfg, list(range(B)), True))[0]
pickle.dump({"tok": [list(t) for t in o["clean_tokens"]],
             "pay": [p.clone() for p in o["payload"]]}, open(out_path, "wb"))
print("dumped", out_path, "full_reserve=", bool(os.environ.get("ES_KV_FULL_RESERVE")))
