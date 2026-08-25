"""Where do BP-OPD's 32.48 s of compute_rm_score go?

Replays ONE reward-model micro-batch (8 seqs x 1112 tok, the shipped
reward.micro_batch_size_per_gpu=8) on the real teacher and times the transformer
forward separately from each full-vocab post-processing stage that
RewardModelWorker._forward_micro_batch runs on the logits.
"""
import time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL = "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"
B, T, K = 8, 1112, 16          # micro_batch_size_per_gpu=8, ~71127/64 tok/seq, log_prob_top_k=16
NMB = 8                        # 64 seqs / 8 = 8 micro-batches per step

# --- verbatim from verl/workers/fsdp_workers.py (neither touches self) -------
def _compute_entropy_safe(logits, chunk_size=4096):
    vocab = logits.shape[-1]
    flat = logits.view(-1, vocab)
    out = []
    for i in range(0, flat.size(0), chunk_size):
        c = flat[i:i+chunk_size]
        lp = F.log_softmax(c, dim=-1)
        p = torch.exp(lp)
        out.append(-torch.sum(p * lp, dim=-1))
    return torch.cat(out, 0).view(logits.shape[:-1])

def _teacher_top_k(logits, student_ids, top_k, chunk_size=1024):
    res, tid, tlp, ov, tis = [], [], [], [], []
    for s in range(0, logits.size(0), chunk_size):
        e = min(s+chunk_size, logits.size(0))
        lc, sc = logits[s:e], student_ids[s:e]
        t_logits, t_ids = torch.topk(lc, k=top_k, dim=-1)
        lse = torch.logsumexp(lc, dim=-1, keepdim=True)
        tlp.append(t_logits - lse); tid.append(t_ids)
        m = (sc.unsqueeze(-1) == t_ids.unsqueeze(-2))
        ov.append(m.any(-1).float()); tis.append(m.any(-2).float())
        res.append(torch.gather(lc, -1, sc) - lse)
    return [torch.cat(x, 0) for x in (res, ov, tid, tlp, tis)]
# ----------------------------------------------------------------------------

def tick():
    torch.cuda.synchronize(); return time.perf_counter()

dev = "cuda"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
V = m.config.vocab_size
ids = torch.randint(0, V, (B, T), device=dev)
sids = torch.randint(0, V, (B*T, K), device=dev)
labels = torch.randint(0, V, (B*T,), device=dev)
from verl.utils.torch_functional import logprobs_from_logits

rows = []
for it in range(2):                      # iter 0 warms up, iter 1 is reported
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = tick(); logits = m(input_ids=ids, use_cache=False).logits.view(B*T, V); t1 = tick()
        logits.div_(1.0);                                             t2 = tick()
        ent = _compute_entropy_safe(logits);                          t3 = tick()
        lp  = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=True); t4 = tick()
        out = _teacher_top_k(logits, sids, K);                        t5 = tick()
    rows = [("transformer fwd + lm_head", t1-t0), ("logits.div_(T)", t2-t1),
            ("_compute_entropy_safe (logging only)", t3-t2),
            ("logprobs_from_logits", t4-t3), ("teacher top-K + overlap", t5-t4)]
    del logits, ent, lp, out; torch.cuda.empty_cache()

tot = sum(v for _, v in rows)
print(f"\nlogits tensor per micro-batch: {B*T} x {V} bf16 = {B*T*V*2/2**30:.2f} GiB\n")
print(f"{'stage':<40}{'ms/micro-batch':>16}{'s/step (x8)':>14}{'share':>8}")
for k, v in rows:
    print(f"{k:<40}{v*1e3:>16.1f}{v*NMB:>14.2f}{v/tot*100:>7.1f}%")
print(f"{'TOTAL':<40}{tot*1e3:>16.1f}{tot*NMB:>14.2f}{100.0:>7.1f}%")
print(f"\nmeasured compute_rm_score in the BP run: 32.48 s")
