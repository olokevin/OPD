"""Phase 4 -- LM-head benchmark (opd_profile_plan.md §25).

Qwen3-1.7B lm_head [151936, 2048] bf16. For (B, R), M = B*R rows:
  full      cuBLAS [M, V] bf16 logits -> .float() -> logsumexp + gather
            (= es_token today, compute_logits + the payload block)
  indep     R independent LM heads of [B, d] (plan baseline B)
  chunked   cuBLAS in vocab chunks + online (max, sum) in torch; never a full
            [M, V] tensor but ~6 kernels per chunk
  stream    Triton streaming kernel (lm_head_kernel.py): clean logits for the
            B clean rows, LSE for all rows, never materialises [M, V]
Graphed timing (payload path included), bytes, correctness vs fp32 reference.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_profile/phase4_lm_head.py
"""
import argparse

import torch

from common import QWEN3_1P7B_ATTN, cuda_time_graphed, gpu_info, md_table, save_json
from verl.trainer.es_token.lm_head_kernel import (LMHeadWorkspace, lm_head_gather_logit,
                                                  lm_head_stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,4,8,16,32,64")
    ap.add_argument("--Rs", default="1,2,4,8,16,32")
    ap.add_argument("--max-M", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    dev = torch.device("cuda")
    V, d = QWEN3_1P7B_ATTN["vocab"], QWEN3_1P7B_ATTN["hidden"]
    Bs = [int(x) for x in args.Bs.split(",")]
    Rs = [int(x) for x in args.Rs.split(",")]
    print("[gpu]", gpu_info(), flush=True)
    W = (torch.randn(V, d, dtype=torch.bfloat16, device=dev) * 0.02)
    w_bytes = W.numel() * 2

    recs, checks = [], []
    for B in Bs:
        base = None
        for R in Rs:
            M = B * R
            if M > args.max_M:
                continue
            X = torch.randn(M, d, dtype=torch.bfloat16, device=dev)
            rows = torch.arange(M, device=dev)
            clean_rows = rows[rows % R == 0].contiguous()
            clean_idx = torch.full((M,), -1, dtype=torch.int32, device=dev)
            clean_idx[clean_rows] = torch.arange(B, dtype=torch.int32, device=dev)
            chosen_b = torch.randint(0, V, (B,), device=dev)
            chosen = chosen_b.repeat_interleave(R)
            ws = LMHeadWorkspace(M, V, B, 128, dev)
            outs = {}

            def f_full():
                logits = torch.matmul(X, W.t())                 # [M, V] bf16
                lf = logits.float()
                lse = torch.logsumexp(lf, dim=-1)
                clean = lf[clean_rows]                          # [B, V]
                tok = lf.gather(1, chosen[:, None])[:, 0]
                outs["full"] = (clean, tok - lse)

            def f_indep():
                Xb = X.view(B, R, d)
                lses, toks = [], []
                for r in range(R):
                    lg = torch.matmul(Xb[:, r], W.t()).float()
                    lses.append(torch.logsumexp(lg, -1))
                    toks.append(lg.gather(1, chosen_b[:, None])[:, 0])
                    if r == 0:
                        clean = lg
                lse = torch.stack(lses, 1).reshape(-1)
                tok = torch.stack(toks, 1).reshape(-1)
                outs["indep"] = (clean, tok - lse)

            clean_buf = torch.empty(B, V, dtype=torch.float32, device=dev)

            def f_chunked():
                m = torch.full((M,), float("-inf"), device=dev)
                s = torch.zeros(M, device=dev)
                for c0 in range(0, V, args.chunk):
                    c1 = min(c0 + args.chunk, V)
                    lg = torch.matmul(X, W[c0:c1].t()).float()   # [M, c]
                    clean_buf[:, c0:c1] = lg[clean_rows]
                    mc = lg.max(dim=1).values
                    mn = torch.maximum(m, mc)
                    s = s * torch.exp(m - mn) + torch.exp(lg - mn[:, None]).sum(1)
                    m = mn
                lse = m + torch.log(s)
                tok = lm_head_gather_logit(X, W, chosen)
                outs["chunked"] = (clean_buf, tok - lse)

            def f_stream():
                clean, lse, _ = lm_head_stream(X, W, clean_idx, ws=ws)
                tok = lm_head_gather_logit(X, W, chosen)
                outs["stream"] = (clean, tok - lse)

            impls = [("full", f_full), ("indep", f_indep), ("chunked", f_chunked), ("stream", f_stream)]
            rec = dict(B=B, R=R, M=M)
            for name, fn in impls:
                try:
                    mn, md = cuda_time_graphed(fn, warmup=2, iters=args.iters)
                except Exception as e:
                    print(f"[warn] B={B} R={R} {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                    mn = md = float("nan")
                rec[name] = mn
                rec[name + "_median"] = md
            if R == Rs[0]:
                base = rec["full"]
            for name, _ in impls:
                rec["rel_" + name] = rec[name] / base
            rec["bytes_full"] = w_bytes + M * V * (2 + 2 + 4 + 4)   # write bf16, read->fp32 write, read for lse
            rec["bytes_stream"] = w_bytes + B * V * 4 + M * ws.n_vt * 8
            rec["flops"] = 2.0 * M * d * V
            rec["tflops_stream"] = rec["flops"] / (rec["stream"] * 1e-3) / 1e12
            recs.append(rec)
            print(f"[B={B:2d} R={R:2d} M={M:4d}] " + "  ".join(f"{n}={rec[n]:.3f}" for n, _ in impls) +
                  f"  rel(full)={rec['rel_full']:.2f} rel(stream)={rec['rel_stream']:.2f}"
                  f"  stream {rec['tflops_stream']:.0f} TF", flush=True)
            # correctness vs fp32 reference
            with torch.no_grad():
                ref_logits = X.float() @ W.float().t()
                ref_lse = torch.logsumexp(ref_logits, -1)
                ref_clean = ref_logits[clean_rows]
                ref_tok = ref_logits.gather(1, chosen[:, None])[:, 0] - ref_lse
                for name, fn in impls:
                    fn()
                    torch.cuda.synchronize()
                    clean, logp = outs[name]
                    checks.append(dict(B=B, R=R, impl=name,
                                       clean_max_abs=(clean.float() - ref_clean).abs().max().item(),
                                       logp_max_abs=(logp - ref_tok).abs().max().item(),
                                       clean_argmax_match=bool((clean.float().argmax(1) == ref_clean.argmax(1)).all())))
            del X, ws, clean_buf
    save_json("phase4_lm_head.json", dict(gpu=gpu_info(), recs=recs, checks=checks))
    print("\n## correctness vs fp32 reference (worst over the sweep)")
    worst = {}
    for c in checks:
        w = worst.get(c["impl"])
        if w is None or c["logp_max_abs"] > w["logp_max_abs"]:
            worst[c["impl"]] = c
    print(md_table(["impl", "worst |d logp|", "worst |d clean logit|", "argmax match", "at (B,R)"],
                   [[n, w["logp_max_abs"], w["clean_max_abs"], all(c["clean_argmax_match"] for c in checks if c["impl"] == n),
                     f"({w['B']},{w['R']})"] for n, w in worst.items()],
                   fmt={"worst |d logp|": "{:.5f}", "worst |d clean logit|": "{:.5f}"}))
    print("\n## ms: full | chunked | stream   (rows B, cols R)")
    rows = []
    for B in Bs:
        rs = {r["R"]: r for r in recs if r["B"] == B}
        rows.append([B] + [f"{rs[R]['full']:.3f} / {rs[R]['chunked']:.3f} / {rs[R]['stream']:.3f}" if R in rs else "-"
                           for R in Rs])
    print(md_table(["B \\ R"] + [str(R) for R in Rs], rows))


if __name__ == "__main__":
    main()
