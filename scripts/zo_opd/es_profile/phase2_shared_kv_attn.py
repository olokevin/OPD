"""Phase 2 -- shared-KV attention microbenchmark (opd_profile_plan.md §23).

Qwen3-1.7B attention (16 q heads, 8 kv heads, D=128, 16-token pages). For
each (clean batch B, total rails R, context L):
  rows    every rail row is its own FA3 request (= es_token today; KV re-read
          once per rail)
  fold    rails folded into the head axis, stock FA3 (and FA2) with GQA packing
  shared  the Triton shared-KV rail kernel (rail_attn_kernel.py)
All graphed; min/median ms; unique-KV GB/s; T(B,R)/T(B,1); correctness vs an
fp32 torch reference.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_profile/phase2_shared_kv_attn.py
"""
import argparse
import math

import torch

from common import QWEN3_1P7B_ATTN, cuda_time_graphed, gpu_info, md_table, save_json
from verl.trainer.es_token.rail_attn_kernel import (
    RailAttnWorkspace, pick_num_splits, rail_attention_fold, rail_attention_reference,
    rail_attention_rows, rail_attention_shared)


def build_cache(B, L, n_kv, D, bs, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    bps = math.ceil(L / bs)
    nblk = B * bps + 8
    k = torch.randn(nblk, bs, n_kv, D, dtype=torch.bfloat16, device=dev, generator=g)
    v = torch.randn(nblk, bs, n_kv, D, dtype=torch.bfloat16, device=dev, generator=g)
    perm = torch.randperm(nblk - 1, device=dev, generator=g)[: B * bps] + 1
    bt = perm.view(B, bps).to(torch.int32).contiguous()
    sl = torch.full((B,), L, dtype=torch.int32, device=dev)
    return k, v, bt, sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,4,8,16,64")
    ap.add_argument("--Rs", default="1,2,4,8,16,24,32")
    ap.add_argument("--Ls", default="128,512,2048,8192,32768")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-kv-gb", type=float, default=16.0)
    ap.add_argument("--check-max-L", type=int, default=2048)
    args = ap.parse_args()
    dev = torch.device("cuda")
    a = QWEN3_1P7B_ATTN
    n_q, n_kv, D, bs = a["n_q"], a["n_kv"], a["head_dim"], 16
    G = n_q // n_kv
    scale = D ** -0.5
    Bs = [int(x) for x in args.Bs.split(",")]
    Rs = [int(x) for x in args.Rs.split(",")]
    Ls = [int(x) for x in args.Ls.split(",")]
    print("[gpu]", gpu_info(), flush=True)

    recs, checks = [], []
    for L in Ls:
        for B in Bs:
            kv_gb = B * L * n_kv * D * 2 * 2 / 2**30
            if kv_gb > args.max_kv_gb:
                print(f"[skip] B={B} L={L}: KV {kv_gb:.1f} GB > cap", flush=True)
                continue
            k, v, bt, sl = build_cache(B, L, n_kv, D, bs, dev)
            base = {}
            for R in Rs:
                q = torch.randn(B, R, n_q, D, dtype=torch.bfloat16, device=dev)
                out = torch.empty_like(q)
                bt_rows = bt.repeat_interleave(R, dim=0).contiguous()
                sl_rows = sl.repeat_interleave(R).contiguous()
                cu_rows = torch.arange(B * R + 1, dtype=torch.int32, device=dev)
                cu_b = torch.arange(B + 1, dtype=torch.int32, device=dev)
                qf = torch.empty(B, n_kv * R * G, D, dtype=torch.bfloat16, device=dev)
                of = torch.empty_like(qf)
                ns = pick_num_splits(B, n_kv)
                ws = RailAttnWorkspace(B, n_kv, R, G, ns, D, dev)

                impls = {
                    "rows": lambda: rail_attention_rows(q, k, v, bt_rows, sl_rows, scale, L, out=out,
                                                        cu_seqlens_q=cu_rows),
                    "fold": lambda: rail_attention_fold(q, k, v, bt, sl, scale, L, out=out, fa_version=3,
                                                        cu_seqlens_q=cu_b, q_fold_buf=qf, o_fold_buf=of),
                    "fold_fa2": lambda: rail_attention_fold(q, k, v, bt, sl, scale, L, out=out, fa_version=2,
                                                            cu_seqlens_q=cu_b, q_fold_buf=qf, o_fold_buf=of),
                    "shared": lambda: rail_attention_shared(q, k, v, bt, sl, scale, out=out, ws=ws,
                                                            num_splits=ns),
                }
                rec = dict(B=B, R=R, L=L, M=B * R, num_splits=ns, block_rg=ws.block_rg)
                for name, fn in impls.items():
                    try:
                        mn, md = cuda_time_graphed(fn, warmup=3, iters=args.iters)
                    except Exception as e:
                        print(f"[warn] B={B} R={R} L={L} {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                        mn = md = float("nan")
                    rec[name] = mn
                    rec[name + "_median"] = md
                    if R == Rs[0]:
                        base[name] = mn
                    rec["rel_" + name] = mn / base[name]
                uniq = B * L * n_kv * D * 2 * 2
                rec["kv_bytes_unique"] = uniq
                rec["kv_bytes_rows"] = uniq * R
                for name in impls:
                    rec[f"gbps_unique_{name}"] = uniq / (rec[name] * 1e-3) / 1e9
                rec["gbps_rows_actual"] = uniq * R / (rec["rows"] * 1e-3) / 1e9
                recs.append(rec)
                print(f"[L={L:5d} B={B:2d} R={R:2d}] " + "  ".join(
                    f"{n}={rec[n]:.4f}({rec['rel_'+n]:.2f}x)" for n in impls) +
                    f"  splits={ns}", flush=True)

                if L <= args.check_max_L and B <= 8:
                    ref = rail_attention_reference(q, k, v, bt, sl, scale)
                    for name, fn in impls.items():
                        out.zero_()
                        fn()
                        torch.cuda.synchronize()
                        d = (out.float() - ref.float()).abs()
                        checks.append(dict(B=B, R=R, L=L, impl=name, max_abs=d.max().item(),
                                           mean_abs=d.mean().item(),
                                           ref_max=ref.float().abs().max().item()))
                del q, out, bt_rows, sl_rows, qf, of, ws
            del k, v
            torch.cuda.empty_cache()

    save_json("phase2_shared_kv_attn.json", dict(gpu=gpu_info(), recs=recs, checks=checks,
                                                 Bs=Bs, Rs=Rs, Ls=Ls))
    print("\n## correctness vs fp32 reference (bf16 outputs)")
    worst = {}
    for c in checks:
        w = worst.get(c["impl"])
        if w is None or c["max_abs"] > w["max_abs"]:
            worst[c["impl"]] = c
    print(md_table(["impl", "worst max_abs", "mean_abs", "at (B,R,L)", "ref_max"],
                   [[n, w["max_abs"], w["mean_abs"], f"({w['B']},{w['R']},{w['L']})", w["ref_max"]]
                    for n, w in worst.items()],
                   fmt={"worst max_abs": "{:.4f}", "mean_abs": "{:.6f}", "ref_max": "{:.2f}"}))
    for L in Ls:
        print(f"\n## L={L}: T(B,R)/T(B,1) -- rows | fold(FA3) | shared(Triton)")
        rows = []
        for B in Bs:
            rs = {r["R"]: r for r in recs if r["B"] == B and r["L"] == L}
            if not rs:
                continue
            rows.append([B] + [f"{rs[R]['rel_rows']:.2f} / {rs[R]['rel_fold']:.2f} / {rs[R]['rel_shared']:.2f}"
                               for R in Rs])
        print(md_table(["B \\ R"] + [str(R) for R in Rs], rows))
        print(f"\n## L={L}: absolute ms at R=1 / R=8 / R=32 (rows | fold | shared)")
        rows = []
        for B in Bs:
            rs = {r["R"]: r for r in recs if r["B"] == B and r["L"] == L}
            if not rs:
                continue
            rows.append([B] + [f"{rs[R]['rows']:.3f} / {rs[R]['fold']:.3f} / {rs[R]['shared']:.3f}"
                               for R in Rs if R in rs and R in (1, 8, 32)])
        print(md_table(["B", "R=1", "R=8", "R=32"], rows))


if __name__ == "__main__":
    main()
