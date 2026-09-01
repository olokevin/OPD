"""Phase 1 -- linear rail microbenchmark (opd_profile_plan.md §22).

For each Qwen3-1.7B linear shape and (clean batch B, total rails R):
  serial      R separate cuBLAS GEMMs of [B, d_in]          (naive reference)
  flat        one cuBLAS GEMM of [B*R, d_in]                (weight reuse)
  flat+rail   flat + the shipping fused Triton rank-1 pass  (= es_token today)
  fused       Triton GEMM with the rank-1 rail in the epilogue (rail_gemm_kernel)
  triton      the same Triton GEMM without the epilogue     (control)

All variants are CUDA-graph captured and timed on replay (GPU time, no launch
overhead -- what the decode graph sees). Output: latency, TFLOP/s, effective
GB/s, T(B,R)/T(B,1) and N_free at 5% / 10% for every (shape, B).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree>/verl python \
        scripts/zo_opd/es_profile/phase1_linear_rails.py
"""
import argparse

import torch

from common import (QWEN3_1P7B_LINEARS, cuda_time_graphed, gpu_info, md_table,
                    save_json)
from verl.trainer.es_token.rail_kernel import apply_rail
from verl.trainer.es_token.rail_gemm_kernel import rail_gemm


def make_rail_state(B, R, d_in, d_out, dev):
    """Packed layout: row b*R + r; r=0 clean."""
    d_total = d_out + d_in
    noise = torch.randint(0, 2, (B, d_total), device=dev).to(torch.bfloat16) * 2 - 1
    signs = torch.randint(0, 2, (max(R - 1, 1), d_total), device=dev).to(torch.bfloat16) * 2 - 1
    sigma = torch.full((1,), 0.01, dtype=torch.bfloat16, device=dev)
    rows = torch.arange(B * R, device=dev)
    r = rows % R
    b = rows // R
    pri = rows[r > 0].contiguous()
    rail = (r[r > 0] - 1).contiguous()
    pidx = b[r > 0].contiguous()
    rail_of_row = torch.where(r > 0, r - 1, torch.full_like(r, -1)).to(torch.int32)
    slot_of_row = b.to(torch.int32)
    return dict(noise=noise, signs=signs, sigma=sigma, pri=pri, rail=rail, pidx=pidx,
                rail_of_row=rail_of_row, slot_of_row=slot_of_row,
                off_u=0, off_v=d_out, d_total=d_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", default="1,2,4,8,16,32,64")
    ap.add_argument("--Rs", default="1,2,4,8,12,16,20,24,32")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-fused", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    Bs = [int(x) for x in args.Bs.split(",")]
    Rs = [int(x) for x in args.Rs.split(",")]
    print("[gpu]", gpu_info(), flush=True)

    recs = []
    checks = []
    for name, d_in, d_out in QWEN3_1P7B_LINEARS:
        W = torch.randn(d_out, d_in, dtype=torch.bfloat16, device=dev) * 0.02
        w_bytes = W.numel() * 2
        for B in Bs:
            base = None
            for R in Rs:
                M = B * R
                X = torch.randn(M, d_in, dtype=torch.bfloat16, device=dev)
                st = make_rail_state(B, R, d_in, d_out, dev)
                Y = torch.empty(M, d_out, dtype=torch.bfloat16, device=dev)
                Xb = X.view(B, R, d_in)

                def f_serial():
                    for r in range(R):
                        torch.matmul(Xb[:, r], W.t(), out=Y.view(B, R, d_out)[:, r])

                def f_flat():
                    torch.matmul(X, W.t(), out=Y)

                def f_flat_rail():
                    torch.matmul(X, W.t(), out=Y)
                    if R > 1:
                        apply_rail(X, Y, st["noise"], st["signs"], st["sigma"],
                                   st["pri"], st["rail"], st["pidx"],
                                   st["off_u"], d_out, st["off_v"], d_in,
                                   st["noise"].stride(0), st["signs"].stride(0))

                Y2 = torch.empty_like(Y)

                def f_fused():
                    rail_gemm(X, W, st["noise"], st["signs"], st["sigma"],
                              st["rail_of_row"], st["slot_of_row"],
                              st["off_u"], st["off_v"], out=Y2, fuse=R > 1)

                def f_triton():
                    rail_gemm(X, W, st["noise"], st["signs"], st["sigma"],
                              st["rail_of_row"], st["slot_of_row"],
                              st["off_u"], st["off_v"], out=Y2, fuse=False)

                variants = [("serial", f_serial), ("flat", f_flat), ("flat+rail", f_flat_rail)]
                if not args.no_fused:
                    variants += [("fused", f_fused), ("triton", f_triton)]
                rec = dict(shape=name, d_in=d_in, d_out=d_out, B=B, R=R, M=M)
                for vname, fn in variants:
                    try:
                        mn, md = cuda_time_graphed(fn, warmup=3, iters=args.iters)
                    except Exception as e:  # e.g. Triton compile failure
                        print(f"[warn] {name} B={B} R={R} {vname}: {e}", flush=True)
                        mn = md = float("nan")
                    rec[vname] = mn
                    rec[vname + "_median"] = md
                flops = 2.0 * M * d_in * d_out
                rec["tflops_flat"] = flops / (rec["flat"] * 1e-3) / 1e12
                rec["gbps_flat"] = (w_bytes + (M * d_in + M * d_out) * 2) / (rec["flat"] * 1e-3) / 1e9
                if R == 1:
                    base = rec["flat"]
                rec["rel_flat_rail"] = rec["flat+rail"] / base
                rec["rel_flat"] = rec["flat"] / base
                if not args.no_fused:
                    rec["rel_fused"] = rec["fused"] / base
                recs.append(rec)
                # correctness of the fused epilogue vs shipping path (once per shape/B)
                if not args.no_fused and R > 1 and R == Rs[-1]:
                    f_flat_rail()
                    f_fused()
                    torch.cuda.synchronize()
                    d = (Y.float() - Y2.float()).abs()
                    ref = Y.float().abs().max().item()
                    checks.append(dict(shape=name, B=B, R=R, max_abs=d.max().item(),
                                       max_ref=ref, mean_abs=d.mean().item()))
                print(f"[{name} B={B:2d} R={R:2d} M={M:4d}] " + "  ".join(
                    f"{v}={rec[v]:.4f}" for v, _ in variants) +
                    f"  rel(flat+rail)={rec['rel_flat_rail']:.3f}" +
                    (f" rel(fused)={rec['rel_fused']:.3f}" if not args.no_fused else ""),
                    flush=True)
                del X, Y, Y2, st
        del W
        torch.cuda.empty_cache()

    # N_free extraction per (shape, B): largest R with rel <= 1+eps
    nfree = []
    for name, _, _ in QWEN3_1P7B_LINEARS:
        for B in Bs:
            rs = [r for r in recs if r["shape"] == name and r["B"] == B]
            row = dict(shape=name, B=B)
            for key, lab in (("rel_flat_rail", "flat+rail"), ("rel_fused", "fused"), ("rel_flat", "flat")):
                if key not in rs[0]:
                    continue
                for eps in (0.05, 0.10, 0.25):
                    ok = [r["R"] for r in rs if r[key] <= 1 + eps]
                    row[f"{lab}_N{int(eps*100)}"] = (max(ok) - 1) if ok else 0
            nfree.append(row)

    save_json("phase1_linear_rails.json", dict(gpu=gpu_info(), recs=recs, nfree=nfree,
                                               checks=checks, Bs=Bs, Rs=Rs))
    print("\n## fused-epilogue correctness vs cuBLAS+rail (bf16)")
    print(md_table(["shape", "B", "R", "max_abs", "max_ref", "mean_abs"],
                   [[c["shape"], c["B"], c["R"], c["max_abs"], c["max_ref"], c["mean_abs"]] for c in checks],
                   fmt={"max_abs": "{:.4f}", "max_ref": "{:.2f}", "mean_abs": "{:.5f}"}))
    print("\n## N_free (extra rails N=R-1 within eps of the clean GEMM, flat+rail path)")
    print(md_table(["shape", "B", "N5%", "N10%", "N25%", "fused N5%", "fused N10%"],
                   [[r["shape"], r["B"], r.get("flat+rail_N5"), r.get("flat+rail_N10"), r.get("flat+rail_N25"),
                     r.get("fused_N5", "-"), r.get("fused_N10", "-")] for r in nfree]))
    print("\n## relative latency T(B,R)/T(B,1), flat+rail (rows) x R (cols)")
    for name, _, _ in QWEN3_1P7B_LINEARS:
        print(f"\n### {name}")
        rows = []
        for B in Bs:
            rs = {r["R"]: r for r in recs if r["shape"] == name and r["B"] == B}
            rows.append([B] + [rs[R]["rel_flat_rail"] for R in Rs])
        print(md_table(["B \\ R"] + [str(R) for R in Rs], rows, fmt={str(R): "{:.2f}" for R in Rs}))


if __name__ == "__main__":
    main()
