"""Correctness gate for the es_token rail-op rewrites.

Every optimised variant must reproduce the shipping `v0_current` rail
contribution. Variants that use the [clean | perturbed] row layout are compared
against v0 run in THAT layout (v0 is layout-agnostic -- it addresses rows via
pri/rail/pidx), otherwise the two sides would read different activation rows.

  CUDA_VISIBLE_DEVICES=6 python scripts/zo_opd/es_token_checks/check_rail_op_parity.py
"""
import sys

import torch

sys.path.insert(0, "scripts/zo_opd/es_token_checks")
import bench_rail_op as B


def clone_inputs(dst, src):
    dst.noise_buf.copy_(src.noise_buf)
    dst.signs_flat.copy_(src.signs_flat)
    dst.signs_sig.copy_(src.signs_flat * src.sigma_flat)
    for li in range(len(dst.layout)):
        dst.S[li].copy_(src.S[li])
        dst.R[li].copy_(src.R[li])


def main():
    dev = torch.device("cuda")
    dtype = torch.float32          # fp32 so the gate measures math, not rounding
    ok = True
    for N in (1, 8):
        print(f"\n=== n_sample={N}, bucket=4, {len(B.build_layers()[0])} layers ===")
        seed_state = B.State(4, N, dtype, dev, contig=False)
        for name, (fn, need_contig) in B.VARIANTS.items():
            if name == "v0_current":
                continue
            # reference in the SAME row layout as the variant under test
            ref = B.State(4, N, dtype, dev, contig=need_contig)
            clone_inputs(ref, seed_state)
            y0 = [t.clone() for t in ref.y]
            B.v0_current(ref)
            expect = [(ref.y[li] - y0[li])[ref.pri] for li in range(len(ref.layout))]

            st = B.State(4, N, dtype, dev, contig=need_contig)
            clone_inputs(st, seed_state)
            for li in range(len(st.layout)):
                st.x[li].copy_(ref.x[li])
                st.y[li].copy_(y0[li])
            fn(st)

            worst, worst_l = 0.0, -1
            for li in range(len(st.layout)):
                got = (st.y[li] - y0[li])[st.pri]
                exp = expect[li]
                scale = max(exp.abs().max().item(), 1e-8)
                rel = (got - exp).abs().max().item() / scale
                if rel > worst:
                    worst, worst_l = rel, li
            good = worst < 1e-4
            ok &= good
            lay = "contig" if need_contig else "shipping"
            verdict = "PASS" if good else "FAIL"
            print(f"  {name:<18s} layout={lay:<8s}  "
                  f"max rel err = {worst:.3e} (layer {worst_l})  {verdict}")
            del ref, st
            torch.cuda.empty_cache()
        del seed_state
        torch.cuda.empty_cache()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
