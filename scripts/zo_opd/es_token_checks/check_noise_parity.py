"""es_token noise gate: decode and assembly must regenerate IDENTICAL bytes.

The whole design rests on never storing or shipping noise -- the decode draws
(u_t, v_t) for token t of rollout rid, and `es_assemble_and_apply` regenerates
exactly those bytes later from (global_seed, t, rollout_id) alone. This gate
exercises both call paths of the new direct-Rademacher fill:

  decode-side   : seeds taken from the per-wave table (build_seed_table), sliced
                  per token, already resident on the device
  assembly-side : seeds derived per chunk on the host and uploaded

plus the properties the estimator needs (values exactly +-1, zero-mean, distinct
across t / rollout, and stable across repeated regeneration).

  CUDA_VISIBLE_DEVICES=6 python scripts/zo_opd/es_token_checks/check_noise_parity.py
"""
import argparse

import torch

from verl.trainer.es_token.noise_kernel import (
    HAVE_TRITON, fill_rademacher_rows, impl_name)
from verl.trainer.es_token.seeding import build_seed_table, es_token_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-total", type=int, default=917504)  # Qwen3-1.7B
    ap.add_argument("--bucket", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--global-seed", type=int, default=42)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    print(f"impl={impl_name()} triton={HAVE_TRITON} device={dev} dtype={dt}")
    print(f"d_total={args.d_total} bucket={args.bucket} tokens={args.tokens}")

    rollout_ids = list(range(100, 100 + args.bucket))
    tbl_dev, tbl_host = build_seed_table(args.global_seed, args.tokens,
                                         rollout_ids, dev)
    ok = True

    # ---- 1. decode-side fill (table slice) vs assembly-side fill (host seeds) ----
    dec = torch.empty(args.bucket, args.d_total, device=dev, dtype=dt)
    asm = torch.empty(args.bucket, args.d_total, device=dev, dtype=dt)
    worst_t = None
    for t in range(args.tokens):
        fill_rademacher_rows(dec, tbl_host[t], tbl_dev[t])          # decode path
        seeds = [es_token_seed(args.global_seed, t, int(r)) for r in rollout_ids]
        fill_rademacher_rows(asm, seeds)                            # assembly path
        if not torch.equal(dec, asm):
            ok = False
            worst_t = t if worst_t is None else worst_t
    print(f"  [1] decode == assembly, all {args.tokens} tokens: "
          f"{'PASS' if worst_t is None else f'FAIL at t={worst_t}'}")

    # ---- 2. assembly chunk ordering: a [M, d] chunk row j == its own (t, rid) ----
    recs = [(t, int(rollout_ids[p])) for t in range(args.tokens)
            for p in range(args.bucket)]
    chunk = torch.empty(len(recs), args.d_total, device=dev, dtype=dt)
    fill_rademacher_rows(chunk, [es_token_seed(args.global_seed, t, r)
                                 for (t, r) in recs])
    single = torch.empty(1, args.d_total, device=dev, dtype=dt)
    bad = []
    for j, (t, r) in enumerate(recs):
        fill_rademacher_rows(single, [es_token_seed(args.global_seed, t, r)])
        if not torch.equal(chunk[j], single[0]):
            bad.append(j)
    ok &= not bad
    print(f"  [2] chunk row j == its own record ({len(recs)} records): "
          f"{'PASS' if not bad else f'FAIL rows {bad[:5]}'}")

    # ---- 3. value / distribution properties ----
    vals = torch.unique(chunk.float())
    exact = vals.numel() == 2 and set(v.item() for v in vals) == {-1.0, 1.0}
    means = chunk.float().mean(dim=1).abs()
    balanced = bool((means < 0.02).all().item())
    ok &= exact and balanced
    print(f"  [3] values exactly {{-1,+1}}: {'PASS' if exact else f'FAIL {vals.tolist()}'}"
          f"   |mean| < 0.02 per row: {'PASS' if balanced else f'FAIL max={means.max():.4f}'}")

    # ---- 4. distinctness across t and across rollout ----
    d_t = not torch.equal(chunk[0], chunk[args.bucket])       # same slot, t vs t+1
    d_r = not torch.equal(chunk[0], chunk[1])                 # same t, rid vs rid+1
    ok &= d_t and d_r
    print(f"  [4] distinct across t: {'PASS' if d_t else 'FAIL'}   "
          f"across rollout: {'PASS' if d_r else 'FAIL'}")

    # ---- 5. repeated regeneration is bit-identical ----
    again = torch.empty_like(chunk)
    fill_rademacher_rows(again, [es_token_seed(args.global_seed, t, r)
                                 for (t, r) in recs])
    stable = torch.equal(chunk, again)
    ok &= stable
    print(f"  [5] regeneration bit-identical: {'PASS' if stable else 'FAIL'}")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
