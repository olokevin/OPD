"""Measure per-layer weight displacement of a checkpoint vs the base model."""
import sys, os, glob, json
import torch
from safetensors.torch import load_file

def load_sd(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise SystemExit(f"no safetensors in {path}")
        sd = {}
        for f in files:
            sd.update(load_file(f))
        return sd
    raise SystemExit(f"not a dir: {path}")

def resolve(p):
    if os.path.isdir(p):
        return p
    # HF hub cache
    import huggingface_hub as hh
    return hh.snapshot_download(p, allow_patterns=["*.safetensors","*.json"])

base = load_sd(resolve(sys.argv[1]))
other = load_sd(resolve(sys.argv[2]))
tag = sys.argv[3]

rows = []
tot_d2 = 0.0; tot_n = 0; tot_w2 = 0.0
changed = 0; unchanged = 0
for k in sorted(base):
    if not k.endswith(".weight"): continue
    if k not in other:
        print(f"  MISSING in other: {k}"); continue
    a = base[k].float(); b = other[k].float()
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH {k}: {a.shape} vs {b.shape}"); continue
    d = (b - a)
    n = d.numel()
    d2 = float((d*d).sum())
    w2 = float((a*a).sum())
    tot_d2 += d2; tot_n += n; tot_w2 += w2
    rms_d = (d2/n)**0.5
    rms_w = (w2/n)**0.5
    frac_changed = float((d != 0).sum())/n
    if d2 == 0: unchanged += 1
    else: changed += 1
    rows.append((k, tuple(a.shape), rms_d, rms_w, rms_d/max(rms_w,1e-30), frac_changed))

print(f"=== {tag} ===")
print(f"tensors changed={changed} unchanged={unchanged}")
print(f"GLOBAL over {tot_n/1e9:.3f}B params: RMS(dW)={ (tot_d2/tot_n)**0.5 :.4e}  RMS(W)={ (tot_w2/tot_n)**0.5 :.4e}  ratio={ ((tot_d2/tot_n)/(tot_w2/tot_n))**0.5 :.4e}")
print(f"{'layer':<52} {'shape':<16} {'RMS(dW)':>11} {'RMS(W)':>10} {'rel':>10} {'frac!=0':>8}")
for k, sh, rd, rw, rel, fc in rows[:6] + rows[len(rows)//2: len(rows)//2+4] + rows[-6:]:
    print(f"{k:<52} {str(sh):<16} {rd:11.4e} {rw:10.4e} {rel:10.4e} {fc:8.4f}")
json.dump({"tag":tag,"rms_dW":(tot_d2/tot_n)**0.5,"rms_W":(tot_w2/tot_n)**0.5,
           "n_params":tot_n,"changed":changed,"unchanged":unchanged,
           "rows":[(k,rd,rw,rel,fc) for k,_,rd,rw,rel,fc in rows]},
          open(f"/tmp/claude-1002/-home-yequan-Project-compression-OPD/9111112e-b13e-415f-af03-658943404176/scratchpad/wdiff_{tag}.json","w"))
