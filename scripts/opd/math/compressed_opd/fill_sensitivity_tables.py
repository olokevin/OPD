"""Regenerate the two per-layer sensitivity tables in compressed_opd.md.

Reads the incremental shard JSONs (layer_sens_qwen3-4b-base_shard*.json), and
rewrites the two markdown tables (self_attn, mlp) between sentinel markers in
docs/results/compressed_opd.md. Idempotent — safe to run repeatedly as cells land.

Cells not yet run show as '–'. Accuracies shown as percentages.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RES = REPO / "scripts/opd/math/compressed_opd/results"
DOC = REPO / "docs/results/compressed_opd.md"

LAYER_ORDER = [35, 34, 33, 32, 31, 30, 20, 19, 18, 17, 16, 15, 5, 4, 3, 2, 1, 0]
RATIOS = [0.9, 0.8, 0.6, 0.5]
MODULES = ["self_attn", "mlp"]

ATTN_BEGIN = "<!-- ATTN_TABLE_BEGIN -->"
ATTN_END = "<!-- ATTN_TABLE_END -->"
MLP_BEGIN = "<!-- MLP_TABLE_BEGIN -->"
MLP_END = "<!-- MLP_TABLE_END -->"


def load_cells():
    acc = {m: {l: {r: None for r in RATIOS} for l in LAYER_ORDER} for m in MODULES}
    n_done = 0
    for fp in glob.glob(str(RES / "layer_sens_qwen3-4b-base_shard*.json")):
        d = json.load(open(fp))
        for c in d.get("cells", []):
            if c.get("math500_acc") is None:
                continue
            m, l, r = c["module"], c["layer"], c["ratio"]
            if m in acc and l in acc[m] and r in acc[m][l]:
                acc[m][l][r] = c["math500_acc"]
                n_done += 1
    return acc, n_done


def fmt(v):
    return "–" if v is None else f"{v*100:.0f}"


def table(acc_mod):
    lines = ["| layer | retain 0.9 | retain 0.8 | retain 0.6 | retain 0.5 |",
             "|------:|:----------:|:----------:|:----------:|:----------:|"]
    for l in LAYER_ORDER:
        cells = " | ".join(fmt(acc_mod[l][r]) for r in RATIOS)
        lines.append(f"| {l:>2} | {cells} |")
    return "\n".join(lines)


def splice(text, begin, end, body):
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    repl = f"{begin}\n{body}\n{end}"
    if pat.search(text):
        return pat.sub(repl, text)
    raise SystemExit(f"markers {begin}/{end} not found in {DOC}")


def main():
    acc, n_done = load_cells()
    text = DOC.read_text()
    text = splice(text, ATTN_BEGIN, ATTN_END, table(acc["self_attn"]))
    text = splice(text, MLP_BEGIN, MLP_END, table(acc["mlp"]))
    DOC.write_text(text)
    print(f"Updated tables: {n_done} cells filled.")


if __name__ == "__main__":
    main()
