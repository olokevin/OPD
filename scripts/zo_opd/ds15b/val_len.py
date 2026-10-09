import json,sys,glob,os
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
for d in sys.argv[1:]:
    print("==",os.path.basename(d))
    for f in sorted(glob.glob(d+"/*.jsonl"), key=lambda x:int(os.path.basename(x)[:-6])):
        rows=[json.loads(l) for l in open(f)]
        m=[r for r in rows if "MATH" in str(r.get("data_source","")) or "math" in str(r.get("data_source","")).lower()] or rows
        keys=rows[0].keys()
        L=[len(tok(r["output"]).input_ids) for r in m[:500]]
        acc=sum(float(r.get("score",r.get("acc",0)))>0 for r in m[:500])/len(m[:500])
        cap=sum(l>=7160 for l in L)/len(L)
        okL=[l for l,r in zip(L,m) if float(r.get("score",0))>0]
        print(f"step {os.path.basename(f)[:-6]:>4}  n={len(m)} acc={acc:.3f} meanlen={sum(L)/len(L):.0f} caphit={cap:.2f} correct_meanlen={sum(okL)/max(1,len(okL)):.0f}")
print(list(keys))
