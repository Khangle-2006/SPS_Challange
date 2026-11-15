"""Fold 0 seed 2027 CPU-only inference contrast with fixed trained weights."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
coverage=json.loads((here/"matched_fold0.json").read_text())
assert coverage["complete"]
rows=json.loads((here/"matched_rows.json").read_text())
out={}
for key,counts in coverage["conditions"].items():
    common=[r for r in rows if r["condition"]==key and r["replicate"] is not None]
    by=defaultdict(list)
    for r in common:
        y=r["reference"]
        by[r["subject"]].append(abs(r["replicate"]-y)-abs(r["original"]-y))
    deltas=np.array([np.mean(by[k]) for k in sorted(by)])
    rng=np.random.default_rng(0)
    ci=np.quantile(deltas[rng.integers(len(by),size=(10000,len(by)))].mean(1),[.025,.975])
    out[key]=dict(**counts,people=len(by),delta=float(deltas.mean()),ci95=ci.tolist(),
                  original_at_or_above179=sum(r["original"]>=179 for r in common),
                  replicate_at_or_above179=sum(r["replicate"]>=179 for r in common))
(here/"paired_summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
