"""Verify the frozen GPU baseline predicts the same HR on the CPU used in the A/B."""
from pathlib import Path
import sys,json
import torch
import numpy as np
from test_static_tsm import EfficientPhys
from t8_evaluate import decode_difference
from t8_plan import read,sha
from polarity_loader import load_canonical_clip

here=Path(__file__).resolve().parent
root=here.parent.parents[1]
torch.set_num_threads(1);torch.set_num_interop_threads(1)
joint=root/"t8_polarity_v3_20261102/fold0/seed2027/joint"
assert sha(joint/"final.pt")==read(joint/"status.json")["checkpoint_sha256"]
model=EfficientPhys(frame_depth=10,img_size=36).eval()
model.load_state_dict(torch.load(joint/"final.pt",map_location="cpu",weights_only=False)["model"])
roots=read(root/"t8_oval_20261031/training_index.json")["roots"]
rows=read(here/"matched_rows.json")
assert len(rows)==870
changes=[]
for i,r in enumerate(rows):
    condition=r["condition"]
    ds="MCD" if condition.startswith("MCD") else "UBFC2"
    variant="f10_b500" if "C5" in condition else "native"
    x,*_=load_canonical_clip(Path(roots[ds])/r["record"]/variant/"crop",r["start"],ds)
    with torch.inference_mode(): wave=model(torch.from_numpy(x[None]))[0].numpy()
    hr=decode_difference(wave)
    if hr!=r["original"]:
        changes.append(dict(subject=r["subject"],record=r["record"],start=r["start"],condition=condition,cpu=hr,gpu=r["original"]))
(here/"cpu_parity.json").write_text(json.dumps(dict(checked=len(rows),changed=changes),indent=2))
print(len(rows),len(changes),flush=True)
