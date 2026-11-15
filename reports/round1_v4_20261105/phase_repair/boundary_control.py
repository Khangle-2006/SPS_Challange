"""Static-input intervention on in-memory TSM padding only; no checkpoint changes."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,json,types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,"/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102")
from EfficientPhys_fixed import EfficientPhys, TSM
from t8_evaluate import decode_difference
torch.set_num_threads(2)
model=EfficientPhys(frame_depth=10,img_size=36).eval()
ck=HERE.parent/"v4_phase/fold0/seed2027/joint/final.pt"
model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"])
# Every time-constant RGB clip has identically zero first differences.
x=torch.zeros(1,301,3,36,36)
def circular_shift(self,features):
    nt,c,h,w=features.shape; fold=c//self.fold_div
    blocks=features.reshape(-1,self.n_segment,c,h,w)
    return torch.cat([blocks[:,:,:fold].roll(-1,1),blocks[:,:,fold:2*fold].roll(1,1),blocks[:,:,2*fold:]],2).reshape_as(features)
with torch.no_grad():
    old=model(x)[0].numpy().astype(float)
    for layer in model.modules():
        if isinstance(layer,TSM):layer.forward=types.MethodType(circular_shift,layer)
    changed=model(x)[0].numpy().astype(float)
result={"original_static_raw_ptp":float(np.ptp(old)),"original_static_hr":decode_difference(old),
        "circular_static_raw_ptp":float(np.ptp(changed)),"circular_static_raw_std":float(changed.std())}
try: result["circular_static_hr"]=decode_difference(changed)
except ValueError as exc: result["circular_static_abstain"]=str(exc)
(HERE/"boundary_control.json").write_text(json.dumps(result,indent=2))
print(result)
