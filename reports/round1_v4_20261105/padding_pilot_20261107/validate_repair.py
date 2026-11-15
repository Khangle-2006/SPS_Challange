import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'code'))
sys.path.insert(0,'/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
from phase_train import INDEX, pools, draw, baseline, ORIGINAL, guard, V3, read, sha
# phase_train sets CUDA_VISIBLE_DEVICES before any CUDA call; override for CPU tests.
os.environ['CUDA_VISIBLE_DEVICES']=''
from EfficientPhys_fixed import EfficientPhys
from repair_loss import phase_loss,cardiac_wave
torch.set_num_threads(2)
index=read(INDEX)
assert sha(INDEX)==guard.EXPECTED_INDEX
pool=pools(index,0)
picks=[p for step in range(4) for p in draw(pool,2027,0,0,step,'joint')]
x,y=baseline.batch_arrays(index,picks)
target=torch.from_numpy(y)
t=torch.arange(300)/30
probe=(torch.sin(2*torch.pi*1.3*t)+.3*torch.sin(2*torch.pi*2.6*t)).diff(prepend=torch.zeros(1))[None]
cases={k:phase_loss(v,probe).item() for k,v in [('identical',probe),('inverted',-probe),('shift10',probe.roll(10,-1)),('dc_bias',probe+2),('high_freq',probe+torch.sin(2*torch.pi*6*t))]}
assert cases['identical']<1e-5 and min(v for k,v in cases.items() if k!='identical')>.1
flat=torch.zeros_like(target,requires_grad=True)
phase_loss(flat,target).mean().backward()
assert torch.isfinite(flat.grad).all() and flat.grad.abs().sum()>0
import random
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
from replicate_tsm import use_replicate_padding
model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
init=baseline.model_hash(model)
assert init==read(V3/'fold0/seed2027/joint/provenance.json')['initial_model_sha256']
# All eight draw/input records are verified. A pair suffices for actual CPU backward.
pred=model(torch.from_numpy(x[:2]))
loss=phase_loss(pred,target[:2]).mean()
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0)
opt.step()
assert all(torch.isfinite(p).all() for p in model.parameters())
receipt=dict(passed=True,tests=cases,actual_clips=len(x),cpu_pair_backward=float(loss.detach()),initial_model_sha256=init,files={n:sha(HERE/n) for n in ['repair_loss.py','pilot_runner.py','pilot_gate.py','validate_repair.py','replicate_tsm.py','test_static_tsm.py']},source_files={n:sha(HERE.parent/'code'/n) for n in ['phase_train.py','phase_loss.py','phase_evaluate.py']})
(HERE/'VALIDATION.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
