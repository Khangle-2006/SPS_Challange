"""Actual frozen dev-cache inputs and initial model; CPU finite backward only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import hashlib,json,random,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;V3=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102');sys.path.insert(0,str(V3))
from t8_plan import read,sha,write,pools,steps_per_epoch,draw,code_fingerprint
from polarity_loader import load_canonical_clip
from EfficientPhys_fixed import EfficientPhys
from phase_loss import phase_loss,grouped_draws
from polarity_guard import EXPECTED_INDEX
ROOT=Path('/mnt/samsung_t5/khangle/facevitals');index=read(ROOT/'t8_oval_20261031/training_index.json')
assert sha(ROOT/'t8_oval_20261031/training_index.json')==EXPECTED_INDEX
assert code_fingerprint(V3)==read(V3/'FROZEN_SHA256.json')
torch.set_num_threads(2);torch.set_num_interop_threads(1)
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
model=EfficientPhys(frame_depth=10,img_size=36);h=hashlib.sha256()
for k,v in model.state_dict().items():h.update(k.encode());h.update(v.numpy().tobytes())
base=ROOT/'t8_polarity_v3_20261102/fold0/seed2027/joint'
assert h.hexdigest()==read(base/'provenance.json')['initial_model_sha256']
pool=pools(index,0);steps=steps_per_epoch(pool);order=hashlib.sha256()
for pairs,picks in grouped_draws(pool,2027,0,0,steps,8,draw):
 for pair in pairs:order.update(json.dumps(pair).encode())
assert order.hexdigest()==read(base/'history.json')[0]['draw_sha256']
picks=[p for step in range(4) for p in draw(pool,2027,0,0,step,'joint')];xs=[];ys=[];checks=[]
for ds,rid,start,variant in picks:
 root=Path(index['roots'][ds])/rid/variant/'crop';x,y,_,receipt=load_canonical_clip(root,start,ds)
 xs.append(x);ys.append(y);checks.append(dict(dataset=ds,record=rid,start=start,variant=variant,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),target_sha256=hashlib.sha256(y.tobytes()).hexdigest(),metadata_sha256=sha(root/'meta.npz'),normalization=receipt))
loss_values={}
for gn in (False,True):
 if gn:model.batch_norm=torch.nn.GroupNorm(1,3)
 model.zero_grad(set_to_none=True);pred=model(torch.from_numpy(np.stack(xs[:2])))
 loss=phase_loss(pred,torch.from_numpy(np.stack(ys[:2]))).mean();assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 loss_values['GN' if gn else 'BN']=float(loss.detach())
write(HERE.parent/'ACTUAL_CACHE_CPU_VALIDATION.json',dict(passed=True,actual_cache_loss_backward_passed=True,device='CPU',initial_model_sha256=h.hexdigest(),first_epoch_exact_draw_sha256=order.hexdigest(),actual_eight_clip_checks=checks,finite_backward_losses=loss_values,phase_files={p.name:sha(p) for p in HERE.glob('phase_*.py')},D3_opened=False,test_opened=False))
print('PASS actual8clips; BN/GN CPU finite backward; initial-model and full-epoch draw parity',flush=True)
