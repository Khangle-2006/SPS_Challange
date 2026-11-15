"""CPU only: five initializations, actual B8 backward, fold isolation and BTC aux."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import io,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
from sweep import make_model,read,save,P,INDEX,Provider,phase_loss,weak_ce,sinc_loss,static_result,BTCData
plan=read(P/'PLAN.json');results=[]
for run in plan['runs']:
 model=make_model(run,'cpu');results.append(dict(fold=run['fold'],checkpoint_sha256=run['checkpoint_sha256'],static=static_result(model)))
 del model
run=plan['runs'][0];model=make_model(run,'cpu').train();provider=Provider(read(INDEX),run['btc_bpp_quartiles']);xs=[];ys=[]
for ds in ['UBFC2','MCD']*4:
 j=next(j for j in run['public_jobs'][ds] if j['recipe']=='h264' and j['rates'][1]>=1.2-1e-8)
 x,y,r=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],0,1.);xs.append(x);ys.append(y)
before={k:v.detach().clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
pred=model(torch.from_numpy(np.stack(xs)));loss=phase_loss(pred,torch.from_numpy(np.stack(ys))).mean();assert torch.isfinite(loss);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
opt.step();assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items());model.eval();assert static_result(model)['hr'] is None
bio=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=1),bio);last_size=bio.tell();bio=io.BytesIO();torch.save(dict(model=model.state_dict(),epoch=10),bio);final_size=bio.tell()
aux=[]
for fold in [2,3,4]:
 btc=BTCData(fold);assert set(btc.people).isdisjoint(plan['runs'][fold]['btc_val_people']);x,pr,freq,ids=btc.aux(np.random.default_rng(2027),fold==2)
 assert len({i[0] for i in ids})==4 and x.shape==(4,301,3,36,36)
 model.zero_grad(set_to_none=True);d=model(torch.from_numpy(x))
 l=weak_ce(d,torch.tensor(pr,dtype=d.dtype)) if fold==2 else sinc_loss(d,freq)[0]
 assert torch.isfinite(l);l.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
 aux.append(dict(fold=fold,loss=float(l.detach()),people=[i[0] for i in ids],input_hashes=btc.hashes));del btc
save(P/'CPU_VALIDATION.json',dict(passed=True,initializations=results,real_batch_size=8,supervised_loss=float(loss.detach()),finite_nonzero_backward=True,optimizer_step=True,post_step_static=static_result(model),auxiliary_checks=aux,last_checkpoint_bytes=last_size,final_checkpoint_bytes=final_size,CUDA_initialized=torch.cuda.is_initialized(),plan_sha256=hashlib.sha256((P/'PLAN.json').read_bytes()).hexdigest()))
assert not torch.cuda.is_initialized()
print('CPU VALIDATION PASS: five safe initializations, B8 real cache, backward, optimizer, static, CE and SiNC; GPU not started.',flush=True)
