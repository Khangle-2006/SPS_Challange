"""New-interface CPU checks using real cached clips; no saved trained weights."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import numpy as np
from common import *
from sweep import make_model
from training_provider import DiskProvider
from aipex_data import AiPEX
from domain_loss import weak_ce,sinc_loss
from repair_loss import phase_loss
from model_eval import weak_gate
def main():
 import torch
 torch.set_num_threads(2);lease=lock('sweep_cpu_validation.lock');plan=read(S/'PLAN.json');index=read(R/'t8_oval_20261031/training_index.json');checks=[]
 # Changed C5 encoder is exercised on actual saved cache, not regenerated.
 choices=[]
 for f in sorted((P/'training_cache/0').glob('*.json')):
  a=read(f)
  if a['encoding']['recipe']=='c5_proxy':choices.append((f,a))
 assert choices,'Wait for one real C5 training crop'
 f,a=choices[0];ds,rid,start,recipe,lowres,fold=json.loads(a['key']);provider=DiskProvider(index,plan['runs'][fold]['btc_bpp_quartiles'],300000000)
 j=next(j for j in plan['runs'][fold]['public_jobs'][ds] if (j['recording'],j['start'],j['recipe'],j['lowres'])==(rid,start,recipe,lowres))
 tensors=[]
 for rate in [j['rates'][0],sum(j['rates'])/2,j['rates'][1]]:
  x,y,receipt=provider.clip(ds,rid,start,recipe,lowres,fold,rate);assert x.shape==(301,3,36,36) and y.shape==(300,) and np.isfinite(x).all() and np.isfinite(y).all();assert receipt['fs']<=10+1e-8 and receipt['whole_file_kbps']<=500
  tensors.append((x,y));checks.append(dict(kind='real_C5_clip',rate=rate,source=receipt['source_sha256'],cache_receipt_sha256=sha(f),fs=receipt['fs'],kbps=receipt['whole_file_kbps']))
 btc=BTCData(2);aipex=AiPEX();rng=np.random.default_rng(2027)
 bx,bp,bf,bi=btc.aux(rng,True);ax,ap,af,ai=aipex.sample(rng,True,n=4)
 assert set(p for p,r,s in bi).isdisjoint(plan['runs'][2]['btc_val_people'])
 xx=np.concatenate([bx,ax]);assert xx.shape==(8,301,3,36,36) and np.isfinite(xx).all()
 model=make_model(plan['runs'][2],'cpu');before=model_static(model);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0)
 optimizer.zero_grad(set_to_none=True);out=model(torch.from_numpy(xx));loss=weak_ce(out,torch.tensor(bp+ap,dtype=out.dtype));assert torch.isfinite(loss);(.1*loss).backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters());assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step();after=model_static(model)
 checks.append(dict(kind='BTC_AiPEX_weak_real_batch8',loss=float(loss.detach()),train_people=[p for p,r,s in bi],AiPEX_people=ai,static_before=before,static_after=after))
 model.train();optimizer.zero_grad(set_to_none=True);out=model(torch.from_numpy(xx));loss,parts=sinc_loss(out,bf+af);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 checks.append(dict(kind='BTC_AiPEX_SiNC_real_batch8',loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()},artifact_hz=bf+af))
 x,y=tensors[1];optimizer.zero_grad(set_to_none=True);out=model(torch.from_numpy(np.stack([x,x])));loss=phase_loss(out,torch.from_numpy(np.stack([y,y]))).mean();loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 checks.append(dict(kind='changed_C5_supervised_backward',loss=float(loss.detach())))
 t=torch.arange(301)/30;d=torch.diff(torch.sin(2*torch.pi*1.2*t))[None]
 assert weak_ce(d,torch.tensor([72.]))<weak_ce(d,torch.tensor([130.]))
 # Collapse guard must reject near-constant predictions even with finite MAE.
 base=[dict(person=str(i),recording=str(i),truth=60+10*i,prediction=60+10*i) for i in range(5)];flat=[dict(r,prediction=85.) for r in base];assert not weak_gate(base,flat)['passed']
 save(P/'SWEEP_CPU_VALIDATION.json',dict(passed=True,utc=now(),checks=checks,code=hashes(),inherited_balanced_public_model_checks_sha256=sha(S/'CPU_VALIDATION.json'),no_checkpoint_saved=True,holdout_evaluations=0))
 print('SWEEP_CPU_PASS',flush=True)
if __name__=='__main__':main()
