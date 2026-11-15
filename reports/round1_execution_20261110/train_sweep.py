"""Exactly five final-epoch runs after B/A3, with per-epoch static contract."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import time,gc,traceback
import numpy as np
from common import *
from sweep import make_model,deadline
from repair_loss import phase_loss
from domain_loss import weak_ce,sinc_loss
from test_static_tsm import static_result
from training_provider import DiskProvider
from aipex_data import AiPEX
from model_eval import c5_rows,weak_gate

def one(plan,run,device):
 import torch
 import t8_run as base
 fold=run['fold'];dest=P/'sweep'/f'fold{fold}'
 assert not dest.exists(),'Inspect any prior partial; never automatically restart'
 deadline(plan);guard();base.seed_all(2027,fold);dest.mkdir(parents=True)
 model=make_model(run,device);btc=BTCData(fold);aipex=AiPEX() if fold>=2 else None
 assert set(btc.people)<=set(run['btc_train_people']) and set(btc.people).isdisjoint(run['btc_val_people'])
 before=infer_native(model,btc,fold,'padding_fixed_before_sweep',device);save(dest/'BEFORE_NATIVE.json',before)
 provider=DiskProvider(read(R/'t8_oval_20261031/training_index.json'),run['btc_bpp_quartiles'],300000000)
 optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
 save(dest/'provenance.json',dict(plan_sha256=sha(P/'PLAN.json'),initial=run,split_sha256=SPLIT,official_MATLAB_parity=False,selection='final_epoch_only',AiPEX_assumed_fps=15))
 history=[]
 for epoch in range(1,plan['epochs']+1):
  deadline(plan);guard();model.train();started=time.monotonic();rng=np.random.default_rng(np.random.SeedSequence([2027,fold,epoch]));losses=[];draws=[];components=[]
  for step in range(run['steps_per_epoch']):
   deadline(plan);xs=[];ys=[]
   for ds in ['UBFC2','MCD']*4:
    choices=run['public_jobs'][ds];j=choices[int(rng.integers(len(choices)))];rate=float(rng.uniform(*j['rates']))
    x,y,receipt=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],fold,rate);xs.append(x);ys.append(y);draws.append(dict(**j,rate=rate))
   optimizer.zero_grad(set_to_none=True)
   output=model(torch.from_numpy(np.stack(xs)).to(device));sup=phase_loss(output,torch.from_numpy(np.stack(ys)).to(device)).mean();assert torch.isfinite(sup);sup.backward();auxvalue=0.
   del output,xs,ys
   if aipex:
    xx,pr,freq,ids=btc.aux(rng,fold==2);ax,ap,af,ai=aipex.sample(rng,fold==2,n=4)
    xx=np.concatenate([xx,ax]);pr+=ap;freq+=af;output=model(torch.from_numpy(xx).to(device))
    if fold==2:aux=weak_ce(output,torch.tensor(pr,device=device,dtype=output.dtype));parts={}
    else:aux,parts=sinc_loss(output,freq)
    weighted=plan['btc_aux_weight'][str(fold)]*aux;assert torch.isfinite(weighted);weighted.backward();auxvalue=float(weighted.detach());components.append({k:float(v.detach()) for k,v in parts.items()})
    draws.append(dict(BTC=ids,AiPEX=ai,auxiliary='weak_CE' if fold==2 else 'SiNC',artifact_hz=freq));del output,xx,ax,weighted,aux
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'nonfinite_gradient'
   optimizer.step();losses.append([float(sup.detach()),auxvalue]);save(dest/'status.json',dict(state='training',epoch=epoch,step=step+1,steps=run['steps_per_epoch'],utc=now(),pid=os.getpid()))
  entry=dict(epoch=epoch,supervised_loss=float(np.mean(losses,0)[0]),weighted_aux_loss=float(np.mean(losses,0)[1]),total_loss=float(np.mean(losses,0).sum()),seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
  model.cpu();static=static_result(model);save(dest/f'STATIC_EPOCH{epoch}.json',static);entry['static_pass']=static['hr'] is None;history.append(entry)
  save(dest/'history.json',history);save(dest/f'epoch{epoch}_draws.json',draws)
  base.save_checkpoint(dict(model=model.state_dict(),epoch=epoch,fold=fold,seed=2027,plan_sha256=sha(P/'PLAN.json')),dest/'last.pt')
  print(json.dumps(dict(fold=fold,**entry)),flush=True)
  if not entry['static_pass']:
   save(dest/'status.json',dict(state='rejected_static',epoch=epoch,utc=now(),checkpoint_sha256=sha(dest/'last.pt')));return
  model.to(device)
 base.save_checkpoint(dict(model=model.cpu().state_dict(),epoch=plan['epochs'],fold=fold,seed=2027,plan_sha256=sha(P/'PLAN.json')),dest/'final.pt');model.to(device)
 rows=infer_native(model,btc,fold,'sweep_'+run['recipe_name'],device);save(dest/'NATIVE.json',rows)
 gate=weak_gate(before,rows) if fold==2 else dict(passed=True,not_applicable=True);save(dest/'WEAK_COLLAPSE_GATE.json',gate)
 save(dest/'C5PRIME.json',c5_rows(model,fold,device,'sweep_'+run['recipe_name']))
 from phase_evaluate import evaluate
 evaluate(model,read(R/'t8_oval_20261031/training_index.json'),fold,2027,dest/'public_evaluation',R/'t8_polarity_v3_20261102',device)
 save(dest/'status.json',dict(state='complete' if gate['passed'] else 'rejected_weak_collapse',completed_epochs=plan['epochs'],checkpoint_sha256=sha(dest/'final.pt'),static_pass=True,weak_gate=gate,utc=now(),holdout_evaluations=0))
 del model,optimizer,provider,btc,aipex;gc.collect();torch.cuda.empty_cache()

def main():
 lease=lock('gpu_execution.lock');leases=[]
 for path in [R/'t8_oval_20261031/runs/queue.lock',R/'t8_polarity_v3_20261102/queue.lock',V/'v4_repaired/queue.lock',V/'v4_padding_pilot/queue.lock',S/'queue.lock']:
  h=path.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(h)
 assert (P/'adabn/C5_COMPLETE.json').exists() and (P/'C5_EVALUATION_COMPLETE.json').exists(),'B and A3 must finish first'
 assert (P/'AIPEX_COMPLETE.json').exists() and (P/'TRAIN_CACHE_COMPLETE.json').exists(),'Prepared inputs required'
 plan=read(P/'PLAN.json');deadline(plan)
 frozen=read(P/'FROZEN.json')
 for file,h in frozen.items():assert sha(file)==h,('Frozen dependency changed',file)
 assert read(P/'SWEEP_CPU_VALIDATION.json')['passed']
 guard(True);device=gpu();save(P/'SWEEP_STATUS.json',dict(state='running',utc=now(),pid=os.getpid()))
 for run in plan['runs']:one(plan,run,device)
 save(P/'SWEEP_STATUS.json',dict(state='complete',runs=5,utc=now(),holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'SWEEP_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
