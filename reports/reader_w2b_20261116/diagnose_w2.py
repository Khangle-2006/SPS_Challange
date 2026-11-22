import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,json,fcntl,traceback,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;W=P.parent/'w2_btc_20261116';E=P.parent/'round1_execution_20261110'
sys.path[:0]=[str(W),str(E)]
import train_w2 as w
import torch,t8_run as base
from runtime_shared_v4 import gpu
def save(name,x):w.save(P/name,x)
def norms(model):
 return {group:float(torch.sqrt(sum(p.grad.detach().square().sum() for n,p in model.named_parameters() if p.grad is not None and (group=='all' or 'final_dense_2' in n))).cpu()) for group in ['all','final_dense_2']}
def main():
 lock=(E/'gpu_execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'T1_COMPLETE.json').exists();device=gpu();base.seed_all(2027,2);plan=json.load(open(E/'PLAN_ELIGIBLE.json'));run=next(r for r in plan['runs'] if r['fold']==2);data=w.RateBTC(2)
 rng=np.random.default_rng(0);fixed=[];xs=[];labels=[]
 for person in rng.choice(data.people,16,replace=False):
  v,start=data.by_person[person][0];xs.append(data.clip(v,start));labels.append(v['pr_bpm']);fixed.append(dict(recording=v['recording'],person=str(person),start=start,PR=v['pr_bpm']))
 assert len({r['person'] for r in fixed})==16;save('FIXED16.json',fixed);X=torch.from_numpy(np.stack(xs)).to(device);Y=torch.tensor(labels,dtype=torch.float32,device=device)
 model=w.make_model(run,device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 # T2: same initial weights/buffers for independent objectives, no update.
 provider=w.DiskProvider(w.common.read(w.common.R/'t8_oval_20261031/training_index.json'),run['btc_bpp_quartiles'],300_000_000);xx=[];yy=[]
 for ds in ['UBFC2','MCD']*4:
  jobs=run['public_jobs'][ds];j=jobs[int(rng.integers(len(jobs)))];rate=float(rng.uniform(*j['rates']));x,y,_=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],2,rate);xx.append(x);yy.append(y)
 model.train();loss=w.phase_loss(model(torch.from_numpy(np.stack(xx)).to(device)),torch.from_numpy(np.stack(yy)).to(device)).mean();loss.backward();pg=norms(model);pv=float(loss.detach());model.zero_grad(set_to_none=True);model.load_state_dict(initial)
 bx,by,draws=data.batch(rng);loss=w.weak_ce(model(torch.from_numpy(bx).to(device)),torch.from_numpy(by).to(device));loss.backward();wg=norms(model)
 save('T2.json',dict(phase_loss=pv,weak_ce=float(loss.detach()),phase_gradient=pg,weak_gradient=wg,weak_over_phase={k:wg[k]/pg[k] if pg[k]>0 else None for k in pg},same_initial_state=True))
 model.zero_grad(set_to_none=True);model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=plan['weight_decay']);history=[]
 def measure(step,last=None):
  model.eval();pred=[];losses=[]
  with torch.inference_mode():
   for a in [0,8]:
    z=model(X[a:a+8]);losses.append(float(w.weak_ce(z,Y[a:a+8])));pred.extend(float(w.decode_difference(d)) for d in z.cpu().numpy())
  row=dict(step=step,weak_ce=float(np.mean(losses)),last_training_ce=last,MAE=float(np.mean(np.abs(np.array(pred)-np.array(labels)))),predictions=pred);history.append(row);save('T1_HISTORY.json',history);model.train();return row
 measure(0)
 for step in range(1,401):
  model.train();optimizer.zero_grad(set_to_none=True);losses=[]
  for a in [0,8]:
   loss=w.weak_ce(model(X[a:a+8]),Y[a:a+8]);assert torch.isfinite(loss);(loss*.5).backward();losses.append(float(loss.detach()))
  optimizer.step()
  if step in [100,200,400]:measure(step,float(np.mean(losses)))
  save('T1_STATUS.json',dict(step=step,total=400,pid=os.getpid()))
 # Baselines measured on the exact loss grid, not treated as hardcoded expected constants.
 f,p=w.weak_ce.__globals__['spectrum'](torch.zeros(16,300,device=device));mask=(f>=.7)&(f<=3);target=torch.exp(-.5*((f[mask][None]*60-Y[:,None])/5)**2);target/=target.sum(-1,keepdim=True)
 final=history[-1]['weak_ce'];save('T1_COMPLETE.json',dict(final_weak_ce=final,run_W2b=final<4.,closed=final>=4.5,intermediate_no_launch=4<=final<4.5,flat_CE=float(np.log(int(mask.sum()))),perfect_target_CE=float(-(target*target.clamp_min(1e-30).log()).sum(-1).mean()),microbatch=8,effective_fixed_batch=16,static_not_a_selection_test_for_memorization=True))
if __name__=='__main__':
 try:main()
 except Exception:save('T1_ERROR.json',dict(traceback=traceback.format_exc()));raise
