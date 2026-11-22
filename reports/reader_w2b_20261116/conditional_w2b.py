import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,json,time,inspect,fcntl,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent/'w2_btc_20261116';E=P.parent/'round1_execution_20261110';sys.path[:0]=[str(W),str(E)]
import train_w2 as w
def read(p):return json.loads(p.read_text())
def save(name,x):w.save(P/name,x)
def main():
 lock=(P/'conditional.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (P/'T1_COMPLETE.json').exists():
  if (P/'T1_ERROR.json').exists():save('W2B_BLOCKED.json',dict(reason='T1 technical error; inspect'));return
  time.sleep(15)
 # User's optimization diagnostic: training weak CE, not BatchNorm eval loss.
 value=read(P/'T1_HISTORY.json')[-1]['last_training_ce'];gate=dict(T1_training_CE_step400=value,launch_pair=value<4.,closed=value>=4.5,intermediate_no_launch=4<=value<4.5,threshold_fixed_before_T1_complete=True)
 save('T1_TRAIN_GATE.json',gate)
 if value>=4.:
  save('W2B_DISPOSITION.json',dict(state='closed' if value>=4.5 else 'not_authorized_intermediate',reason='T1 training CE did not pass <4.0; no additional training'));return
 import torch
 from runtime_shared_v4 import gpu
 gpu_lock=(E/'gpu_execution.lock').open('a')
 # T1 writes its final receipt immediately before process exit/releasing the GPU lease.
 fcntl.flock(gpu_lock,fcntl.LOCK_EX)
 assert time.time()-read(P/'PREFLIGHT.json')['measured_epoch']<3600,'Fresh inventory needed before W2b'
 device=gpu();plan=read(E/'PLAN_ELIGIBLE.json');dest=P/'W2b';dest.mkdir(exist_ok=True)
 source=inspect.getsource(w.one)
 a="optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=plan['weight_decay']);history=[]"
 b="optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=plan['weight_decay']);weak_optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=plan['weight_decay']);history=[]"
 assert source.count(a)==1;source=source.replace(a,b)
 a='sup.backward();del output,xs,ys';b='sup.backward();optimizer.step();weak_optimizer.zero_grad(set_to_none=True);del output,xs,ys';assert source.count(a)==1;source=source.replace(a,b)
 a='optimizer.step();losses.append';assert source.count(a)==1;source=source.replace(a,'weak_optimizer.step();losses.append')
 source=source.replace('del model,optimizer,provider,data','del model,optimizer,weak_optimizer,provider,data')
 source=source.replace('shuffle=shuffle,shuffle_seed=0','shuffle=shuffle,weak_lr=3e-4,two_optimizers=True,shuffle_seed=0')
 (P/'W2b_one_frozen.py.txt').write_text(source);ns=dict(vars(w));ns['P']=dest;ns['__file__']=str(P/'W2b_one_frozen.py.txt')
 def budget():
  assert sum(f.stat().st_size for f in P.rglob('*') if f.is_file())<200_000_000
  w.deadline(plan)
 ns['check_budget']=budget;exec(compile(source,'W2b_two_adamw','exec'),ns);one=ns['one'];save('W2B_STARTED.json',dict(pid=os.getpid(),two_independent_AdamW=True,phase_lr=1e-4,weak_lr=3e-4))
 for shuffled in [False,True]:one(plan,2,shuffled,device)
 if not all((dest/f/ 'COMPLETE.json').exists() for f in ['fold2','fold2_shuffle']):save('W2B_DISPOSITION.json',dict(state='closed',reason='static validity failure'));return
 h=read(dest/'fold2/history.json');true=read(dest/'fold2/COMPLETE.json');shuf=read(dest/'fold2_shuffle/COMPLETE.json');drop=h[0]['weak_ce']-h[-1]['weak_ce'];improvement=shuf['MAE']-true['MAE'];passed=drop>=.3 and improvement>=1
 save('W2B_EXPANSION_GATE.json',dict(passed=passed,training_CE_drop=drop,true_MAE=true['MAE'],shuffle_MAE=shuf['MAE'],true_improvement=improvement))
 if passed:
  for fold in [0,1,3,4]:one(plan,fold,False,device)
 save('W2B_DISPOSITION.json',dict(state='fivefold_queue_terminal' if passed else 'closed_after_pair',no_extra_training=True))
if __name__=='__main__':
 try:main()
 except Exception:save('W2B_ERROR.json',dict(traceback=traceback.format_exc()));raise
