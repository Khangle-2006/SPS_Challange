"""BTC train-person-only BN recalibration; parameters remain bitwise identical."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,json,time,traceback
import numpy as np
from common import *
from sweep import make_model
def setup_bn(model):
 import torch
 model.eval();bn=[]
 for name,m in model.named_modules():
  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
   assert m.track_running_stats;m.reset_running_stats();m.momentum=None;m.train();bn.append((name,m))
 assert bn
 return bn
def main():
 import torch
 lease=lock('gpu_execution.lock');oldlease=(S/'queue.lock').open('a');fcntl.flock(oldlease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard(True);device=gpu();plan=read(S/'PLAN.json');out=P/'adabn';out.mkdir(exist_ok=True)
 assert not (out/'STARTED.json').exists(),'Inspect prior launch; never duplicate/restart silently'
 save(out/'STARTED.json',dict(utc=now(),pid=os.getpid(),source_plan_sha256=sha(S/'PLAN.json'),code=hashes(),split_sha256=SPLIT))
 for run in plan['runs']:
  fold=run['fold'];dest=out/f'fold{fold}';dest.mkdir();data=BTCData(fold)
  assert set(data.people).isdisjoint(run['btc_val_people'])
  model=make_model(run,'cpu');before=model_static(model);params={n:p.detach().cpu().clone() for n,p in model.named_parameters()};model.to(device)
  # Report the unadapted counterpart on exactly this BTC fold, for a paired comparison.
  base=infer_native(model,data,fold,'padding_fixed_before_AdaBN',device);save(dest/'BEFORE_NATIVE.json',base)
  bn=setup_bn(model);records=sorted(data.train,key=lambda x:(x[0]['recording'],x[1]));used=[];started=time.monotonic()
  for offset in range(0,len(records),2):
   selected=records[offset:offset+2];xs=[data.clip(v,s) for v,s in selected]
   with torch.no_grad():w=model(torch.from_numpy(np.stack(xs)).to(device));assert torch.isfinite(w).all()
   used.extend(dict(person=v['person'],recording=v['recording'],start=s) for v,s in selected)
   save(out/'STATUS.json',dict(state='adapting',fold=fold,windows=len(used),total=len(records),utc=now(),pid=os.getpid()))
  assert all(torch.equal(p.detach().cpu(),params[n]) for n,p in model.named_parameters()),'AdaBN changed a parameter'
  model.eval();after=model_static(model);torch.save(dict(model=model.state_dict(),fold=fold,seed=2027,method='AdaBN',initial_checkpoint_sha256=run['checkpoint_sha256']),dest/'final.pt')
  model.to(device);rows=infer_native(model,data,fold,'AdaBN',device);save(dest/'NATIVE.json',rows)
  save(dest/'COMPLETE.json',dict(utc=now(),seconds=time.monotonic()-started,fold=fold,training_windows=used,training_people=data.people,validation_people=run['btc_val_people'],parameters_unchanged=True,BN=[dict(name=n,num_batches=int(m.num_batches_tracked),mean=m.running_mean.cpu().tolist(),variance=m.running_var.cpu().tolist()) for n,m in bn],static_before=before,static_after=after,checkpoint_sha256=sha(dest/'final.pt'),input_hashes=data.hashes,holdout_evaluations=0,C5prime='pending_A3'))
  print(json.dumps(dict(fold=fold,state='native_complete',utc=now())),flush=True)
  del model,data;torch.cuda.empty_cache();guard()
 save(out/'NATIVE_COMPLETE.json',dict(utc=now(),folds=5,holdout_evaluations=0,C5prime='pending_A3'))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'adabn/ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
