"""Finish a pre-outcome codec eligibility inventory; no model results are read."""
import os,time,traceback,hashlib,copy
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common import *

def main():
 lease=lock('finalize_eligibility.lock');guard(True);assert not (P/'constraint_preparation/FINALIZER_STARTED.json').exists()
 save(P/'constraint_preparation/FINALIZER_STARTED.json',dict(utc=now(),pid=os.getpid(),source_plan_sha256=sha(P/'PLAN.json')))
 original=read(P/'PLAN.json');total=sum(len(j) for r in original['runs'] for j in r['public_jobs'].values());assert total==8566
 while True:
  ready=[];missing=[];failures=[]
  for run in original['runs']:
   for ds,jobs in run['public_jobs'].items():
    for j in jobs:
     key=json.dumps([ds,j['recording'],float(j['start']),j['recipe'],bool(j['lowres']),run['fold']],separators=(',',':'));digest=hashlib.sha256(key.encode()).hexdigest();cache=P/'training_cache'/str(run['fold'])/(digest+'.json');failure=P/'constraint_preparation/failures'/(digest+'.json')
     assert not (cache.exists() and failure.exists()),'Conflicting eligibility'
     if cache.exists():ready.append((run['fold'],ds,j,cache))
     elif failure.exists():failures.append(failure)
     else:missing.append(key)
  save(P/'constraint_preparation/ELIGIBILITY_STATUS.json',dict(utc=now(),eligible=len(ready),constraint_rejected=len(failures),missing=len(missing),total=total,pid=os.getpid()))
  if not missing:break
  time.sleep(30)
 guard();plan=copy.deepcopy(original);counts=[]
 for run in plan['runs']:
  for ds in run['public_jobs']:
   kept=[j for f,d,j,c in ready if f==run['fold'] and d==ds];assert kept and any(j['recipe']=='c5_proxy' for j in kept),'Training domain or C5 augmentation lost'
   counts.append(dict(fold=run['fold'],dataset=ds,before=len(run['public_jobs'][ds]),after=len(kept)));run['public_jobs'][ds]=kept
 checks={}
 for fold,ds,j,file in ready:
  a=read(file);assert sha(file.with_suffix('.npz'))==a['sha256'];assert a['encoding']['recipe']!='c5_proxy' or a['encoding']['whole_file_kbps']<=500
  checks[str(file)]=sha(file)
 for f in failures:
  a=read(f);assert a['eligible'] is False and a['whole_file_kbps']>500 and a['no_retuning'] and a['no_metric_selection'];checks[str(f)]=sha(f)
 assert not (P/'PLAN_ELIGIBLE.json').exists()
 plan['eligibility_amendment']=dict(source_plan_sha256=sha(P/'PLAN.json'),reason='Exclude deterministic codec samples exceeding unchanged whole-file500kbps criterion; no CRF/rate/model/gate tuning',counts=counts,excluded_receipts=[str(f) for f in failures],steps_epochs_seeds_folds_unchanged=True)
 save(P/'PLAN_ELIGIBLE.json',plan);save(P/'TRAIN_CACHE_ELIGIBLE_COMPLETE.json',dict(utc=now(),eligible=len(ready),constraint_rejected=len(failures),total=total,input_hashes=checks,plan_sha256=sha(P/'PLAN_ELIGIBLE.json')))
 frozen=read(P/'FROZEN.json')
 for f,h in frozen.items():assert sha(f)==h
 frozen.update({str(P/f):sha(P/f) for f in ['PLAN_ELIGIBLE.json','TRAIN_CACHE_ELIGIBLE_COMPLETE.json','train_sweep_eligible_v2.py','start_sweep_eligible_v2.py','EXECUTION_FROZEN_V2.json']})
 save(P/'FROZEN_ELIGIBLE.json',frozen);save(P/'constraint_preparation/FINALIZER_COMPLETE.json',dict(utc=now(),eligible=len(ready),constraint_rejected=len(failures),plan_sha256=sha(P/'PLAN_ELIGIBLE.json'),no_metric_selection=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'constraint_preparation/FINALIZER_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
