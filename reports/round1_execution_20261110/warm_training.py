"""Prepare deterministic lossless crop cache; independent of GPU and outcome."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import traceback
from common import *
from training_provider import DiskProvider
def main():
 lease=lock('training_cache.lock');guard(True);plan=read(S/'PLAN.json');index=read(R/'t8_oval_20261031/training_index.json')
 save(P/'TRAIN_CACHE_START.json',dict(pid=os.getpid(),utc=now(),code=hashes(),source_plan_sha256=sha(S/'PLAN.json')))
 for run in plan['runs']:
  provider=DiskProvider(index,run['btc_bpp_quartiles'],max_bytes=300000000);done=0
  for ds,jobs in run['public_jobs'].items():
   for j in jobs:
    guard();provider.prepare(ds,j['recording'],j['start'],j['recipe'],j['lowres'],run['fold']);done+=1
    save(P/'TRAIN_CACHE_STATUS.json',dict(fold=run['fold'],completed=done,total=sum(len(z) for z in run['public_jobs'].values()),utc=now(),pid=os.getpid()))
  del provider
 save(P/'TRAIN_CACHE_COMPLETE.json',dict(utc=now(),folds=5))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'TRAIN_CACHE_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
