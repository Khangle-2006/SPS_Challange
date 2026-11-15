"""Disjoint CPU shards; same frozen encoder functions and identical cache keys."""
import os,sys,time,traceback,signal
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common import *

def worker(kind,shard,count):
 if kind=='c5':
  from c5prime_v2 import jobs,one
  todo=[(j,v) for j in jobs() for v in ['default','bf0']]
  for i,(j,v) in enumerate(todo):
   if i%count!=shard:continue
   one(j,v);save(P/f'parallel/c5_{shard}_STATUS.json',dict(index=i,total=len(todo),recording=j['recording'],variant=v,utc=now(),pid=os.getpid()))
 else:
  from training_provider import DiskProvider
  plan=read(S/'PLAN.json');index=read(R/'t8_oval_20261031/training_index.json');ordinal=0
  for run in plan['runs']:
   provider=DiskProvider(index,run['btc_bpp_quartiles'],300000000)
   for ds,jobs in run['public_jobs'].items():
    for j in jobs:
     i=ordinal;ordinal+=1
     if i%count!=shard:continue
     guard();provider.prepare(ds,j['recording'],j['start'],j['recipe'],j['lowres'],run['fold'])
     save(P/f'parallel/warm_{shard}_STATUS.json',dict(fold=run['fold'],global_index=i,utc=now(),pid=os.getpid()))
   del provider
 save(P/f'parallel/{kind}_{shard}_COMPLETE.json',dict(utc=now(),pid=os.getpid()))

def coordinator():
 leases=[lock('c5prime.lock'),lock('training_cache.lock'),lock('parallel_prepare.lock')];guard(True)
 assert not (P/'parallel/STARTED.json').exists(),'No automatic restart'
 ram=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
 assert ram>=24000000000 and os.cpu_count()>=32,'Parallel resource guard'
 children=[];code=hashes()
 for kind,count,first in [('c5',4,10),('warm',8,16)]:
  for shard in range(count):
   cpus=str(first+shard) if kind=='c5' else f'{first+2*shard},{first+2*shard+1}'
   cmd=['nice','-n','15','taskset','-c',cpus,sys.executable,str(P/'parallel_prepare.py'),kind,str(shard),str(count)]
   d=P/'parallel';d.mkdir(exist_ok=True)
   with (d/f'{kind}_{shard}.log').open('ab') as log:p=subprocess.Popen(cmd,cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   children.append((kind,shard,p));save(d/'STARTED.json',dict(utc=now(),pid=os.getpid(),children=[dict(kind=k,shard=s,pid=q.pid) for k,s,q in children],code=code))
 while True:
  for kind in ['c5','warm']:
   group=[p for k,s,p in children if k==kind]
   if all(p.poll()==0 for p in group):
    if kind=='c5' and not (P/'C5_COMPLETE.json').exists():
     from c5prime_v2 import jobs
     rows=[]
     for j in jobs():
      for v in ['default','bf0']:
       a=read(P/'c5prime'/j['dataset']/j['recording']/v/'COMPLETE.json');rows.append(dict(dataset=j['dataset'],recording=j['recording'],variant=v,state=a['state'],bitrate_pass=a.get('bitrate_pass')))
     assert len(rows)==750;save(P/'C5_COMPLETE.json',dict(utc=now(),rows=rows,holdout_evaluations=0))
    if kind=='warm' and not (P/'TRAIN_CACHE_COMPLETE.json').exists():save(P/'TRAIN_CACHE_COMPLETE.json',dict(utc=now(),folds=5,shards=8))
  statuses=[p.poll() for k,s,p in children]
  if all(s is not None for s in statuses):
   assert all(s==0 for s in statuses),str(statuses)
   save(P/'parallel/COMPLETE.json',dict(utc=now()));break
  # A failed shard is preserved; other disjoint workers may finish useful work.
  if any(s not in [None,0] for s in statuses):save(P/'parallel/ERROR.json',dict(utc=now(),exit_codes=statuses,action='inspect failed shard; no automatic restart'))
  time.sleep(20)
if __name__=='__main__':
 try:
  if len(sys.argv)==1:coordinator()
  else:worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
 except Exception as e:
  suffix='_'.join(sys.argv[1:3]) or 'controller';save(P/f'parallel/{suffix}_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
