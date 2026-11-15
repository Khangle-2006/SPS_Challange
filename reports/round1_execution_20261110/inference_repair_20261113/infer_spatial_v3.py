"""Same spatial factorial, four bounded RAM producers, one exclusive inference consumer."""
from infer_spatial import *
from classical_repair_v2 import run as classical
from concurrent.futures import ThreadPoolExecutor

def producer(j,c,slot,began):
 while not (P/'orientation'/j['key']/'COMPLETE.json').exists():
  assert time.time()-began<36*3600 and not list(P.glob('ORIENTATION*_ERROR.json'))
  save(P/f'SPATIAL_PRODUCER{slot}.json',dict(utc=now(),key=j['key'],condition=c,state='waiting_orientation'));time.sleep(10)
 if j['dataset']=='UBFC2' and c=='c5':
  while not (P/'legacy_sources'/(j['recording']+'.mp4')).exists():
   assert time.time()-began<36*3600;save(P/f'SPATIAL_PRODUCER{slot}.json',dict(utc=now(),key=j['key'],condition=c,state='waiting_verified_legacy_source_relay'));time.sleep(10)
 out=P/'spatial'/j['key']/c;assert not out.exists();out.mkdir(parents=True)
 try:old,meta,d=cached(j,c)
 except ValueError as e:save(out/'EXCLUDED.json',dict(reason=str(e)));return j,c,out,None,None
 save(P/f'SPATIAL_PRODUCER{slot}.json',dict(utc=now(),key=j['key'],condition=c,state='source_reproduction'))
 command=['nice','-n','15','taskset','-c',str(4+slot),'/data_pool/khangle/facevitals/envs/storage_pilot/bin/python','-u',str(P/'spatial_stream.py'),j['key'],c,str(out/'prepare')]
 with (out/'prepare.log').open('wb') as log:raw=subprocess.check_output(command,stderr=log)
 a=dict(np.load(io.BytesIO(raw)));save(P/f'SPATIAL_PRODUCER{slot}.json',dict(utc=now(),key=j['key'],condition=c,state='ready_in_RAM'));return j,c,out,a,meta

def fresh_wait(began):
 while (datetime.now(timezone.utc)-datetime.fromisoformat(read(E/'A100_INVENTORY.json')['measured_utc'])).total_seconds()>3600:
  assert time.time()-began<36*3600;save(P/'SPATIAL_STATUS.json',dict(utc=now(),state='waiting_fresh_cross_server_inventory',no_GPU_start=True));time.sleep(15)

def main():
 controller=lock('spatial_controller.lock');assert not (P/'SPATIAL_START.json').exists();budget(True);save(P/'SPATIAL_START.json',dict(pid=os.getpid(),utc=now(),code=sha(__file__),no_training=True,CPU_producers=4,bounded_RAM_jobs=4));jobs=[(j,c) for j in manifest() for c in CONDITIONS];began=time.time();inv=None;models=None;gpu_lease=None;next_job=0;done=0
 with ThreadPoolExecutor(max_workers=4) as pool:
  pending={}
  for slot in range(4):
   j,c=jobs[next_job];pending[slot]=pool.submit(producer,j,c,slot,began);next_job+=1
  while pending:
   # Consume completed producers; never launch more than four RAM jobs.
   slot=next((s for s,f in pending.items() if f.done()),None)
   if slot is None:assert time.time()-began<36*3600;time.sleep(2);continue
   j,c,out,a,meta=pending.pop(slot).result()
   if a is not None:
    if models is None:
     while not (P/'ORIGINAL_COMPLETE.json').exists():
      assert not (P/'ORIGINAL_ERROR.json').exists() and time.time()-began<36*3600
      save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,state='waiting_original_GPU_inference'));time.sleep(10)
     fresh_wait(began);gpu_lease=lock('inference_gpu.lock');samples=setup();inv=inventory();models=load_models(inv);save(P/'SPATIAL_GPU_STARTED.json',dict(utc=now(),pid=os.getpid(),free_MiB=samples,no_training=True))
    k=int(a['quarter_turn_ccw']);aliases={}
    for variant in ['rotate','mask','rotate_mask']:
     target=out/variant
     if variant=='rotate' and k==0:aliases[variant]=str(P/'original'/j['key']/c);continue
     if variant=='rotate_mask' and k==0:aliases[variant]=str(out/'mask');continue
     save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,total=1488,key=j['key'],condition=c,input=variant,state='GPU_inference'))
     evaluate(j,c,models,inv,a[variant],a['t'],a['valid'],meta,target);rows=read(target/'PREDICTIONS.json')
     for r in rows:r['input']=variant
     save(target/'PREDICTIONS.json',rows);receipt=read(target/'COMPLETE.json');receipt.update(rows_sha256=sha(target/'PREDICTIONS.json'),input=variant);save(target/'COMPLETE.json',receipt);classical(j,c,variant,a[variant],a['t'],a['valid'],target)
    save(out/'COMPLETE.json',dict(utc=now(),aliases=aliases,orientation_sha256=sha(P/'orientation'/j['key']/'COMPLETE.json'),prepared_sha256=sha(out/'prepare/PREPARED.json'),zero_turn_aliases_verified=True));del a
   done+=1
   if done%4==0:budget()
   if next_job<len(jobs):
    jj,cc=jobs[next_job];pending[slot]=pool.submit(producer,jj,cc,slot,began);next_job+=1
 save(P/'SPATIAL_COMPLETE.json',dict(utc=now(),completed=done,no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'SPATIAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
