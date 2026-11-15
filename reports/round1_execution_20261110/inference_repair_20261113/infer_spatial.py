from infer_original import *
from classical_repair import run as classical
from crop_io import cached
import io

def main():
 controller=lock('spatial_controller.lock');assert not (P/'SPATIAL_START.json').exists();save(P/'SPATIAL_START.json',dict(pid=os.getpid(),utc=now(),code=sha(__file__),no_training=True));inv=None;models=None;gpu_lease=None;done=0;began=time.time()
 for j in manifest():
  while not (P/'orientation'/j['key']/'COMPLETE.json').exists():
   assert time.time()-began<24*3600,'24H_OPERATION_BOUND'
   assert not list(P.glob('ORIENTATION*_ERROR.json')),'ORIENTATION_WORKER_FAILED'
   save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,total=1488,key=j['key'],state='waiting_orientation'));time.sleep(10)
  for c in CONDITIONS:
   out=P/'spatial'/j['key']/c;assert not out.exists();out.mkdir(parents=True)
   try:old,meta,d=cached(j,c)
   except ValueError as e:save(out/'EXCLUDED.json',dict(reason=str(e)));done+=1;continue
   save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,total=1488,key=j['key'],condition=c,state='CPU_source_reproduction'))
   command=['nice','-n','15','taskset','-c','14','/data_pool/khangle/facevitals/envs/storage_pilot/bin/python','-u',str(P/'spatial_stream.py'),j['key'],c,str(out/'prepare')]
   with (out/'prepare.log').open('wb') as log:raw=subprocess.check_output(command,stderr=log)
   a=dict(np.load(io.BytesIO(raw)));del raw
   if models is None:
    while not (P/'ORIGINAL_COMPLETE.json').exists():
     assert not (P/'ORIGINAL_ERROR.json').exists(),'ORIGINAL_INFERENCE_FAILED'
     assert time.time()-began<24*3600
     save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,key=j['key'],condition=c,state='waiting_original_GPU_inference'));time.sleep(10)
    gpu_lease=lock('inference_gpu.lock');samples=setup();inv=inventory();models=load_models(inv);save(P/'SPATIAL_GPU_STARTED.json',dict(utc=now(),pid=os.getpid(),free_MiB=samples,no_training=True))
   k=int(a['quarter_turn_ccw']);aliases={}
   for variant in ['rotate','mask','rotate_mask']:
    target=out/variant
    if variant=='rotate' and k==0:aliases[variant]=str(P/'original'/j['key']/c);continue
    if variant=='rotate_mask' and k==0:aliases[variant]=str(out/'mask');continue
    save(P/'SPATIAL_STATUS.json',dict(utc=now(),completed=done,total=1488,key=j['key'],condition=c,input=variant,state='GPU_inference'))
    evaluate(j,c,models,inv,a[variant],a['t'],a['valid'],meta,target)
    rows=read(target/'PREDICTIONS.json')
    for r in rows:r['input']=variant
    save(target/'PREDICTIONS.json',rows);receipt=read(target/'COMPLETE.json');receipt['rows_sha256']=sha(target/'PREDICTIONS.json');receipt['input']=variant;save(target/'COMPLETE.json',receipt)
    classical(j,c,variant,a[variant],a['t'],a['valid'],target)
   save(out/'COMPLETE.json',dict(utc=now(),aliases=aliases,orientation_sha256=sha(P/'orientation'/j['key']/'COMPLETE.json'),prepared_sha256=sha(out/'prepare/PREPARED.json'),zero_turn_aliases_verified=True));done+=1;del a
  budget()
 save(P/'SPATIAL_COMPLETE.json',dict(utc=now(),completed=done,no_training=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'SPATIAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
