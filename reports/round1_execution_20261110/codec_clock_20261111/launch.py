import os,sys,json,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
lease=(P/'launch.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (P/'READOUT_AUDIT.json').exists()
assert not (P/'WORKERS.json').exists(),'Inspect existing launch, never duplicate'
from audit_clock import guard,sha,binding,read
assert binding()==read(P/'PLAN.json')['bindings']
receipt=guard(True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
workers=[]
for mode,cpu in [('real',7),('synthetic',8)]:
 with (P/(mode+'.log')).open('xb') as log:
  args=['nice','-n','15','taskset','-c',str(cpu),sys.executable,str(P/'audit_clock.py'),mode]
  p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=P)
  workers.append(dict(mode=mode,pid=p.pid,args=args))
out=dict(utc=datetime.now(timezone.utc).isoformat(),workers=workers,guard=receipt,plan_sha=sha(P/'PLAN.json'),no_GPU=True)
(P/'WORKERS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
