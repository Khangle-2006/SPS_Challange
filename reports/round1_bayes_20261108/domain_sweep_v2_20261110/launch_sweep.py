"""No wait daemon or hidden retries: authorized fresh preflight, then one queue."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import fcntl,subprocess,sys,json,math
from pathlib import Path
from datetime import datetime,timezone
from sweep import P,ROOT,OLD,read,sha,budget,deadline,save,now
def main():
 plan=read(P/'PLAN.json');deadline(plan)
 assert not (P/'ERROR.json').exists() and not (P/'LAUNCH.json').exists(),'Inspect prior launch/error, no automatic restart'
 for file,h in read(P/'FROZEN.json').items():assert sha(Path(file) if Path(file).is_absolute() else P/file)==h
 assert read(P/'CPU_VALIDATION.json')['passed'] and read(P/'CPU_VALIDATION.json')['plan_sha256']==sha(P/'PLAN.json') and read(P/'CODEC_SMOKE.json')['passed']
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  assert str(P/'sweep.py') not in cmd,'Existing sweep process; inspect instead of duplicate'
 leases=[]
 for path in [ROOT/'t8_oval_20261031/runs/queue.lock',ROOT/'t8_polarity_v3_20261102/queue.lock',OLD/'v4_repaired/queue.lock',OLD/'v4_padding_pilot/queue.lock',P.parent/'domain_sweep_20261109/queue.lock',P/'queue.lock']:
  h=path.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(h)
 report=budget(plan,True)
 import t8_run
 resources=t8_run.resource_guard()
 for h in leases:h.close() # Worker reacquires nonblocking; a competing launcher fails closed.
 seconds=max(1,math.floor((datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()))
 log=(P/'training.log').open('ab')
 cmd=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s','nice','-n','10','taskset','-c','0,1',sys.executable,str(P/'sweep.py')]
 proc=subprocess.Popen(cmd,cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(P/'LAUNCH.json',dict(supervisor_pid=proc.pid,utc=now(),command=cmd,storage=report,resources=resources,plan_sha256=sha(P/'PLAN.json'),status='spawned_actual_GPU_start_requires_process_observation'))
 print(json.dumps(read(P/'LAUNCH.json')))
if __name__=='__main__':main()
