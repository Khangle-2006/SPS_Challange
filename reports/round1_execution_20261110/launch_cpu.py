"""Launch one independent CPU task with a receipt; never restart a partial task."""
import os,sys,subprocess
from common import *
jobs={'aipex':('aipex_data.py','4','storage_pilot'), 'warm':('warm_training.py','2,3','storage_pilot'), 'c5eval':('evaluate_c5.py','8','hr')}
task=sys.argv[1];script,cpus,environment=jobs[task]
guard(True)
extra=read(P/'BUDGET_AIPEX_ADDENDUM.json')
assert extra['new_total_estimate_bytes']<=84000000000
assert not (P/(task+'_LAUNCH.json')).exists(),'Inspect existing launch; no automatic restart'
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 assert str(P/script) not in cmd,'Duplicate CPU worker'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['nice','-n','15','taskset','-c',cpus,f'/data_pool/khangle/facevitals/envs/{environment}/bin/python',str(P/script)]
with (P/(task+'.log')).open('ab') as log:
 p=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(P/(task+'_LAUNCH.json'),dict(pid=p.pid,utc=now(),command=cmd,code=hashes(),holdout_evaluations=0))
print(task,p.pid,flush=True)
