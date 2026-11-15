import sys,os,fcntl,subprocess,json
from pathlib import Path
import worker_budget_v3 as worker
P=Path(__file__).resolve().parent;server=sys.argv[1]
assert server in ['P100','A100']
assert not (P/('FEATURES_COMPLETE_'+server+'.json')).exists()
assert not (P/('FEATURE_ERROR_'+server+'.json')).exists()
lock=(P/('features_'+server+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for pid in Path('/proc').iterdir():
    if not pid.name.isdigit():continue
    try:cmd=(pid/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    assert not (str(P/'features_budget_v3.py')+' '+server in cmd)
for name,h in worker.read(P/'FEATURE_CODE_HASHES_BUDGET_V3.json').items():assert worker.sha(P/name)==h
budget=worker.guard(server,True);cpu=8 if server=='P100' else 3
assert cpu in os.sched_getaffinity(0)
lock.close();log=(P/('features_'+server+'.log')).open('ab')
proc=subprocess.Popen(['nice','-n','15','taskset','-c',str(cpu),sys.executable,str(P/'features_budget_v3.py'),server],cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
a=dict(pid=proc.pid,utc=worker.now(),cpu=cpu,server=server,budget=budget,code_sha256=worker.sha(P/'FEATURE_CODE_HASHES_BUDGET_V3.json'))
worker.save(P/('FEATURE_LAUNCH_'+server+'.json'),a);print(json.dumps(a))
