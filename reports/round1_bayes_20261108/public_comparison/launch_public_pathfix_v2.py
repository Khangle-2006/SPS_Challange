"""Launch one bounded CPU worker after actual lock, source bundle and budget checks."""
import sys,os,fcntl,subprocess,json
from pathlib import Path
import worker_pathfix_v2 as worker
P=Path(__file__).resolve().parent
server=sys.argv[1]
assert server in ['P100','A100']
smoke=(P/'mesa_smoke.log').read_text()
assert 'renderer: llvmpipe' in smoke and 'delegate for CPU' in smoke
assert not (P/('COMPLETE_'+server+'.json')).exists()
assert not (P/('ERROR_'+server+'.json')).exists(), 'Inspect prior failure; never automatic restart'
lock=(P/('worker_'+server+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for pid in Path('/proc').iterdir():
    if not pid.name.isdigit():continue
    try:cmd=(pid/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    assert not (str(P/'worker_pathfix_v2.py')+' '+server in cmd), 'worker process already present'
for name,h in worker.read(P/'CODE_HASHES_PATHFIX_V2.json').items():assert worker.sha(P/name)==h
budget=worker.guard(server,True)
cpu=7 if server=='P100' else 2
assert cpu in os.sched_getaffinity(0)
log=(P/('worker_'+server+'.log')).open('ab')
lock.close()
proc=subprocess.Popen(['nice','-n','15','taskset','-c',str(cpu),sys.executable,str(P/'worker_pathfix_v2.py'),server],cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,utc=worker.now(),server=server,cpu=cpu,budget=budget,code_manifest_sha256=worker.sha(P/'CODE_HASHES_PATHFIX_V2.json'))
worker.save(P/('LAUNCH_'+server+'.json'),receipt)
print(json.dumps(receipt))
