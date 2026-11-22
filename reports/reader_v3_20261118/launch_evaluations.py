import os,json,hashlib,subprocess,sys,datetime,fcntl
from pathlib import Path
P=Path(__file__).resolve().parent
lock=(P/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (P/'LAUNCH_20261120.json').exists()
for name,digest in json.loads((P/'UPLOAD_MANIFEST_20261120.json').read_text()).items():
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
pre=json.loads((P/'PREFLIGHT_120GB_20261120.json').read_text())
now=datetime.datetime.now(datetime.timezone.utc)
assert (now-datetime.datetime.fromisoformat(pre['measured_at_utc'])).total_seconds()<3600
assert pre['scope_conservative_bytes']<120000000000 and pre['project_conservative_bytes']<500000000000
R=P.parent
jobs=[('conditions_20261120',10,['evaluate_conditions.py','--reports-root',str(R),'--numeric-root',str(R/'w2_btc_20261116/numeric_dev'),'--source-metadata',str(R/'claude_research_20261118/out/dev_video_meta.csv')]),('public_20261120',11,['evaluate_public.py','--spectra',str(R/'claude_research_20261118/out/public_spectra.npz')])]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
receipt=[]
for name,cpu,args in jobs:
 free=subprocess.check_output(['free','-g'],text=True);assert int(next(l for l in free.splitlines() if l.startswith('Mem:')).split()[-1])>=15
 out=P/name;out.mkdir(exist_ok=False)
 cmd=['taskset','-c',str(cpu),'nice','-n','10',sys.executable,str(P/'run_recorded.py'),str(out),sys.executable,str(P/args[0]),*args[1:],'--out',str(out)]
 log=(out/'process.log').open('xb')
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True,cwd=P)
 receipt.append(dict(name=name,pid=proc.pid,command=cmd,started_utc=now.isoformat(),free_g=free))
 (P/'LAUNCH_20261120.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
