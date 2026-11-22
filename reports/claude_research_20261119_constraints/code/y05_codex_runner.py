"""Operational runner for the user's unchanged y05 script. Writes only under W."""
import os,sys,json,time,subprocess,fcntl,hashlib,re,traceback,ast
from pathlib import Path
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
LOG=W/'logs'; SCRIPT=W/'code/y05_openrppg.py'; PY=W/'venv_orppg/bin/python'
LOCK=(LOG/'y05_codex.lock').open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
tree=ast.parse(SCRIPT.read_text());MODELS=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in n.targets))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
FROZEN={str(p.relative_to(W)):sha(p) for p in [SCRIPT,W/'PREREG.md']}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',KERAS_BACKEND='jax',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
for key,d in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('KERAS_HOME','keras'),('MPLCONFIGDIR','matplotlib'),('JAX_COMPILATION_CACHE_DIR','jax')]:
 p=W/'out/y05_runtime'/d;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
STATE=dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),frozen=FROZEN,mode='preflight',jobs={},errors=[],holdout_evaluations=0)
def save():
 STATE['updated_epoch']=time.time()
 p=LOG/'y05_codex_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(STATE,indent=2));tmp.replace(p)
def avail():
 raw=subprocess.check_output(['free','-g'],text=True)
 value=int(next(l for l in raw.splitlines() if l.startswith('Mem:')).split()[-1])
 return value,raw
def guard():
 n,raw=avail()
 with (LOG/'y05_resource_checks.log').open('a') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n'+raw)
 return n>=15
def rss(pid):
 try:return int(next(l for l in Path('/proc/'+str(pid)+'/status').read_text().splitlines() if l.startswith('VmRSS:')).split()[1])*1024
 except (FileNotFoundError,StopIteration):return 0
def launch(mode,name=None):
 while not guard():STATE['mode']='waiting_RAM_15GB';save();time.sleep(20)
 for p,h in FROZEN.items():assert sha(W/p)==h,'Input code/prereg changed'
 suffix=mode if name is None else 'run_'+name
 log=LOG/('y05_'+suffix+'.log')
 f=log.open('x')
 args=['taskset','-c','0-11' if mode=='eval' else '0-11,36-47','nice','-n','0',str(PY),'-u',str(SCRIPT),mode]+([] if name is None else [name])
 p=subprocess.Popen(args,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
 key=suffix;STATE['jobs'][key]=dict(pid=p.pid,args=args,started=time.time(),log=str(log),peak_rss=0);save()
 return p,key
def inspect(p,key,limited=False):
 j=STATE['jobs'][key];r=rss(p.pid);j['peak_rss']=max(j['peak_rss'],r)
 if limited and r>4_000_000_000 and p.poll() is None:
  j['stop_reason']='RSS exceeded 4 GB';p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  STATE['errors'].append(dict(job=key,error=j['stop_reason'],rss=r))
 if p.poll() is not None:
  j.update(exit_code=p.returncode,finished=time.time());return True
 return False
def main():
 assert len(list((W/'out/x10').glob('*.npz')))==201
 assert not list((W/'out/y05').glob('*.npz')),'Existing output: inspect, do not overwrite'
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   c=(d/'cmdline').read_bytes().split(b'\0')
   assert not (len(c)>2 and str(SCRIPT).encode() in c),'Existing y05 process'
  except (FileNotFoundError,PermissionError):pass
 STATE['mode']='pilot';p,key=launch('pilot')
 while not inspect(p,key):save();time.sleep(3)
 save();text=(LOG/'y05_pilot.log').read_text(errors='replace')
 if p.returncode!=0:raise RuntimeError('Pilot process failed; preserve log, no automatic rerun')
 kept=[];decisions=[]
 for name in MODELS:
  lines=[l for l in text.splitlines() if l.startswith(name+' ')]
  if not lines:reason='No completed pilot row'
  elif any('ERROR' in l for l in lines):reason='ERROR in pilot'
  else:
   secs=[float(x) for x in re.findall(r': ([0-9.]+)s peak ',lines[-1])]
   reason='PASS' if len(secs)==2 and max(secs)<=60 else 'over 60 seconds/video or missing times'
  decisions.append(dict(model=name,decision=reason,rows=lines))
  if reason=='PASS':kept.append(name)
 (LOG/'y05_models_kept.txt').write_text(''.join(n+'\n' for n in kept))
 (LOG/'y05_pilot_decisions.json').write_text(json.dumps(decisions,indent=2))
 STATE.update(mode='full_runs',kept=kept);save();pending=list(kept);active={};successful=[]
 while pending or active:
  while pending and len(active)<3:
   if not guard():break
   name=pending.pop(0);p,key=launch('run',name);active[name]=(p,key)
  for name,(p,key) in list(active.items()):
   if inspect(p,key,True):
    out=W/'out/y05'/(name+'.npz')
    if p.returncode==0 and out.exists():
     import numpy as np
     with np.load(out,allow_pickle=False) as z:
      assert z['S'].shape==(218,277) and len(z['keys'])==218
      STATE['jobs'][key]['finite_videos']=int(np.isfinite(z['S']).all(1).sum())
     STATE['jobs'][key]['output_sha256']=sha(out);successful.append(name)
    else:
     STATE['errors'].append(dict(job=key,exit_code=p.returncode,error='Full run did not complete'))
     if out.exists():
      dest=W/'out/y05_rejected';dest.mkdir(exist_ok=True);out.rename(dest/out.name)
    del active[name]
  STATE.update(mode='full_runs',active=list(active),pending=pending,successful=successful);save();time.sleep(3)
 if not successful:raise RuntimeError('No successful full models for eval')
 STATE['mode']='eval';p,key=launch('eval')
 while not inspect(p,key):save();time.sleep(5)
 assert p.returncode==0,'Eval failed; preserve log'
 for fn,h in FROZEN.items():assert sha(W/fn)==h
 STATE.update(mode='complete',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
 (LOG/'y05_CODEX_COMPLETE.json').write_text(json.dumps(STATE,indent=2))
try:main()
except Exception:
 STATE.update(mode='error',traceback=traceback.format_exc());save();print(STATE['traceback'],flush=True)

