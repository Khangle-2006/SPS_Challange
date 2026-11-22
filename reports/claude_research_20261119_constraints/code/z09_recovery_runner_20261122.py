"""Operational launcher only; y05 and z09 user scripts are unchanged."""
import os,json,time,hashlib,subprocess,fcntl,signal,traceback,re
from pathlib import Path
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints');L=W/'logs'
P=L/'z09_recovery_STATUS_20261122.json'
lock=(L/'z09_recovery_20261122.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

assert not P.exists(),'Existing status; no duplicate or blind restart'
pre=json.loads((L/'y05z09_PREFLIGHT_20261122.json').read_text());assert pre['pass_budget']
base=json.loads((L/'y13_y14_PREFLIGHT.json').read_text())
HR=Path('/data_pool/khangle/facevitals/envs/hr/bin/python');PY=W/'venv_orppg/bin/python'
scripts={'y05':W/'code/y05_openrppg.py','z09':W/'code/z09_factorizephys_pilot.py'}
first=[]
slow=[]
weights=['iBVP_FactorizePhys_FSAM_Res','PURE_FactorizePhys_FSAM_Res','SCAMPS_FactorizePhys_FSAM_Res','UBFC-rPPG_FactorizePhys_FSAM_Res']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
state=dict(pid=os.getpid(),started_utc=now(),mode='preflight',jobs={},errors=[],frozen=pre['frozen'],holdout_evaluations=0,max_parallel=3,rss_cap=8000000000)
def save():
 state['updated_utc']=now();write(P,state)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',KERAS_BACKEND='jax',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
for k,d in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('KERAS_HOME','keras'),('MPLCONFIGDIR','matplotlib'),('JAX_COMPILATION_CACHE_DIR','jax'),('TORCH_HOME','torch'),('HF_HOME','huggingface')]:
 p=W/'out/y05_runtime'/d;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
assert json.loads((L/'z09_DEPENDENCY_IMPORT_PASS_20261122.json').read_text())['import_pass']
env['PYTHONPATH']=str(W/'out/z09_dependency_overlay_20261122')
prior={str(W/'out/y05'/n):a['sha256'] for n,a in pre['existing'].items()}
prior[str(L/'y05_eval.log')]=pre['old_eval_sha']
for p in (W/'out/z09').glob('*/*.json'):prior[str(p)]=sha(p)
write(L/'z09_recovery_PRIOR_HASHES_20261122.json',prior)
for m in first+slow:assert not (W/'out/y05'/(m+'.npz')).exists(),'Unexpected existing target model; inspect first'


inputs={}
for p in list((W/'out/x10').glob('BTC_*.npz'))+[W/'out/x15/stabsq14'/(k+'.npz') for k in pre['pilot_keys']]:
 st=p.stat();inputs[str(p)]=[st.st_size,st.st_mtime_ns]
write(L/'z09_recovery_INPUT_STATS_20261122.json',inputs)
def procs():
 out={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   a=dict(x.split(':',1) for x in (d/'status').read_text().splitlines() if ':' in x)
   out[int(d.name)]=dict(ppid=int(a['PPid']),rss=int(a.get('VmRSS','0 kB').split()[0])*1024,args=(d/'cmdline').read_bytes().split(b'\0'))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return out
def externals(active):
 own={p.pid for p,j in active.values()};out=set()
 for pid,v in procs().items():
  if any(Path(x.decode(errors='replace')).name in [p.name for p in scripts.values()] for x in v['args'] if x):
   try:g=os.getpgid(pid)
   except ProcessLookupError:continue
   if g not in own:out.add(g)
 return out
def guard():
 raw=subprocess.check_output(['free','-g'],text=True);available=int(next(x for x in raw.splitlines() if x.startswith('Mem:')).split()[-1])
 with (L/'z09_recovery_RESOURCES_20261122.log').open('a') as f:f.write(now()+'\n'+raw)
 if available<15:state['wait_reason']='available RAM below15GB';return False
 from datetime import datetime,timezone
 a=json.loads((L/'y05z09_A100_INVENTORY.json').read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'].replace('Z','+00:00'))).total_seconds()
 if age>3600:state['wait_reason']='fresh A100 inventory required';return False
 pp=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in [W.parent.parent,Path('/data_pool/khangle/facevitals')])
 scope=base['scope_conservative_bytes']+max(0,pp-base['p100_project_bytes'])+max(0,a['a100_bytes']-base['a100_bytes'])+600000000
 project=pp+a['a100_bytes']+base['reserved_bytes']+base['scratch_bytes']+600000000
 state['last_budget']=dict(utc=now(),scope_bytes=scope,project_bytes=project,available_gb=available,inventory_age_s=age)
 if scope>160000000000 or project>500000000000:state['wait_reason']='Storage cap exceeded; no new launch';return False
 for p,h in pre['frozen'].items():assert sha(Path(p))==h,'Source/weight changed: '+p
 state.pop('wait_reason',None);return True
def key(j):return '_'.join(x.replace('/','_') for x in j if x)
def logpath(j):
 st,mode,m,c=j
 if mode!='run':return L/('y05_eval_all.log' if st=='y05' else 'z09_score.log')
 return L/('y05_run_'+m+'.log' if st=='y05' else 'z09_run_'+m+'_'+c.replace('/','_')+'.log')
def launch(j):
 st,mode,m,c=j;k=key(j);log=logpath(j)
 if log.exists():
  archive=L/'z09_recovery_PREVIOUS_LOGS_20261122';archive.mkdir(exist_ok=True);dest=archive/log.name
  assert not dest.exists();dest.write_bytes(log.read_bytes())
  with log.open('a') as f:f.write('\n=== Authorized 8GB retry '+now()+' ===\n')
 offset=log.stat().st_size if log.exists() else 0
 args=['taskset','-c','0-23,36-59','nice','-n','0',str(PY if st=='y05' and mode=='run' else HR),'-u','-B',str(scripts[st]),mode]
 if mode=='run':args += [m] if st=='y05' else [m,c,'scratch/x19_pilot_keys.txt']
 elif st=='z09':args+=['scratch/x19_pilot_keys.txt']
 with log.open('a') as f:p=subprocess.Popen(args,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 assert os.getpriority(os.PRIO_PROCESS,p.pid)==10
 state['jobs'][k]=dict(pid=p.pid,args=args,log=str(log),log_start_offset=offset,started_utc=now(),peak_rss=0,peak_tree_rss=0,effective_nice=10)
 save();return p,j
def finish(p,j):
 st,mode,m,c=j;k=key(j);a=state['jobs'][k];a.update(exit_code=p.returncode,finished_utc=now())
 txt=Path(a['log']).read_bytes()[a['log_start_offset']:].decode(errors='replace')
 if st=='z09' and mode=='run':
  matches=re.findall(r'missing (\d+) unexpected (\d+)',txt);a['weight_load_counts']=matches
 if p.returncode!=0:state['errors'].append(dict(job=k,reason=a.get('stop_reason','nonzero exit'),exit_code=p.returncode))
 elif mode=='run':
  if st=='y05':
   out=W/'out/y05'/(m+'.npz')
   with np.load(out,allow_pickle=False) as z:
    assert z['S'].shape==(218,277)
    a.update(videos_with_output=int(np.isfinite(z['S']).all(1).sum()),video_rows=len(z['keys']),output_sha256=sha(out))
   assert 'done; videos with output' in txt
  else:
   out=W/'out/z09'/c.replace('/','_')/(m+'.json');r=json.loads(out.read_text())
   assert all(k in r for k in pre['pilot_keys'])
   a.update(videos_with_output=int(sum(r[k] is not None and np.isfinite(np.asarray(r[k])).all() for k in pre['pilot_keys'])),video_rows=len(r),output_sha256=sha(out))
def main():
 pending=[('y05','run',m,'') for m in first]+[('z09','run',m,c) for m in weights for c in ['x10','x15/stabsq14']]+[('y05','run',m,'') for m in slow]
 active={};scored={'y05'}
 while pending or active or len(scored)<2:
  table=procs()
  for k,(p,j) in list(active.items()):
   ids={p.pid}
   while True:
    more={i for i,a in table.items() if a['ppid'] in ids}
    if more<=ids:break
    ids|=more
   rr=[table[i]['rss'] for i in ids if i in table];a=state['jobs'][k]
   a['peak_rss']=max([a['peak_rss']]+rr);a['peak_tree_rss']=max(a['peak_tree_rss'],sum(rr))
   if rr and max(rr)>8000000000 and p.poll() is None:
    a['stop_reason']='A process RSS exceeded8GB';os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   if p.poll() is not None:finish(p,j);del active[k]
  ext=externals(active)
  orig=json.loads((L/'y05z09_STATUS_20261122.json').read_text())
  ready=not orig.get('pending') and 'z09' in orig.get('scored',[]) and all('exit_code' in a for k,a in orig['jobs'].items() if k.startswith('z09'))
  if not ready:state['wait_reason']='Wait original queue to dispatch all six models and finish failed z09 attempts'
  elif len(active)+len(ext)>=3:state['wait_reason']='Three total y05/z09 inference or score jobs active'
  else:
   # Score each independent branch when all its inference attempts are terminal.
   for st,mode in [('z09','score'),('y05','eval')]:
    if st not in scored and not any(j[0]==st for j in pending) and not any(j[0]==st for p,j in active.values()) and len(active)+len(ext)<3 and not any(j[1]!='run' for p,j in active.values()):
     if guard():j=(st,mode,'','');active[key(j)]=launch(j);scored.add(st)
   while pending and len(active)+len(ext)<3:
    if not guard():break
    j=pending.pop(0);active[key(j)]=launch(j)
  state.update(mode='running',active=list(active),pending=[key(j) for j in pending],scored=sorted(scored));save();time.sleep(2)
 for p,h in pre['frozen'].items():assert sha(Path(p))==h
 for p,h in prior.items():assert sha(Path(p))==h,'Prior output/log changed: '+p
 for p,v in inputs.items():
  st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==v,'Input changed: '+p
 state.update(mode='complete',completed_utc=now(),prior_hashes_preserved=True,input_stats_preserved=True);save();write(L/'z09_recovery_COMPLETE_20261122.json',state)
try:main()
except Exception:
 state.update(mode='error',traceback=traceback.format_exc());save();print(state['traceback'],flush=True)
