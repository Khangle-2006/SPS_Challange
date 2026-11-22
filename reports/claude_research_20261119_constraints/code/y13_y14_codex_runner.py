"""Operational queue only. Supplied y13/y14 scripts remain unchanged."""
import os,sys,json,time,subprocess,fcntl,hashlib,traceback,signal
from pathlib import Path
import numpy as np
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
L=W/'logs';ROOT=W.parent.parent
LOCK=(L/'y13_y14_codex.lock').open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (L/'y13_y14_CODEX_COMPLETE.json').exists()
assert not (L/'y13_y14_codex_status.json').exists(),'Existing state: inspect, do not duplicate'
MODELS=['FacePhys.rlap','ME-chunk.pure','ME-flow.pure']
CONDS=['c5_split10','c5_split10_jpg60','c4_qvga','c6_us1','aidtrack_c1_g25gv01','c2_vit','track_c1_g25p','track_c7_blue','c1_g50p','c1_g12p','c1_g25gv001','c7_white']
PY=W/'venv_orppg/bin/python';HR=Path('/data_pool/khangle/facevitals/envs/hr/bin/python')
SCRIPTS={n:W/'code'/('y14_btc_split_jpeg.py' if n=='y14' else 'y13_openrppg_public_conditions.py') for n in ['y14','y13']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,a):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(a,indent=2));q.replace(p)
frozen={str(p):sha(p) for p in [*SCRIPTS.values(),W/'code/y05_openrppg.py',W/'PREREG.md']}
for nm in ['y11_openrppg_conditions.py','y12_openrppg_public.py','y01_btc_improve.py','y06_reader_quality.py','y08_fps_sim.py','x05_public_score.py','y02_public_improve.py','y07_public_prior.py']:
 p=W/'code'/nm
 if p.exists():frozen[str(p)]=sha(p)
pre=json.loads((L/'y13_y14_PREFLIGHT.json').read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',KERAS_BACKEND='jax',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
for key,d in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('KERAS_HOME','keras'),('MPLCONFIGDIR','matplotlib'),('JAX_COMPILATION_CACHE_DIR','jax')]:
 p=W/'out/y05_runtime'/d;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
state=dict(pid=os.getpid(),started_utc=now(),mode='preflight',jobs={},errors=[],frozen=frozen,holdout_evaluations=0,maximum_parallel=8)
def save():state['updated_utc']=now();write(L/'y13_y14_codex_status.json',state)
def procs():
 out={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   st=dict(line.split(':',1) for line in (d/'status').read_text().splitlines() if ':' in line)
   out[int(d.name)]=dict(ppid=int(st['PPid']),rss=int(st.get('VmRSS','0 kB').split()[0])*1024,args=(d/'cmdline').read_bytes().split(b'\0'))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return out
def external_groups(active):
 known={p.pid for p,job in active.values()};pg=set()
 for pid,a in procs().items():
  if any(any(Path(x.decode(errors='replace')).name==s.name for x in a['args'] if x) for s in SCRIPTS.values()):
   try:g=os.getpgid(pid)
   except ProcessLookupError:continue
   if g not in known:pg.add(g)
 return pg
def guard():
 raw=subprocess.check_output(['free','-g'],text=True);available=int(next(x for x in raw.splitlines() if x.startswith('Mem:')).split()[-1])
 with (L/'y13_y14_resources.log').open('a') as f:f.write(now()+'\n'+raw)
 a=json.loads((L/'y13_y14_A100_INVENTORY.json').read_text())
 from datetime import datetime,timezone
 age=(datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'])).total_seconds()
 if age>3600:state['wait_reason']='fresh A100 inventory required';return False
 if available<15:state['wait_reason']='available RAM below15GB';return False
 # Bound scope growth conservatively by ALL project growth since the measured preflight.
 pp=sum(int(subprocess.check_output(['du','-sb',str(x)],text=True).split()[0]) for x in [ROOT,Path('/data_pool/khangle/facevitals')])
 growth=max(0,pp-pre['p100_project_bytes'])+max(0,a['a100_bytes']-pre['a100_bytes'])
 scope=pre['scope_conservative_bytes']+growth
 project=pp+a['a100_bytes']+pre['reserved_bytes']+pre['scratch_bytes']
 if scope>160_000_000_000 or project>500_000_000_000:
  state['wait_reason']='Storage cap exceeded; no new launch/deletion';return False
 state['last_budget']=dict(utc=now(),scope_conservative_bytes=scope,project_conservative_bytes=project,a100_age_seconds=age,available_gb=available)
 for p,h in frozen.items():assert sha(Path(p))==h,'Source changed: '+p
 state.pop('wait_reason',None);return True
def output(job):
 stage,mode,model,cond=job
 if mode!='run':return None
 return W/'out/y11/c5_split10_jpg60'/(model+'.npz') if stage=='y14' else W/'out/y13'/cond/model
prior={}
for m in MODELS:
 p=W/'out/y11/c5_split10_jpg60'/(m+'.npz')
 if p.exists():
  with np.load(p,allow_pickle=False) as z:
   ids=np.flatnonzero(z['done']);prior[str(p)]=dict(kind='y14',ids=ids.tolist(),sha=sha(p),values_sha=hashlib.sha256(z['S'][ids].tobytes()+z['seconds'][ids].tobytes()).hexdigest())
for c in CONDS:
 for m in MODELS:
  for p in (W/'out/y13'/c/m).glob('*.npz'):
   if not p.name.endswith('.tmp.npz'):prior[str(p)]=dict(kind='y13',sha=sha(p))
write(L/'y13_y14_EXISTING_OUTPUTS.json',prior)
def key(job):return '_'.join(x for x in job if x)
def launch(job):
 stage,mode,m,c=job;assert guard()
 k=key(job);name=(stage+'_'+mode+'_'+m+('_'+c if stage=='y13' else '')+'.log') if mode=='run' else stage+('_eval.log' if stage=='y14' else '_score.log')
 log=L/name
 if log.exists():
  with log.open('a') as f:f.write('\n=== Authorized resume '+now()+' ===\n')
 args=['taskset','-c','0-23,36-59' if mode=='run' else ('0-15,36-51' if stage=='y14' else '0-19,36-55'),'nice','-n','0',str(PY if mode=='run' else HR),'-u','-B',str(SCRIPTS[stage]),mode]
 args+=([m,'60'] if stage=='y14' else [m,c]) if mode=='run' else ([','.join(MODELS),'60'] if stage=='y14' else [','.join(MODELS)])
 with log.open('a') as fh:p=subprocess.Popen(args,cwd=W,env=env,stdin=subprocess.DEVNULL,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
 assert os.getpriority(os.PRIO_PROCESS,p.pid)==10
 state['jobs'][k]=dict(pid=p.pid,effective_nice=10,nice_note='inherits controller nice10; nice -n0 preserves10',command=args,log=str(log),started_utc=now(),peak_rss=0,peak_tree_rss=0)
 save();return p,job
def finish(p,job):
 k=key(job);j=state['jobs'][k];j.update(exit_code=p.returncode,finished_utc=now())
 if p.returncode!=0:state['errors'].append(dict(job=k,reason=j.get('stop_reason','nonzero exit'),exit_code=p.returncode))
 dest=output(job)
 if dest is not None and dest.exists():
  if job[0]=='y14':
   with np.load(dest,allow_pickle=False) as z:
    assert z['S'].shape==(218,277) and z['done'].shape==(218,)
    j.update(videos_with_output=int(np.isfinite(z['S']).all(1).sum()),attempted_videos=int(z['done'].sum()),sha256=sha(dest))
  else:
   ff=[f for f in dest.glob('*.npz') if not f.name.endswith('.tmp.npz')];j['recordings_total']=len(ff);j['output_hashes']={f.name:sha(f) for f in ff}
   j['recordings_with_finite_output']=sum(bool(np.isfinite(np.load(f,allow_pickle=False)['spec']).all(1).any()) for f in ff)
 if p.returncode==0:
  text=Path(j['log']).read_text(errors='replace')
  if job[1]=='run' and ('done; videos with output' if job[0]=='y14' else 'done; recordings written') not in text:
   state['errors'].append(dict(job=k,reason='Missing done marker'))

def crop_gate():
 p=L/'x16_boxcrops.jsonl'
 if not p.exists():return 'waiting'
 lines=p.read_text().splitlines();rows=[]
 for i,line in enumerate(lines):
  if not line.strip():continue
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:
   if i==len(lines)-1:return 'waiting'
   state['crop_gate_error']='Malformed complete log line';return 'failed'
 bad=[r for r in rows if r.get('state') in ('error','failed') or r.get('error') or any((k.startswith('max_abs_diff_vs_stored_') or k.startswith('valid_mismatch_')) and v!=0 for k,v in r.items())]
 done=[r for r in rows if r.get('state')=='done']
 live=[pid for pid,a in procs().items() if any(Path(x.decode(errors='replace')).name=='x16_boxcrops_public.py' for x in a['args'] if x)]
 state['crop_gate']=dict(done_rows=len(done),unique_done=len({r['key'] for r in done}),live_pids=live,bad=bad)
 if bad:
  state['crop_gate_error']='x16 error/nonzero comparison';return 'failed'
 if live:return 'waiting'
 if len(done)!=73 or len({r['key'] for r in done})!=73:
  state['crop_gate_error']='x16 ended without exactly73 distinct done rows';return 'failed'
 keys={r['key'] for r in done}
 for c in CONDS[2:]:
  missing=[k for k in keys if not (W/'out/x16'/c/(k+'.npz')).exists()]
  if missing:state['crop_gate_error']='Missing crop files '+c;return 'failed'
 state['crop_gate']['log_sha256']=sha(p)
 return 'ready'

def main():
 active={};pending=[('y14','run',m,'60') for m in MODELS]+[('y13','run',m,c) for c in CONDS for m in MODELS]
 eval_launched=set()
 while pending or active or len(eval_launched)<2:
  for k,(p,job) in list(active.items()):
   table=procs();ids={p.pid}
   while True:
    more={pid for pid,a in table.items() if a['ppid'] in ids}
    if more<=ids:break
    ids|=more
   rr=[table[x]['rss'] for x in ids if x in table];j=state['jobs'][k]
   j['peak_rss']=max([j['peak_rss']]+rr);j['peak_tree_rss']=max(j['peak_tree_rss'],sum(rr))
   if rr and max(rr)>4_000_000_000 and p.poll() is None:
    j['stop_reason']='A process RSS exceeded4GB';os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   if p.poll() is not None:finish(p,job);del active[k]
  gate=crop_gate()
  if gate=='failed' and 'y13' not in eval_launched:
   state['errors'].append(dict(stage='y13',reason=state['crop_gate_error']))
   pending=[j for j in pending if j[0]!='y13']
   for kk,(pp,jj) in list(active.items()):
    if jj[0]=='y13' and pp.poll() is None:
     state['jobs'][kk]['stop_reason']='Public stopped: crop gate failed'
     os.killpg(pp.pid,signal.SIGTERM)
   eval_launched.add('y13')
   state['public_score_blocked']=state['crop_gate_error']
  for stage,mode in [('y14','eval'),('y13','score')]:
   if stage not in eval_launched and not any(j[0]==stage and j[1]=='run' for j in pending) and not any(j[0]==stage and j[1]=='run' for p,j in active.values()):
    # Do not overlap the two scoring pools. They count as one top-level job each.
    if not any(j[1]!='run' for p,j in active.values()) and not external_groups(active) and len(active)<8 and guard():
     job=(stage,mode,'','');active[key(job)]=launch(job);eval_launched.add(stage)
  while pending and len(active)+len(external_groups(active))<8:
   if external_groups(active):state['wait_reason']='Existing external y13/y14 process; no duplicate';break
   eligible=[i for i,j in enumerate(pending) if j[0]=='y14' or j[3].startswith('c5_split10') or gate=='ready']
   if not eligible:state['wait_reason']='Waiting x16 exact73 and all comparisons zero';break
   if not guard():break
   job=pending.pop(eligible[0]);active[key(job)]=launch(job)
  state.update(mode='running',active=list(active),pending=[key(j) for j in pending],eval_launched=sorted(eval_launched),wait=state.get('wait_reason'));save()
  time.sleep(2)
 for p,a in prior.items():
  f=Path(p)
  if a['kind']=='y13':assert sha(f)==a['sha'],'Existing public output changed'
  else:
   with np.load(f,allow_pickle=False) as z:
    ix=np.array(a['ids'],int);assert hashlib.sha256(z['S'][ix].tobytes()+z['seconds'][ix].tobytes()).hexdigest()==a['values_sha'],'Existing BTC rows changed'
 for p,h in frozen.items():assert sha(Path(p))==h
 state.update(mode='complete',completed_utc=now(),existing_outputs_preserved=True);save();write(L/'y13_y14_CODEX_COMPLETE.json',state)
try:main()
except Exception:
 state.update(mode='error',traceback=traceback.format_exc());save();print(state['traceback'],flush=True)
