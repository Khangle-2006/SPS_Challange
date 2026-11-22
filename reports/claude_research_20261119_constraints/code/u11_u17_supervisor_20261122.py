import os,sys,time,json,subprocess,hashlib,re,signal
from pathlib import Path
W=Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
os.chdir(W)
L=W/'logs'; runtime=W/'out/u11_u17_runtime_20261122'; runtime.mkdir(parents=True,exist_ok=True)
status=L/'u11_u17_status_20261122.json'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',KERAS_BACKEND='jax',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
for k,v in dict(HOME='home',TMPDIR='tmp',XDG_CACHE_HOME='cache',KERAS_HOME='keras',MPLCONFIGDIR='matplotlib',JAX_COMPILATION_CACHE_DIR='jax',TORCH_HOME='torch',HF_HOME='huggingface').items():
 p=runtime/v;p.mkdir(exist_ok=True);env[k]=str(p)
env['PYTHONPATH']=str(W/'out/z09_dependency_overlay_20261122')
HR='/data_pool/khangle/facevitals/envs/hr/bin/python'
state=dict(start=time.time(),jobs={},hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'code/u17_openrppg_bvp.py',W/'code/u11_mmrphys_btc.py']})
def save():
 status.write_text(json.dumps(state,indent=2))
def available():
 return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
def launch(name,args,log):
 assert available()>=15000000000
 pth=L/log
 if pth.exists(): pth.rename(pth.with_name(pth.name+'.previous.'+str(time.time_ns())))
 unit='u11u17-'+name.replace('.','-')+'-'+str(os.getpid())
 cmd=['systemd-run','--user','--quiet','--wait','--pipe','--unit='+unit,'-p','MemoryMax=4000000000','-p','MemorySwapMax=0','-p','ProtectSystem=strict','-p','ProtectHome=read-only','-p','ReadWritePaths='+str(W),'-p','WorkingDirectory='+str(W),'/usr/bin/env']+[k+'='+v for k,v in env.items()]+['nice','-n','10','taskset','-c','24-35']+args
 f=pth.open('w');p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);f.close()
 state['jobs'][name]=dict(unit=unit,log=str(pth),start=time.time(),peak_memory_bytes=0)
 save()
 return p
models=['MMRPhys.ibvp','MMRPhys.pure','MMRPhys.ubfc','MMRPhys.bp4d']
active={}
active['FacePhys.rlap']=launch('FacePhys.rlap',[str(W/'venv_orppg/bin/python'),'-u','-B','code/u17_openrppg_bvp.py','FacePhys.rlap','scratch/u02_btc_all_keys.txt'],'u17_FacePhys.rlap.log')
for m in models:
 active[m]=launch(m,[HR,'-u','-B','code/u11_mmrphys_btc.py','run',m,'scratch/u02_btc_all_keys.txt'],'u11_'+m+'.log')
npz_started=False
while active:
 av=available();state['available_bytes']=av
 for name,p in list(active.items()):
  j=state['jobs'][name];txt=Path(j['log']).read_text(errors='replace')
  match=re.search(r'^missing (\d+).*$',txt,re.M)
  if match: j['load_state_line']=match.group()
  reason=('missing model parameters' if match and int(match[1])>0 else 'available RAM below 15 GB' if av<15000000000 else None)
  if reason and p.poll() is None:
   j['stop_reason']=reason;subprocess.run(['systemctl','--user','stop',j['unit']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  if p.poll() is None:
   q=subprocess.run(['systemctl','--user','show',j['unit'],'-p','MemoryCurrent','-p','MemoryPeak'],capture_output=True,text=True).stdout
   nums=[int(x.split('=')[1]) for x in q.splitlines() if x.split('=')[-1].isdigit()]
   if nums:j['peak_memory_bytes']=max(j['peak_memory_bytes'],*nums)
  else:
   j['returncode']=p.returncode;j['end']=time.time()
   j['last_line']=txt.strip().splitlines()[-1] if txt.strip() else ''
   j['success']=p.returncode==0 and (('done; files 218' in txt) if name=='FacePhys.rlap' else ('218 / 218' in txt) if name in models else True)
   del active[name]
 if not npz_started and all('returncode' in state['jobs'][m] for m in models):
  npz_started=True
  if all(state['jobs'][m]['success'] for m in models):
   active['npz']=launch('npz',[HR,'-u','-B','code/u11_mmrphys_btc.py','npz'],'u11_npz.log')
 state['updated']=time.time();save();time.sleep(1)
state['finished']=time.time()
state['scripts_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in state['hashes'].items())
save()
