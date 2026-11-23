"""R1: reset at each recording boundary; preserve each original spectral operator."""
import os, sys
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
W=OUT.parent/'claude_research_20261119_constraints'
NC=OUT.parent.parent/'git_push_20261122/newcomer_20261123'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',KERAS_BACKEND='jax',PYTHONDONTWRITEBYTECODE='1',
 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2',
 NUMBA_CACHE_DIR=str(OUT/'cache/numba'),KERAS_HOME=str(OUT/'cache/keras'),XDG_CACHE_HOME=str(OUT/'cache'),
 JAX_COMPILATION_CACHE_DIR=str(OUT/'cache/jax'),TMPDIR=str(OUT/'tmp'))
sys.dont_write_bytecode=True
for p in [NC/'runtime',W.parent/'claude_research_20261118/code',W/'code']:sys.path.insert(0,str(p))
import copy, hashlib, json, time
import numpy as np
from btc_spectral_math import supported,window_spectrum
MODELS=['FacePhys.rlap','ME-chunk.pure','ME-flow.pure']
WEIGHTS={'FacePhys.rlap.weights.h5':'6cd99510bcc5771f9cebcdb0f47a5cb3692d19f1e3d6e2444af5e8fac0ee3c07',
 'ME.pure.weights.h5':'2135995cb4161d1d4c7c3aa45d333dc6b1f5818b0652d11da9fc0fb0b6e11bf0',
 'state.pkl':'d894487c5f0162dd798a8e7f8aabedb958b48cab8e4a31abdd433f97826c57f4'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify_weights():
 import importlib.util
 p=Path(importlib.util.find_spec('rppg').origin).parent/'weights'
 actual={n:sha(p/n) for n in WEIGHTS};assert actual==WEIGHTS,actual
 return actual
def selected_mode():
 d=json.loads((OUT/'MODE.json').read_text());assert sha(W/'out/z18_selfwarm_score.json')==d['source_sha256']
 assert d['mode'] in ('reset','selfwarm');return d['mode']
class Expert:
 def __init__(self,name):
  import rppg
  self.model=rppg.Model(name);self.initial=copy.deepcopy(self.model.state)
 def waveform(self,crops,fps,mode):
  assert mode in ('reset','selfwarm')
  self.model.state=copy.deepcopy(self.initial)
  start=time.monotonic()
  for _ in range(2 if mode=='selfwarm' else 1):self.model.process_faces_tensor(np.ascontiguousarray(crops),fps=float(fps))
  bvp,ts=self.model.bvp(raw=True)
  return np.asarray(bvp,float),np.asarray(ts,float),time.monotonic()-start
def btc_spectrum(expert,crops,t,valid,mode,split=False,legacy_operator=False):
 fps=(len(t)-1)/(t[-1]-t[0])
 if split:
  k=max(1,int(np.ceil(fps/10-1e-3)));t,valid,crops,fps=t[::k],valid[::k],crops[::k],fps/k
 bvp,ts,sec=expert.waveform(crops,fps,mode)
 if len(bvp)<(100 if split else 300):return None,sec,0
 ts=ts-ts[0];powers=[]
 if legacy_operator:
  import y01_btc_improve as Y
  op=lambda w:Y.spec2(w)[0]
 else:op=window_spectrum
 for s in np.arange(0,max(0.,t[-1]-10+1e-8),5.):
  if not supported(t,valid,s,s+10) or s+10>ts[-1]+.2:continue
  w=np.interp(s+np.arange(300)/30.,ts,bvp)
  if np.isfinite(w).all() and w.std()>1e-12:powers.append(op(w))
 return (np.mean(powers,0) if powers else None),sec,len(powers)
def public_spectrum(expert,crops,t,valid,job,mode,condition):
 from y12_openrppg_public import band_spectrum
 if condition=='native':
  with np.load(job['meta']) as z:mt,mv=z['t'],z['valid'].astype(bool)
 else:
  # y13 casts timestamps, decimates, and recomputes fps from retained timestamps.
  t=t.astype(float)
  if condition=='c5_split10':
   fps0=(len(t)-1)/(t[-1]-t[0]);k=max(1,int(np.ceil(fps0/10-1e-3)))
   t,valid,crops=t[::k],valid[::k],crops[::k]
  mt,inv=np.unique(np.asarray(t,float),return_inverse=True);mv=np.ones(len(mt),bool);np.logical_and.at(mv,inv,np.asarray(valid,bool))
 fps=(len(t)-1)/(t[-1]-t[0]);bvp,ts,sec=expert.waveform(crops,fps,mode)
 ts=ts-ts[0]+t[0] if len(ts) else ts;starts=[];truth=[];spec=[]
 for w in job['windows']:
  if not w['eligible']:continue
  s=float(w['start']);p=None
  if len(bvp)>=(100 if condition=='c5_split10' else 300) and supported(mt,mv,s,s+10) and s>=ts[0]-1e-6 and s+10<=ts[-1]+.2:
   p=band_spectrum(np.interp(s+np.arange(300)/30.,ts,bvp))
  starts.append(s);truth.append(float(w['truth']));spec.append(np.full(314,np.nan) if p is None else p)
 return dict(start=np.array(starts),truth=np.array(truth),spec=np.array(spec,np.float64).reshape(len(starts),314),seconds=sec,frames=len(t))
