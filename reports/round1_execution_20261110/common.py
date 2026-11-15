import os,sys,json,hashlib,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;R=P.parent.parent;B=P.parent/'round1_bayes_20261108';V=P.parent/'round1_v4_20261105';S=B/'domain_sweep_v2_20261110'
sys.path.insert(0,str(S))
from codec_provider import read,sha,Provider
from btc_data import BTCData
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def lock(name):
 h=(P/name).open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def guard(fresh=False):
 a=read(P/'APPROVAL.json');assert a['T8_cap_bytes']==84000000000 and a['status']=='AUTHORIZED_BY_USER'
 q=read(P/'BUDGET_ESTIMATE.json');assert (datetime.now(timezone.utc)-datetime.fromisoformat(q['utc'])).total_seconds()<3600 or not fresh
 assert q['estimated_T8_whole_bytes']<a['T8_cap_bytes']
 # Independent workers share a single predeclared root budget, measured each time.
 used=sum(f.stat().st_size for f in P.rglob('*') if f.is_file());reserve=420000000
 assert q['existing_T8_whole_bytes']+used+reserve<a['T8_cap_bytes'],'STOP_STORAGE_T8'
 # Cross-server sizes are refreshed before launch; grow accounting for own output.
 project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (R,Path('/data_pool/khangle/facevitals')))+read(P/'A100_INVENTORY.json')['a100_bytes']
 assert project+reserve<500000000000,'STOP_STORAGE_PROJECT'
 return dict(utc=now(),new_root_bytes=used,project_bytes=project)
def hashes():return {p.name:sha(p) for p in P.glob('*.py')}
def model_static(m):
 import torch
 from test_static_tsm import static_result
 m=m.cpu();out=static_result(m);assert out['hr'] is None,'STATIC_FAILED';return out
def gpu():
 import torch
 import time
 for _ in range(3):
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();a=[int(x.strip()) for x in rows[0].split(',')]
  assert a[0]==0 and a[1]>=6000 and a[2]==0,f'WAIT_GPU0:{a}'
  time.sleep(2)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
 return 'cuda:0'
def infer_native(model,data,fold,method,device):
 import numpy as np,torch
 from t8_evaluate import decode_difference
 model.eval();rows=[]
 for v in data.rows:
  if v['fold']!=fold:continue
  for start in sorted({r['start'] for r in data.candidates(v,'native') if r['scheme']=='10s'}):
   pred=None;reason=None
   try:
    x=data.clip(v,start)
    with torch.inference_mode():w=model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
    assert np.isfinite(w).all(),'Nonfinite inference'
    pred=decode_difference(w)
   except ValueError as e:reason=str(e)
   rows.append(dict(recording=v['recording'],person=v['person'],fold=fold,group=v['group'],task=v['task'],truth=v['pr_bpm'],condition='native',method=method,start=start,prediction=pred,reason=reason))
 return rows
