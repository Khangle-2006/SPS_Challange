import os,sys,json,hashlib,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent;B=R/'reports/round1_bayes_20261108';OLD=R/'reports/round1_v4_20261105/btc_zeroshot';N=E/'final_priorities_20261111'
T8=Path('/data_pool/khangle/facevitals/code/t8_oval_20261031');V3=T8.parent/'t8_polarity_v3_20261102';BACK=T8.parent/'ppg_polarity_backend_20261102'
for d in [T8,V3,B,N,E]:sys.path.insert(0,str(d))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def lock(name):
 h=(P/name).open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def budget(fresh=False):
 cuda=os.environ.get('CUDA_VISIBLE_DEVICES','');sys.path.insert(0,str(E/'v5_multiview_20261112'));import runtime as rt
 os.environ['CUDA_VISIBLE_DEVICES']=cuda
 b=rt.budget(fresh=fresh);own=rt.old.size(P);remaining=max(0,2_000_000_000-own)
 assert own<2_000_000_000 and b['scope_actual_bytes']+b['remaining_reserved_bytes']+remaining+420_000_000<84_000_000_000
 assert b['project_actual_bytes']+b['remaining_reserved_bytes']+remaining+420_000_000<500_000_000_000
 return dict(**b,visual_bytes=own,visual_remaining=remaining)
def checkpoint(f):
 root=R/'reports/round1_v4_20261105/v4_padding_pilot' if f==0 else N/'padding_fixed'
 p=root/f'fold{f}/seed2027/joint/final.pt';s=read(p.parent/'status.json');assert s['state']=='complete' and s['completed_epochs']==30 and sha(p)==s['checkpoint_sha256'];return p
def manifest():
 f=read(P/'FROZEN.json');assert sha(P/'SELECTION.json')==f['selection_sha256'] and sha(P/'PROTOCOL.md')==f['protocol_sha256'];return read(P/'SELECTION.json')
