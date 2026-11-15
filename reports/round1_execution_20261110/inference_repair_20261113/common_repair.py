import os,sys,json,hashlib,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent;V=R/'reports/round1_v4_20261105';O=V/'btc_zeroshot';B=R/'reports/round1_bayes_20261108';N=E/'final_priorities_20261111';VIS=E/'visual_diagnostics_20261112'
T8=Path('/data_pool/khangle/facevitals/code/t8_oval_20261031');V3=T8.parent/'t8_polarity_v3_20261102'
for p in [T8,V3,E,B,N,V/'padding_pilot_20261107']:sys.path.insert(0,str(p))
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def lock(n):
 f=(P/n).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def budget(fresh=False):
 cuda=os.environ.get('CUDA_VISIBLE_DEVICES','');sys.path.insert(0,str(E/'v5_multiview_20261112'));import runtime
 os.environ['CUDA_VISIBLE_DEVICES']=cuda
 b=runtime.budget(fresh=fresh);used=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());remaining=max(0,2_000_000_000-used)
 assert used<2_000_000_000 and b['scope_actual_bytes']+b['remaining_reserved_bytes']+remaining+420_000_000<84_000_000_000
 assert b['project_actual_bytes']+remaining+420_000_000<500_000_000_000
 return dict(**b,repair_bytes=used,repair_remaining=remaining)
def manifest():
 f=read(P/'FROZEN.json');assert sha(P/'MANIFEST.json')==f['manifest_sha256'];assert sha(P/'PROTOCOL.md')==f['protocol_sha256'];return read(P/'MANIFEST.json')
