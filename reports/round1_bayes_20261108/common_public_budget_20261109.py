"""Versioned CPU operations. No holdout discovery, GPU, deletion or old-result writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,hashlib,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
OLD=ROOT/'reports/round1_v4_20261105/btc_zeroshot'
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def csvout(p,rr):
 if not rr:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file()) if Path(p).exists() else 0
def guard(fresh=True,reserve=50_000_000):
 r=read(OUT/'INVENTORY.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['measured_utc'])).total_seconds()
 if fresh:assert -60<=age<=3600,'Fresh cross-server receipt required before launch'
 sys.path.insert(0,'/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
 from polarity_guard import storage_guard
 b=storage_guard(read(ROOT/'t8_oval_20261031/training_index.json'),ROOT/'t8_oval_20261031',ROOT/'t8_polarity_v3_20261102')
 whole=b['total_bytes']+size(OLD.parent)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+r['a100_v4_bytes']+size(OUT)
 project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (ROOT,Path('/data_pool/khangle/facevitals')))+r['a100_bytes']
 assert whole+250_000_000+reserve<65_050_000_000,'T8 limit including new reports and reserve'
 assert project+reserve<499_900_000_000,'project limit'
 assert size(OLD)+size(OUT)+reserve<1_350_000_000,'BTC derivative limit includes new Bayes outputs conservatively'
 return dict(utc=now(),T8_bytes=whole,project_bytes=project,BTC_bytes=size(OLD)+size(OUT),reserve=reserve,inventory_sha256=sha(OUT/'INVENTORY.json'))
def dev_manifest():
 assert sha(OLD/'btc_split.json')==SPLIT
 m=read(OLD/'dev_manifest.json');s=read(OLD/'btc_split.json')
 assert m['split_sha256']==SPLIT and len(m['videos'])==221
 assert all(s['people'][v['person']]['partition']=='dev' for v in m['videos'])
 return m['videos']
def prereg():
 r=read(OUT/'PREREGISTRATION.json')
 for f,h in r['files'].items():assert sha(OUT/f)==h,('preregistration changed',f)
 return r
