"""Verified evidence export, explicitly not a final scientific/paired-capture pass."""
from common_causal import *
import zipfile,shutil

def export_if_ready():
 required=['BATCH_COMPLETE.json','ROUTE_COMPLETE.json','METADATA_COMPLETE.json','physical_v3/LADDER_COMPLETE.json','FRONTEND_COMPARISON_COMPLETE.json']
 if not all((C/n).exists() for n in required) or not (P/'report_spatial/COMPLETE.json').exists():return
 if (C/'EXPORT_COMPLETE.json').exists():return
 lease=lock('export.lock');budget(True)
 H=Path('/mnt/hdd1t/khangle/facevitals_backup_20261112/addenda');K=Path('/mnt/samsung_t5/khangle/facevitals_backup_20261112/addenda')
 assert os.stat(H).st_dev!=os.stat(C).st_dev and os.stat(H).st_dev!=os.stat(K).st_dev
 pairs=[]
 for f in C.rglob('*'):
  if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts or f.suffix in ['.tmp','.lock','.staged','.xlsx']:continue
  pairs.append((f,'causal_audit/'+f.relative_to(C).as_posix()))
 for f in (P/'report_spatial').rglob('*'):
  if f.is_file() and not f.is_symlink() and f.suffix in ['.json','.csv','.md','.txt']:pairs.append((f,'spatial_AB/'+f.relative_to(P/'report_spatial').as_posix()))
 for n in ['PROTOCOL.md','FROZEN.json','BASELINE_AUDIT.json']:
  if (P/n).exists():pairs.append((P/n,'spatial_AB/provenance/'+n))
 size=sum(f.stat().st_size for f,_ in pairs)
 for d in [H,K]:assert shutil.disk_usage(d).free>20_000_000_000+size
 dest=H/'BTC_CAUSE_EXPORT_V1.zip';other=K/dest.name;assert not dest.exists() and not other.exists(),'PRESERVE_EXISTING_EXPORT'
 entries=[]
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for f,rel in sorted(pairs,key=lambda x:x[1]):
   data=f.read_bytes();entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));z.writestr(rel,data)
  z.writestr('README.txt','Recorded causal-audit evidence. No training; closed holdout not evaluated.\nSynthetic tests and existing BTC A/B are complete, but independent scientific review and real paired webcam/phone capture remain pending. This archive does not establish acquisition loss and does not itself change FREEZE.\nRead causal_audit/REPORT.md, REFERENCE_CLOCK_ADDENDUM.md, CAUSAL_LEDGER.json and SOURCE_GRAPH.md. Failed attempts are preserved and clearly named.\nOriginal data/checkpoint files remain at their hashed source paths; they are not all bundled.\n')
  z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(utc=now(),files=entries,scientific_review_pending=True,paired_capture_pending=True,holdout_evaluations=0,FREEZE_changed=False),indent=2))
 with zipfile.ZipFile(dest) as z:
  for a in entries:assert hashlib.sha256(z.read(a['path'])).hexdigest()==a['sha256']
 digest=sha(dest);shutil.copyfile(dest,other);assert sha(other)==digest
 receipt=dict(utc=now(),files=len(entries),source_bytes=size,archive_bytes=dest.stat().st_size,sha256=digest,copies=[str(dest),str(other)],all_members_verified=True,cross_physical_disks=True,scientific_review_pending=True,paired_capture_pending=True,no_deletion=True)
 save(C/'EXPORT_COMPLETE.json',receipt)
 for d in [H,K]:save(d/'BTC_CAUSE_EXPORT_V1.zip.json',receipt)

if __name__=='__main__':export_if_ready()
