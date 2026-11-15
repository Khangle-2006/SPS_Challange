"""Cross-disk verified snapshot of completed causal evidence; never claims final."""
from common_causal import *
import zipfile,shutil
def main():
 lease=lock('backup_progress_v2.lock');assert not (C/'BACKUP_PROGRESS_V2_COMPLETE.json').exists()
 H=Path('/mnt/hdd1t/khangle/facevitals_backup_20261112/addenda');K=Path('/mnt/samsung_t5/khangle/facevitals_backup_20261112/addenda')
 assert os.stat(H).st_dev!=os.stat(C).st_dev and os.stat(H).st_dev!=os.stat(K).st_dev
 files=[]
 for folder in ['batch','routes','production_sources','supplied','tool_validation']:
  files.extend(f for f in (C/folder).rglob('*') if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts)
 for branch in ['ladder','physical_v3/ladder','reference_clock','localization_all','orientation_localization']:
  root=C/branch
  completed=[f.parent for f in root.glob('*/*/COMPLETE.json')] if 'ladder' in branch or branch=='reference_clock' else [f.parent for f in root.glob('*/COMPLETE.json')]
  for folder in completed:files.extend(f for f in folder.rglob('*') if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts)
 files+=list(C.glob('*.py'))+list(C.glob('*.md'))+list(C.glob('*.json'))
 for n in ['PROTOCOL.json','BATCH_COMPLETE.json','ROUTE_COMPLETE.json','V5_TARGET_EVIDENCE.json','MODEL_LOAD_RECEIPT.json','PRODUCTION_SOURCES.json','PACKAGE_RECEIPT.json','CURRENT_ANNOTATION_SELECTED.json','CURRENT_ANNOTATION_AUDIT.json','WORKBOOK_FETCH.json']:
  if (C/n).exists():files.append(C/n)
 files=sorted(set(files));size=sum(f.stat().st_size for f in files)
 for p in [H,K]:assert shutil.disk_usage(p).free>20_000_000_000+size
 name='BTC_CAUSE_EXPORT_V1_PROGRESS2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip';dest=H/name;assert not dest.exists();entries=[]
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for f in files:
   data=f.read_bytes();rel=str(f.relative_to(C));entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));z.writestr(rel,data)
  z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(utc=now(),partial=True,pending=['ladder','metadata_complete','full_spatial_AB','paired_team_capture','scientific_review'],files=entries),indent=2))
 with zipfile.ZipFile(dest) as z:
  for a in entries:assert hashlib.sha256(z.read(a['path'])).hexdigest()==a['sha256']
 digest=sha(dest);copy=K/name;assert not copy.exists();shutil.copyfile(dest,copy);assert sha(copy)==digest
 receipt=dict(utc=now(),files=len(entries),source_bytes=size,archive_bytes=dest.stat().st_size,archive_sha256=digest,copies=[str(dest),str(copy)],all_members_SHA256_verified=True,cross_physical_disks=True,partial=True,no_deletion=True)
 save(C/'BACKUP_PROGRESS_V2_COMPLETE.json',receipt);save(H/(name+'.json'),receipt);save(K/(name+'.json'),receipt)
if __name__=='__main__':main()
