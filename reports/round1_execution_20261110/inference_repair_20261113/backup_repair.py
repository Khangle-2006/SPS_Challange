"""Preserve completed inference evidence across disks, then a final versioned addendum."""
from common_repair import *
import tarfile,shutil,time,traceback
H=Path('/mnt/hdd1t/khangle/facevitals_backup_20261112/addenda')
K=Path('/mnt/samsung_t5/khangle/facevitals_backup_20261112/addenda')

def archive(stage):
 assert os.stat(H).st_dev!=os.stat(P).st_dev and os.stat(H).st_dev!=os.stat(K).st_dev
 if stage=='baseline':
  files=[]
  for name in ['original','classical_original','legacy_sources','CHROM_BAND_ATTEMPT1','OPERATIONAL_ATTEMPT1','preflight_visual']:
   files.extend(f for f in (P/name).rglob('*') if f.is_file() and not f.is_symlink())
  files += list(P.glob('*.py'))
  files += [P/name for name in ['PROTOCOL.md','FROZEN.json','MANIFEST.json','READOUT_NATIVE_COMPLETE.json','READOUT_PUBLIC_COMPLETE.json','BASELINE_AUDIT.json','REFERENCE_06.json','REFERENCE_COMPLETE.json','READOUT_PROBE.json','READOUT_PROBE_WAVES.npz','LEGACY_RELAY_COMPLETE.json']]
 else:
  assert (P/'REVIEW_READY.json').exists();files=[f for f in P.rglob('*') if f.is_file() and not f.is_symlink() and not f.name.startswith('REPAIR_BACKUP_')]
 files=sorted(set(files));size=sum(f.stat().st_size for f in files);assert size<2_000_000_000
 for root in [H,K]:assert shutil.disk_usage(root).free>20_000_000_000+size
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');name='INFERENCE_REPAIR_'+stage.upper()+'_'+stamp+'.tar.gz';a=H/name;b=K/name;assert not a.exists() and not b.exists()
 manifest=[dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f)) for f in files];mf=P/('REPAIR_BACKUP_'+stage.upper()+'_SOURCE_MANIFEST.json');assert not mf.exists();save(mf,manifest)
 with tarfile.open(a,'w:gz',compresslevel=1) as tar:
  for f in files+[mf]:tar.add(f,arcname=P.name+'/'+str(f.relative_to(P)),recursive=False)
 expected={r['path']:r['sha256'] for r in manifest};expected[mf.name]=sha(mf)
 with tarfile.open(a) as tar:
  for member in tar.getmembers():
   stream=tar.extractfile(member);digest=hashlib.sha256()
   for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
   assert digest.hexdigest()==expected.pop(member.name.split('/',1)[1]),member.name
 assert not expected
 digest=sha(a);shutil.copyfile(a,b);assert sha(b)==digest
 receipt=dict(utc=now(),stage=stage,source_files=len(files)+1,source_bytes=size+mf.stat().st_size,archive_bytes=a.stat().st_size,archive_sha256=digest,copies=[str(a),str(b)],all_members_verified=True,cross_physical_disks=True,no_deletion=True,whole_repair_complete=stage=='final')
 for dest in [P/('REPAIR_BACKUP_'+stage.upper()+'_COMPLETE.json'),H/(name+'.json'),K/(name+'.json')]:save(dest,receipt)
 return receipt

def main():
 lease=lock('repair_backup.lock');assert not (P/'REPAIR_BACKUP_START.json').exists();save(P/'REPAIR_BACKUP_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));assert (P/'ORIGINAL_COMPLETE.json').exists() and (P/'CLASSICAL_ORIGINAL_COMPLETE.json').exists() and (P/'LEGACY_RELAY_COMPLETE.json').exists();archive('baseline');began=time.time()
 while not (P/'REVIEW_READY.json').exists():
  assert time.time()-began<36*3600,'36H_BOUND';save(P/'REPAIR_BACKUP_STATUS.json',dict(utc=now(),state='completed_evidence_baseline_verified_waiting_final',final_not_complete=True));time.sleep(60)
 archive('final')
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'REPAIR_BACKUP_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
