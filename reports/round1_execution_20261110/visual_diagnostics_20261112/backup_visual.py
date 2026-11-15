"""Versioned cross-physical-disk addendum. Never removes or overwrites a backup."""
from pathlib import Path
import json,hashlib,tarfile,shutil,os,datetime
P=Path(__file__).resolve().parent
H=Path('/mnt/hdd1t/khangle/facevitals_backup_20261112/addenda')
K=Path('/mnt/samsung_t5/khangle/facevitals_backup_20261112/addenda')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert os.stat(H).st_dev!=os.stat(P).st_dev
assert os.stat(H).st_dev!=os.stat(K).st_dev
for name in ['MODEL_COMPLETE.json','SIGNAL_COMPLETE.json','CODEC_V3_COMPLETE.json']:
    assert json.loads((P/name).read_text())['errors']==[]
files=sorted(p for p in P.rglob('*') if p.is_file() and not p.is_symlink())
size=sum(p.stat().st_size for p in files)
assert size<2_000_000_000
for root in [H,K]:assert shutil.disk_usage(root).free>20_000_000_000+size
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
name='VISUAL_DIAGNOSTICS_'+stamp+'.tar.gz';a=H/name;b=K/name
assert not a.exists() and not b.exists()
manifest=[dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
mf=P/'VIS_BACKUP_SOURCE_MANIFEST.json';assert not mf.exists();mf.write_text(json.dumps(manifest,indent=2))
with tarfile.open(a,'w:gz',compresslevel=1) as t:
    for p in files+[mf]:t.add(p,arcname='visual_diagnostics_20261112/'+str(p.relative_to(P)),recursive=False)
# Verify every archived member against the measured source, not merely copy success.
expected={m['path']:m['sha256'] for m in manifest};expected[mf.name]=sha(mf)
with tarfile.open(a) as t:
    for m in t.getmembers():
        rel=m.name.split('/',1)[1];f=t.extractfile(m);h=hashlib.sha256()
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==expected.pop(rel)
assert not expected
digest=sha(a);shutil.copyfile(a,b);assert sha(b)==digest
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(P),source_files=len(files)+1,source_bytes=size+mf.stat().st_size,archive_bytes=a.stat().st_size,archive_sha256=digest,copies=[str(a),str(b)],all_archive_member_hashes_verified=True,cross_physical_disks=True,no_deletion=True)
for out in [P/'VIS_BACKUP_COMPLETE.json',H/(name+'.json'),K/(name+'.json')]:out.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
