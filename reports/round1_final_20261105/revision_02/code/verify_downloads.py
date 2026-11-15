import csv,json,tarfile,hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]);receipt={}
for host in ('P100','A100'):
    a=r/f'final_receipts_{host}.tar.gz';rows=list(csv.DictReader((r/f'REMOTE_FILES_{host}.csv').open()))
    h=next(x['sha256'] for x in rows if x['path'].endswith(a.name));assert hashlib.sha256(a.read_bytes()).hexdigest()==h
    dest=r/('receipts_'+host);dest.mkdir(exist_ok=False)
    with tarfile.open(a) as t:
        for m in t.getmembers():
            assert not m.issym() and not m.islnk() and (dest/m.name).resolve().is_relative_to(dest.resolve())
            t.extract(m,dest)
    manifest=json.loads((dest/f'RECEIPT_HASHES_{host}.json').read_text())
    for p,sha in manifest.items():assert hashlib.sha256((dest/p).read_bytes()).hexdigest()==sha,p
    receipt[host]=dict(archive_sha256=h,verified_files=len(manifest),passed=True)
for name in ('COHORT_COUNTS.json','COMMON_SUPPORT_ASSERTIONS.json','SENSOR_CADENCE_CLARIFICATION.json'):
    p=r/name;assert not p.exists();p.write_bytes((r/'receipts_P100'/name).read_bytes())
(r/'DOWNLOAD_VALIDATION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
