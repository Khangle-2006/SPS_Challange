import json,hashlib,tarfile,csv,sys
from pathlib import Path
r=Path(sys.argv[1]);host=sys.argv[2]
selected=[]
for p in sorted(r.rglob('*')):
    if not p.is_file() or p.is_symlink() or 'venv' in p.parts or '__pycache__' in p.parts:continue
    if p.suffix in ('.json','.csv','.md','.log','.txt') and p.stat().st_size<5_000_000 and not p.name.startswith(('REMOTE_FILES','RECEIPT_HASHES')):selected.append(p)
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
manifest=r/f'RECEIPT_HASHES_{host}.json';assert not manifest.exists();manifest.write_text(json.dumps(hashes,indent=2))
archive=r/f'final_receipts_{host}.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in selected+[manifest]:tar.add(p,arcname=p.relative_to(r))
rows=[]
for p in sorted(r.rglob('*')):
    if not p.is_file() and not p.is_symlink():continue
    h=hashlib.sha256()
    isvenv='venv' in p.parts
    if not isvenv and not p.is_symlink():
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    rows.append(dict(host=host,path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest() if not isvenv and not p.is_symlink() else '',note='venv_or_symlink_not_hashed' if isvenv or p.is_symlink() else ''))
with (r/f'REMOTE_FILES_{host}.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),selected=len(selected),files=len(rows))))
