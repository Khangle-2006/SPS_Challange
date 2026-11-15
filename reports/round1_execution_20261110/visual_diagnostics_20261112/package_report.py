from pathlib import Path
import tarfile,hashlib,json
P=Path(__file__).resolve().parent
assert json.loads((P/'SIGNAL_COMPLETE.json').read_text())['errors']==[]
files=[]
for p in P.iterdir():
    if p.is_file() and p.suffix in ['.py','.md','.json','.png','.txt','.npz']: files.append(p)
for folder in ['signal','model','data','codec_v3']:
    for p in (P/folder).rglob('*'):
        if not p.is_file():continue
        if p.name in ['clips.npz','bitstream_maps.npz']:continue
        if p.suffix in ['.json','.png','.npz']:files.append(p)
target=P/'REPORT_ARTIFACTS.tar.gz'
with tarfile.open(target,'w:gz') as t:
    for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(P)),recursive=False)
h=hashlib.sha256(target.read_bytes()).hexdigest()
receipt=dict(file=target.name,bytes=target.stat().st_size,sha256=h,files=len(set(files)))
(P/'REPORT_ARCHIVE.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
