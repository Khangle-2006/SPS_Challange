from pathlib import Path
import tarfile,json,hashlib,shutil
P=Path(__file__).resolve().parent
r=json.loads((P/'REPORT_ARCHIVE.json').read_text())
a=P/r['file'];assert hashlib.sha256(a.read_bytes()).hexdigest()==r['sha256']
with tarfile.open(a) as t:
    for m in t.getmembers():
        assert m.isfile() and not m.issym() and not m.islnk()
        assert (P/m.name).resolve().is_relative_to(P.resolve())
    for m in t.getmembers():
        dest=P/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with t.extractfile(m) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
print('Archive SHA verified and safely extracted',r['files'])
