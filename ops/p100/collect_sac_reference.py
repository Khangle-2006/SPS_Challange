import urllib.request,json,hashlib
from pathlib import Path
out=Path('papers/pdf');out.mkdir(exist_ok=True)
dest=out/'13_SAC_2606.04198.pdf'
url='https://arxiv.org/pdf/2606.04198'
if not dest.exists():
    data=urllib.request.urlopen(url,timeout=60).read()
    assert data.startswith(b'%PDF');dest.write_bytes(data)
data=dest.read_bytes()
Path('reports/mcd_mapping_20261030/sac_reference.json').write_text(json.dumps({
    'url':url,'local_pdf':str(dest),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
    'status':'Preprint; abstract reviewed only in this task, full text not yet audited; no reproduction'},indent=2))
print(len(data))
