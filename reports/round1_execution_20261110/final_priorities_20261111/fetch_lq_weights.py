import urllib.request,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent/'sources';repo=P/'LQ-rPPG'
tree=json.loads((P/'LQ-rPPG_TREE.json').read_text());commit=next(x['commit'] for x in json.loads((P/'SOURCE_RECEIPTS.json').read_text()) if x['repo'].endswith('/LQ-rPPG'))
files=[f for f in tree['tree'] if f['type']=='blob' and (f['path'].startswith('PreTrainedModels/MMPD_MMPD_') or f['path'] in ['setup/mamba/mamba_ssm/modules/mamba_simple.py','setup/mamba/mamba_ssm/ops/selective_scan_interface.py','setup/mamba/LICENSE','neural_methods/loss/NegPearsonLoss.py','dataset/haarcascade_frontalface_default.xml','requirements.txt','setup.sh'])]
assert sum(f['size'] for f in files)<10_000_000
receipts=[]
for f in files:
    url=f"https://raw.githubusercontent.com/junseonglee-repo/LQ-rPPG/{commit}/{f['path']}"
    data=urllib.request.urlopen(url,timeout=60).read();assert len(data)==f['size']
    assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==f['sha']
    target=repo/f['path'];target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(data)
    receipts.append(dict(path=f['path'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob_verified=True,url=url))
(P/'LQ_DOWNLOAD_RECEIPT.json').write_text(json.dumps(dict(commit=commit,files=receipts,total_bytes=sum(x['bytes'] for x in receipts),code_license='MIT',weights_not_yet_executed=True),indent=2));print('Downloaded',len(receipts),sum(x['bytes'] for x in receipts))
