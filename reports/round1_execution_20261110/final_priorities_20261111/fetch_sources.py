"""Retrieve source metadata only, never execute third-party setup."""
import urllib.request,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent/'sources';P.mkdir(exist_ok=True)
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FaceVitals-source-audit'}),timeout=40).read()
receipts=[]
for repo,branch in [('junseonglee-repo/LQ-rPPG','main'),('ksyegorov/mcd_rppg','main'),('phuselab/pyVHR','master')]:
    head=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
    tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{head}?recursive=1'))
    name=repo.split('/')[-1];(P/(name+'_TREE.json')).write_text(json.dumps(tree,indent=2))
    files=[]
    for f in tree['tree']:
        path=f['path']
        wanted=path in ['README.md','LICENSE','configs/MMPD_MMPD_LQC2F.yaml','dataset/data_loader/BaseLoader.py','dataset/data_loader/MMPDLoader.py','neural_methods/trainer/LQC2FTrainer.py','neural_methods/model/LQC2F.py','pyVHR/BVP/methods.py']
        if wanted:
            data=get(f'https://raw.githubusercontent.com/{repo}/{head}/{path}')
            target=P/name/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            files.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    receipts.append(dict(repo=repo,commit=head,files=files))
(P/'SOURCE_RECEIPTS.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps(receipts,indent=2))
