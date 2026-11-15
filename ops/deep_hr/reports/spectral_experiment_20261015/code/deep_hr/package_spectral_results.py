import json,hashlib,tarfile,subprocess,sys
from pathlib import Path
r=Path('/data_pool/khangle/facevitals');out=r/'results/spectral_ablation_v1'
assert json.loads((out/'evaluation_complete.json').read_text())['complete']
subprocess.run([sys.executable,str(r/'code/deep_hr/diagnose_spectral_waveforms.py')],check=True)
archive=r/'results/spectral_ablation_v1_completed.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(out.rglob('*')):
    if p.is_file() and p.name!='last.pt':t.add(p,arcname=str(p.relative_to(r)))
 for pat in ['*spectral*','SPECTRAL*','diagnose_h264_waveforms.py']:
    for p in sorted((r/'code/deep_hr').glob(pat)):
        if p.is_file():t.add(p,arcname=str(p.relative_to(r)))
 for name in ['spectral_ablation_v1.log','spectral_evaluation_v1.log']:
    p=r/'logs'/name
    if p.exists():t.add(p,arcname=str(p.relative_to(r)))
 for rate in [250,1000]:
    base=r/f'datasets/UBFC-rPPG/efficientphys36_val_h264_{rate}k_v1'
    for name in ['inventory.json','encoding_report.json','clips.json','subject_split.json']:t.add(base/name,arcname=str((base/name).relative_to(r)))
with archive.open('rb') as f:print(hashlib.file_digest(f,'sha256').hexdigest())
print(archive.stat().st_size)
