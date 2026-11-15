"""Archive completed H264 runs, best checkpoints, and reproducibility metadata."""
import hashlib,json,tarfile
from pathlib import Path
root=Path('/data_pool/khangle/facevitals')
base=root/'results/h264_ablation_v1'
for mode in ['control','augmentation']:
    for seed in [2027,2028,2029]:
        p=base/f'{mode}_{seed}'
        status=json.loads((p/'status.json').read_text())
        assert status['state']=='complete' and status['reload_verified']
        with (p/'best.pt').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==status['checkpoint_sha256']
out=root/'results/h264_ablation_v1_completed.tar.gz'
with tarfile.open(out,'w:gz') as tar:
    for p in sorted(base.rglob('*')):
        if p.is_file() and p.name!='last.pt':tar.add(p,arcname=str(p.relative_to(root)))
    for p in sorted((root/'code/deep_hr').glob('*h264*')):
        if p.is_file():tar.add(p,arcname=str(p.relative_to(root)))
    for p in (root/'logs/h264_controls_v1.log',root/'logs/h264_augmentation_v1.log'):
        tar.add(p,arcname=str(p.relative_to(root)))
    for cache in ['efficientphys36_trainval_v1','efficientphys36_train_h264_500k_v1','efficientphys36_val_h264_500k_v1']:
        for name in ['clips.json','subject_split.json','inventory.json','encoding_report.json']:
            p=root/'datasets/UBFC-rPPG'/cache/name
            if p.exists():tar.add(p,arcname=str(p.relative_to(root)))
with out.open('rb') as f:print(hashlib.file_digest(f,'sha256').hexdigest())
print(out.stat().st_size)
