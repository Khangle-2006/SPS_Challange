"""Verify the released file hashes; optional synthetic CPU checkpoint smoke test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('Invalid or empty manifest')
    seen = set()
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in row['path'] or ':' in row['path'] or row['path'] in seen:
            raise ValueError('Invalid manifest path')
        p = (bundle/rel).resolve()
        if bundle not in p.parents or not p.is_file():raise ValueError(f'Missing/unsafe file: {rel}')
        seen.add(row['path'])
        if p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Hash mismatch: {rel}')
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != seen|{'MANIFEST.json'}:raise ValueError('Unexpected or missing files')
    return len(seen)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--archive',type=Path,help='Also check ZIP against the hash pinned in the repository')
    a=p.parse_args()
    if a.archive:
        release=json.loads((Path(__file__).resolve().parents[1]/'configs/inference-assets-20261123.json').read_text())
        if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=release['sha256']:
            raise ValueError('ZIP differs from the pinned release hash')
        print('PASS: ZIP matches the release hash pinned in the repository')
    print(f'PASS: {verify(a.bundle)} asset file hashes')
    if a.smoke:
        for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ[k]=v
        sys.dont_write_bytecode=True
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0,str(a.bundle.resolve()/'code'))
        from EfficientPhys_fixed import EfficientPhys
        from replicate_tsm import use_replicate_padding
        x=torch.randn((1,301,3,36,36),generator=torch.Generator().manual_seed(0))
        for fold in range(5):
            data=torch.load(a.bundle/'checkpoints'/f'padding_fixed_fold{fold}_seed2027.pt',map_location='cpu',weights_only=True)
            assert set(data)=={'model','epoch'} and data['epoch']==30
            m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36)).eval()
            m.load_state_dict(data['model'],strict=True)
            with torch.inference_mode():y=m(x);s=m(torch.zeros_like(x))
            assert y.shape==(1,300) and torch.isfinite(y).all()
            assert torch.isfinite(s).all() and s.double().std(unbiased=False)<1e-10
            print(f'fold{fold}: strict load, finite synthetic waveform, static flat waveform PASS')
        print('Synthetic checks only; not a real-video end-to-end/accuracy validation.')

if __name__=='__main__':main()
