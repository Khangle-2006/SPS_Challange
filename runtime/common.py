"""Small runtime helpers. No dataset paths, labels or fitting at inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    'FacePhys.rlap.weights.h5': '6cd99510bcc5771f9cebcdb0f47a5cb3692d19f1e3d6e2444af5e8fac0ee3c07',
    'ME.pure.weights.h5': '2135995cb4161d1d4c7c3aa45d333dc6b1f5818b0652d11da9fc0fb0b6e11bf0',
    'state.pkl': 'd894487c5f0162dd798a8e7f8aabedb958b48cab8e4a31abdd433f97826c57f4',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify_weights():
    import importlib.util
    base = Path(importlib.util.find_spec('rppg').origin).parent/'weights'
    for name, expected in WEIGHTS.items():
        actual = sha(base/name)
        if actual != expected: raise RuntimeError(f'Expert weight hash mismatch: {name}')
    return dict(WEIGHTS)


def load_model(fold=None):
    import numpy as np
    name = 'btc_reader_v4_deploy.json' if fold is None else 'btc_reader_v4_folds.json'
    manifest = json.loads((ROOT/'models/MANIFEST.json').read_text())
    path = ROOT/'models'/name
    if sha(path) != manifest['sha256'][name]: raise RuntimeError(f'Reader hash mismatch: {name}')
    model = json.loads(path.read_text())
    if fold is not None: model = model[str(fold)]
    return dict(prior=np.asarray(model['prior'],float), beta={int(k): v for k,v in model['beta'].items()},
                off={int(k): v for k,v in model['off'].items()}), manifest['sha256'][name]
