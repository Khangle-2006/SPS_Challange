"""Explicit, hashed dev inputs for joint-reader fitting and replay; no root scans."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from reader_v4 import MODELS

VIDEOS = 'w2_btc_20261116/export_extracted/VIDEOS.json'
META = 'w2_btc_20261116/delivery/claude_server_20261118/dev_video_meta.csv'
NATIVE = 'matlab_native_20261124/fixtures/btc_native.npz'
SHIFTED = 'post_s1_improvement/dev_tta/DEV_SPECTRAL_VIEWS_PRIVATE.npz'
ROUTER_REFERENCE = 'post_s1_gradient_20261124/headroom/DEV_PRIVATE.npz'
FALLBACK_REFERENCE = 'post_s1_round2/fallback/DEV_OOF.npz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid bundle path: {name}')
    root = Path(root).resolve()
    result = (root / name).resolve()
    if root not in result.parents:
        raise ValueError(f'Bundle path escapes reports root: {name}')
    return result


def load_videos(root):
    rows = json.loads(resolve(root, VIDEOS).read_bytes())
    if not isinstance(rows, list) or len(rows) != 218:
        raise ValueError('Expected the locked 218-video BTC dev cohort')
    seen, person_folds = set(), {}
    for row in rows:
        key = row['recording']
        if not isinstance(key, str) or not key or PurePosixPath(key).name != key or any(c in key for c in '\\:') or key in {'.', '..'} or key in seen:
            raise ValueError('Invalid or duplicate dev recording key')
        seen.add(key)
        fold = row['fold']
        if type(fold) is not int or fold not in range(5):
            raise ValueError('Expected dev fold 0..4')
        person = row['person']
        if person in person_folds and person_folds[person] != fold:
            raise ValueError('A person occurs in multiple folds')
        person_folds[person] = fold
        if not np.isfinite(row['pr_bpm']):
            raise ValueError('Non-finite dev reference')
    if len(person_folds) != 158 or set(person_folds.values()) != set(range(5)):
        raise ValueError('Expected 158 dev people in all five folds')
    return rows


def required_paths(rows, purposes):
    if not purposes or len(set(purposes)) != len(purposes) or not set(purposes) <= {'fit', 'replay'}:
        raise ValueError('Purpose must be fit, replay, or both')
    paths = [VIDEOS, META, NATIVE]
    paths += ['w2_btc_20261116/export_extracted/blocks/' + r['recording'] + '.npz' for r in rows]
    if 'fit' in purposes:
        paths.append(SHIFTED)
    if 'replay' in purposes:
        paths += [ROUTER_REFERENCE, FALLBACK_REFERENCE]
    return sorted(paths)


def create_manifest(root, output, purposes):
    output = Path(output)
    if output.exists():
        raise ValueError('Manifest output already exists; choose a new path')
    rows = load_videos(root)
    files = []
    for name in required_paths(rows, purposes):
        path = resolve(root, name)
        if not path.is_file():
            raise ValueError(f'Missing dev asset: {name}')
        files.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
    doc = dict(schema_version=1, kind='btc-joint-dev', scope='dev-only',
               purposes=sorted(purposes), videos=218, people=158, files=files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    return doc


def verify(root, manifest, purpose):
    doc = json.loads(Path(manifest).read_bytes())
    if (doc.get('schema_version'), doc.get('kind'), doc.get('scope'), doc.get('videos'), doc.get('people')) != (1, 'btc-joint-dev', 'dev-only', 218, 158):
        raise ValueError('Expected a version 1 BTC joint dev-only manifest (218 videos/158 people)')
    purposes = doc.get('purposes', [])
    if purpose not in purposes:
        raise ValueError(f'Bundle is not declared complete for {purpose}')
    seen = set()
    for row in doc['files']:
        name = row['path']
        if name in seen:
            raise ValueError(f'Duplicate bundle path: {name}')
        seen.add(name)
        p = resolve(root, name)
        if not p.is_file():
            raise ValueError(f'Missing dev asset: {name}')
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError(f'Dev asset hash mismatch: {name}')
    if VIDEOS not in seen:
        raise ValueError('Dev video list is missing from the manifest')
    rows = load_videos(root)
    if seen != set(required_paths(rows, purposes)):
        raise ValueError('Bundle file list is incomplete or contains unexpected files')
    return rows


def load_fit_inputs(root, manifest):
    from .fallback_reader import capture_features, trace_features
    rows = verify(root, manifest, 'fit')
    with np.load(resolve(root, NATIVE), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ('C', 'P', *MODELS, 'person', 'fold', 'truth', 'low', 'task')}
    for field, key in [('person', 'person'), ('fold', 'fold'), ('truth', 'pr_bpm')]:
        if not np.array_equal(data[field], [r[key] for r in rows]):
            raise ValueError(f'Native cache and dev list disagree on {field}')
    for flag in ['low', 'task']:
        if data[flag].shape != (218,) or not np.isin(data[flag], [0, 1]).all():
            raise ValueError(f'Invalid native {flag} vector')
    data['low'], data['task'] = data['low'].astype(int), data['task'].astype(bool)
    expected_task = [r['task'].startswith('Task') for r in rows]
    if not np.array_equal(data['task'], expected_task):
        raise ValueError('Activity metadata and native cache disagree')
    with np.load(resolve(root, SHIFTED), allow_pickle=False) as z:
        shifted = z['shift10']
    if shifted.shape != (218, 2, 277):
        raise ValueError('Expected shift10 spectra with shape (218, 2, 277)')
    data['sC'], data['sP'] = shifted[:, 0], shifted[:, 1]
    for key in ('C', 'P', 'sC', 'sP', *MODELS):
        a = data[key]
        if a.shape != (218, 277) or not np.isfinite(a).all() or (a < 0).any() or (a.sum(1) <= 0).any():
            raise ValueError(f'Invalid dev spectra: {key}')
    with resolve(root, META).open(newline='', encoding='utf-8-sig') as f:
        metadata_rows = list(csv.DictReader(f))
    metadata = {r['recording']: r for r in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError('Duplicate capture metadata recording')
    capture, trace = [], []
    for i, row in enumerate(rows):
        key = row['recording']
        m = metadata[key]
        bit_rate = int(m['bit_rate'])
        if (bit_rate < 4e6) != bool(data['low'][i]):
            raise ValueError('Capture class disagrees with original bitrate')
        capture.append(capture_features(bit_rate, int(m['width']), int(m['height']), float(m['avg_fps']), m['codec']))
        path = resolve(root, 'w2_btc_20261116/export_extracted/blocks/' + key + '.npz')
        with np.load(path, allow_pickle=False) as z:
            t, valid, rgb, boxes = (z[k] for k in ('t', 'valid', 'rgb_mean', 'boxes'))
            if t.ndim != 1 or len(t) < 2 or valid.shape != t.shape or rgb.shape != (len(t), 3) or boxes.shape != (len(t), 4) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
                raise ValueError('Invalid dev trace dimensions or clock')
            trace.append(trace_features(t, valid, rgb, boxes))
    data['capture'], data['trace'] = np.stack(capture), np.stack(trace)
    return data
