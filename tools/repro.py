"""Prepare a private, relocated research workspace without editing archived sources.

No downloads, model fitting or inference occur in init/doctor/prepare/plan.
Only `run` executes an explicitly named research script, on CPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WREL = Path('reports/claude_research_20261119_constraints')
OLD_ROOTS = ('/mnt/samsung_t5/khangle/facevitals', '/data_pool/khangle/facevitals', '/data/khangle/facevitals')
PROFILES = ('core', 'frontend', 'torch-cpu', 'experts')
# Explicit entrypoints, not training/controllers. Arguments remain the archived CLI.
STAGES = {
    'btc-reader-v3': ('core', 'reports/claude_research_20261118/code/a11_reader_v3.py'),
    'btc-front': ('frontend', str(WREL / 'code/x01_extract.py')),
    'box-crops': ('frontend', str(WREL / 'code/x10_boxcrop.py')),
    'btc-spectra': ('core', str(WREL / 'code/x03_btc_spectra.py')),
    'btc-experts': ('experts', str(WREL / 'code/y05_openrppg.py')),
    'btc-conditions': ('experts', str(WREL / 'code/y11_openrppg_conditions.py')),
    'btc-score': ('core', str(WREL / 'code/y11_openrppg_conditions.py')),
    'public-cnn': ('torch-cpu', str(WREL / 'code/x02_infer.py')),
    'public-experts': ('experts', str(WREL / 'code/y12_openrppg_public.py')),
    'public-conditions': ('experts', str(WREL / 'code/y13_openrppg_public_conditions.py')),
    'public-score': ('core', str(WREL / 'code/z13_public_p4_prefilter.py')),
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def relative_path(value):
    # Reject both POSIX and Windows traversal even when running on Linux.
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Asset destination must be a relative POSIX path')
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('reports', 'code', 'data', 'checkpoints'):
        raise ValueError(f'Unsafe asset destination: {value}')
    if any('holdout' in part.lower() for part in p.parts):
        raise ValueError('Holdout assets are outside this dev-only launcher')
    return p

def config(path):
    path = Path(path).resolve()
    c = read_json(path)
    if c.get('schema_version') != 1:
        raise ValueError('Unsupported config schema')
    def resolve(s):
        return (path.parent / Path(s).expanduser()).resolve()
    c['workspace'] = resolve(c['workspace'])
    c['assets_manifest'] = resolve(c['assets_manifest'])
    c['python'] = {k: str(resolve(v)) for k, v in c['python'].items()}
    if c['workspace'] == ROOT or ROOT in c['workspace'].parents or c['workspace'] in ROOT.parents:
        raise ValueError('Workspace must be outside the checkout, not its ancestor')
    # Known production roots must never be used as a writable workspace.
    for root in OLD_ROOTS:
        if str(c['workspace']) == root:
            raise ValueError('Refusing the original production root')
    return c

def mappings(c):
    w = c['workspace'].as_posix()
    d = {r: w for r in OLD_ROOTS}
    for r in OLD_ROOTS:
        d[r + '/code/t8_oval_20261031'] = w + '/ops/t8_oval_20261031'
        d[r + '/code/t8_polarity_v3_20261102'] = w + '/ops/t8_polarity_v3_20261102'
        d[r + '/envs/hr/bin/python'] = c['python']['torch-cpu']
        d[r + '/envs/storage_pilot/bin/python'] = c['python']['frontend']
    for old, new in c.get('path_map', {}).items():
        if not old.startswith('/') or not Path(new).is_absolute():
            raise ValueError('path_map entries must be absolute prefixes')
        d[old.rstrip('/')] = str(new).rstrip('/')
    return d

def relocate(text, mapping):
    # Single pass, longest prefix first: replacements cannot be replaced again.
    pattern = '(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r''')(?=/|$|[\s"';])'''
    return re.sub(pattern, lambda m: mapping[m.group()], text)

def assets(c):
    p = c['assets_manifest']
    if not p.is_file():
        return None, [f'Missing asset manifest: {p}; see docs/ASSETS.md']
    a = read_json(p)
    if a.get('schema_version') != 1 or a.get('scope') != 'dev-only':
        raise ValueError('Only a schema 1 dev-only asset bundle is accepted')
    errors, seen = [], set()
    for row in a.get('files', []):
        dest = str(relative_path(row['destination']))
        if dest in seen:
            raise ValueError(f'Duplicate asset destination: {dest}')
        seen.add(dest)
        if not re.fullmatch('[a-fA-F0-9]{64}', row.get('sha256', '')):
            errors.append(f'Expected maintainer SHA256: {dest}')
            continue
        src = (p.parent / row['source']).resolve()
        if any('holdout' in part.lower() for part in src.parts):
            raise ValueError('Holdout source is outside this dev-only launcher')
        if not src.is_file():
            errors.append(f'Missing asset: {dest}')
        elif sha(src) != row['sha256'].lower():
            errors.append(f'SHA256 mismatch: {dest}')
    if not a.get('files'):
        errors.append('Empty private asset bundle')
    return a, errors

def doctor(c, stage=None, prepared=False):
    errors = []
    profiles = [STAGES[stage][0]] if stage else list(PROFILES)
    for profile in profiles:
        if not Path(c['python'].get(profile, '')).is_file():
            errors.append(f'Missing interpreter {profile}; run tools/setup_env.py {profile}')
    if stage and STAGES[stage][0] == 'frontend':
        for exe in ('ffmpeg', 'ffprobe'):
            if not shutil.which(exe):
                errors.append(f'Missing system program: {exe}')
    a, problems = assets(c)
    errors.extend(problems)
    if stage and a and stage not in a.get('complete_for', []):
        errors.append(f'Maintainer has not marked the private bundle complete for {stage}')
    if prepared:
        receipt = c['workspace'] / 'RELOCATION.json'
        if not receipt.is_file():
            errors.append('Run prepare before plan/run')
        else:
            r = read_json(receipt)
            if r['asset_manifest_sha256'] != sha(c['assets_manifest']):
                errors.append('Asset manifest changed after prepare; use a new workspace')
            for row in r['files']:
                path = c['workspace'] / row['path']
                if not path.is_file() or sha(path) != row['prepared_sha256']:
                    errors.append(f'Prepared file changed: {row["path"]}')
    return errors

def prepare(c):
    # Validate every asset before the first write. Never merge into an existing workspace.
    a, errors = assets(c)
    if errors:
        raise ValueError('\n'.join(errors))
    w = c['workspace']
    if w.exists():
        raise ValueError('Workspace already exists; choose a new empty path (no overwrite/delete)')
    source_manifest = read_json(ROOT / 'docs/SOURCE_MANIFEST.json')
    for row in source_manifest['files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'Archived source hash mismatch: {row["path"]}')
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    copies = [p for p in tracked if p and p.startswith(('reports/', 'ops/', 'runtime/'))]
    if any(row['destination'] in copies for row in a['files']):
        raise ValueError('Asset bundle must not overwrite archived code/docs')
    mapping = mappings(c)
    w.mkdir(parents=True)
    records = []
    for rel in copies:
        src, dst = ROOT / rel, w / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        b = src.read_bytes()
        # Only operational path literals in working copies; algorithms and original archive stay unchanged.
        new = relocate(b.decode('utf-8-sig'), mapping).encode('utf-8') if src.suffix in ('.py', '.sh', '.m') else b
        dst.write_bytes(new)
        records.append({'path': rel, 'source_sha256': hashlib.sha256(b).hexdigest(),
                        'prepared_sha256': sha(dst), 'path_relocated': b != new})
    for row in a['files']:
        src = (c['assets_manifest'].parent / row['source']).resolve()
        dst = w / relative_path(row['destination'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if row.get('relocate_json_paths'):
            if src.suffix.lower() != '.json':
                raise ValueError('Only JSON metadata may request path relocation')
            b = src.read_text(encoding='utf-8-sig')
            def walk(v):
                if isinstance(v, str): return relocate(v, mapping)
                if isinstance(v, list): return [walk(x) for x in v]
                if isinstance(v, dict): return {k: walk(x) for k, x in v.items()}
                return v
            save(dst, walk(json.loads(b)))
        else:
            shutil.copyfile(src, dst)
        records.append({'path': row['destination'], 'source_sha256': row['sha256'].lower(),
                        'prepared_sha256': sha(dst), 'path_relocated': bool(row.get('relocate_json_paths'))})
    for sub in ('logs', 'out', 'scratch'):
        (w / WREL / sub).mkdir(parents=True, exist_ok=True)
    save(w / 'RELOCATION.json', {'schema_version': 1, 'scope': 'dev-only',
        'asset_manifest_sha256': sha(c['assets_manifest']), 'mapping': mapping, 'files': records,
        'warning': 'Relocated working copies are not byte-identical frozen releases. No hash assertions disabled.'})
    print(f'Prepared {len(records)} files in private workspace: {w}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/paths.local.json')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('init')
    sub.add_parser('doctor').add_argument('--stage', choices=STAGES)
    sub.add_parser('prepare')
    for name in ('plan', 'run'):
        s = sub.add_parser(name)
        s.add_argument('stage', choices=STAGES)
        s.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    try:
        if a.action == 'init':
            a.config.parent.mkdir(parents=True, exist_ok=True)
            example = read_json(ROOT/'configs/paths.example.json')
            # Resolve template paths relative to the template location, not arbitrary output directory.
            example['workspace'] = str(ROOT.parent / 'facevitals-reproduction')
            example['assets_manifest'] = str(ROOT.parent / 'private-assets/assets.json')
            for profile in PROFILES:
                example['python'][profile] = str(ROOT/'.venv'/profile/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
            with a.config.open('x', encoding='utf-8') as f:
                json.dump(example, f, indent=2); f.write('\n')
            print(f'Created {a.config}; edit private paths, then run doctor.')
            return
        c = config(a.config)
        if a.action == 'prepare':
            prepare(c); return
        errors = doctor(c, getattr(a, 'stage', None), prepared=a.action in ('plan', 'run'))
        if errors:
            raise ValueError('\n'.join(errors))
        if a.action == 'doctor':
            print('PASS: interpreters and listed asset hashes present. This is not an end-to-end pipeline validation.');return
        profile, script = STAGES[a.stage]
        command = [c['python'][profile], str(c['workspace']/script), *a.args]
        print(json.dumps({'cwd':str(c['workspace']/WREL),'command':command},ensure_ascii=False),flush=True)
        if a.action == 'run':
            if not c.get('allow_dev_reproduction'):
                raise ValueError('Set allow_dev_reproduction=true only for an authorized dev-only bundle')
            if sys.platform != 'linux':
                raise ValueError('Archived research entrypoints require Linux; use Linux/WSL2')
            mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0]) < 15*1024**2:
                raise ValueError('Need at least 15 GiB available RAM; retry later')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', KERAS_BACKEND='jax',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
            for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
            raise SystemExit(subprocess.call(['nice','-n','19',*command],cwd=c['workspace']/WREL,env=env))
    except (ValueError, KeyError, OSError) as e:
        p.exit(2, f'BLOCKED: {e}\n')

if __name__ == '__main__':
    main()
