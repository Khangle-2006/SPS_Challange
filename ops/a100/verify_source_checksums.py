"""Compare all manifest files with authenticated Drive metadata; no network here."""
import hashlib
import json
from pathlib import Path
ROOT = Path('/data/khangle/facevitals')
DATA = ROOT / 'datasets/UBFC-rPPG/DATASET_2'
OUT = ROOT / 'results/ubfc_pos_green_v1'
manifest = json.loads((DATA.parent / 'drive_manifest.json').read_text())
source = json.loads((ROOT / 'logs/ubfc_source_metadata.json').read_text())
remote = {r['Path']: r for r in source if not r.get('IsDir')}
expected = {r['path'] for r in manifest}
assert expected == set(remote), {'missing': sorted(expected - set(remote)), 'extra': sorted(set(remote) - expected)}
results = []
for item in manifest:
    name = item['path']
    src = remote[name]
    dest = DATA / name
    hashes = {k.lower(): v for k, v in src.get('Hashes', {}).items() if v}
    assert hashes.get('md5'), 'Missing source MD5: ' + name
    h = hashlib.md5()
    with dest.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    record = {'path': name, 'id_matches': src.get('ID') == item['id'],
              'bytes': dest.stat().st_size, 'source_bytes': src['Size'],
              'md5': h.hexdigest(), 'source_md5': hashes['md5']}
    record['ok'] = record['id_matches'] and record['bytes'] == record['source_bytes'] and record['md5'] == record['source_md5']
    results.append(record)
    print(json.dumps({'done': len(results), 'path': name, 'ok': record['ok']}), flush=True)
report = {'expected': 84, 'matched': sum(r['ok'] for r in results), 'complete': all(r['ok'] for r in results), 'files': results}
(OUT / 'source_checksum_verification.json').write_text(json.dumps(report, indent=2))
summary_path = OUT / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    summary['source_checksum_status'] = {'matched': report['matched'], 'expected': 84, 'complete': report['complete'], 'algorithm': 'MD5', 'identity_and_size_checked': True}
    summary_path.write_text(json.dumps(summary, indent=2))
raise SystemExit(0 if report['complete'] else 2)
