import json,hashlib,time
from pathlib import Path
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=ROOT.parent.parent/'storage_pilot_20261030'
manifest={f['path']:f for f in json.loads((ROOT/'download_manifest.json').read_text())['files']}
rows=json.loads((OUT/'clock_audit.json').read_text())
for r in rows:
    f=ROOT/r['video'];h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    mark=json.loads(Path(str(f)+'.verified.json').read_text())
    lines=(ROOT/r['meta']).read_text().splitlines()
    r.update(metadata_nonempty_rows=sum(bool(x.strip()) for x in lines),
        local_sha256=h.hexdigest(),manifest_sha256=manifest[r['video']]['lfs_sha256'],
        sha_matches_marker=h.hexdigest()==mark['sha256'],
        sha_matches_publisher=h.hexdigest()==manifest[r['video']]['lfs_sha256'])
    r['frame_deficit']=r['metadata_nonempty_rows']-r['pyav_frames']
(OUT/'clock_audit.json').write_text(json.dumps(rows,indent=2))
total_meta=sum(sum(bool(x.strip()) for x in p.open()) for p in (ROOT/'meta').glob('*.txt'))
summary={'state':'blocked_alignment_audit','records_audited':len(rows),
    'all_six_match_publisher_sha256':all(r['sha_matches_publisher'] for r in rows),
    'metadata_nonempty_total':total_meta,
    'nominal_uint8_crop36_bytes_from_metadata':total_meta*36*36*3,
    'note':'Frame count is metadata-based, not a validated decoded frame total; no compressed crop size established. Raw deletion disabled.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
status=json.loads((ROOT/'download_status.json').read_text())
status.update(state='paused_alignment_audit',updated=time.time(),reason='User requested streaming storage. Six pilot videos require frame/clock alignment audit before raw pruning.')
(ROOT/'download_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(summary))
print(json.dumps([{k:r[k] for k in ['video','metadata_nonempty_rows','pyav_frames','frame_deficit','sha_matches_publisher']} for r in rows]))
