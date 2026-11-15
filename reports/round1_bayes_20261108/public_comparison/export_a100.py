"""Complete-only small feature relay package. No media/checkpoints or holdout."""
import json,zipfile,hashlib,gzip
from pathlib import Path
P=Path(__file__).resolve().parent
assert (P/'FEATURES_COMPLETE_A100.json').exists() and (P/'COMPLETE_A100.json').exists()
assert not (P/'A100_FEATURES.zip').exists(), 'Preserve previously exported bundle'
features=sorted((P/'features/UBFC2').glob('*.gz'));assert len(features)==70
for feature in features:
    with gzip.open(feature,'rt',encoding='utf8') as f:obj=json.load(f)
    assert obj['rows'], 'expected eligible public windows'
    first=obj['rows'][0];d=P/'traces/UBFC2'/first['video']/first['condition']
    receipt=json.loads((d/'PUBLIC_COMPLETE.json').read_text())
    for name,key in [('trace.npz','trace_sha256'),('clock.npz','clock_sha256')]:
        assert hashlib.sha256((d/name).read_bytes()).hexdigest()==receipt[key]==obj[key]
files=features+[P/'FEATURES_COMPLETE_A100.json',P/'COMPLETE_A100.json']
files+=sorted((P/'traces/UBFC2').rglob('PUBLIC_COMPLETE.json'))
assert len(files)==142
assert sum(x.stat().st_size for x in P.rglob('*') if x.is_file())+sum(x.stat().st_size for x in files)+100000<20_000_000, 'A10020MB cap with archive headroom'
manifest={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(P/'A100_EXPORT_FILES.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(P/'A100_FEATURES.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files+[P/'A100_EXPORT_FILES.json']:z.write(p,p.relative_to(P).as_posix())
print(json.dumps(dict(bytes=(P/'A100_FEATURES.zip').stat().st_size,sha256=hashlib.sha256((P/'A100_FEATURES.zip').read_bytes()).hexdigest(),files=len(files))))
