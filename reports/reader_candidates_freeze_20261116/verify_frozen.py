import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
a=json.loads((p/'MANIFEST.json').read_text())
bad=[name for name,h in a['sha256'].items() if not (p/name).is_file() or hashlib.sha256((p/name).read_bytes()).hexdigest()!=h]
assert not bad,('Frozen content mismatch',bad)
print('PASS',len(a['sha256']),'files; holdout remains closed')
