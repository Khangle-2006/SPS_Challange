"""Verify archived source hashes without importing/running experiment modules."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'docs/SOURCE_MANIFEST.json').read_text())
errors = []
for row in manifest['files']:
    path = ROOT / row['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append(row['path'])
for path in ROOT.rglob('*.py'):
    if '.venv' in path.parts or '.git' in path.parts:
        continue
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
if errors:
    raise SystemExit('Source mismatch: ' + ', '.join(errors))
print(f"PASS: {len(manifest['files'])} archived source/config/license hashes; Python syntax parsed.")
