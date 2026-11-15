"""Seeded, outcome-independent development and expanded audit selection."""
import csv,hashlib,json,random
from pathlib import Path
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=Path('/mnt/samsung_t5/khangle/facevitals/mapping_audit_20261030');OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'db.csv').open()))
manifest={x['path']:x for x in json.loads((ROOT/'download_manifest.json').read_text())['files']}
groups={}
for r in rows:groups.setdefault(r['patient_id'],[]).append(r)
subjects=sorted([p for p,rs in groups.items() if len(rs)==6 and p!='1020'])
random.Random(20261030).shuffle(subjects);dev=subjects[:60]
assert len(dev)==60
cams=['FullHDwebcam','USBVideo','IriunWebcam']
audit=[]
for i,p in enumerate(dev[:30]):
    camera=cams[i%3];step=['before','after'][(i//3)%2]
    matches=[r for r in groups[p] if r['camera']==camera and r['step']==step]
    assert len(matches)==1;audit+=matches
devrows=[r for p in dev for r in groups[p]]
plan={'seed':20261030,'db_sha256':hashlib.sha256((ROOT/'db.csv').read_bytes()).hexdigest(),
    'pinned_revision':'bf59b3cd1414c2a510d83c3125f16bb87d469ebb',
    'development_subjects':dev,'diagnostic_subjects_also_protected':['1020'],
    'development_records':devrows,'audit_records':audit,
    'development_raw_bytes':sum(manifest[r['video']]['size'] for r in devrows),
    'audit_raw_bytes':sum(manifest[r['video']]['size'] for r in audit),
    'audit_available_now':sum((ROOT/r['video']).exists() for r in audit),
    'development_available_now':sum((ROOT/r['video']).exists() for r in devrows),
    'delete_raw':False,'note':'30 unique people,10 per camera,5before/5after each; all60 keep all6 recordings; no outcomes used.'}
path=OUT/'development_plan.json';assert not path.exists(),'Plan is frozen; do not overwrite'
path.write_text(json.dumps(plan,indent=2))
print(json.dumps({k:v for k,v in plan.items() if k not in ['development_records','audit_records']}),flush=True)
