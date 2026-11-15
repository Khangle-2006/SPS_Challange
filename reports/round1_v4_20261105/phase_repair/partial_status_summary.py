"""Read-only partial comparison; never changes training or selects checkpoints."""
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
from statistics import mean
from datetime import datetime, timezone

root = Path('/mnt/samsung_t5/khangle/facevitals/reports')
out = root/'round1_v4_20261105/v4_repaired'
base = json.loads((root/'round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json').read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):
    return r['subject_id'], r['recording_id'], float(r['window_start_s'])
results=[]
for path in sorted(out.glob('fold*/seed*/joint/status.json')):
    status=json.loads(path.read_text())
    if status['state']!='complete':
        continue
    assert status['completed_epochs']==30
    run=path.parent
    fold=int(run.parent.parent.name[4:]); seed=int(run.parent.name[4:])
    conditions={}
    for folder, label in [('MCD_native','MCD native'),('UBFC2_f10_b500','UBFC C5'),('UBFC2_native','UBFC clean'),('MCD_f10_b500','MCD-C5')]:
        d=run/'evaluation'/folder
        if not d.exists():
            continue
        reuse=json.loads((d/'REFERENCE_REUSE.json').read_text())
        assert sha(reuse['path'])==reuse['sha256']
        assert sha(reuse['raw_path'])==reuse['raw_sha256']
        metrics=json.loads((d/'metrics.json').read_text())
        assert sha(d/'processed_output.csv')==metrics['output_sha256']
        assert metrics['reference_sha256']==reuse['sha256']
        old={key(r):r for r in base if r['branch']=='v3_joint' and r['fold']==fold and r['seed']==seed and r['condition']==label}
        rows=list(csv.DictReader((d/'processed_output.csv').open()))
        assert len(rows)==len(old) and {key(r) for r in rows}==set(old)
        errors=defaultdict(list); previous=defaultdict(list); ceilings=0
        for r in rows:
            b=old[key(r)]
            accepted=b['reference_eligible'] and r['abstain'].lower()=='false'
            assert accepted==b['accepted']
            if accepted:
                h=float(r['hr_bpm']); assert math.isfinite(h)
                errors[r['subject_id']].append(abs(h-b['reference_hr_bpm']))
                previous[r['subject_id']].append(abs(b['prediction_hr_bpm']-b['reference_hr_bpm']))
                ceilings+=h>=179
        v=mean(map(mean,errors.values()))
        assert abs(v-metrics['subject_macro']['mae'])<1e-6
        conditions[label]=dict(v4=v,v3=mean(map(mean,previous.values())),people=len(errors),windows=sum(map(len,errors.values())),ceilings=ceilings)
    results.append(dict(fold=fold,seed=seed,conditions=conditions))
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),partial=True,completed=len(results),total=15,runs=results),indent=2))

